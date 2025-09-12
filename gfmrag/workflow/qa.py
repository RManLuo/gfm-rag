import json
import logging
import os
from multiprocessing.dummy import Pool as ThreadPool

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gfmrag import utils
from gfmrag.models.ultra import query_utils
from gfmrag.prompt_builder import QAPromptBuilder

# A logger for this file
logger = logging.getLogger(__name__)


def ans_prediction(
    cfg: DictConfig,
    output_dir: str,
    documents: dict,
    retrieval_result: list[dict],
) -> str:
    llm = instantiate(cfg.llm)

    prompt_builder = QAPromptBuilder(cfg.qa_prompt)

    def predict(data: dict) -> dict | Exception:
        if "document" in data["predictions"]:
            if len(data["predictions"]["document"]) < cfg.test.top_k:
                logger.warning(
                    f"The number of retrieved documents ({len(data['predictions']['document'])}) is less than top_k ({cfg.test.top_k}) for sample id {data['id']}. Using all retrieved documents."
                )
            docs_prediction = data["predictions"]["document"][
                : min(cfg.test.top_k, len(data["predictions"]["document"]))
            ]
        else:
            raise ValueError("The retrieval results do not contain 'document' key!")

        retrieved_docs: list[dict] = []
        for doc in docs_prediction:
            title = doc[
                0
            ]  # The first element is the title, the second element is the score
            if title in documents:
                retrieved_docs.append(
                    {
                        "title": title,
                        "content": documents[title],
                        "score": doc[1],
                    }
                )
            else:
                logger.warning(
                    f"Document title {title} not found in the provided documents. Skipping this document."
                )

        retrieved_entities: list[dict] = []
        for entity in data["predictions"].get("entity", []):
            entity_name = entity[0]
            retrieved_entities.append(
                {
                    "name": entity_name,
                    "score": entity[1],
                }
            )

        message = prompt_builder.build_input_prompt(
            data["question"], retrieved_docs, retrieved_entities
        )

        response = llm.generate_sentence(message)
        if isinstance(response, Exception):
            return response
        else:
            return {
                "id": data["id"],
                "question": data["question"],
                "answer": data["answer"],
                "answer_aliases": data.get(
                    "answer_aliases", []
                ),  # Some datasets have answer aliases
                "response": response,
                "retrieved_docs": retrieved_docs,
            }

    with open(os.path.join(output_dir, "prediction.jsonl"), "w") as f:
        with ThreadPool(cfg.test.n_threads) as pool:
            for results in tqdm(
                pool.imap(predict, retrieval_result),
                total=len(retrieval_result),
            ):
                if isinstance(results, Exception):
                    logger.error(f"Error: {results}")
                    continue

                f.write(json.dumps(results) + "\n")
                f.flush()

    return os.path.join(output_dir, "prediction.jsonl")


@hydra.main(config_path="config/gfm_rag", config_name="qa_inference", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    torch.manual_seed(cfg.seed + utils.get_rank())

    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")

    if cfg.test.retrieved_result_path and os.path.exists(
        cfg.test.retrieved_result_path
    ):
        logger.info(f"Loading retrieved results from {cfg.test.retrieved_result_path}")
        with open(cfg.test.retrieved_result_path) as f:
            retrieval_result = json.load(f)
        if cfg.test.n_sample > 0:
            retrieval_result = retrieval_result[: cfg.test.n_sample]
        logger.info(f"Loaded {len(retrieval_result)} retrieval results")
    else:
        raise FileNotFoundError("Please provide the retrieved_result_path!")

    if cfg.test.document_path is None:
        raise ValueError("Please provide the document_path for QA inference!")
    else:
        with open(cfg.test.document_path) as f:
            documents = json.load(f)
        logger.info(f"Loaded {len(documents)} documents from {cfg.test.document_path}")

    output_path = ans_prediction(cfg, output_dir, documents, retrieval_result)

    # Evaluation
    evaluator = instantiate(cfg.qa_evaluator, prediction_file=output_path)
    metrics = evaluator.evaluate()
    query_utils.print_metrics(metrics, logger)
    logger.info(f"Saved prediction results to {output_path}")
    return metrics


if __name__ == "__main__":
    main()
