"""
This script is used to answer the questions with the retrieved documents and entities for graphrag bench datasets (G-Bench: Novel and Medical).
When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation
"""

import json
import logging
import os
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gfmrag import utils
from gfmrag.prompt_builder import QAPromptBuilder
from graphrag_benchmark.prompt_utils import load_prompt_builders, resolve_task_type

# A logger for this file
logger = logging.getLogger(__name__)


def ans_prediction(
    cfg: DictConfig,
    output_dir: str,
    documents: dict,
    retrieval_result: list[dict],
) -> None:
    llm = instantiate(cfg.llm)

    prompt_builders = load_prompt_builders(cfg.prompt_map)

    output_file = cfg.test.prediction_result_path
    if output_file:
        # Allow passing a directory or explicit file path.
        if output_file.endswith(os.sep) or os.path.isdir(output_file):
            os.makedirs(output_file, exist_ok=True)
            output_file = os.path.join(output_file, "prediction.jsonl")
        else:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    else:
        output_file = os.path.join(output_dir, "prediction.jsonl")
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    task_groups: dict[str, list[dict]] = {}
    for data in retrieval_result:
        task_type = resolve_task_type(data, cfg.task_type)
        task_groups.setdefault(task_type, []).append(data)

    missing_prompts = set(task_groups) - set(prompt_builders)
    if missing_prompts:
        raise ValueError(
            f"Missing prompt config for task types: {sorted(missing_prompts)}"
        )

    for task_type, data_samples in task_groups.items():
        logger.warning(task_type)
        logger.warning(len(data_samples))
        logger.warning("-" * 100)

    def predict(data: dict, prompt_builder: QAPromptBuilder) -> dict | Exception:
        # Get the appropriate prompt config for this task type
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
            # The first element is the title, the second element is the score
            title = doc[0]
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

        retrieved_result = {"document": retrieved_docs}
        if retrieved_entities:
            retrieved_result["entity"] = retrieved_entities

        message = prompt_builder.build_input_prompt(data["question"], retrieved_result)

        response = llm.generate_sentence(message)
        if isinstance(response, Exception):
            return response

        generated_answer = None
        if isinstance(response, str) and "Answer:" in response:
            generated_answer = response.split("Answer:", maxsplit=1)[1].strip()

        # align with the format of the graphrag benchmark eval data
        return {
            "id": data["id"],
            "question": data["question"],
            "ground_truth": data["answer"],
            "response": response,
            "context": retrieved_docs,
            "question_type": data["question_type"],
            "generated_answer": generated_answer,
            "evidence": data.get("evidence", []),
        }

    with open(output_file, "w") as f:
        for task_type, data_samples in task_groups.items():
            prompt_builder = prompt_builders[task_type]
            logger.info("%s prompt loaded", task_type)

            with ThreadPool(cfg.test.n_threads) as pool:
                predict_with_prompt = partial(predict, prompt_builder=prompt_builder)
                for results in tqdm(
                    pool.imap(predict_with_prompt, data_samples),
                    total=len(data_samples),
                ):
                    if isinstance(results, Exception):
                        logger.error(f"Error: {results}")
                        continue

                    f.write(json.dumps(results) + "\n")
                    f.flush()


@hydra.main(
    config_path="../gfmrag/workflow/config/gfm_rag",
    config_name="graphrag_benchmark_qa_inference",
    version_base=None,
)
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

    ans_prediction(cfg, output_dir, documents, retrieval_result)


if __name__ == "__main__":
    main()
