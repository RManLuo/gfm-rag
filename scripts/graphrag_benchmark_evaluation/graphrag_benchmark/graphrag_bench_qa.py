"""
This script is used to answer the questions with the retrieved documents and entities for graphrag benchmark datasets (G-Bench: CS).
GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation.
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

from .prompt_utils import load_prompt_builders, resolve_task_type

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

    output_root = cfg.test.prediction_result_path or output_dir
    if cfg.test.prediction_result_path and os.path.splitext(output_root)[1]:
        logger.warning(
            "prediction_result_path is expected to be a directory for GraphRAG-Bench; "
            "using its parent directory instead."
        )
        output_root = os.path.dirname(output_root) or "."

    os.makedirs(output_root, exist_ok=True)

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

        # Build retrieved result dict to align with the shared prompt builder API
        retrieved_result = {"document": retrieved_docs}
        if retrieved_entities:
            retrieved_result["entity"] = retrieved_entities

        message = prompt_builder.build_input_prompt(data["question"], retrieved_result)

        # GraphRAG-Bench eval script will extract the answer from the response, so we don't need to do it here.
        response = llm.generate_sentence(message)
        if isinstance(response, Exception):
            return response
        else:
            # align with the format of the graphrag benchmark eval data
            return {
                "id": data["id"],
                "question": data["original_question"],
                "full_question": data["question"],
                "ground_truth": data["answer"],
                "prediction": response,
                "context": retrieved_docs,
                "question_type": task_type,
            }

    for task_type, data_samples in task_groups.items():
        prompt_builder = prompt_builders[task_type]
        logger.info(f"{task_type} prompt loaded")
        logger.info(f"task prompt: {prompt_builder.system_prompt}")

        # Collect all results in a list and save incrementally to JSON
        results_list: list[dict] = []
        output_file = os.path.join(
            output_root, f"{cfg.output_name_map.get(task_type, task_type)}.json"
        )

        with ThreadPool(cfg.test.n_threads) as pool:
            predict_with_prompt = partial(predict, prompt_builder=prompt_builder)
            for results in tqdm(
                pool.imap(predict_with_prompt, data_samples),
                total=len(data_samples),
            ):
                if isinstance(results, Exception):
                    logger.error(f"Error: {results}")
                    continue

                # Append to in-memory list
                results_list.append(results)

                # Immediately save the entire list to JSON (backup in case of crash)
                with open(output_file, "w") as f_json:
                    json.dump(results_list, f_json, indent=4)

        logger.info(f"Saved {len(results_list)} results to {output_file}")


@hydra.main(
    config_path="config",
    config_name="graphrag_bench_cs_qa_inference",
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
