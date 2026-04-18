import ast
import json
import logging
import os
from multiprocessing.dummy import Pool as ThreadPool

import hydra
import pandas as pd
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
    nodes: pd.DataFrame,
    retrieval_result: list[dict],
) -> str:
    llm = instantiate(cfg.llm)

    prompt_builder = QAPromptBuilder(cfg.qa_prompt)

    def predict(data: dict) -> dict | Exception:
        retrieved_result: dict[str, list[dict]] = {}
        for target_type in cfg.test.target_types:
            if target_type not in data["predictions"]:
                raise ValueError(
                    f"The retrieval results do not contain '{target_type}' key!"
                )
            target_prediction = data["predictions"][target_type]
            if len(target_prediction) < cfg.test.top_k:
                logger.warning(
                    f"The number of retrieved {target_type}s ({len(target_prediction)}) is less than top_k ({cfg.test.top_k}) for sample id {data['id']}. Using all retrieved {target_type}s."
                )
            target_prediction = target_prediction[
                : min(cfg.test.top_k, len(target_prediction))
            ]
            retrieved_result[target_type] = []
            for item in target_prediction:
                uid = item[
                    0
                ]  # The first element is the uid or name, the second element is the score
                matched_node = nodes.loc[uid]

                retrieved_result[target_type].append(
                    {"name": matched_node.name, **matched_node.attributes}
                )
        message = prompt_builder.build_input_prompt(data["question"], retrieved_result)

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
                "retrieved_result": retrieved_result,
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

    if cfg.test.node_path is None:
        raise ValueError("Please provide the node_path for QA inference!")
    else:
        nodes = pd.read_csv(cfg.test.node_path, keep_default_na=False)
        logger.info(f"Loaded {len(nodes)} nodes from {cfg.test.node_path}")

        if "uid" in nodes.columns:
            if nodes["uid"].nunique() != len(nodes):
                raise ValueError(
                    f"The 'uid' column must contain unique values. Unique values found: {nodes['uid'].nunique()}, total rows: {len(nodes)}"
                )
            else:
                nodes = nodes.set_index("uid")
        elif "name" in nodes.columns:
            if nodes["name"].nunique() != len(nodes):
                raise ValueError(
                    f"The 'name' column must contain unique values. Unique values found: {nodes['name'].nunique()}, total rows: {len(nodes)}"
                )
            else:
                nodes = nodes.set_index("name")
        # Handle attributes
        nodes["attributes"] = nodes["attributes"].apply(
            lambda x: {} if pd.isna(x) else ast.literal_eval(x)
        )

    output_path = ans_prediction(cfg, output_dir, nodes, retrieval_result)

    # Evaluation
    evaluator = instantiate(cfg.qa_evaluator, prediction_file=output_path)
    metrics = evaluator.evaluate()
    query_utils.print_metrics(metrics, logger)
    logger.info(f"Saved prediction results to {output_path}")
    return metrics


if __name__ == "__main__":
    main()
