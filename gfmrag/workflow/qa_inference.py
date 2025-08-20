import json
import logging
import os
from multiprocessing.dummy import Pool as ThreadPool

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils import data as torch_data
from tqdm import tqdm

from gfmrag import utils
from gfmrag.graph_index_datasets import GraphIndexDatasetV1
from gfmrag.models.ultra import query_utils
from gfmrag.prompt_builder import QAPromptBuilder

# A logger for this file
logger = logging.getLogger(__name__)


@torch.no_grad()
def doc_retrieval(
    cfg: DictConfig,
    model: nn.Module,
    qa_data: GraphIndexDatasetV1,
    device: torch.device,
) -> list[dict]:
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    test_data = qa_data.test_data
    graph = qa_data.graph

    # Retrieve the supporting documents for each query
    sampler = torch_data.DistributedSampler(test_data, world_size, rank, shuffle=False)
    test_loader = torch_data.DataLoader(
        test_data, cfg.test.retrieval_batch_size, sampler=sampler
    )

    model.eval()
    all_predictions: list[dict] = []
    target_types = cfg.test.target_types
    for batch in tqdm(test_loader, disable=not utils.is_main_process()):
        batch = query_utils.cuda(batch, device=device)
        pred = model(graph, batch)

        idx = batch["sample_id"]
        preds_by_type: dict[str, torch.Tensor] = {
            target_type: [] for target_type in target_types
        }
        # Collect predictions and targets for each target type
        for target_type in target_types:
            target_node_ids = graph.nodes_by_type[target_type]  # type: ignore
            target_node_pred = pred[:, target_node_ids]  # type: ignore
            preds_by_type[target_type] = target_node_pred

        for i in range(len(idx)):
            all_predictions.append(
                {
                    "id": idx[i],
                    **{
                        target_type: p[i].cpu()
                        for target_type, p in preds_by_type.items()
                    },
                }
            )

    # Gather the predictions across all processes
    if utils.get_world_size() > 1:
        gathered_predictions = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_predictions, all_predictions)
    else:
        gathered_predictions = [all_predictions]  # type: ignore

    sorted_predictions = sorted(
        [item for sublist in gathered_predictions for item in sublist],  # type: ignore
        key=lambda x: x["id"],
    )
    utils.synchronize()
    return sorted_predictions


def ans_prediction(
    cfg: DictConfig,
    output_dir: str,
    qa_data: GraphIndexDatasetV1,
    retrieval_result: list[dict],
) -> str:
    llm = instantiate(cfg.llm)
    doc_retriever = utils.DocumentRetriever(qa_data.doc, qa_data.id2node)
    test_data = qa_data.raw_test_data
    id2ent = {v: k for k, v in qa_data.id2node.items()}

    prompt_builder = QAPromptBuilder(cfg.qa_prompt)

    def predict(qa_input: tuple[dict, torch.Tensor]) -> dict | Exception:
        data, retrieval_doc = qa_input
        retrieved_ent_idx = torch.topk(
            retrieval_doc["entity"], cfg.test.save_top_k_entity, dim=-1
        ).indices
        retrieved_ent = [id2ent[i.item()] for i in retrieved_ent_idx]
        retrieved_docs = doc_retriever(retrieval_doc["document"], top_k=cfg.test.top_k)

        message = prompt_builder.build_input_prompt(data["question"], retrieved_docs)

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
                "retrieved_ent": retrieved_ent,
                "retrieved_docs": retrieved_docs,
            }

    with open(os.path.join(output_dir, "prediction.jsonl"), "w") as f:
        with ThreadPool(cfg.test.n_threads) as pool:
            for results in tqdm(
                pool.imap(predict, zip(test_data, retrieval_result)),
                total=len(test_data),
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
    utils.init_distributed_mode()
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")

    model, model_config = utils.load_model_from_pretrained(
        cfg.graph_retriever.model_path
    )
    dataset_cls = get_class(cfg.dataset._target_)
    qa_data = dataset_cls(
        **cfg.dataset.cfgs,
        text_emb_model_cfgs=OmegaConf.create(model_config["text_emb_model_config"]),
    )
    device = utils.get_device()
    model = model.to(device)

    qa_data.graph = qa_data.graph.to(device)

    if cfg.test.retrieved_result_path:
        retrieval_result = torch.load(cfg.test.retrieved_result_path, weights_only=True)
    else:
        if cfg.test.prediction_result_path:
            retrieval_result = None
        else:
            retrieval_result = doc_retrieval(cfg, model, qa_data, device=device)
    if utils.is_main_process():
        if cfg.test.save_retrieval and retrieval_result is not None:
            logger.info(
                f"Ranking saved to disk: {os.path.join(output_dir, 'retrieval_result.pt')}"
            )
            torch.save(
                retrieval_result, os.path.join(output_dir, "retrieval_result.pt")
            )
        if cfg.test.prediction_result_path:
            output_path = cfg.test.prediction_result_path
        else:
            output_path = ans_prediction(cfg, output_dir, qa_data, retrieval_result)

        # Evaluation
        evaluator = instantiate(cfg.qa_evaluator, prediction_file=output_path)
        metrics = evaluator.evaluate()
        query_utils.print_metrics(metrics, logger)
        return metrics


if __name__ == "__main__":
    main()
