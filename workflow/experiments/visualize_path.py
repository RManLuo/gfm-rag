import logging
import os
from inspect import cleandoc

import hydra
import torch
import torch.utils
import torch.utils.data
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch_geometric.data import Data

from gfmrag import utils
from gfmrag.datasets import QADataset
from gfmrag.ultra import query_utils

# A logger for this file
logger = logging.getLogger(__name__)


def visualize_path(
    cfg: DictConfig,
    sample: dict,
    model: nn.Module,
    graph: Data,
    ent2docs: torch.Tensor,
    device: torch.device,
) -> dict:
    if cfg.test.init_entities_weight:
        entities_weight = utils.get_entities_weight(ent2docs)
    else:
        entities_weight = None

    model.eval()
    sample = query_utils.cuda(sample, device=device)
    paths_results = model.visualize(graph, sample, entities_weight)
    return paths_results


@hydra.main(
    config_path="../config", config_name="exp_visualize_path", version_base=None
)
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
    qa_data = QADataset(
        **cfg.dataset,
        text_emb_model_cfgs=OmegaConf.create(model_config["text_emb_model_config"]),
    )
    device = utils.get_device()
    model = model.to(device)
    graph = qa_data.kg.to(device)
    ent2id = qa_data.ent2id
    rel2id = qa_data.rel2id
    _, test_data = qa_data._data
    ent2docs = qa_data.ent2docs.to(device)

    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}

    test_data = torch.utils.data.Subset(test_data, range(0, cfg.test.max_sample))
    test_data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
    )
    raw_test_data = qa_data.raw_test_data
    for i, sample in enumerate(test_data_loader):
        sample = query_utils.cuda(sample, device=device)
        raw_sample = raw_test_data[i]
        paths_results = visualize_path(cfg, sample, model, graph, ent2docs, device)

        result_str = (
            cleandoc(f"""
        >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        Question: {raw_sample['question']}
        Answer: {raw_sample['answer']}
        Question Entities: {raw_sample['question_entities']}
        Supporting Facts: {raw_sample['supporting_facts']}
        Supporting Entities: {raw_sample['supporting_entities']}
        Predicted Paths:
        """)
            + "\n"
        )

        for t_index, (paths, weights) in paths_results.items():
            result_str += (
                cleandoc(f"""--------------------------------------------------------
            Target Entity: {id2ent[t_index]}
            """)
                + "\n"
            )
            for path, weight in zip(paths, weights):
                path_str_list = []
                for h, t, r in path:
                    path_str_list.append(f"[ {id2ent[h]}, {id2rel[r]}, {id2ent[t]} ]")
                result_str += f"{weight:.4f}: {' => '.join(path_str_list)}\n"
        logger.info(cleandoc(result_str))


if __name__ == "__main__":
    main()
