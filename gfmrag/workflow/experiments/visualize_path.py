import logging
import os
from inspect import cleandoc

import hydra
import torch
import torch.utils
import torch.utils.data
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist

from gfmrag import utils
from gfmrag.models.ultra import query_utils

# A logger for this file
logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../config/gfm_reasoner",
    config_name="visualize_path",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    utils.init_distributed_mode(cfg.timeout)
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        output_dir = HydraConfig.get().runtime.output_dir
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")
        output_dir_list = [output_dir]
    else:
        output_dir_list = [None]
    if utils.get_world_size() > 1:
        dist.broadcast_object_list(
            output_dir_list, src=0
        )  # Use the output dir from rank 0
    output_dir = output_dir_list[0]

    model, model_config = utils.load_model_from_pretrained(
        cfg.load_model_from_pretrained
    )

    qa_data = instantiate(
        cfg.dataset,
        text_emb_model_cfgs=OmegaConf.create(model_config["text_emb_model_config"]),
        _recursive_=False,
    )

    device = torch.device("cpu")  # utils.get_device()
    model = model.to(device)

    graph = qa_data.graph.to(device)
    test_data = qa_data.test_data

    node_types = graph.node_type
    node_type_names = graph.node_type_names
    id2ent = qa_data.id2node
    rel2id = qa_data.rel2id
    id2rel = {v: k for k, v in rel2id.items()}

    # sample up to cfg.test_max_sample examples from the test set (deterministic per seed+rank)
    if getattr(cfg, "test_max_sample", None) and cfg.test_max_sample > 0:
        n = len(test_data)
        num = min(n, cfg.test_max_sample)
        gen = torch.Generator().manual_seed(cfg.seed + utils.get_rank())
        indices = torch.randperm(n, generator=gen)[:num].tolist()
        test_data = torch.utils.data.Subset(test_data, indices)
    test_data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
    )
    raw_test_data = qa_data.raw_test_data
    for i, sample in enumerate(test_data_loader):
        raw_sample = raw_test_data[i]
        sample = query_utils.cuda(sample, device=device)
        paths_results = model.visualize(graph, sample)
        start_nodes = "Starting Nodes:"
        for node_type in raw_sample["start_nodes"]:
            start_nodes += (
                f"\n{node_type}: {', '.join(raw_sample['start_nodes'][node_type])}"
            )
        target_nodes = "Target Nodes:"
        for node_type in raw_sample["target_nodes"]:
            target_nodes += (
                f"\n{node_type}: {', '.join(raw_sample['target_nodes'][node_type])}"
            )

        result_str = (
            cleandoc(
                f"""
        >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        Question: {raw_sample["question"]}
        Answer: {raw_sample["answer"]}
        Supporting Facts: {raw_sample["supporting_documents"]}
        {start_nodes}
        {target_nodes}
        Predicted Paths:
        """
            )
            + "\n"
        )

        for t_index, (paths, weights) in paths_results.items():
            result_str += (
                cleandoc(
                    f"""--------------------------------------------------------
            Target Nodes: {id2ent[t_index]} ({node_type_names[node_types[t_index].item()]})
            """
                )
                + "\n"
            )
            for path, weight in zip(paths, weights):
                path_str_list = []
                for h, t, r in path:
                    h_type = node_type_names[node_types[h].item()]
                    t_type = node_type_names[node_types[t].item()]
                    path_str_list.append(
                        f"[ {id2ent[h]} ({h_type}), {id2rel[r]}, {id2ent[t]} ({t_type}) ]"
                    )
                result_str += f"{weight:.4f}: {' => '.join(path_str_list)}\n"
        logger.info(cleandoc(result_str))


if __name__ == "__main__":
    main()
