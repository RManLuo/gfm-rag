import json
import os

import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from transformers.utils import cached_file


def save_model_to_pretrained(
    model: torch.nn.Module, cfg: DictConfig, path: str
) -> None:
    os.makedirs(path, exist_ok=True)
    model_config = OmegaConf.to_container(cfg.model)
    model_config["rel_emb_dim"] = model.rel_emb_dim
    config = {
        "text_emb_model_config": OmegaConf.to_container(
            cfg.datasets.cfgs.text_emb_model_cfgs
        ),
        "model_config": model_config,
    }

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    torch.save({"model": model.state_dict()}, os.path.join(path, "model.pth"))


def load_model_from_pretrained(path: str) -> tuple[torch.nn.Module, dict]:
    config_path = cached_file(path, "config.json")
    if config_path is None:
        raise FileNotFoundError(f"config.json not found in {path}")
    with open(config_path) as f:
        config = json.load(f)
    model = instantiate(config["model_config"])
    model_path = cached_file(path, "model.pth")
    if model_path is None:
        raise FileNotFoundError(f"model.pth not found in {path}")
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state["model"])
    return model, config


def init_multi_dataset(cfg: DictConfig, world_size: int, rank: int) -> list:
    """
    Pre-rocess the dataset in each rank
    Args:
        cfg (DictConfig): The config file
        world_size (int): The number of GPUs
        rank (int): The rank of the current GPU
    Returns:
        list: The list of feat_dim in each dataset
    """
    data_name_list = []
    # Remove duplicates in the list
    for data_name in cfg.datasets.train_names + cfg.datasets.valid_names:
        if data_name not in data_name_list:
            data_name_list.append(data_name)

    dataset_cls = get_class(cfg.datasets._target_)
    # Make sure there is no overlap datasets between different ranks
    feat_dim_list = []
    for i, data_name in enumerate(data_name_list):
        if i % world_size == rank:
            dataset = dataset_cls(**cfg.datasets.cfgs, data_name=data_name)
            feat_dim_list.append(dataset.feat_dim)
    # Gather the feat_dim from all processes
    if world_size > 1:
        gathered_lists: list[list[int]] = [[] for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_lists, feat_dim_list)
        # Flatten the list of lists
        all_feat_dim_list = [item for sublist in gathered_lists for item in sublist]
    else:
        all_feat_dim_list = feat_dim_list

    return all_feat_dim_list


def get_entities_weight(ent2docs: torch.Tensor) -> torch.Tensor:
    frequency = torch.sparse.sum(ent2docs, dim=-1).to_dense()
    weights = 1 / frequency
    # Masked zero weights
    weights[frequency == 0] = 0
    return weights
