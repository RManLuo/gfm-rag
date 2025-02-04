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


def get_multi_dataset(cfg: DictConfig) -> dict:
    """
    Return the joint KG datasets
    """
    data_name_list = []
    # Remove duplicates in the list
    for data_name in cfg.datasets.train_names + cfg.datasets.valid_names:
        if data_name not in data_name_list:
            data_name_list.append(data_name)
    dataset_cls = get_class(cfg.datasets._target_)
    dataset_list = {}
    for data_name in data_name_list:
        kg_data = dataset_cls(**cfg.datasets.cfgs, data_name=data_name)
        dataset_list[data_name] = kg_data
    return dataset_list


def get_entities_weight(ent2docs: torch.Tensor) -> torch.Tensor:
    frequency = torch.sparse.sum(ent2docs, dim=-1).to_dense()
    weights = 1 / frequency
    # Masked zero weights
    weights[frequency == 0] = 0
    return weights
