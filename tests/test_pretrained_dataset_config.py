import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from gfmrag.graph_index_datasets import GraphIndexDataset, GraphIndexDatasetV1
from gfmrag.utils.util import save_model_to_pretrained


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)
        self.feat_dim = 768


def test_save_model_to_pretrained_saves_dataset_config_for_v1(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "torch.nn.Identity",
            },
            "datasets": {
                "_target_": "gfmrag.graph_index_datasets.GraphIndexDatasetV1",
                "cfgs": {
                    "root": "./data",
                    "force_reload": False,
                    "target_type": "entity",
                    "use_node_feat": False,
                    "use_relation_feat": True,
                    "use_edge_feat": False,
                    "inverse_relation_feat": "text",
                    "text_emb_model_cfgs": {
                        "_target_": "gfmrag.text_emb_models.BaseTextEmbModel",
                        "text_emb_model_name": "sentence-transformers/all-mpnet-base-v2",
                        "normalize": False,
                        "batch_size": 32,
                        "query_instruct": None,
                        "passage_instruct": None,
                        "model_kwargs": None,
                    },
                },
            },
        }
    )

    save_model_to_pretrained(DummyModel(), cfg, str(tmp_path))

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["dataset_config"] == {
        "class_name": GraphIndexDatasetV1.__name__,
        "text_emb_model_cfgs": {
            "_target_": "gfmrag.text_emb_models.BaseTextEmbModel",
            "text_emb_model_name": "sentence-transformers/all-mpnet-base-v2",
            "normalize": False,
            "batch_size": 32,
            "query_instruct": None,
            "passage_instruct": None,
            "model_kwargs": None,
        },
        "use_node_feat": False,
        "use_relation_feat": True,
        "use_edge_feat": False,
        "inverse_relation_feat": "text",
        "target_type": "entity",
    }


def test_graph_index_dataset_v1_fingerprint_attrs_include_target_type() -> None:
    assert GraphIndexDataset.FINGER_PRINT_ATTRS == [
        "use_node_feat",
        "use_relation_feat",
        "use_edge_feat",
        "inverse_relation_feat",
    ]
    assert GraphIndexDatasetV1.FINGER_PRINT_ATTRS == [
        "use_node_feat",
        "use_relation_feat",
        "use_edge_feat",
        "inverse_relation_feat",
        "target_type",
    ]
