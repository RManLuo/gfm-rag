import hashlib
import json
import logging
import os
import os.path as osp
import sys
import warnings

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data, InMemoryDataset, makedirs
from torch_geometric.data.dataset import _repr, files_exist

from gfmrag.kg_construction.utils import KG_DELIMITER
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils import get_rank

logger = logging.getLogger(__name__)


class KGDataset(InMemoryDataset):
    """A dataset class for processing and managing Knowledge Graph (KG) data.

    This class extends InMemoryDataset to handle knowledge graph data, including entity-relation-entity triplets,
    and supports processing of both direct and inverse relations.

    Args:
        root (str): Root directory where the dataset should be saved.
        data_name (str): Name of the dataset.
        text_emb_model_cfgs (DictConfig): Configuration for the text embedding model.
        force_rebuild (bool, optional): Whether to force rebuilding the processed data. Defaults to False.
        **kwargs (str): Additional keyword arguments.

    Attributes:
        name (str): Name of the dataset.
        fingerprint (str): MD5 hash of the text embedding model configuration.
        delimiter (str): Delimiter used in the KG text file.
        data (Data): Processed graph data object.
        slices (dict): Data slices for batching.

    Note:
        - The class expects a 'kg.txt' file in the raw directory containing triplets.
        - Processes both direct and inverse relations.
        - Generates and stores relation embeddings using the specified text embedding model.
        - Saves processed data along with entity and relation mappings.
    """

    delimiter = KG_DELIMITER

    def __init__(
        self,
        root: str,
        data_name: str,
        text_emb_model_cfgs: DictConfig,
        force_rebuild: bool = False,
        **kwargs: str,
    ) -> None:
        self.name = data_name
        self.force_rebuild = force_rebuild
        # Get fingerprint of the model configuration
        self.fingerprint = hashlib.md5(
            json.dumps(
                OmegaConf.to_container(text_emb_model_cfgs, resolve=True)
            ).encode()
        ).hexdigest()
        self.text_emb_model_cfgs = text_emb_model_cfgs
        super().__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.feat_dim = self._data.rel_emb.size(1)

    @property
    def raw_file_names(self) -> list:
        return ["kg.txt"]

    def load_file(
        self, triplet_file: str, inv_entity_vocab: dict, inv_rel_vocab: dict
    ) -> dict:
        """Load a knowledge graph file and return the processed data."""

        triplets = []  # Triples with inverse relations
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, encoding="utf-8") as fin:
            for line in fin:
                try:
                    u, r, v = (
                        line.split()
                        if self.delimiter is None
                        else line.strip().split(self.delimiter)
                    )
                except Exception as e:
                    logger.error(f"Error in line: {line}, {e}, Skipping")
                    continue
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab),  # entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
        }

    def _process(self) -> None:
        f = osp.join(self.processed_dir, "pre_transform.pt")
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
            self.pre_transform
        ):
            warnings.warn(  # noqa:B028
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first",
                stacklevel=1,
            )

        f = osp.join(self.processed_dir, "pre_filter.pt")
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
            self.pre_filter
        ):
            warnings.warn(
                f"The `pre_filter` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-fitering technique, make sure to "
                f"delete '{self.processed_dir}' first",
                stacklevel=1,
            )

        if self.force_rebuild or not files_exist(self.processed_paths):
            logger.warning(f"Processing KG dataset {self.name} at rank {get_rank()}")
            if self.log and "pytest" not in sys.modules:
                print("Processing...", file=sys.stderr)

            makedirs(self.processed_dir)
            self.process()

            path = osp.join(self.processed_dir, "pre_transform.pt")
            torch.save(_repr(self.pre_transform), path)
            path = osp.join(self.processed_dir, "pre_filter.pt")
            torch.save(_repr(self.pre_filter), path)

            if self.log and "pytest" not in sys.modules:
                print("Done!", file=sys.stderr)

    def process(self) -> None:
        """Process the knowledge graph dataset.

        This method processes the raw knowledge graph file and creates the following:

        1. Loads the KG triplets and vocabulary
        2. Creates edge indices and types for both original and inverse relations
        3. Saves entity and relation mappings to JSON files
        4. Generates relation embeddings using a text embedding model
        5. Builds relation graphs
        6. Saves the processed data and model configurations

        The processed data includes:

        - Edge indices and types for both original and inverse edges
        - Target edge indices and types (original edges only)
        - Number of nodes and relations
        - Relation embeddings
        - Relation graphs

        Files created:

        - ent2id.json: Entity to ID mapping
        - rel2id.json: Relation to ID mapping (including inverse relations)
        - text_emb_model_cfgs.json: Text embedding model configuration
        - Processed graph data file at self.processed_paths[0]
        """
        kg_file = self.raw_paths[0]

        kg_result = self.load_file(kg_file, inv_entity_vocab={}, inv_rel_vocab={})

        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train and 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = kg_result["num_node"]
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = kg_result["num_relation"]

        kg_triplets = kg_result["triplets"]

        train_target_edges = torch.tensor(
            [[t[0], t[1]] for t in kg_triplets], dtype=torch.long
        ).t()
        train_target_etypes = torch.tensor([t[2] for t in kg_triplets])

        # Add inverse edges
        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat(
            [train_target_etypes, train_target_etypes + num_relations]
        )

        with open(self.processed_dir + "/ent2id.json", "w") as f:
            json.dump(kg_result["inv_entity_vocab"], f)
        rel2id = kg_result["inv_rel_vocab"]
        id2rel = {v: k for k, v in rel2id.items()}
        for etype in train_etypes:
            if etype.item() >= num_relations:
                raw_etype = etype - num_relations
                raw_rel = id2rel[raw_etype.item()]
                rel2id["inverse_" + raw_rel] = etype.item()
        with open(self.processed_dir + "/rel2id.json", "w") as f:
            json.dump(rel2id, f)

        # Generate relation embeddings
        logger.info("Generating relation embeddings")
        text_emb_model: BaseTextEmbModel = instantiate(self.text_emb_model_cfgs)
        rel_emb = text_emb_model.encode(list(rel2id.keys()), is_query=False).cpu()

        kg_data = Data(
            edge_index=train_edges,
            edge_type=train_etypes,
            num_nodes=num_node,
            target_edge_index=train_target_edges,
            target_edge_type=train_target_etypes,
            num_relations=num_relations * 2,
            rel_emb=rel_emb,
        )

        torch.save((self.collate([kg_data])), self.processed_paths[0])

        # Save text embeddings model configuration
        with open(self.processed_dir + "/text_emb_model_cfgs.json", "w") as f:
            json.dump(OmegaConf.to_container(self.text_emb_model_cfgs), f, indent=4)

    def __repr__(self) -> str:
        return f"{self.name}()"

    @property
    def num_relations(self) -> int:
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self) -> str:
        return os.path.join(str(self.root), str(self.name), "processed", "stage1")

    @property
    def processed_dir(self) -> str:
        return os.path.join(
            str(self.root),
            str(self.name),
            "processed",
            "stage2",
            self.fingerprint,
        )

    @property
    def processed_file_names(self) -> str:
        return "data.pt"
