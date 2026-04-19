import ast
import json
import logging
import os.path as osp
from typing import Any, ClassVar

import pandas as pd
import torch
from hydra.utils import instantiate
from torch_geometric.data import Data

from gfmrag.graph_index_datasets.graph_index_dataset import GraphIndexDataset
from gfmrag.text_emb_models import BaseTextEmbModel

logger = logging.getLogger(__name__)


class GraphIndexDatasetV1(GraphIndexDataset):
    """
    Version 1 of the Graph Index Dataset for GFM-RAG.

    This is a specialized version of the GraphIndexDataset tailored for the first iteration of the GFM-RAG framework, which predict other types of nodes based on entity predictions.
    It inherits from GraphIndexDataset and can include additional features or modifications specific to version 1
    """

    FINGER_PRINT_ATTRS: ClassVar[list[str]] = GraphIndexDataset.FINGER_PRINT_ATTRS + [
        "target_type"
    ]

    def __init__(self, target_type: str, **kwargs: Any):
        """
        Initialize the GraphIndexDatasetV1 with the specified prediction type and other parameters.
        Args:
            target_type (str): The type of node to be used to construct graph index (e.g., 'entity').
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        self.target_type = target_type
        super().__init__(**kwargs)
        # Additional initialization or modifications for version 1 can be added here.

    def attributes_to_text(  # type: ignore[override]
        self, name: str | None = None, attributes: dict | None = None, **kwargs: dict
    ) -> str:
        """Return a string representation of the attributes. V1 version only encodes the name

        Args:
            name (str): The name of the node or relation to be used as the string representation
            **kwargs (dict): Additional keyword arguments to include in the string. The keys of the dictionary will be used as attribute names.

        Returns:
            str: A formatted string representation of the attributes.

        Examples:
            >>> name = "Node1"
            >>> print(attributes_to_text(name=name))
            Node1

        """
        if name is not None:
            return name

        if attributes is None:
            attributes = {}

        if len(attributes) > 0:
            attr_str = "\n".join(
                f"{key}: {value}"
                for key, value in attributes.items()
                if value is not None
            )
        else:
            attr_str = ""

        if len(kwargs) > 0:
            additional_attrs = "\n".join(
                f"{key}: {value}" for key, value in kwargs.items() if value is not None
            )
        else:
            additional_attrs = ""

        if attr_str and additional_attrs:
            return f"{additional_attrs}\n{attr_str}".strip()
        elif attr_str:
            return attr_str.strip()
        elif additional_attrs:
            return additional_attrs.strip()
        else:
            return ""

    def process_graph(self) -> None:
        """Process the graph index dataset.

        This method processes the raw graph index file and creates the following:

        1. Loads the nodes, edges, and relations from the raw files
        2. Only use edges with the specified target type nodes for graph index construction
        3. Saves entity and relation mappings to JSON files
        4. Generates relation, entity, edges features using a text embedding model
        5. Saves the processed data and model configurations

        The processed data includes:

        - Edge indices and types for both original and inverse edges
        - Target edge indices and types (original edges only)
        - Number of nodes and relations
        - Relation embeddings
        - Entity embeddings
        - Target to other types mapping as a sparse tensor, used for getting the prediction of other types of nodes based on the target type nodes.

        Files created:

        - graph.pt: Contains the processed graph data
        - node2id.json: Node to ID mapping
        - rel2id.json: Relation to ID mapping (including inverse relations)
        """
        node_file, relation_file, edge_file = self.raw_graph
        if (
            not osp.exists(node_file)
            or not osp.exists(relation_file)
            or not osp.exists(edge_file)
        ):
            raise FileNotFoundError(
                f"Required files not found in {self.raw_dir}. "
                "Please ensure 'nodes.csv', 'relations.csv', and 'edges.csv' exist."
            )

        # Load nodes
        nodes_df = self._read_csv_file(node_file)
        node2id = nodes_df["id"].to_dict()  # Map names or uid to continuous IDs
        nodes_type_id, node_type_names = pd.factorize(nodes_df["type"])
        nodes_df["type_id"] = nodes_type_id  # Add type ID column
        # Create a tensor for node types
        node_types = torch.LongTensor(nodes_type_id)
        # Save node ids under each type for fast access
        nodes_by_type = {}
        # Group node id by type
        node_types_group = nodes_df.groupby("type")["id"].apply(list).to_dict()
        for node_type, node_ids in node_types_group.items():
            nodes_by_type[node_type] = torch.LongTensor(node_ids)

        if len(nodes_by_type.get(self.target_type, [])) == 0:
            raise ValueError(
                f"No nodes found for target type '{self.target_type}'. "
                "Please ensure the target type exists in the graph."
            )

        # Load relations
        relations_df = self._read_csv_file(relation_file)
        rel2id = relations_df["id"].to_dict()

        # Load triplets from edges.csv
        edges_df = pd.read_csv(edge_file)
        edges_df["attributes"] = edges_df["attributes"].apply(
            lambda x: {} if pd.isna(x) else ast.literal_eval(x)
        )

        # Vectorized mapping of source, target, and relation to IDs
        edges_df["u"] = edges_df["source"].map(node2id)
        edges_df["v"] = edges_df["target"].map(node2id)
        edges_df["r"] = edges_df["relation"].map(rel2id)

        # Filter out rows with missing node or relation IDs
        valid_edges_df = edges_df.dropna(subset=["u", "v", "r"])

        # Log skipped edges
        skipped_edges = edges_df[edges_df[["u", "v", "r"]].isnull().any(axis=1)]
        for _, row in skipped_edges.iterrows():
            logger.warning(
                f"Skipping edge with missing node or relation: {row['source']}, {row['relation']}, {row['target']}"
            )

        # Apply node type for edges
        valid_edges_df["source_type"] = valid_edges_df["source"].apply(
            lambda x: nodes_df.loc[x, "type"]
        )
        valid_edges_df["target_type"] = valid_edges_df["target"].apply(
            lambda x: nodes_df.loc[x, "type"]
        )

        # Only select edges that the node type of both source and target is the target type
        target_edges_df = valid_edges_df[
            (valid_edges_df["source_type"] == self.target_type)
            & (valid_edges_df["target_type"] == self.target_type)
        ]

        num_nodes = len(node2id)
        num_relations = len(rel2id)

        # Create target type to other types mapping and store as sparse tensor, size: (n nodes, total number of nodes)
        target_to_other_types: dict[str, torch.Tensor] = dict()
        for other_type, group in valid_edges_df[
            valid_edges_df["source_type"] == self.target_type
        ].groupby("target_type"):
            # Skip if the other type is the same as the target type
            if other_type == self.target_type:
                continue
            indices = torch.tensor(group[["u", "v"]].astype(int).values.T)
            target_to_other_mapping = torch.sparse_coo_tensor(
                indices,
                torch.ones(indices.size(1), dtype=torch.float),
                size=(num_nodes, num_nodes),
            )
            target_to_other_types[other_type] = target_to_other_mapping

        # Convert IDs to int and build edge tuples
        edges = list(
            zip(
                target_edges_df["u"].astype(int),
                target_edges_df["v"].astype(int),
                target_edges_df["r"].astype(int),
            )
        )

        train_target_edges = torch.tensor(
            [[t[0], t[1]] for t in edges], dtype=torch.long
        ).t()
        train_target_etypes = torch.tensor([t[2] for t in edges])

        # Add inverse edges
        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat(
            [train_target_etypes, train_target_etypes + num_relations]
        )

        with open(self.processed_dir + "/node2id.json", "w") as f:
            json.dump(node2id, f)
        id2rel = {v: k for k, v in rel2id.items()}
        for etype in train_etypes:
            if etype.item() >= num_relations:
                raw_etype = etype - num_relations
                raw_rel = id2rel[raw_etype.item()]
                rel2id["inverse_" + raw_rel] = etype.item()
        with open(self.processed_dir + "/rel2id.json", "w") as f:
            json.dump(rel2id, f)

        # Instantiate the text embedding model if attributes are used
        if self.use_node_feat or self.use_edge_feat or self.use_relation_feat:
            text_emb_model: BaseTextEmbModel = instantiate(self.text_emb_model_cfgs)

        # Generate relation embeddings
        if self.use_relation_feat:
            logger.info("Generating relation embeddings")
            relation_text_attributes = relations_df.apply(
                lambda row: self.attributes_to_text(
                    attributes=row["attributes"], name=row.name
                ),
                axis=1,
            ).to_list()
            rel_emb = text_emb_model.encode(
                relation_text_attributes, is_query=False
            ).cpu()
            if self.inverse_relation_feat == "inverse":
                # Inverse relations by adding the negative sign to the relation embeddings http://arxiv.org/abs/2505.20422
                rel_emb = torch.cat([rel_emb, -rel_emb], dim=0)
            elif self.inverse_relation_feat == "text":
                inverse_relation_text_attributes = relations_df.apply(
                    lambda row: self.attributes_to_text(
                        attributes=row["attributes"], name="inverse_" + row.name
                    ),
                    axis=1,
                ).to_list()
                inverse_rel_emb = text_emb_model.encode(
                    inverse_relation_text_attributes, is_query=False
                ).cpu()
                rel_emb = torch.cat([rel_emb, inverse_rel_emb], dim=0)
        else:
            rel_emb = None

        # Generate entity embeddings
        if self.use_node_feat:
            node_text_attributes = nodes_df.apply(
                lambda row: self.attributes_to_text(
                    attributes=row["attributes"], name=row.name, type=row["type"]
                ),
                axis=1,
            ).to_list()
            logger.info("Generating entity embeddings")
            node_emb = text_emb_model.encode(node_text_attributes, is_query=False).cpu()
        else:
            node_emb = None

        if self.use_edge_feat:
            logger.info("Generating edge embeddings")
            edge_text_attributes = edges_df.apply(
                lambda row: self.attributes_to_text(
                    attributes=row["attributes"],
                ),
                axis=1,
            ).to_list()
            edge_emb = text_emb_model.encode(edge_text_attributes, is_query=False).cpu()
        else:
            edge_emb = None

        # Get feature dimension
        for emb in [node_emb, rel_emb, edge_emb]:
            if emb is not None:
                if emb.ndim != 2:
                    raise ValueError(
                        f"Expected 2D tensor for embeddings, got {emb.ndim}D tensor."
                    )
                feat_dim = emb.size(1)
                break
        else:
            feat_dim = 0  # No embeddings available

        graph = Data(
            node_type=node_types,
            node_type_names=node_type_names,
            nodes_by_type=nodes_by_type,
            target_to_other_types=target_to_other_types,
            edge_index=train_edges,
            edge_type=train_etypes,
            num_nodes=num_nodes,
            target_edge_index=train_target_edges,
            target_edge_type=train_target_etypes,
            num_relations=num_relations * 2,
            x=node_emb,
            rel_attr=rel_emb,
            edge_attr=edge_emb,
            feat_dim=feat_dim,
        )

        torch.save(graph, self.processed_graph[0])
