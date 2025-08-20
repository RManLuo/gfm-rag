import ast
import hashlib
import json
import logging
import os
import os.path as osp
from typing import Literal

import datasets
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils import data as torch_data
from torch_geometric.data import Data
from torch_geometric.data.dataset import files_exist

from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils import get_rank
from gfmrag.utils.qa_utils import entities_to_mask

logger = logging.getLogger(__name__)


class GraphIndexDataset:
    """A dataset class for processing and managing graph index data.

    GraphDataset provides an unified interface for loading, processing, and managing graph data.

    Args:
        root (str): Root directory where the dataset should be saved.
        data_name (str): Name of the dataset.
        text_emb_model_cfgs (DictConfig): Configuration for the text embedding model.
        force_reload (bool, optional): Whether to force rebuilding the processed data. Defaults to False.
        use_node_feat (bool, optional): Whether to use node features. Defaults to True.
        use_relation_feat (bool, optional): Whether to use relation features. Defaults to True.
        use_edge_feat (bool, optional): Whether to use edge features. Defaults to False.
        inverse_relation_feat (Literal['text', 'inverse'], optional): How to handle inverse relations.
            - 'text': Generate text embeddings for inverse relations by adding "inverse_" prefix to the relation name.
            - 'inverse': Use the negative of the relation embeddings for inverse relations following: http://arxiv.org/abs/2505.20422
        **kwargs (str): Additional keyword arguments.

    Attributes:
        name (str): Name of the dataset.
        fingerprint (str): MD5 hash of the text embedding model configuration.
        graph (Data): Processed graph data object.
        train_data (torch.utils.data.Dataset | None): Training data.
        test_data (torch.utils.data.Dataset | None): Testing data.
        feat_dim (int): Dimension of the entity and relation embeddings.
        node2id (dict): Mapping from node names to continuous IDs.
        rel2id (dict): Mapping from relation names to continuous IDs.
        id2node: Dict[str, int]: Mapping from continuous IDs to node names.
        doc: Dict[str, Any]: The original document data
        raw_train_data (Dict[str, Any]): The raw training data.
        raw_test_data (Dict[str, Any]): The raw testing data.

    Note:
        - The class expects 'edges.csv', 'nodes.csv', 'relations.csv' files in the stage1 directory.
        - Processes both direct and inverse relations.
        - Generates and stores node and relation embeddings using the specified text embedding model.
        - Saves processed data along with entity and relation mappings.

    Files created:
        - graph.pt: Contains the processed graph data.
        - train.pt: Contains the processed training data, if available.
        - test.pt: Contains the processed testing data, if available.
        - node2id.json: Maps node names to continuous IDs.
        - rel2id.json: Maps relation names to continuous IDs (including inverse relations).
        - config.json: Contains the configuration of the text embedding model and dataset attributes.

    """

    FINGER_PRINT_ATTRS = [
        "use_node_feat",
        "use_relation_feat",
        "use_edge_feat",
        "inverse_relation_feat",
    ]
    RAW_GRAPH_NAMES = ["nodes.csv", "relations.csv", "edges.csv"]
    RAW_QA_DATA_NAMES = ["train.json", "test.json"]
    RAW_DOCUMENT_NAME = "documents.json"

    PROCESSED_GRAPH_NAMES = ["graph.pt", "node2id.json", "rel2id.json"]
    PROCESSED_QA_DATA_NAMES = ["train.pt", "test.pt"]

    def __init__(
        self,
        root: str,
        data_name: str,
        text_emb_model_cfgs: DictConfig,
        force_reload: bool = False,
        use_node_feat: bool = True,
        use_relation_feat: bool = True,
        use_edge_feat: bool = False,
        inverse_relation_feat: Literal["text", "inverse"] = "text",
        **kwargs: str,
    ) -> None:
        self.root = root
        self.name = data_name
        self.text_emb_model_cfgs = text_emb_model_cfgs
        self.use_node_feat = use_node_feat
        self.use_relation_feat = use_relation_feat
        self.use_edge_feat = use_edge_feat
        self.inverse_relation_feat = inverse_relation_feat

        # Get fingerprint of the model configuration
        cfgs = OmegaConf.to_container(text_emb_model_cfgs, resolve=True)
        cfgs.pop("batch_size", None)  # Remove batch_size for fingerprinting

        for key in self.FINGER_PRINT_ATTRS:
            cfgs[key] = getattr(self, key, None)

        self.fingerprint = hashlib.md5(
            (self.__class__.__name__ + json.dumps(cfgs)).encode()
        ).hexdigest()

        graph_rebuild = self.load_graph(force_reload)
        qa_rebuild = self.load_qa_data(graph_rebuild, force_reload)

        if any([graph_rebuild, qa_rebuild]):
            # Save the dataset configuration if the graph or QA data was rebuilt
            self.save_config()

    def load_graph(self, force_reload: bool = False) -> bool:
        """Load the processed graph data.

        Setting attributes:
            - self.graph: The processed graph data as a torch_geometric Data object.
            - self.node2id: A dictionary mapping node names to continuous IDs.
            - self.rel2id: A dictionary mapping relation names to continuous IDs.
            - self.id2node: A dictionary mapping continuous IDs back to node names.
            - self.feat_dim: The dimension of the entity and relation embeddings.

        Args:
            force_reload (bool): Whether to force reload the graph data. If True, it will process the graph even if the processed files exist.

        Returns:
            bool: Whether the graph was rebuilt. If the graph was rebuilt, it returns True, otherwise False.
        """

        rebuild_graph = False
        if force_reload or not files_exist(self.processed_graph):
            os.makedirs(self.processed_dir, exist_ok=True)
            logger.warning(f"Processing graph for {self.name} at rank {get_rank()}")
            self.process_graph()

        self.graph = torch.load(self.processed_graph[0], weights_only=False)
        with open(self.processed_graph[1]) as fin:
            self.node2id = json.load(fin)
        with open(self.processed_graph[2]) as fin:
            self.rel2id = json.load(fin)

        self.id2node = {v: k for k, v in self.node2id.items()}
        self.feat_dim = self.graph.feat_dim
        return rebuild_graph

    def load_qa_data(self, graph_rebuild: bool, force_reload: bool = False) -> bool:
        """
        Load the QA data.

        Setting attributes:
            - self.train_data: The processed training data as a torch.utils.data.Dataset.
            - self.test_data: The processed testing data as a torch.utils.data.Dataset.
            - self.raw_train_data: The raw training data as a dictionary.
            - self.raw_test_data: The raw testing data as a dictionary.
            - self.doc: The original document data as a dictionary.

        Args:
            graph_rebuild (bool): Whether the graph was rebuilt.
            force_reload (bool): Whether to force reload the QA data. If True, it will process the QA data even if the processed files exist.
        Returns:
            bool: Whether the QA data was rebuilt.
        """
        rebuild_qa_data = False
        exist_raw_qa_data = []
        # Check if any raw QA data files exist
        for raw_data_name in self.raw_qa_data:
            if os.path.exists(raw_data_name):
                exist_raw_qa_data.append(raw_data_name)

        if len(exist_raw_qa_data) > 0:
            # Process the QA data if it does not exist or if force_reload is True or if the graph is rebuilt
            need_to_process_qa_data = [
                raw_data_name
                for raw_data_name in exist_raw_qa_data
                if not osp.exists(
                    osp.join(
                        self.processed_dir,
                        f"{osp.basename(raw_data_name).split('.')[0]}.pt",
                    )
                )
            ]
            if force_reload or graph_rebuild:
                need_to_process_qa_data = exist_raw_qa_data

            if len(need_to_process_qa_data) > 0:
                logger.warning(
                    f"Processing QA data for {self.name} at rank {get_rank()}"
                )
                self.process_qa_data(need_to_process_qa_data)
                rebuild_qa_data = True

        # Load the processed QA data
        if osp.exists(osp.join(self.processed_dir, "train.pt")):
            self.train_data = torch.load(
                osp.join(self.processed_dir, "train.pt"), weights_only=False
            )
            with open(os.path.join(self.raw_dir, "train.json")) as fin:
                self.raw_train_data = json.load(fin)
        else:
            self.train_data = None
            self.raw_train_data = None

        if osp.exists(osp.join(self.processed_dir, "test.pt")):
            self.test_data = torch.load(
                osp.join(self.processed_dir, "test.pt"), weights_only=False
            )
            with open(os.path.join(self.raw_dir, "test.json")) as fin:
                self.raw_test_data = json.load(fin)
        else:
            self.test_data = None
            self.raw_test_data = None

        with open(
            os.path.join(str(self.root), str(self.name), "raw", self.RAW_DOCUMENT_NAME)
        ) as fin:
            self.doc = json.load(fin)

        return rebuild_qa_data

    def attributes_to_text(self, attributes: dict | None = None, **kwargs: dict) -> str:
        """Return a string representation of the attributes.

        Args:
            attributes (dict | None): A dictionary of attributes.
            **kwargs (dict): Additional keyword arguments to include in the string. The keys of the dictionary will be used as attribute names.

        Returns:
            str: A formatted string representation of the attributes.

        Examples:
            >>> attributes = {"description": "A node in the graph"}
            >>> name = "Node1"
            >>> print(attributes_to_text(attributes, name=name))
            name: Node1
            description: A node in the graph

        """
        if attributes is None:
            attributes = {}
        if len(attributes) == 0 and len(kwargs) == 0:
            raise ValueError(
                "At least 'attributes' or other keyword arguments must be provided."
            )

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

    def _read_csv_file(self, file_path: str) -> pd.DataFrame:
        """Read a CSV file and return a dict for nodes and relations.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: A DataFrame
        """
        if not osp.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        df = pd.read_csv(file_path, keep_default_na=False)
        df["id"] = df.index  # Add an ID column based on the index
        # Change index to 'name' for nodes and 'relation' for relations
        df = df.set_index("name")

        # Handle attributes
        df["attributes"] = df["attributes"].apply(
            lambda x: {} if pd.isna(x) else ast.literal_eval(x)
        )

        return df

    def process_graph(self) -> None:
        """Process the graph index dataset.

        This method processes the raw graph index file and creates the following:

        1. Loads the nodes, edges, and relations from the raw files
        2. Creates edge indices and types for both original and inverse relations
        3. Saves entity and relation mappings to JSON files
        4. Generates relation, entity, edges features using a text embedding model
        5. Saves the processed data and model configurations

        The processed data includes:

        - Edge indices and types for both original and inverse edges
        - Target edge indices and types (original edges only)
        - Number of nodes and relations
        - Relation embeddings
        - Entity embeddings

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
        node2id = nodes_df["id"].to_dict()  # Map names to continuous IDs
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

        # Load relations
        relations_df = self._read_csv_file(relation_file)
        rel2id = relations_df["id"].to_dict()

        # Load triplets from edges.csv
        edges_df = pd.read_csv(edge_file, keep_default_na=False)
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

        # Convert IDs to int and build edge tuples
        edges = list(
            zip(
                valid_edges_df["u"].astype(int),
                valid_edges_df["v"].astype(int),
                valid_edges_df["r"].astype(int),
            )
        )
        # # Sort the edges by source and target for consistency
        # edges.sort(key=lambda x: (x[0], x[1]))

        num_nodes = len(node2id)
        num_relations = len(rel2id)

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
                lambda row: self.attributes_to_text(row["attributes"], name=row.name),
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
                        row["attributes"], name="inverse_" + row.name
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
                    row["attributes"], name=row.name, type=row["type"]
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
                    row["attributes"],
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

    def process_qa_data(self, qa_data_names: list) -> None:
        """Process and prepare the question-answering dataset.

        This method processes raw data files to create a structured dataset for question answering
        tasks. It performs the following main operations:

        1. Loads entity and relation mappings from processed files
        2. Creates entity-document mapping tensors
        3. Processes question samples to generate:
            - Question embeddings
            - Combine masks for start nodes of each type
            - Combine masks for target nodes of each type

        The processed dataset is saved as torch splits containing:

        - Question embeddings
        - Start node masks
        - Target node masks
        - Sample IDs

        Files created:

        - train.pt: Contains the processed training data, if available
        - test.pt: Contains the processed testing data, if available

        Args:
            qa_data_names (list): List of raw QA data file names to process.

        Returns:
            None
        """
        num_nodes = self.graph.num_nodes

        start_nodes_mask = []
        target_nodes_mask = []
        sample_id = []
        questions = []
        num_samples = []

        for data_name in qa_data_names:
            num_sample = 0
            with open(data_name) as fin:
                data = json.load(fin)

                for index, item in enumerate(data):
                    # Get start nodes and target nodes for each node type
                    start_nodes_ids = []
                    target_nodes_ids = []
                    for node in item["start_nodes"].values():
                        start_nodes_ids.extend(
                            [self.node2id[x] for x in node if x in self.node2id]
                        )
                    for node in item["target_nodes"].values():
                        target_nodes_ids.extend(
                            [self.node2id[x] for x in node if x in self.node2id]
                        )

                    # Skip samples if any of the entities or documens are empty
                    if any(
                        len(x) == 0
                        for x in [
                            start_nodes_ids,
                            target_nodes_ids,
                        ]
                    ):
                        continue

                    num_sample += 1
                    sample_id.append(index)
                    question = item["question"]
                    questions.append(question)

                    # Create masks for start nodes and target nodes
                    start_nodes_mask.append(
                        entities_to_mask(start_nodes_ids, num_nodes)
                    )
                    target_nodes_mask.append(
                        entities_to_mask(target_nodes_ids, num_nodes)
                    )

                num_samples.append(num_sample)

        # Generate question embeddings
        logger.info("Generating question embeddings")
        text_emb_model: BaseTextEmbModel = instantiate(self.text_emb_model_cfgs)
        question_embeddings = text_emb_model.encode(
            questions,
            is_query=True,
        ).cpu()
        sample_id = torch.tensor(sample_id, dtype=torch.long)
        start_nodes_mask = torch.stack(start_nodes_mask)
        target_nodes_mask = torch.stack(target_nodes_mask)

        dataset = datasets.Dataset.from_dict(
            {
                "question_embeddings": question_embeddings,
                "start_nodes_mask": start_nodes_mask,
                "target_nodes_mask": target_nodes_mask,
                "sample_id": sample_id,
            }
        ).with_format("torch")

        offset = 0
        for raw_data_name, num_sample in zip(qa_data_names, num_samples):
            split = torch_data.Subset(dataset, range(offset, offset + num_sample))
            split_name = osp.basename(raw_data_name).split(".")[0]
            processed_split_path = osp.join(self.processed_dir, f"{split_name}.pt")
            torch.save(split, processed_split_path)
            offset += num_sample

    def __repr__(self) -> str:
        return f"{self.name}()"

    @property
    def num_relations(self) -> int:
        return self.graph.num_edge_types

    @property
    def raw_dir(self) -> str:
        return os.path.join(str(self.root), str(self.name), "processed", "stage1")

    @property
    def raw_graph(self) -> list:
        return [osp.join(self.raw_dir, name) for name in self.RAW_GRAPH_NAMES]

    @property
    def raw_qa_data(self) -> list:
        return [osp.join(self.raw_dir, name) for name in self.RAW_QA_DATA_NAMES]

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
    def processed_graph(self) -> list[str]:
        r"""The names of the processed files in the dataset."""
        return [
            osp.join(self.processed_dir, name) for name in self.PROCESSED_GRAPH_NAMES
        ]

    @property
    def processed_qa_data(self) -> list[str]:
        return [
            osp.join(self.processed_dir, name) for name in self.PROCESSED_QA_DATA_NAMES
        ]

    def save_config(self) -> None:
        """Save the configuration of the dataset to a JSON file."""
        config = {
            "class_name": self.__class__.__name__,
            "text_emb_model_cfgs": OmegaConf.to_container(
                self.text_emb_model_cfgs, resolve=True
            ),
        }
        for key in self.FINGER_PRINT_ATTRS:
            config[key] = getattr(self, key, None)
        with open(osp.join(self.processed_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
