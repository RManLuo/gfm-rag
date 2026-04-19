import ast
import logging
import os

import pandas as pd
import torch
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from gfmrag import utils
from gfmrag.graph_index_construction.entity_linking_model import BaseELModel
from gfmrag.graph_index_construction.graph_constructors import BaseGraphConstructor
from gfmrag.graph_index_construction.ner_model import BaseNERModel
from gfmrag.graph_index_datasets import GraphIndexDataset
from gfmrag.models.base_model import BaseGNNModel
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils.qa_utils import entities_to_mask

logger = logging.getLogger(__name__)

# Captured at import time so that patching GraphIndexDataset in tests
# does not affect stage1 file list resolution.
_STAGE1_GRAPH_NAMES: list[str] = GraphIndexDataset.RAW_GRAPH_NAMES


class GFMRetriever:
    """Graph Foundation Model (GFM) Retriever for document retrieval.

    Attributes:
        qa_data (GraphIndexDataset): Dataset containing the knowledge graph and mappings.
        graph: Knowledge graph structure.
        text_emb_model (BaseTextEmbModel): Model for text embedding.
        ner_model (BaseNERModel): Named Entity Recognition model.
        el_model (BaseELModel): Entity Linking model.
        graph_retriever (BaseGNNModel): GNN-based retriever (GNNRetriever or GraphReasoner).
        node_info (pd.DataFrame): Node attributes from nodes.csv, indexed by node name/uid.
        device (torch.device): Device to run computations on.
        num_nodes (int): Number of nodes in the knowledge graph.

    Examples:
        >>> retriever = GFMRetriever.from_index(
        ...     data_dir="./data",
        ...     data_name="my_dataset",
        ...     model_path="rmanluo/GFM-RAG-8M",
        ...     ner_model=ner_model,
        ...     el_model=el_model,
        ... )
        >>> results = retriever.retrieve("Who is the president of France?", top_k=5)
    """

    def __init__(
        self,
        qa_data: GraphIndexDataset,
        text_emb_model: BaseTextEmbModel,
        ner_model: BaseNERModel,
        el_model: BaseELModel,
        graph_retriever: BaseGNNModel,
        node_info: pd.DataFrame,
        device: torch.device,
    ) -> None:
        self.qa_data = qa_data
        self.graph = qa_data.graph
        self.text_emb_model = text_emb_model
        self.ner_model = ner_model
        self.el_model = el_model
        self.graph_retriever = graph_retriever
        self.node_info = node_info
        self.device = device
        self.num_nodes = self.graph.num_nodes

    @torch.no_grad()
    def retrieve(
        self,
        query: str,
        top_k: int,
        target_types: list[str] | None = None,
    ) -> dict[str, list[dict]]:
        """Retrieve nodes from the graph based on the given query.

        Args:
            query (str): Input query text.
            top_k (int): Number of results to return per target type.
            target_types (list[str] | None): Node types to retrieve. Each type must exist
                in graph.nodes_by_type. Defaults to ["document"].

        Returns:
            dict[str, list[dict]]: Results keyed by target type. Each entry contains
                dicts with keys: id, type, attributes, score.
        """
        if target_types is None:
            target_types = ["document"]

        from gfmrag.models.ultra import (
            query_utils,  # deferred to avoid circular import at module load
        )

        graph_retriever_input = self.prepare_input_for_graph_retriever(query)
        graph_retriever_input = query_utils.cuda(
            graph_retriever_input, device=self.device
        )

        pred = self.graph_retriever(self.graph, graph_retriever_input)  # 1 x num_nodes

        results: dict[str, list[dict]] = {}
        for target_type in target_types:
            node_ids = self.graph.nodes_by_type[
                target_type
            ]  # raises KeyError if missing
            type_pred = pred[:, node_ids].squeeze(0)
            topk = torch.topk(type_pred, k=min(top_k, len(node_ids)))
            original_ids = node_ids[topk.indices]
            results[target_type] = [
                {
                    "id": self.qa_data.id2node[nid.item()],
                    "type": target_type,
                    "attributes": self.node_info.loc[
                        self.qa_data.id2node[nid.item()], "attributes"
                    ],
                    "score": score.item(),
                }
                for nid, score in zip(original_ids, topk.values)
            ]
        return results

    def prepare_input_for_graph_retriever(self, query: str) -> dict:
        """
        Prepare input for the graph retriever model by processing the query through entity detection, linking and embedding generation. The function performs the following steps:

        1. Detects entities in the query using NER model
        2. Links detected entities to knowledge graph entities
        3. Converts entities to node masks
        4. Generates question embeddings
        5. Combines embeddings and masks into input format

        Args:
            query (str): Input query text to process

        Returns:
            dict: Dictionary containing processed inputs with keys:

                - question_embeddings: Embedded representation of the query
                - start_nodes_mask: Binary mask tensor indicating entity nodes (shape: 1 x num_nodes)

        Notes:
            - If no entities are detected in query, the full query is used for entity linking
            - Only linked entities that exist in qa_data.ent2id are included in masks
            - Entity masks and embeddings are formatted for graph retriever model input
        """

        # Prepare input for deep graph retriever
        mentioned_entities = self.ner_model(query)
        if len(mentioned_entities) == 0:
            logger.warning(
                "No mentioned entities found in the query. Use the query as is for entity linking."
            )
            mentioned_entities = [query]
        linked_entities = self.el_model(mentioned_entities, topk=1)
        entity_ids = [
            self.qa_data.node2id[ent[0]["entity"]]
            for ent in linked_entities.values()
            if ent[0]["entity"] in self.qa_data.node2id
        ]
        start_nodes_mask = (
            entities_to_mask(entity_ids, self.num_nodes).unsqueeze(0).to(self.device)
        )  # 1 x num_nodes
        question_embedding = self.text_emb_model.encode(
            [query],
            is_query=True,
            show_progress_bar=False,
        )
        graph_retriever_input = {
            "question_embeddings": question_embedding,
            "start_nodes_mask": start_nodes_mask,
        }
        return graph_retriever_input

    @staticmethod
    def _load_qa_data_from_model_config(
        data_dir: str,
        data_name: str,
        model_config: dict,
        force_reindex: bool,
    ) -> GraphIndexDataset:
        dataset_config = model_config.get("dataset_config")
        if dataset_config is None:
            raise ValueError("dataset_config not found in model config")

        dataset_cls = get_class(
            f"gfmrag.graph_index_datasets.{dataset_config['class_name']}"
        )
        assert issubclass(dataset_cls, GraphIndexDataset)

        dataset_kwargs = {
            key: value for key, value in dataset_config.items() if key != "class_name"
        }
        dataset_kwargs["text_emb_model_cfgs"] = OmegaConf.create(
            dataset_kwargs["text_emb_model_cfgs"]
        )
        return dataset_cls(
            root=data_dir,
            data_name=data_name,
            force_reload=force_reindex,
            **dataset_kwargs,
        )

    @staticmethod
    def from_index(
        data_dir: str,
        data_name: str,
        model_path: str,
        ner_model: BaseNERModel,
        el_model: BaseELModel,
        graph_constructor: BaseGraphConstructor | None = None,
        force_reindex: bool = False,
    ) -> "GFMRetriever":
        """Construct a GFMRetriever from a data directory.

        Detects whether processed/stage1/ exists. If not, uses graph_constructor
        to build it from raw/documents.json. Then restores the stage2 dataset from
        the checkpoint dataset config when available, indexes the entity linking
        model, and assembles the retriever.

        Args:
            data_dir: Root data directory (contains data_name/ subdirectory).
            data_name: Dataset subdirectory name.
            model_path: HuggingFace model ID or local path (e.g. "rmanluo/GFM-RAG-8M").
            ner_model: Instantiated NER model.
            el_model: Instantiated EL model. index() is called internally.
            graph_constructor: Required only when stage1/ does not exist.
            force_reindex: Force rebuild of stage2 processed files.

        Returns:
            Fully initialized GFMRetriever.

        Raises:
            FileNotFoundError: If raw/documents.json is missing.
            ValueError: If stage1/ is missing and graph_constructor is None.
        """
        stage1_dir = os.path.join(data_dir, data_name, "processed", "stage1")
        stage1_files = [os.path.join(stage1_dir, name) for name in _STAGE1_GRAPH_NAMES]

        if not utils.check_all_files_exist(stage1_files):
            raw_docs = os.path.join(data_dir, data_name, "raw", "documents.json")
            if not os.path.exists(raw_docs):
                raise FileNotFoundError(
                    f"raw/documents.json not found at {raw_docs}. "
                    "Provide documents.json or pre-built stage1/ CSV files."
                )
            if graph_constructor is None:
                raise ValueError(
                    "processed/stage1/ not found. Provide a graph_constructor "
                    "to build the graph from raw/documents.json."
                )
            logger.info(f"Building graph index for {data_name}")
            os.makedirs(stage1_dir, exist_ok=True)
            graph = graph_constructor.build_graph(data_dir, data_name)
            pd.DataFrame(graph["nodes"]).to_csv(
                os.path.join(stage1_dir, "nodes.csv"), index=False
            )
            pd.DataFrame(graph["edges"]).to_csv(
                os.path.join(stage1_dir, "edges.csv"), index=False
            )
            pd.DataFrame(graph["relations"]).to_csv(
                os.path.join(stage1_dir, "relations.csv"), index=False
            )
            logger.info(f"Stage1 graph files saved to {stage1_dir}")

        graph_retriever, model_config = utils.load_model_from_pretrained(model_path)
        graph_retriever.eval()

        qa_data = GFMRetriever._load_qa_data_from_model_config(
            data_dir=data_dir,
            data_name=data_name,
            model_config=model_config,
            force_reindex=force_reindex,
        )

        device = utils.get_device()
        graph_retriever = graph_retriever.to(device)
        qa_data.graph = qa_data.graph.to(device)

        el_model.index(list(qa_data.node2id.keys()))

        nodes_csv = os.path.join(stage1_dir, "nodes.csv")
        nodes_df = pd.read_csv(nodes_csv, keep_default_na=False)
        nodes_df["attributes"] = nodes_df["attributes"].apply(
            lambda x: {} if x == "" else ast.literal_eval(x)
        )
        id_col = "uid" if "uid" in nodes_df.columns else "name"
        nodes_df = nodes_df.set_index(id_col)

        text_emb_model = instantiate(qa_data.text_emb_model_cfgs)

        return GFMRetriever(
            qa_data=qa_data,
            text_emb_model=text_emb_model,
            ner_model=ner_model,
            el_model=el_model,
            graph_retriever=graph_retriever,
            node_info=nodes_df,
            device=device,
        )
