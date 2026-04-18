import logging

import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gfmrag import utils
from gfmrag.graph_index_construction.entity_linking_model import BaseELModel
from gfmrag.graph_index_construction.ner_model import BaseNERModel
from gfmrag.graph_index_datasets import GraphIndexDataset
from gfmrag.models.base_model import BaseGNNModel
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils.qa_utils import entities_to_mask

logger = logging.getLogger(__name__)


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
    def from_config(cfg: DictConfig) -> "GFMRetriever":
        """
        Constructs a GFMRetriever instance from a configuration dictionary.

        This factory method initializes all necessary components for the GFM retrieval system including:
        - Graph retrieval model
        - Question-answering dataset
        - Named Entity Recognition (NER) model
        - Entity Linking (EL) model
        - Document ranking and retrieval components
        - Text embedding model

        Args:
            cfg (DictConfig): Configuration dictionary containing settings for:

                - graph_retriever: Model path and NER/EL model configurations
                - dataset: Dataset parameters
                - Optional entity weight initialization flag

        Returns:
            GFMRetriever: Fully initialized retriever instance with all components loaded and
                          moved to appropriate device (CPU/GPU)

        Note:
            Deprecated: This method is broken (node_info is an empty placeholder) and will be
            replaced by from_index() in the next task. Calling retrieve() on an instance
            created by this method will fail.

            The configuration must contain valid paths and parameters for all required models
            and dataset components. Models are automatically moved to available device (CPU/GPU).
        """
        import warnings

        warnings.warn(
            "from_config() is deprecated and will be replaced by from_index() in the next task. "
            "The node_info parameter is a placeholder and retrieve() will fail if called.",
            DeprecationWarning,
            stacklevel=2,
        )
        graph_retriever, model_config = utils.load_model_from_pretrained(
            cfg.graph_retriever.model_path
        )
        graph_retriever.eval()
        qa_data = GraphIndexDataset(
            **cfg.dataset,
            text_emb_model_cfgs=OmegaConf.create(model_config["text_emb_model_config"]),
        )
        device = utils.get_device()
        graph_retriever = graph_retriever.to(device)

        qa_data.graph = qa_data.graph.to(device)

        ner_model = instantiate(cfg.graph_retriever.ner_model)
        el_model = instantiate(cfg.graph_retriever.el_model)

        el_model.index(list(qa_data.node2id.keys()))

        text_emb_model = instantiate(
            OmegaConf.create(model_config["text_emb_model_config"])
        )

        # node_info is not available via from_config; placeholder until Task 4 replaces this method
        node_info = pd.DataFrame()

        return GFMRetriever(
            qa_data=qa_data,
            text_emb_model=text_emb_model,
            ner_model=ner_model,
            el_model=el_model,
            graph_retriever=graph_retriever,
            node_info=node_info,
            device=device,
        )
