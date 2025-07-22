import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gfmrag import utils
from gfmrag.graph_index_construction.entity_linking_model import BaseELModel
from gfmrag.graph_index_construction.ner_model import BaseNERModel
from gfmrag.graph_index_datasets import GraphIndexDataset
from gfmrag.models import GNNRetriever
from gfmrag.models.ultra import query_utils
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils.qa_utils import entities_to_mask

logger = logging.getLogger(__name__)


class GFMRetriever:
    """Graph Foundation Model (GFM) Retriever for document retrieval.

    This class implements a document retrieval system that combines named entity recognition,
    entity linking, graph neural networks, and document ranking to retrieve relevant documents
    based on a query.

    Attributes:
        qa_data (GraphIndexDataset): Dataset containing the knowledge graph, documents and mappings
        graph (torch.Tensor): Knowledge graph structure
        text_emb_model (BaseTextEmbModel): Model for text embedding
        ner_model (BaseNERModel): Named Entity Recognition model
        el_model (BaseELModel): Entity Linking model
        graph_retriever (GNNRetriever): Graph Neural Network based retriever
        doc_retriever (DocumentRetriever): Map nodes id to documents
        target_type (str): Type of target for retrieval (e.g., "document")
        device (torch.device): Device to run computations on
        num_nodes (int): Number of nodes in the knowledge graph

    Examples:
        >>> retriever = GFMRetriever.from_config(cfg)
        >>> docs = retriever.retrieve("Who is the president of France?", top_k=5)
    """

    def __init__(
        self,
        qa_data: GraphIndexDataset,
        text_emb_model: BaseTextEmbModel,
        ner_model: BaseNERModel,
        el_model: BaseELModel,
        graph_retriever: GNNRetriever,
        doc_retriever: utils.DocumentRetriever,
        target_type: str,
        device: torch.device,
    ) -> None:
        self.qa_data = qa_data
        self.graph = qa_data.graph
        self.text_emb_model = text_emb_model
        self.ner_model = ner_model
        self.el_model = el_model
        self.graph_retriever = graph_retriever
        self.doc_retriever = doc_retriever
        self.device = device
        self.num_nodes = self.graph.num_nodes
        self.target_type = target_type

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """
        Retrieve documents from the corpus based on the given query.

        1. Prepares the query input for the graph retriever
        2. Executes the graph retriever forward pass to get entity predictions
        3. Ranks documents based on entity predictions
        4. Retrieves the top-k supporting documents

        Args:
            query (str): input query
            top_k (int): number of documents to retrieve

        Returns:
            list[dict]: A list of retrieved documents, where each document is represented as a dictionary
                        containing document metadata and content
        """

        # Prepare input for deep graph retriever
        graph_retriever_input = self.prepare_input_for_graph_retriever(query)
        graph_retriever_input = query_utils.cuda(
            graph_retriever_input, device=self.device
        )

        # Graph retriever forward pass
        pred = self.graph_retriever(self.graph, graph_retriever_input)
        target_pred = pred[:, self.graph.nodes_by_type[self.target_type]]

        # Retrieve the supporting documents
        retrieved_docs = self.doc_retriever(target_pred.cpu(), top_k=top_k)

        return retrieved_docs

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
            The configuration must contain valid paths and parameters for all required models
            and dataset components. Models are automatically moved to available device (CPU/GPU).
        """
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

        # Map id to name
        doc_retriever = utils.DocumentRetriever(qa_data.doc, qa_data.id2node)

        text_emb_model = instantiate(
            OmegaConf.create(model_config["text_emb_model_config"])
        )

        return GFMRetriever(
            qa_data=qa_data,
            text_emb_model=text_emb_model,
            ner_model=ner_model,
            el_model=el_model,
            graph_retriever=graph_retriever,
            doc_retriever=doc_retriever,
            target_type=cfg.graph_retriever.target_type,
            device=device,
        )
