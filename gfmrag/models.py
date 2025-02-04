from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data

from gfmrag.ultra.models import EntityNBFNet, QueryNBFNet


class QueryGNN(nn.Module):
    """A neural network module for query embedding in graph neural networks.

    This class implements a query embedding model that combines relation embeddings with an entity-based graph neural network
    for knowledge graph completion tasks.

    Args:
        entity_model (EntityNBFNet): The entity-based neural network model for reasoning on graph structure.
        rel_emb_dim (int): Dimension of the relation embeddings.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        rel_emb_dim (int): Dimension of relation embeddings.
        entity_model (EntityNBFNet): The entity model instance.
        rel_mlp (nn.Linear): Linear transformation layer for relation embeddings.

    Methods:
        forward(data: Data, batch: torch.Tensor) -> torch.Tensor:
            Forward pass of the query GNN model.

            Args:
                data (Data): Graph data object containing the knowledge graph structure and features.
                batch (torch.Tensor): Batch of triples with shape (batch_size, 1+num_negatives, 3),
                                    where each triple contains (head, tail, relation) indices.

            Returns:
                torch.Tensor: Scoring tensor for the input triples.
    """

    def __init__(
        self, entity_model: EntityNBFNet, rel_emb_dim: int, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the model.

        Args:
            entity_model (EntityNBFNet): The entity model component
            rel_emb_dim (int): Dimension of relation embeddings
            *args (Any): Variable length argument list
            **kwargs (Any): Arbitrary keyword arguments

        """

        super().__init__()
        self.rel_emb_dim = rel_emb_dim
        self.entity_model = entity_model
        self.rel_mlp = nn.Linear(rel_emb_dim, self.entity_model.dims[0])

    def forward(self, data: Data, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            data (Data): Graph data object containing entity embeddings and graph structure.
            batch (torch.Tensor): Batch of triple indices with shape (batch_size, 1+num_negatives, 3),
                                where each triple contains (head_idx, tail_idx, relation_idx).

        Returns:
            torch.Tensor: Scores for the triples in the batch.

        Notes:
            - Relations are assumed to be the same across all positive and negative triples
            - Easy edges are removed before processing to encourage learning of non-trivial paths
            - The batch tensor contains both positive and negative samples where the first sample
              is positive and the rest are negative samples
        """
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        batch_size = len(batch)
        relation_representations = (
            self.rel_mlp(data.rel_emb).unsqueeze(0).expand(batch_size, -1, -1)
        )
        h_index, t_index, r_index = batch.unbind(-1)
        # to make NBFNet iteration learn non-trivial paths
        data = self.entity_model.remove_easy_edges(data, h_index, t_index, r_index)
        score = self.entity_model(data, relation_representations, batch)

        return score


class GNNRetriever(QueryGNN):
    """A Query-dependent Graph Neural Network-based retriever that processes questions and entities for information retrieval.

    This class extends QueryGNN to implement a GNN-based retrieval system that processes question
    embeddings and entity information to retrieve relevant information from a graph.

    Attributes:
        question_mlp (nn.Linear): Linear layer for transforming question embeddings.

    Args:
        entity_model (QueryNBFNet): The underlying query-dependent GNN for reasoning on graph.
        rel_emb_dim (int): Dimension of relation embeddings.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Methods:
        forward(graph, batch, entities_weight=None):
            Processes the input graph and question embeddings to generate retrieval scores.

            Args:
                graph (Data): The input graph structure.
                batch (dict[str, torch.Tensor]): Batch of input data containing question embeddings and masks.
                entities_weight (torch.Tensor, optional): Optional weights for entities.

            Returns:
                torch.Tensor: Output scores for retrieval.

        visualize(graph, sample, entities_weight=None):
            Generates visualization data for the model's reasoning process.

            Args:
                graph (Data): The input graph structure.
                sample (dict[str, torch.Tensor]): Single sample data containing question embeddings and masks.
                entities_weight (torch.Tensor, optional): Optional weights for entities.

            Returns:
                dict[int, torch.Tensor]: Visualization data for each reasoning step.

    Note:
        The visualization method currently only supports batch size of 1.
    """

    """Wrap the GNN model for retrieval."""

    def __init__(
        self, entity_model: QueryNBFNet, rel_emb_dim: int, *args: Any, **kwargs: Any
    ) -> None:
        """
        Initialize the RelGFM model.

        Args:
            entity_model (QueryNBFNet): Model for entity embedding and message passing
            rel_emb_dim (int): Dimension of relation embeddings
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            None

        Note:
            This constructor initializes the base class with entity_model and rel_emb_dim,
            and creates a linear layer to project question embeddings to entity dimension.
        """

        super().__init__(entity_model, rel_emb_dim)
        self.question_mlp = nn.Linear(self.rel_emb_dim, self.entity_model.dims[0])

    def forward(
        self,
        graph: Data,
        batch: dict[str, torch.Tensor],
        entities_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        This method processes a graph and question embeddings to produce entity-level reasoning output.

        Args:
            graph (Data): A PyTorch Geometric Data object containing the graph structure and features.
            batch (dict[str, torch.Tensor]): A dictionary containing:
                - question_embeddings: Tensor of question embeddings
                - question_entities_masks: Tensor of masks for question entities
            entities_weight (torch.Tensor | None, optional): Optional weight tensor for entities. Defaults to None.

        Returns:
            torch.Tensor: The output tensor representing entity-level reasoning results.

        Notes:
            The forward pass includes:
            1. Processing question embeddings through MLP
            2. Expanding relation representations
            3. Applying optional entity weights
            4. Computing entity-question interaction
            5. Running entity-level reasoning model
        """

        question_emb = batch["question_embeddings"]
        question_entities_mask = batch["question_entities_masks"]

        question_embedding = self.question_mlp(question_emb)  # shape: (bs, emb_dim)
        batch_size = question_embedding.size(0)
        relation_representations = (
            self.rel_mlp(graph.rel_emb).unsqueeze(0).expand(batch_size, -1, -1)
        )

        # initialize the input with the fuzzy set and question embs
        if entities_weight is not None:
            question_entities_mask = question_entities_mask * entities_weight.unsqueeze(
                0
            )

        input = torch.einsum(
            "bn, bd -> bnd", question_entities_mask, question_embedding
        )

        # GNN model: run the entity-level reasoner to get a scalar distribution over nodes
        output = self.entity_model(
            graph, input, relation_representations, question_embedding
        )

        return output

    def visualize(
        self,
        graph: Data,
        sample: dict[str, torch.Tensor],
        entities_weight: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        """Visualizes attention weights and intermediate states for the model.

        This function generates visualization data for understanding how the model processes
        inputs and generates entity predictions. It is designed for debugging and analysis purposes.

        Args:
            graph (Data): The input knowledge graph structure containing entity and relation information
            sample (dict[str, torch.Tensor]): Dictionary containing:
                - question_embeddings: Tensor of question text embeddings
                - question_entities_masks: Binary mask tensor indicating question entities
            entities_weight (torch.Tensor | None, optional): Optional tensor of entity weights to apply.
                Defaults to None.

        Returns:
            dict[int, torch.Tensor]: Dictionary mapping layer indices to attention weight tensors,
                allowing visualization of attention patterns at different model depths.

        Note:
            Currently only supports batch size of 1 for visualization purposes.

        Raises:
            AssertionError: If batch size is not 1
        """

        question_emb = sample["question_embeddings"]
        question_entities_mask = sample["question_entities_masks"]
        question_embedding = self.question_mlp(question_emb)  # shape: (bs, emb_dim)
        batch_size = question_embedding.size(0)

        assert batch_size == 1, "Currently only supports batch size 1 for visualization"

        relation_representations = (
            self.rel_mlp(graph.rel_emb).unsqueeze(0).expand(batch_size, -1, -1)
        )

        # initialize the input with the fuzzy set and question embs
        if entities_weight is not None:
            question_entities_mask = question_entities_mask * entities_weight.unsqueeze(
                0
            )

        input = torch.einsum(
            "bn, bd -> bnd", question_entities_mask, question_embedding
        )
        return self.entity_model.visualize(
            graph,
            sample,
            input,
            relation_representations,
            question_embedding,  # type: ignore
        )
