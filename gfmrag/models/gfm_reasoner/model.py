from typing import Any, Literal

import torch
from torch import nn
from torch_geometric.data import Data

from gfmrag.models.base_model import BaseGNNModel
from gfmrag.ultra.models import QueryNBFNet


class QueryGNN(BaseGNNModel):
    """A neural network module for query embedding in graph neural networks.

    This class implements a query embedding model that combines relation embeddings with an entity-based graph neural network
    for knowledge graph completion tasks.

    Args:
        entity_model (EntityNBFNet): The entity-based neural network model for reasoning on graph structure.
        feat_dim (int): Dimension of the entity and relation embeddings.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        feat_dim (int): Dimension of entity and relation embeddings.
        entity_model (EntityNBFNet): The entity model instance.
        rel_mlp (nn.Linear): Linear transformation layer for relation embeddings.
        use_ent_emb (Literal[None, "early-fusion", "late-fusion"]): Specifies how to use entity embeddings.
            - None: No entity embeddings used
            - "early-fusion": Entity embeddings are fused early in the model
            - "late-fusion": Entity embeddings are fused late in the model

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
        self,
        entity_model: QueryNBFNet,
        feat_dim: int,
        use_ent_emb: Literal[
            None, "early-fusion", "late-fusion", "early-late-fusion"
        ] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the model.

        Args:
            entity_model (QueryNBFNet): The entity model component
            feat_dim (int): Dimension of relation embeddings
            use_ent_emb (Literal[None, "early-fusion", "late-fusion"]): Specifies how to use entity embeddings.
                - None: No entity embeddings used
                - "early-fusion": Entity embeddings are fused early in the model
                - "late-fusion": Entity embeddings are fused late in the model
            *args (Any): Variable length argument list
            **kwargs (Any): Arbitrary keyword arguments

        """

        super().__init__()
        self.feat_dim = feat_dim
        self.use_ent_emb = use_ent_emb
        self.entity_model = entity_model
        self.rel_mlp = nn.Linear(feat_dim, self.entity_model.dims[0])
        self.question_mlp = nn.Linear(self.feat_dim, self.entity_model.dims[0])

        if self.use_ent_emb is not None:
            self.ent_mlp = nn.Linear(feat_dim, self.entity_model.dims[0])

        if (
            self.use_ent_emb == "early-fusion"
            or self.use_ent_emb == "early-late-fusion"
        ):
            self.early_fuse_mlp = nn.Sequential(
                nn.Linear(self.entity_model.dims[0] * 2, self.entity_model.dims[0]),
                nn.ReLU(),
                nn.Linear(self.entity_model.dims[0], self.entity_model.dims[0]),
            )

        if self.use_ent_emb == "late-fusion" or self.use_ent_emb == "early-late-fusion":
            self.predict_mlp = nn.Sequential(
                nn.Linear(self.entity_model.dims[0] * 3, self.entity_model.dims[0]),
                nn.ReLU(),
                nn.Linear(self.entity_model.dims[0], 1),
            )

    def get_input_node_feature(
        self, graph: Data, query_head: torch.Tensor, query_representation: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the input node features for the GNN model.

        Args:
            graph (Data): Graph data object containing entity embeddings and graph structure.
            query_head (torch.Tensor): Tensor of head indices for the query.
            query_representation (torch.Tensor): query relation representations.

        Returns:
            torch.Tensor: The input node features for GNN
        """
        batch_size = len(query_head)
        index = query_head.unsqueeze(-1).expand_as(query_representation)
        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(
            batch_size,
            graph.num_nodes,
            self.entity_model.dims[0],
            device=query_head.device,
        )
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query_representation.unsqueeze(1))

        if self.use_ent_emb == "early-fusion":
            # if we use early-fusion, we add entity embeddings to the boundary condition
            ent_emb = self.ent_mlp(graph.x)
            boundary += ent_emb.unsqueeze(0).expand_as(boundary)

        return boundary

    def forward(self, graph: Data, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            graph (Data): Graph data object containing entity embeddings and graph structure.
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
            self.rel_mlp(graph.rel_attr).unsqueeze(0).expand(batch_size, -1, -1)
        )
        h_index, t_index, r_index = batch.unbind(-1)

        # Obtain entity embeddings
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.entity_model.negative_sample_to_tail(
            h_index, t_index, r_index, num_direct_rel=graph.num_relations // 2
        )
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        query_head = h_index[
            :, 0
        ]  # take the first head index for all triples in the batch
        query_relation = r_index[
            :, 0
        ]  # take the first relation index for all triples in the batch

        # Get the input embedding for the query head and relation
        raw_rel_emb = graph.rel_attr.unsqueeze(0).expand(batch_size, -1, -1)
        query_relation_emb = raw_rel_emb[
            torch.arange(batch_size, device=r_index.device), query_relation
        ]
        query_embedding = self.question_mlp(query_relation_emb)  # shape: (bs, emb_dim)
        node_embedding = self.get_input_node_feature(graph, query_head, query_embedding)

        # to make NBFNet iteration learn non-trivial paths
        graph = self.entity_model.remove_easy_edges(graph, h_index, t_index, r_index)
        score = self.entity_model(
            graph, node_embedding, relation_representations, query_embedding
        )

        return score


class GraphReasoner(QueryGNN):
    """A Query-dependent Graph Neural Network that reasons over the graph structure to identify relevant information.

    This class extends QueryGNN to implement a GNN-based reasoner system that processes question embeddings and entity information to find relevant information from a graph.

    Attributes:
        question_mlp (nn.Linear): Linear layer for transforming question embeddings.

    Args:
        entity_model (QueryNBFNet): The underlying query-dependent GNN for reasoning on graph.
        feat_dim (int): Dimension of relation embeddings.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Methods:
        forward(graph, batch, entities_weight=None):
            Processes the input graph and question embeddings to generate reasoning scores.

            Args:
                graph (Data): The input graph structure.
                batch (dict[str, torch.Tensor]): Batch of input data containing question embeddings and masks.
                entities_weight (torch.Tensor, optional): Optional weights for entities.

            Returns:
                torch.Tensor: Output scores.

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

    """Wrap the GNN model for reasoning."""

    def forward(  # type: ignore[override]
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
                - start_nodes_mask: Tensor of masks for question entities
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
        question_entities_mask = batch["start_nodes_mask"]

        question_embedding = self.question_mlp(question_emb)  # shape: (bs, emb_dim)
        batch_size = question_embedding.size(0)
        relation_representations = (
            self.rel_mlp(graph.rel_attr).unsqueeze(0).expand(batch_size, -1, -1)
        )

        # initialize the input with the fuzzy set and question embs
        if entities_weight is not None:
            question_entities_mask = question_entities_mask * entities_weight.unsqueeze(
                0
            )

        node_embedding = torch.einsum(
            "bn, bd -> bnd", question_entities_mask, question_embedding
        )
        if (
            self.use_ent_emb == "early-fusion"
            or self.use_ent_emb == "early-late-fusion"
        ):
            # if we use early-fusion, we add entity embeddings to the input
            ent_emb = self.ent_mlp(graph.x)
            # node_embedding += ent_emb.unsqueeze(0).expand_as(node_embedding)
            node_embedding = self.early_fuse_mlp(
                torch.cat(
                    [node_embedding, ent_emb.unsqueeze(0).expand_as(node_embedding)],
                    dim=-1,
                )
            )

        # GNN model: run the entity-level reasoner to get a scalar distribution over nodes
        output = self.entity_model(
            graph, node_embedding, relation_representations, question_embedding
        )  # shape: (bs, num_nodes, emb_dim)
        if self.use_ent_emb == "late-fusion" or self.use_ent_emb == "early-late-fusion":
            ent_late_emb = (
                self.ent_mlp(graph.x).unsqueeze(0).expand(batch_size, -1, -1)
            )  # shape: (bs, num_nodes, emb_dim)
            output = self.predict_mlp(
                torch.cat([output, ent_late_emb], dim=-1)
            ).squeeze(-1)  # shape: (bs, num_nodes)
        elif self.use_ent_emb == "early-late-fusion":
            output = self.predict_mlp(
                torch.cat(
                    [output, ent_emb.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1
                )
            ).squeeze(-1)  # shape: (bs, num_nodes)

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
                - start_nodes_mask: Binary mask tensor indicating question entities
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
        question_entities_mask = sample["start_nodes_mask"]
        question_embedding = self.question_mlp(question_emb)  # shape: (bs, emb_dim)
        batch_size = question_embedding.size(0)

        assert batch_size == 1, "Currently only supports batch size 1 for visualization"

        relation_representations = (
            self.rel_mlp(graph.rel_attr).unsqueeze(0).expand(batch_size, -1, -1)
        )

        # initialize the input with the fuzzy set and question embs
        if entities_weight is not None:
            question_entities_mask = question_entities_mask * entities_weight.unsqueeze(
                0
            )

        node_embedding = torch.einsum(
            "bn, bd -> bnd", question_entities_mask, question_embedding
        )
        if self.use_ent_emb == "early-fusion":
            # if we use early-fusion, we add entity embeddings to the input
            ent_emb = self.ent_mlp(graph.x)
            node_embedding += ent_emb.unsqueeze(0).expand_as(node_embedding)

        return self.entity_model.visualize(
            graph,
            sample,
            node_embedding,
            relation_representations,
            question_embedding,  # type: ignore
        )
