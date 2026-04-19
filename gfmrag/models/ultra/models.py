# mypy: ignore-errors
import torch
from torch import autograd, nn

from . import layers
from .base_nbfnet import BaseNBFNet


class EntityNBFNet(BaseNBFNet):
    """Neural Bellman-Ford Network for entity prediction."""

    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):
        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)
        self.return_hidden = kwargs.get("return_hidden", False)
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i],
                    self.dims[i + 1],
                    num_relation,
                    self.dims[0],
                    self.message_func,
                    self.aggregate_func,
                    self.layer_norm,
                    self.activation,
                    dependent=False,
                    project_relations=True,
                )
            )

        feature_dim = (
            sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]
        ) + input_dim
        if not self.return_hidden:
            self.mlp = nn.Sequential()
            mlp = []
            for i in range(self.num_mlp_layers - 1):
                mlp.append(nn.Linear(feature_dim, feature_dim))
                mlp.append(nn.ReLU())
            mlp.append(nn.Linear(feature_dim, 1))
            self.mlp = nn.Sequential(*mlp)

    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(
            batch_size, data.num_nodes, self.dims[0], device=h_index.device
        )
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(
                layer_input,
                query,
                boundary,
                data.edge_index,
                data.edge_type,
                size,
                edge_weight,
            )
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(
            -1, data.num_nodes, -1
        )  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch):
        h_index, t_index, r_index = batch.unbind(-1)

        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        # if self.training:
        # Edge dropout in the training mode
        # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
        # to make NBFNet iteration learn non-trivial paths
        # data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index, num_direct_rel=data.num_relations // 2
        )
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(
            data, h_index[:, 0], r_index[:, 0]
        )  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(
            1, index
        )  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)


class QueryNBFNet(EntityNBFNet):
    """
    The entity-level reasoner for UltraQuery-like complex query answering pipelines.

    This class extends EntityNBFNet to handle query-specific reasoning in knowledge graphs.
    Key differences from EntityNBFNet include:

    1. Initial node features are provided during forward pass rather than read from triples batch
    2. Query comes from outer loop
    3. Returns distribution over all nodes (assuming t_index covers all nodes)

    Attributes:
        layers: List of neural network layers for message passing
        short_cut: Boolean flag for using residual connections
        concat_hidden: Boolean flag for concatenating hidden states
        mlp: Multi-layer perceptron for final scoring
        num_beam: Beam size for path search
        path_topk: Number of top paths to return

    Methods:
        bellmanford(data, node_features, query, separate_grad=False):
            Performs Bellman-Ford message passing iterations.
            Args:
                data: Graph data object containing edge information
                node_features: Initial node representations
                query: Query representation
                separate_grad: Whether to track gradients separately for edges
            Returns:
                dict: Contains node features and edge weights

        forward(data, node_features, relation_representations, query):
            Main forward pass of the model.
            Args:
                data: Graph data object
                node_features: Initial node features
                relation_representations: Representations for relations
                query: Query representation
            Returns:
                torch.Tensor: Scores for each node

        visualize(data, sample, node_features, relation_representations, query):
            Visualizes reasoning paths for given entities.
            Args:
                data: Graph data object
                sample: Dictionary containing entity masks
                node_features: Initial node features
                relation_representations: Representations for relations
                query: Query representation
            Returns:
                dict: Contains paths and weights for target entities
    """

    def bellmanford(self, data, node_features, query, separate_grad=False):
        import torch.distributed as dist

        dist_context = getattr(data, "dist_context", None)
        is_dist = dist_context is not None and dist.is_initialized()
        boundary_mode = getattr(data, "boundary_mode", False)

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=query.device, dtype=query.dtype)

        hiddens = []
        edge_weights = []

        if is_dist and boundary_mode:
            # ------------------------------------------------------------------
            # Boundary-only AllGather (METIS partition)
            # ------------------------------------------------------------------
            # Scalable variant: only communicate boundary source node states
            # instead of the full hidden tensor.  The compact tensor layout is
            # [local_nodes | boundary_nodes] with edges pre-remapped to this
            # space by partition_graph_metis().
            # ------------------------------------------------------------------
            rank, world_size = dist_context
            num_nodes = data.num_nodes
            local_nodes = data.local_nodes
            boundary_nodes = data.boundary_nodes
            compact_size = data.compact_size
            node2part = data.node2part
            local_N = local_nodes.shape[0]
            boundary_N = boundary_nodes.shape[0]

            # Collect local_nodes from all ranks for scatter-back.
            all_local_nodes_list = [None] * world_size
            dist.all_gather_object(all_local_nodes_list, local_nodes.cpu())
            all_local_N = [len(ns) for ns in all_local_nodes_list]
            max_local_N = max(all_local_N)

            def _scatter_allgather(local_t):
                """AllGather local outputs and scatter to global positions."""
                B, loc_n, D = local_t.shape
                if loc_n < max_local_N:
                    pad = local_t.new_zeros(B, max_local_N - loc_n, D)
                    padded = torch.cat([local_t, pad], dim=1).contiguous()
                else:
                    padded = local_t.contiguous()
                chunks = [torch.zeros_like(padded) for _ in range(world_size)]
                dist.all_gather(chunks, padded)
                # Scatter each rank's results to correct global positions.
                output = local_t.new_zeros(B, num_nodes, D)
                for r in range(world_size):
                    r_nodes = all_local_nodes_list[r].to(local_t.device)
                    r_data = chunks[r][:, : all_local_N[r], :]
                    output[:, r_nodes, :] = r_data
                return output

            # Precompute boundary exchange info (once per forward).
            boundary_owners = node2part[boundary_nodes]
            all_boundary_nodes_list = [None] * world_size
            dist.all_gather_object(all_boundary_nodes_list, boundary_nodes.cpu())

            local_nodes_set = set(local_nodes.cpu().tolist())
            send_indices = {}
            for r in range(world_size):
                if r == rank:
                    continue
                their_boundary = set(all_boundary_nodes_list[r].tolist())
                needed_from_us = their_boundary & local_nodes_set
                if needed_from_us:
                    needed_t = torch.tensor(sorted(needed_from_us), device=query.device)
                    idx = torch.searchsorted(local_nodes, needed_t)
                    send_indices[r] = idx

            recv_indices = {}
            for r in range(world_size):
                if r == rank:
                    continue
                mask = boundary_owners == r
                if mask.any():
                    recv_indices[r] = mask.nonzero(as_tuple=True)[0]

            # Initial local hidden states and compact boundary condition.
            local_layer_input = node_features[:, local_nodes, :].clone()
            compact_boundary = torch.cat([
                node_features[:, local_nodes, :],
                node_features[:, boundary_nodes, :],
            ], dim=1)

            compact_edge_size = (compact_size, compact_size)

            for layer in self.layers:
                if separate_grad:
                    edge_weight = edge_weight.clone().requires_grad_()

                # Exchange boundary states.
                B, _, D = local_layer_input.shape
                boundary_hidden = local_layer_input.new_zeros(B, boundary_N, D)

                send_data = {}
                for r, idx in send_indices.items():
                    send_data[r] = local_layer_input[:, idx, :].contiguous()

                all_send_data = [None] * world_size
                dist.all_gather_object(all_send_data, send_data)

                for r in range(world_size):
                    if r == rank:
                        continue
                    if rank in all_send_data[r]:
                        states = all_send_data[r][rank].to(query.device)
                        boundary_hidden[:, recv_indices[r], :] = states

                compact_input = torch.cat(
                    [local_layer_input, boundary_hidden], dim=1
                )

                hidden = layer(
                    compact_input,
                    query,
                    compact_boundary,
                    data.edge_index,
                    data.edge_type,
                    compact_edge_size,
                    edge_weight,
                )

                local_hidden = hidden[:, :local_N, :]
                if self.short_cut and local_hidden.shape == local_layer_input.shape:
                    local_hidden = local_hidden + local_layer_input
                hiddens.append(local_hidden)
                edge_weights.append(edge_weight)
                local_layer_input = local_hidden

            node_query_local = (
                query.unsqueeze(1).expand(-1, local_N, -1).contiguous()
            )
            if self.concat_hidden:
                local_output = torch.cat(hiddens + [node_query_local], dim=-1)
            else:
                local_output = torch.cat([hiddens[-1], node_query_local], dim=-1)

            output = _scatter_allgather(local_output)

        elif is_dist:
            # ------------------------------------------------------------------
            # Distributed split-graph inference
            # ------------------------------------------------------------------
            # Strategy (mathematically exact):
            #   1. Each rank owns nodes [local_start, local_end) and the edges
            #      whose *target* falls in that slice (set by partition_graph_edges).
            #   2. Before each layer: AllGather local hidden states â†’ full (B,N,D).
            #   3. Run the layer with the full source states but local-only edges.
            #      The layer output is correct at local target positions; non-local
            #      positions contain boundary-only values and are discarded.
            #   4. Slice to local portion, apply residual, store.
            #   5. After all layers: AllGather the local concatenated output once
            #      to reconstruct the full result on every rank.
            # ------------------------------------------------------------------
            rank, world_size = dist_context
            num_nodes = data.num_nodes
            base_N = (num_nodes + world_size - 1) // world_size  # ceiling division
            local_start = rank * base_N
            local_end = min((rank + 1) * base_N, num_nodes)
            local_N = local_end - local_start

            # Collect each rank's actual local_N once (for uneven last partition).
            local_N_t = torch.tensor(local_N, device=query.device)
            all_local_N_list = [torch.zeros_like(local_N_t) for _ in range(world_size)]
            dist.all_gather(all_local_N_list, local_N_t)
            all_local_N = [int(x.item()) for x in all_local_N_list]
            max_local_N = max(all_local_N)  # == base_N

            def _allgather(local_t: torch.Tensor) -> torch.Tensor:
                """AllGather (B, local_N, D) across ranks into (B, N, D).

                Handles the case where the last rank may have fewer nodes than
                the others by zero-padding to max_local_N before gathering,
                then slicing each chunk to its actual size before concatenation.
                """
                B, loc_n, D = local_t.shape
                if loc_n < max_local_N:
                    pad = local_t.new_zeros(B, max_local_N - loc_n, D)
                    padded = torch.cat([local_t, pad], dim=1).contiguous()
                else:
                    padded = local_t.contiguous()
                chunks = [torch.zeros_like(padded) for _ in range(world_size)]
                dist.all_gather(chunks, padded)
                return torch.cat(
                    [chunks[r][:, : all_local_N[r], :] for r in range(world_size)],
                    dim=1,
                )  # (B, N, D)

            # Local slice of the initial boundary / layer input.
            local_layer_input = node_features[:, local_start:local_end, :].clone()

            for layer in self.layers:
                if separate_grad:
                    edge_weight = edge_weight.clone().requires_grad_()

                # AllGather â†’ full source-node states on each rank.
                global_input = _allgather(local_layer_input)  # (B, N, D)

                # Layer forward with full input but local-target edges.
                # hidden[v] is correct for v in [local_start, local_end);
                # non-local positions contain boundary-only values (discarded).
                hidden = layer(
                    global_input,
                    query,
                    node_features,  # boundary: full (B, N, D), same on all ranks
                    data.edge_index,
                    data.edge_type,
                    size,
                    edge_weight,
                )

                # Slice to local, apply residual, then store local hidden.
                local_hidden = hidden[:, local_start:local_end, :]  # (B, local_N, D)
                if self.short_cut and local_hidden.shape == local_layer_input.shape:
                    local_hidden = local_hidden + local_layer_input
                hiddens.append(local_hidden)
                edge_weights.append(edge_weight)
                local_layer_input = local_hidden

            # Concatenate local hidden slices (+ local node_query) then AllGather.
            node_query_local = (
                query.unsqueeze(1).expand(-1, local_N, -1).contiguous()
            )  # (B, local_N, input_dim)
            if self.concat_hidden:
                local_output = torch.cat(hiddens + [node_query_local], dim=-1)
            else:
                local_output = torch.cat([hiddens[-1], node_query_local], dim=-1)
            # local_output: (B, local_N, out_dim)

            output = _allgather(local_output)  # (B, N, out_dim)

        else:
            # ------------------------------------------------------------------
            # Standard single-process path (unchanged)
            # ------------------------------------------------------------------
            layer_input = node_features
            for layer in self.layers:
                if separate_grad:
                    edge_weight = edge_weight.clone().requires_grad_()

                # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
                hidden = layer(
                    layer_input,
                    query,
                    node_features,
                    data.edge_index,
                    data.edge_type,
                    size,
                    edge_weight,
                )
                if self.short_cut and hidden.shape == layer_input.shape:
                    # residual connection here
                    hidden = hidden + layer_input
                hiddens.append(hidden)
                edge_weights.append(edge_weight)
                layer_input = hidden

            # original query (relation type) embeddings
            node_query = query.unsqueeze(1).expand(
                -1, data.num_nodes, -1
            )  # (batch_size, num_nodes, input_dim)
            if self.concat_hidden:
                output = torch.cat(hiddens + [node_query], dim=-1)
            else:
                output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, node_features, relation_representations, query):
        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        # we already did traversal_dropout in the outer loop of UltraQuery
        # if self.training:
        #     # Edge dropout in the training mode
        #     # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
        #     # to make NBFNet iteration learn non-trivial paths
        #     data = self.remove_easy_edges(data, h_index, t_index, r_index)

        # node features arrive in shape (bs, num_nodes, dim)
        # NBFNet needs batch size on the first place
        output = self.bellmanford(
            data, node_features, query
        )  # (num_nodes, batch_size, feature_dim）
        if self.return_hidden:
            return output["node_feature"]
        else:
            score = self.mlp(output["node_feature"]).squeeze(-1)  # (bs, num_nodes)
            # return only the score
            return score

    def visualize(self, data, sample, node_features, relation_representations, query):
        for layer in self.layers:
            layer.relation = relation_representations

        output = self.bellmanford(
            data, node_features, query, separate_grad=True
        )  # (num_nodes, batch_size, feature_dim）
        node_feature = output["node_feature"]
        edge_weights = output["edge_weights"]
        question_entities_mask = sample["start_nodes_mask"]
        target_entities_mask = sample["target_nodes_mask"]
        query_entities_index = question_entities_mask.nonzero(as_tuple=True)[1]
        target_entities_index = target_entities_mask.nonzero(as_tuple=True)[1]

        paths_results = {}
        for t_index in target_entities_index:
            index = (
                t_index.unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(-1, -1, node_feature.shape[-1])
            )
            feature = node_feature.gather(1, index).squeeze(0)
            score = self.mlp(feature).squeeze(-1)

            edge_grads = autograd.grad(score, edge_weights, retain_graph=True)
            distances, back_edges = self.beam_search_distance(
                data, edge_grads, query_entities_index, t_index, self.num_beam
            )
            paths, weights = self.topk_average_length(
                distances, back_edges, t_index, self.path_topk
            )
            paths_results[t_index.item()] = (paths, weights)
        return paths_results
