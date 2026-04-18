import torch
import torch.distributed as dist
from torch_geometric.data import Data


def partition_graph_metis(graph: Data, rank: int, world_size: int) -> Data:
    """Partition a graph using METIS and precompute boundary information.

    Each rank owns a set of nodes determined by METIS graph partitioning.
    Edges are assigned to the rank that owns their target (destination) node.
    Boundary source nodes (remote sources needed by local edges) are identified
    so that ``bellmanford`` can AllGather only those states instead of the full
    hidden tensor.

    The returned graph has remapped edge indices into a compact local space
    ``[0, local_N + boundary_N)`` where the first ``local_N`` positions are
    owned nodes and the remaining ``boundary_N`` positions are remote sources.

    Attributes attached to the returned graph:
        - ``dist_context``: ``(rank, world_size)``
        - ``local_nodes``: global IDs of owned nodes, shape ``(local_N,)``
        - ``boundary_nodes``: global IDs of boundary sources, shape ``(boundary_N,)``
        - ``compact_size``: ``local_N + boundary_N`` (size of compact tensor)
        - ``node2part``: partition assignment for all nodes, shape ``(N,)``
        - ``boundary_mode``: ``True`` — signals bellmanford to use boundary-only path

    Edge convention note
    --------------------
    Same as ``partition_graph_edges``: ``edge_index[0]`` is destination,
    ``edge_index[1]`` is source (reversed PyG, matching rspmm kernel).
    """
    import pymetis

    num_nodes: int = graph.num_nodes  # type: ignore[assignment]
    edge_index: torch.Tensor = graph.edge_index  # type: ignore[assignment]
    edge_type: torch.Tensor = graph.edge_type  # type: ignore[assignment]
    dst = edge_index[0].cpu()
    src = edge_index[1].cpu()

    # --- METIS partitioning (rank 0 computes, broadcasts to all) ---
    def _compute_metis() -> list[int]:
        adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
        for s, d in zip(src.tolist(), dst.tolist()):
            adjacency[s].append(d)
            adjacency[d].append(s)
        adjacency = [list(set(neighbors)) for neighbors in adjacency]
        _, parts = pymetis.part_graph(world_size, adjacency)
        return parts

    if dist.is_initialized() and dist.get_world_size() > 1:
        if dist.get_rank() == 0:
            node2part_list = _compute_metis()
        else:
            node2part_list = None
        # broadcast_object_list works over the gloo/CPU backend automatically.
        container = [node2part_list]
        dist.broadcast_object_list(container, src=0)
        node2part = torch.tensor(container[0], dtype=torch.long)
    else:
        node2part = torch.tensor(_compute_metis(), dtype=torch.long)

    # --- Identify local nodes and local edges ---
    local_nodes = (node2part == rank).nonzero(as_tuple=True)[0]  # global IDs
    local_n = local_nodes.shape[0]

    # Edges whose target is owned by this rank
    target_part = node2part[dst]
    edge_mask = target_part == rank
    local_dst = dst[edge_mask]
    local_src = src[edge_mask]
    local_edge_type = edge_type[edge_mask.to(edge_type.device)]

    # --- Identify boundary source nodes ---
    src_part = node2part[local_src]
    remote_src_mask = src_part != rank
    boundary_nodes = local_src[remote_src_mask].unique().sort().values  # global IDs
    boundary_n = boundary_nodes.shape[0]

    # --- Build compact index mapping: [local_nodes | boundary_nodes] ---
    # global_id -> compact_id
    compact_size = local_n + boundary_n
    global_to_compact = torch.full((num_nodes,), -1, dtype=torch.long)
    global_to_compact[local_nodes] = torch.arange(local_n)
    global_to_compact[boundary_nodes] = torch.arange(local_n, compact_size)

    # Remap edge indices to compact space
    compact_dst = global_to_compact[local_dst]
    compact_src = global_to_compact[local_src]
    assert (compact_dst >= 0).all() and (compact_src >= 0).all(), (
        "Edge remapping failed: some nodes not in compact set"
    )
    compact_edge_index = torch.stack([compact_dst, compact_src], dim=0)

    # --- Build output graph ---
    local_graph = graph.clone()
    local_graph.edge_index = compact_edge_index.to(graph.edge_index.device)
    local_graph.edge_type = local_edge_type
    local_graph.num_edges = int(edge_mask.sum().item())

    # Metadata for bellmanford
    local_graph.dist_context = (rank, world_size)
    local_graph.local_nodes = local_nodes.to(graph.edge_index.device)
    local_graph.boundary_nodes = boundary_nodes.to(graph.edge_index.device)
    local_graph.compact_size = compact_size
    local_graph.node2part = node2part.to(graph.edge_index.device)
    local_graph.boundary_mode = True

    return local_graph


def partition_graph_edges(graph: Data, rank: int, world_size: int) -> Data:
    """Return a copy of *graph* with edges filtered to those whose target node
    falls in the local partition for *rank*.

    Partitioning uses ceiling division so that each rank gets at most
    ``ceil(N / world_size)`` nodes.  The last rank may receive slightly fewer.

    A ``dist_context`` attribute ``(rank, world_size)`` is attached so that
    ``QueryNBFNet.bellmanford`` knows to run distributed message passing.

    Edge convention note
    --------------------
    The rspmm CUDA kernel uses a reversed PyG convention: ``edge_index[0]`` is
    the *destination* (output) node and ``edge_index[1]`` is the *source*
    (input) node.  Partitioning therefore filters by ``edge_index[0]`` so that
    each rank holds all edges that write output into its local node slice.

    Correctness argument
    --------------------
    Each rank maintains the hidden states for its local node slice.  Before
    every bellmanford layer each rank AllGathers the full hidden state, runs
    the layer with its local edges (giving correct output only at local target
    positions), then slices back to the local portion.  After all layers the
    local output slices are AllGathered once to reconstruct the full result.
    This is mathematically identical to single-process inference.
    """
    num_nodes: int = graph.num_nodes  # type: ignore[assignment]
    base_n = (num_nodes + world_size - 1) // world_size  # ceiling division
    local_start = rank * base_n
    local_end = min((rank + 1) * base_n, num_nodes)

    edge_index: torch.Tensor = graph.edge_index  # type: ignore[assignment]
    edge_type: torch.Tensor = graph.edge_type  # type: ignore[assignment]

    # The rspmm kernel uses edge_index[0] as the output (target/destination) index
    # and edge_index[1] as the input (source) index â€” reversed from standard PyG.
    # Keep only edges whose target (edge_index[0]) falls in the local partition.
    target_nodes = edge_index[0]
    mask = (target_nodes >= local_start) & (target_nodes < local_end)

    local_graph = graph.clone()
    local_graph.edge_index = edge_index[:, mask]
    local_graph.edge_type = edge_type[mask]
    local_graph.num_edges = int(mask.sum().item())
    local_graph.dist_context = (rank, world_size)

    return local_graph
