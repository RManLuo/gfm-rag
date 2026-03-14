import torch
from torch_geometric.data import Data


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
    base_N = (num_nodes + world_size - 1) // world_size  # ceiling division
    local_start = rank * base_N
    local_end = min((rank + 1) * base_N, num_nodes)

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
