"""Profile split-graph partitioning to evaluate boundary-only AllGather feasibility.

Measures boundary node ratios for contiguous and METIS partitions across datasets
and GPU counts. The key metric is |boundary_sources| / N — if this is small,
boundary-only AllGather can dramatically reduce communication volume.

Usage:
    python scripts/profile_split_graph.py --root /path/to/new_graph_interface

Requires: pymetis (pip install pymetis)
"""

import argparse
from pathlib import Path

import torch

try:
    import pymetis

    HAS_METIS = True
except ImportError:
    HAS_METIS = False


def load_graph(dataset_dir: Path) -> torch.Tensor | None:
    """Load graph.pt from the first available stage2 hash dir."""
    stage2 = dataset_dir / "processed" / "stage2"
    if not stage2.exists():
        return None
    for hash_dir in stage2.iterdir():
        graph_pt = hash_dir / "graph.pt"
        if graph_pt.exists():
            return torch.load(graph_pt, map_location="cpu", weights_only=False)
    return None


def contiguous_partition(
    num_nodes: int, edge_index: torch.Tensor, world_size: int
) -> list[dict[str, int]]:
    """Partition using contiguous node slicing (PR #19/20 strategy).

    Returns per-rank stats: (local_edges, boundary_sources, local_N).
    """
    # edge_index[0] = destination (target), edge_index[1] = source (rspmm convention)
    dst = edge_index[0]
    src = edge_index[1]
    base_n = (num_nodes + world_size - 1) // world_size

    stats: list[dict[str, int]] = []
    for rank in range(world_size):
        local_start = rank * base_n
        local_end = min((rank + 1) * base_n, num_nodes)

        # Edges whose target falls in this rank's slice
        mask = (dst >= local_start) & (dst < local_end)
        local_edge_count = mask.sum().item()

        # Source nodes for those edges that are NOT in this rank's slice
        local_sources = src[mask]
        remote_mask = (local_sources < local_start) | (local_sources >= local_end)
        boundary_sources = local_sources[remote_mask].unique().numel()

        stats.append(
            {
                "rank": rank,
                "local_N": local_end - local_start,
                "local_edges": local_edge_count,
                "boundary_sources": boundary_sources,
            }
        )
    return stats


def metis_partition(
    num_nodes: int, edge_index: torch.Tensor, world_size: int
) -> list[dict[str, int]] | None:
    """Partition using METIS graph partitioning.

    Returns per-rank stats: (local_edges, boundary_sources, local_N).
    """
    if not HAS_METIS:
        return None

    # Build adjacency list for METIS (undirected)
    src = edge_index[1].numpy()
    dst = edge_index[0].numpy()

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        adjacency[s].append(d)
        adjacency[d].append(s)

    # Deduplicate
    adjacency = [list(set(neighbors)) for neighbors in adjacency]

    _, node2part = pymetis.part_graph(world_size, adjacency)
    node2part = torch.tensor(node2part)

    stats = []
    for rank in range(world_size):
        local_nodes = (node2part == rank).nonzero(as_tuple=True)[0]
        local_node_set = set(local_nodes.tolist())
        local_n = len(local_node_set)

        # Edges whose target is in this partition
        target_mask = node2part[dst] == rank
        local_edge_count = target_mask.sum().item()

        # Source nodes for those edges that are NOT in this partition
        local_sources = torch.from_numpy(src)[target_mask]
        remote_mask = node2part[local_sources] != rank
        boundary_sources = local_sources[remote_mask].unique().numel()

        stats.append(
            {
                "rank": rank,
                "local_N": local_n,
                "local_edges": local_edge_count,
                "boundary_sources": boundary_sources,
            }
        )
    return stats


def print_stats(
    dataset_name: str,
    num_nodes: int,
    num_edges: int,
    partition_stats: list[dict[str, int]],
    method: str,
    world_size: int,
) -> None:
    """Print a formatted table of partition stats."""
    avg_boundary = sum(s["boundary_sources"] for s in partition_stats) / len(
        partition_stats
    )
    avg_local_edges = sum(s["local_edges"] for s in partition_stats) / len(
        partition_stats
    )
    max_boundary = max(s["boundary_sources"] for s in partition_stats)
    ratio = avg_boundary / num_nodes

    print(f"\n  {method} partition, K={world_size}:")
    print(f"    Avg edges/rank:     {avg_local_edges:,.0f} / {num_edges:,} total")
    print(f"    Avg boundary nodes: {avg_boundary:,.0f} / {num_nodes:,} total")
    print(f"    Max boundary nodes: {max_boundary:,}")
    print(f"    Boundary/N ratio:   {ratio:.1%}")
    print(
        f"    AllGather savings:  {1 - ratio:.1%} "
        f"(current: B×{num_nodes}×D, boundary-only: B×{int(avg_boundary)}×D)"
    )

    # Memory estimates (bf16, B=8, D=1024, L=6 layers)
    batch_size, hidden_dim, num_layers = 8, 1024, 6
    bytes_per_elem = 2  # bf16
    current_comm = num_layers * batch_size * num_nodes * hidden_dim * bytes_per_elem
    boundary_comm = (
        num_layers * batch_size * int(avg_boundary) * hidden_dim * bytes_per_elem
    )
    print(
        f"    Comm/fwd (bf16, B=8, D=1024, L=6): "
        f"current={current_comm / 1e9:.2f} GB, "
        f"boundary={boundary_comm / 1e9:.2f} GB"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to new_graph_interface directory",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "hotpotqa_test",
            "hotpotqa_train0",
            "musique_test",
            "2wikimultihopqa_test",
        ],
    )
    parser.add_argument(
        "--world-sizes",
        nargs="+",
        type=int,
        default=[2, 4, 8],
    )
    args = parser.parse_args()

    root = Path(args.root)

    for dataset_name in args.datasets:
        dataset_dir = root / dataset_name
        graph = load_graph(dataset_dir)
        if graph is None:
            print(f"\n{'=' * 60}")
            print(f"Dataset: {dataset_name} — SKIPPED (no graph.pt)")
            continue

        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        edge_index = graph.edge_index
        avg_degree = num_edges / num_nodes

        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"  Nodes: {num_nodes:,}")
        print(f"  Edges: {num_edges:,}")
        print(f"  Avg degree: {avg_degree:.1f}")
        print(f"  Edge density (E/N): {avg_degree:.1f}")

        for ws in args.world_sizes:
            if ws > num_nodes:
                continue

            # Contiguous partition
            contiguous_stats = contiguous_partition(num_nodes, edge_index, ws)
            print_stats(
                dataset_name,
                num_nodes,
                num_edges,
                contiguous_stats,
                "Contiguous",
                ws,
            )

            # METIS partition
            if HAS_METIS:
                metis_stats = metis_partition(num_nodes, edge_index, ws)
                if metis_stats:
                    print_stats(
                        dataset_name, num_nodes, num_edges, metis_stats, "METIS", ws
                    )
            else:
                print(f"\n  METIS partition, K={ws}: SKIPPED (pymetis not installed)")


if __name__ == "__main__":
    main()
