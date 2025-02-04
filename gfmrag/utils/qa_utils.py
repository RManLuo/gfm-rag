# mypy: ignore-errors

import torch
from torch import distributed as dist
from torch_scatter import scatter_add

from gfmrag.ultra import variadic


class DocumentRetriever:
    """
    Return documents based on document ranking
    """

    def __init__(self, docs: dict, id2doc: dict) -> None:
        self.docs = docs
        self.id2doc = id2doc

    def __call__(self, doc_ranking: torch.Tensor, top_k: int = 1) -> list:
        top_k_docs = doc_ranking.topk(top_k).indices
        norm_doc_scors = mini_max_scale(doc_ranking)
        return [
            {
                "title": self.id2doc[doc.item()],
                "content": self.docs[self.id2doc[doc.item()]],
                "score": doc_ranking[doc].item(),
                "norm_score": norm_doc_scors[doc].item(),
            }
            for doc in top_k_docs
        ]


def mini_max_scale(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def entities_to_mask(entities, num_nodes):
    mask = torch.zeros(num_nodes)
    mask[entities] = 1
    return mask


def evaluate(pred, target, metrics):
    ranking, num_pred = pred
    answer_ranking, num_hard = target

    metric = {}
    for _metric in metrics:
        if _metric == "mrr":
            answer_score = 1 / ranking.float()
            query_score = variadic.variadic_mean(answer_score, num_hard)
        elif _metric.startswith("recall@"):
            threshold = int(_metric[7:])
            answer_score = (ranking <= threshold).float()
            query_score = (
                variadic.variadic_sum(answer_score, num_hard) / num_hard.float()
            )
        elif _metric.startswith("hits@"):
            threshold = int(_metric[5:])
            answer_score = (ranking <= threshold).float()
            query_score = variadic.variadic_mean(answer_score, num_hard)
        elif _metric == "mape":
            query_score = (num_pred - num_hard).abs() / (num_hard).float()
        else:
            raise ValueError(f"Unknown metric `{_metric}`")

        score = query_score.mean()
        name = _metric
        metric[name] = score.item()

    return metric


def gather_results(pred, target, rank, world_size, device):
    # for multi-gpu setups: join results together
    # for single-gpu setups: doesn't do anything special
    ranking, num_pred = pred
    answer_ranking, num_target = target

    all_size_r = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_ar = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_p = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_r[rank] = len(ranking)
    all_size_ar[rank] = len(answer_ranking)
    all_size_p[rank] = len(num_pred)
    if world_size > 1:
        dist.all_reduce(all_size_r, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_ar, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_p, op=dist.ReduceOp.SUM)

    # obtaining all ranks
    cum_size_r = all_size_r.cumsum(0)
    cum_size_ar = all_size_ar.cumsum(0)
    cum_size_p = all_size_p.cumsum(0)

    all_ranking = torch.zeros(all_size_r.sum(), dtype=torch.long, device=device)
    all_num_pred = torch.zeros(all_size_p.sum(), dtype=torch.long, device=device)
    all_answer_ranking = torch.zeros(all_size_ar.sum(), dtype=torch.long, device=device)
    all_num_target = torch.zeros(all_size_p.sum(), dtype=torch.long, device=device)

    all_ranking[cum_size_r[rank] - all_size_r[rank] : cum_size_r[rank]] = ranking
    all_num_pred[cum_size_p[rank] - all_size_p[rank] : cum_size_p[rank]] = num_pred
    all_answer_ranking[cum_size_ar[rank] - all_size_ar[rank] : cum_size_ar[rank]] = (
        answer_ranking
    )
    all_num_target[cum_size_p[rank] - all_size_p[rank] : cum_size_p[rank]] = num_target

    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_pred, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_answer_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_target, op=dist.ReduceOp.SUM)

    return (all_ranking.cpu(), all_num_pred.cpu()), (
        all_answer_ranking.cpu(),
        all_num_target.cpu(),
    )


def batch_evaluate(pred, target, limit_nodes=None):
    num_target = target.sum(dim=-1)

    # answer2query = functional._size_to_index(num_answer)
    answer2query = torch.repeat_interleave(num_target)

    num_entity = pred.shape[-1]

    # in inductive (e) fb_ datasets, the number of nodes in the graph structure might exceed
    # the actual number of nodes in the graph, so we'll mask unused nodes
    if limit_nodes is not None:
        # print(f"Keeping only {len(limit_nodes)} nodes out of {num_entity}")
        keep_mask = torch.zeros(num_entity, dtype=torch.bool, device=limit_nodes.device)
        keep_mask[limit_nodes] = 1
        # keep_mask = F.one_hot(limit_nodes, num_entity)
        pred[:, ~keep_mask] = float("-inf")

    order = pred.argsort(dim=-1, descending=True)

    range = torch.arange(num_entity, device=pred.device)
    ranking = scatter_add(range.expand_as(order), order, dim=-1)

    target_ranking = ranking[target]
    # unfiltered rankings of all answers
    order_among_answer = variadic.variadic_sort(target_ranking, num_target)[1]
    order_among_answer = (
        order_among_answer + (num_target.cumsum(0) - num_target)[answer2query]
    )
    ranking_among_answer = scatter_add(
        variadic.variadic_arange(num_target), order_among_answer
    )

    # filtered rankings of all answers
    ranking = target_ranking - ranking_among_answer + 1
    ends = num_target.cumsum(0)
    starts = ends - num_target
    hard_mask = variadic.multi_slice_mask(starts, ends, ends[-1])
    # filtered rankings of hard answers
    ranking = ranking[hard_mask]

    return ranking, target_ranking
