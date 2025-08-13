from abc import ABC, abstractmethod

import torch


class BaseDocRanker(ABC):
    """
    Abstract class for document ranker
    """

    @abstractmethod
    def __call__(self, ent_pred: torch.Tensor, ent2doc: torch.Tensor) -> torch.Tensor:
        pass


class SimpleRanker(BaseDocRanker):
    """
    Rank documents based on entity prediction without any weighting
    """

    def __call__(self, ent_pred: torch.Tensor, ent2doc: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on entity prediction

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)
            ent2doc (torch.Tensor): Sparse tensor mapping entities to documents, shape (n_entities, n_docs)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        doc_pred = torch.sparse.mm(ent_pred, ent2doc)
        return doc_pred


class IDFWeightedRanker(BaseDocRanker):
    """
    Rank documents based on entity prediction with IDF weighting
    """

    def __call__(self, ent_pred: torch.Tensor, ent2doc: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on entity prediction with IDF weighting

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """

        frequency = torch.sparse.sum(ent2doc, dim=-1).to_dense()
        idf_weight = 1 / frequency
        idf_weight[frequency == 0] = 0

        doc_pred = torch.sparse.mm(ent_pred * idf_weight.unsqueeze(0), ent2doc)
        return doc_pred


class TopKRanker(BaseDocRanker):
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def __call__(self, ent_pred: torch.Tensor, ent2doc: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on top-k entity prediction

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)
            ent2doc (torch.Tensor): Sparse tensor mapping entities to documents, shape (n_entities, n_docs)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        top_k_ent_pred = torch.topk(ent_pred, self.top_k, dim=-1)
        masked_ent_pred = torch.zeros_like(ent_pred, device=ent_pred.device)
        masked_ent_pred.scatter_(1, top_k_ent_pred.indices, 1)
        doc_pred = torch.sparse.mm(masked_ent_pred, ent2doc)
        return doc_pred


class IDFWeightedTopKRanker(BaseDocRanker):
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def __call__(self, ent_pred: torch.Tensor, ent2doc: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on top-k entity prediction

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)
            ent2doc (torch.Tensor): Sparse tensor mapping entities to documents, shape (n_entities, n_docs)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        frequency = torch.sparse.sum(ent2doc, dim=-1).to_dense()
        idf_weight = 1 / frequency
        idf_weight[frequency == 0] = 0

        top_k_ent_pred = torch.topk(ent_pred, self.top_k, dim=-1)
        idf_weight = torch.gather(
            idf_weight.expand(ent_pred.shape[0], -1), 1, top_k_ent_pred.indices
        )
        masked_ent_pred = torch.zeros_like(
            ent_pred, device=ent_pred.device, dtype=idf_weight.dtype
        )
        masked_ent_pred.scatter_(1, top_k_ent_pred.indices, idf_weight)
        doc_pred = torch.sparse.mm(masked_ent_pred, ent2doc)
        return doc_pred
