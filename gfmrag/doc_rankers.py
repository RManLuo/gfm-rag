from abc import ABC, abstractmethod

import torch


class BaseDocRanker(ABC):
    """
    Abstract class for document ranker

    Args:
        ent2doc (torch.Tensor): Mapping from entity to document
    """

    def __init__(self, ent2doc: torch.Tensor) -> None:
        self.ent2doc = ent2doc

    @abstractmethod
    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        pass


class SimpleRanker(BaseDocRanker):
    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on entity prediction

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        doc_pred = torch.sparse.mm(ent_pred, self.ent2doc)
        return doc_pred


class IDFWeightedRanker(BaseDocRanker):
    """
    Rank documents based on entity prediction with IDF weighting
    """

    def __init__(self, ent2doc: torch.Tensor) -> None:
        super().__init__(ent2doc)
        frequency = torch.sparse.sum(ent2doc, dim=-1).to_dense()
        self.idf_weight = 1 / frequency
        self.idf_weight[frequency == 0] = 0

    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on entity prediction with IDF weighting

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        doc_pred = torch.sparse.mm(
            ent_pred * self.idf_weight.unsqueeze(0), self.ent2doc
        )
        return doc_pred


class TopKRanker(BaseDocRanker):
    def __init__(self, ent2doc: torch.Tensor, top_k: int) -> None:
        super().__init__(ent2doc)
        self.top_k = top_k

    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on top-k entity prediction

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        top_k_ent_pred = torch.topk(ent_pred, self.top_k, dim=-1)
        masked_ent_pred = torch.zeros_like(ent_pred, device=ent_pred.device)
        masked_ent_pred.scatter_(1, top_k_ent_pred.indices, 1)
        doc_pred = torch.sparse.mm(masked_ent_pred, self.ent2doc)
        return doc_pred


class IDFWeightedTopKRanker(BaseDocRanker):
    def __init__(self, ent2doc: torch.Tensor, top_k: int) -> None:
        super().__init__(ent2doc)
        self.top_k = top_k
        frequency = torch.sparse.sum(ent2doc, dim=-1).to_dense()
        self.idf_weight = 1 / frequency
        self.idf_weight[frequency == 0] = 0

    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on top-k entity prediction

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        top_k_ent_pred = torch.topk(ent_pred, self.top_k, dim=-1)
        idf_weight = torch.gather(
            self.idf_weight.expand(ent_pred.shape[0], -1), 1, top_k_ent_pred.indices
        )
        masked_ent_pred = torch.zeros_like(ent_pred, device=ent_pred.device)
        masked_ent_pred.scatter_(1, top_k_ent_pred.indices, idf_weight)
        doc_pred = torch.sparse.mm(masked_ent_pred, self.ent2doc)
        return doc_pred
