from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from torch.nn import functional as F  # noqa:N812

from gfmrag.models.ultra.variadic import variadic_softmax


class BaseLoss(ABC):
    """Abstract interface for loss functions."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        pass


class BCELoss(BaseLoss):
    """
    Binary Cross Entropy loss function with adversarial temperature.
    """

    def __init__(
        self, adversarial_temperature: float = 0, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the loss function.

        Args:
            adversarial_temperature (float, optional): Temperature parameter for adversarial loss scaling. Defaults to 0.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None
        """
        self.adversarial_temperature = adversarial_temperature

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        """Calculate the weighted binary cross-entropy loss with adversarial temperature.

        This method implements a custom loss function that applies different weights to positive
        and negative samples. For negative samples, it can optionally use adversarial temperature
        to compute softmax-based weights.

        Args:
            pred (torch.Tensor): The predicted logits tensor
            target (torch.Tensor): The target tensor with binary labels (0 or 1)
            *args (Any): Variable length argument list
            **kwargs (Any): Arbitrary keyword arguments

        Returns:
            Any: The computed loss value

        The loss calculation involves:

        1. Computing binary cross entropy loss
        2. Identifying positive and negative samples
        3. Applying weights to negative samples based on adversarial_temperature
        4. Computing weighted average of the losses
        """
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        is_positive = target > 0.5
        is_negative = target <= 0.5
        num_positive = is_positive.sum(dim=-1)
        num_negative = is_negative.sum(dim=-1)

        neg_weight = torch.zeros_like(pred, dtype=pred.dtype)
        neg_weight[is_positive] = (
            (1 / num_positive).repeat_interleave(num_positive).to(pred.dtype)
        )

        if self.adversarial_temperature > 0:
            with torch.no_grad():
                logit = pred[is_negative] / self.adversarial_temperature
                neg_weight[is_negative] = variadic_softmax(logit, num_negative).to(
                    pred.dtype
                )
                # neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
        else:
            neg_weight[is_negative] = (
                (1 / num_negative).repeat_interleave(num_negative).to(pred.dtype)
            )
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()
        return loss


class ListCELoss(BaseLoss):
    """Ranking loss for multi-label target lists."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        """Compute the normalized listwise cross-entropy loss."""
        target_sum = target.sum(dim=-1)
        non_zero_target_mask = target_sum != 0  # Skip empty target
        target_sum = target_sum[non_zero_target_mask]
        pred = pred[non_zero_target_mask]
        target = target[non_zero_target_mask]
        pred_prob = torch.sigmoid(pred)  # B x N
        pred_prob_sum = pred_prob.sum(dim=-1, keepdim=True)  # B x 1
        loss = -torch.log((pred_prob / (pred_prob_sum + 1e-5)) + 1e-5) * target
        loss = loss.sum(dim=-1) / target_sum
        loss = loss.mean()
        return loss


class KLDivLoss(BaseLoss):
    """Kullback-Leibler divergence loss."""

    def __init__(
        self,
        reduction: Literal["sum", "mean", "batchmean"] = "batchmean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if reduction not in ["sum", "mean", "batchmean"]:
            raise ValueError(
                f"Invalid reduction mode: {reduction}. Supported modes are 'sum', 'mean', and 'batchmean'."
            )
        self.reduction = reduction
        self.eps = 1e-6  # Small epsilon to avoid log(0)

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        """Compute KL divergence between the predicted and target distributions."""
        pred_prob = F.sigmoid(pred)
        target_prob = (target + 1) / 2
        # Ensure prob is not zero to avoid log(0)
        student_prob = torch.clamp(pred_prob, min=self.eps, max=1 - self.eps)
        target_prob = torch.clamp(target_prob, min=self.eps, max=1 - self.eps)
        # Compute the KL divergence loss
        loss = target_prob * (torch.log(target_prob) - torch.log(student_prob)) + (
            1 - target_prob
        ) * (torch.log(1 - target_prob) - torch.log(1 - student_prob))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "batchmean":
            return loss.sum() / pred.size(0)


class MSELoss(BaseLoss):
    """Mean squared error loss."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        """Compute mean squared error after normalizing both tensors."""
        # Normalize the pred and target to [0, 1]
        norm_pred = F.sigmoid(pred)
        norm_target = (target + 1) / 2
        return F.mse_loss(norm_pred, norm_target, reduction="mean")
