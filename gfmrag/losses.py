from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.nn import functional as F  # noqa:N812

from gfmrag.ultra.variadic import variadic_softmax


class BaseLoss(ABC):
    """Base abstract class for all loss functions.

    This class serves as a template for implementing custom loss functions. All loss
    functions should inherit from this class and implement the required abstract methods.

    Methods:
        __init__(*args: Any, **kwargs: Any) -> None
            Initialize the loss function with given parameters.

        __call__(pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any) -> Any
            Calculate the loss between predicted and target values.

    Args:
        pred : torch.Tensor
            The predicted values from the model
        target : torch.Tensor
            The ground truth values
        *args : Any
            Variable length argument list
        **kwargs : Any
            Arbitrary keyword arguments

    Returns:
        Any: The computed loss value
    """

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

        neg_weight = torch.zeros_like(pred)
        neg_weight[is_positive] = (1 / num_positive.float()).repeat_interleave(
            num_positive
        )

        if self.adversarial_temperature > 0:
            with torch.no_grad():
                logit = pred[is_negative] / self.adversarial_temperature
                neg_weight[is_negative] = variadic_softmax(logit, num_negative)
                # neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
        else:
            neg_weight[is_negative] = (1 / num_negative.float()).repeat_interleave(
                num_negative
            )
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()
        return loss


class ListCELoss(BaseLoss):
    """Ranking loss for multi-label target lists.


    Args:
        pred (torch.Tensor): Predicted logits tensor of shape (B x N) where B is batch size
            and N is number of possible labels.
        target (torch.Tensor): Binary target tensor of shape (B x N) where 1 indicates positive
            labels and 0 indicates negative labels.
        *args: Additional positional arguments (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        torch.Tensor: Scalar tensor containing the mean loss value.

    Notes:
        - Empty targets (all zeros) are automatically skipped in loss computation
        - Small epsilon values (1e-5) are added to prevent numerical instability
        - The loss is normalized by the number of positive labels per sample
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        """Compute the ranking loss

        This loss function first normalizes the predictions using sigmoid and sum, then calculates
        a negative log likelihood loss weighted by the target values. Empty targets are skipped.

        Args:
            pred (torch.Tensor): Prediction tensor of shape (B x N) containing logits
            target (torch.Tensor): Target tensor of shape (B x N) containing ground truth values
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            torch.Tensor: Scalar loss value averaged over non-empty batch elements

        Note:
            - B represents batch size
            - N represents the number of elements per sample
            - A small epsilon (1e-5) is added for numerical stability
        """
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
