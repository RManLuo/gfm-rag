from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseGNNModel(ABC, torch.nn.Module):
    """
    Base class for GNN models used in the GFM-RAG framework.
    """

    @abstractmethod
    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass of the GNN model.
        """
        pass
