import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseSFTConstructor(ABC):
    """Abstract interface for constructing supervised fine-tuning data."""

    @abstractmethod
    def prepare_data(self, data_root: str, data_name: str, file: str) -> list[dict]:
        """Prepare processed samples for ``file`` under the given dataset."""
        pass
