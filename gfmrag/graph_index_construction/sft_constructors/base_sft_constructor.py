import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseSFTConstructor(ABC):
    """An abstract base class for constructing supervised fine-tuning datasets.
    It would generate the target nodes and start nodes for the fine-tuning datasets.

    Attributes:
        None

    Methods:
        prepare_data:
            Abstract method that must be implemented by subclasses to prepare fine-tuning data.
            Takes data location parameters and returns processed data as a list of dictionaries.

    """

    @abstractmethod
    def prepare_data(self, data_root: str, data_name: str, file: str) -> list[dict]:
        """
        Prepare QA data for training and evaluation

        Args:
            data_root (str): path to the dataset
            data_name (str): name of the dataset
            file (str): file name to process
        Returns:
            list[dict]: list of processed data
        """
        pass
