from abc import ABC, abstractmethod
from typing import Any


class BaseELModel(ABC):
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def index(self, entity_list: list) -> None:
        """Build the entity index from ``entity_list``."""
        pass

    @abstractmethod
    def __call__(self, ner_entity_list: list, topk: int = 1) -> dict:
        """Link named entities to indexed entities and return the top matches."""
        pass
