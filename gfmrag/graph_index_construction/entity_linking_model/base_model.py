from abc import ABC, abstractmethod
from typing import Any


class BaseELModel(ABC):
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def index(self, entity_list: list) -> None:
        """
        This method creates an index for the provided list of entities to enable efficient entity linking and searching capabilities.

        Args:
            entity_list (list): A list of entities to be indexed. Each entity should be a string or dictionary containing
                               the entity text and other relevant metadata.

            None: This method modifies the internal index structure but does not return anything.

        Raises:
            ValueError: If entity_list is empty or contains invalid entity formats.
            TypeError: If entity_list is not a list type.

        Examples:
            >>> model = EntityLinkingModel()
            >>> entities = ["Paris", "France", "Eiffel Tower"]
            >>> model.index(entities)
        """
        pass

    @abstractmethod
    def __call__(self, ner_entity_list: list, topk: int = 1) -> dict:
        """
        Link entities in the given text to the knowledge graph.

        Args:
            ner_entity_list (list): list of named entities
            topk (int): number of linked entities to return

        Returns:
            dict: dict of linked entities in the knowledge graph

                - key (str): named entity
                - value (list[dict]): list of linked entities

                    - entity: linked entity
                    - score: score of the entity
                    - norm_score: normalized score of the entity
        """
        pass
