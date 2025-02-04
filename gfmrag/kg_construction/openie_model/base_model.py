from abc import ABC, abstractmethod
from typing import Any


class BaseOPENIEModel(ABC):
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __call__(self, text: str) -> dict:
        """
        Perform OpenIE on the given text.

        Args:
            text (str): input text

        Returns:
            dict: dict of passage, extracted entities, extracted_triples

                - passage (str): input text
                - extracted_entities (list): list of extracted entities
                - extracted_triples (list): list of extracted triples

        Examples:
            >>> openie_model = OPENIEModel()
            >>> result = openie_model("Emmanuel Macron is the president of France")
            >>> print(result)
            {'passage': 'Emmanuel Macron is the president of France', 'extracted_entities': ['Emmanuel Macron', 'France'], 'extracted_triples': [['Emmanuel Macron', 'president of', 'France']]}
        """
        pass
