from abc import ABC, abstractmethod
from typing import Any


class BaseNERModel(ABC):
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __call__(self, text: str) -> list:
        """
        This method implements the callable functionality of the class to perform Named Entity Recognition
        on input text. When an instance of the class is called directly, this method is invoked.

        Args:
            text (str): The input text to perform NER analysis on.

        Returns:
            list: A list of named entities found in the text. Each entity is represented
                  according to the model's output format.

        Examples:
            >>> ner = NERModel()
            >>> entities = ner("This is a sample text")
            >>> print(entities)
            [{'text': 'sample', 'label': 'EXAMPLE'}]

        Note:
            This is an abstract method that should be implemented by subclasses.
        """
        pass
