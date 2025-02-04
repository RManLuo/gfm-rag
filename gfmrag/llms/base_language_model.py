from abc import ABC, abstractmethod


class BaseLanguageModel(ABC):
    """
    Base lanuage model. Define how to generate sentence by using a LM
    """

    @abstractmethod
    def __init__(self, model_name_or_path: str):
        pass

    @abstractmethod
    def token_len(self, text: str) -> int:
        """
        Return tokenized length of text

        Args:
            text (str): input text
        """
        pass

    @abstractmethod
    def generate_sentence(
        self, llm_input: str | list, system_input: str = ""
    ) -> str | Exception:
        """
        Generate sentence by using a LM

        Args:
            lm_input (LMInput): input for LM
        """
        pass
