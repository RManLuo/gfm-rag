from abc import ABC, abstractmethod


class BaseLanguageModel(ABC):
    """Abstract interface for language-model backends."""

    @abstractmethod
    def __init__(self, model_name_or_path: str):
        pass

    @abstractmethod
    def token_len(self, text: str) -> int:
        """Return the tokenized length of ``text``."""
        pass

    @abstractmethod
    def generate_sentence(
        self, llm_input: str | list, system_input: str = ""
    ) -> str | Exception:
        """Generate text from ``llm_input`` with an optional system prompt."""
        pass
