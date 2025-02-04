from typing import Any

import torch

from .base_model import BaseTextEmbModel


class NVEmbedV2(BaseTextEmbModel):
    """A text embedding model class that extends BaseTextEmbModel specifically for Nvidia models.

    This class customizes the base embedding model by:
    1. Setting a larger max sequence length of 32768
    2. Setting right-side padding
    3. Adding EOS tokens to input text

    Args:
        text_emb_model_name (str): Name or path of the text embedding model
        normalize (bool): Whether to normalize the output embeddings
        batch_size (int): Batch size for processing
        query_instruct (str, optional): Instruction prefix for query texts. Defaults to "".
        passage_instruct (str, optional): Instruction prefix for passage texts. Defaults to "".
        model_kwargs (dict | None, optional): Additional keyword arguments for model initialization. Defaults to None.

    Methods:
        add_eos: Adds EOS token to each input example
        encode: Encodes text by first adding EOS tokens then calling parent encode method

    Attributes:
        text_emb_model: The underlying text embedding model with customized max_seq_length and padding_side
    """

    def __init__(
        self,
        text_emb_model_name: str,
        normalize: bool,
        batch_size: int,
        query_instruct: str = "",
        passage_instruct: str = "",
        model_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            text_emb_model_name,
            normalize,
            batch_size,
            query_instruct,
            passage_instruct,
            model_kwargs,
        )
        self.text_emb_model.max_seq_length = 32768
        self.text_emb_model.tokenizer.padding_side = "right"

    def add_eos(self, input_examples: list[str]) -> list[str]:
        input_examples = [
            input_example + self.text_emb_model.tokenizer.eos_token
            for input_example in input_examples
        ]
        return input_examples

    def encode(self, text: list[str], *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Encode a list of text strings into embeddings with added EOS token.

        This method adds an EOS (end of sequence) token to each text string before encoding.

        Args:
            text (list[str]): List of text strings to encode
            *args (Any): Additional positional arguments passed to parent encode method
            **kwargs (Any): Additional keyword arguments passed to parent encode method

        Returns:
            torch.Tensor: Encoded text embeddings tensor

        Examples:
            >>> encoder = NVEmbedder()
            >>> texts = ["Hello world", "Another text"]
            >>> embeddings = encoder.encode(texts)
        """
        return super().encode(self.add_eos(text), *args, **kwargs)
