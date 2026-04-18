from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
from tqdm import tqdm

from .base_model import BaseTextEmbModel
from .configuration_nvembed import NVEmbedConfig
from .modeling_nvembed import NVEmbedModel


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
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.text_emb_model_name = text_emb_model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.query_instruct = query_instruct
        self.passage_instruct = passage_instruct
        self.model_kwargs: dict[str, Any] | None = (
            dict(model_kwargs) if model_kwargs is not None else None
        )

        config = NVEmbedConfig.from_pretrained(self.text_emb_model_name)
        self.text_emb_model = cast(
            NVEmbedModel,
            NVEmbedModel.from_pretrained(
                self.text_emb_model_name,
                config=config,
                device_map="auto",
                **dict(self.model_kwargs or {}),
            ),
        )

        self.max_seq_length = 32768
        self.text_emb_model.padding_side = "right"
        tokenizer = self.text_emb_model.tokenizer
        if tokenizer is None:
            raise ValueError("NVEmbedModel tokenizer must be initialized.")
        tokenizer.padding_side = "right"

    def add_eos(self, input_examples: list[str]) -> list[str]:
        tokenizer = self.text_emb_model.tokenizer
        if tokenizer is None or tokenizer.eos_token is None:
            raise ValueError("NVEmbedModel tokenizer EOS token must be initialized.")
        return [input_example + tokenizer.eos_token for input_example in input_examples]

    def encode(
        self, text: list[str], is_query: bool = False, show_progress_bar: bool = True
    ) -> torch.Tensor:
        """
        Encode a list of text strings into embeddings with added EOS token.

        This method adds an EOS (end of sequence) token to each text string before encoding.

        Args:
            text (list[str]): List of text strings to encode
            is_query (bool): Whether the text is being encoded as a query.
            show_progress_bar (bool): Whether to display a progress bar during encoding.

        Returns:
            torch.Tensor: Encoded text embeddings tensor

        Examples:
            >>> encoder = NVEmbedder()
            >>> texts = ["Hello world", "Another text"]
            >>> embeddings = encoder.encode(texts)
        """
        if len(text) == 0:
            return torch.empty((0, 0), dtype=torch.float32)

        prompt = self.query_instruct if is_query else self.passage_instruct
        prompt = prompt or ""
        all_embeddings: list[torch.Tensor] = []
        for i in tqdm(
            range(0, len(text), self.batch_size), disable=not show_progress_bar
        ):
            batch = text[i : min(i + self.batch_size, len(text))]
            raw_batch_embeddings = self.text_emb_model.encode(
                batch,
                instruction=prompt,
                max_length=self.max_seq_length,
            )
            if isinstance(raw_batch_embeddings, np.ndarray):
                batch_embeddings = torch.from_numpy(raw_batch_embeddings).float()
            else:
                batch_embeddings = raw_batch_embeddings.float()

            if self.normalize:
                batch_embeddings = torch.nn.functional.normalize(
                    batch_embeddings, p=2, dim=1
                )

            all_embeddings.append(batch_embeddings.cpu())

            del batch_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0)
