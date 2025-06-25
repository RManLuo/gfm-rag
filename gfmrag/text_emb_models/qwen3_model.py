import torch
from vllm import LLM, PoolingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from .base_model import BaseTextEmbModel


class Qwen3TextEmbModel(BaseTextEmbModel):
    """A text embedding model class that extends BaseTextEmbModel specifically for Qwen3 embedding models.

    Args:
        text_emb_model_name (str): Name or path of the SentenceTransformer model to use
        normalize (bool, optional): Whether to L2-normalize the embeddings. Defaults to False.
        batch_size (int, optional): Batch size for encoding. Defaults to 32.
        query_instruct (str | None, optional): Instruction/prompt to prepend to queries. Defaults to None.
        passage_instruct (str | None, optional): Instruction/prompt to prepend to passages. Defaults to None.
        truncate_dim (int | None, optional): Dimension to truncate the embeddings to. Defaults to None.
        model_kwargs (dict | None, optional): Additional keyword arguments for the model. Defaults to None.
        tokenizer_kwargs (dict | None, optional): Additional keyword arguments for the tokenizer. Defaults to None.

    Attributes:
        text_emb_model (LLM): The underlying text embedding model
        text_emb_model_name (str): Name of the model being used
        normalize (bool): Whether embeddings are L2-normalized
        batch_size (int): Batch size used for encoding
        query_instruct (str | None): Instruction text for queries
        passage_instruct (str | None): Instruction text for passages
        truncate_dim (int | None): Dimension to truncate the embeddings to
        model_kwargs (dict | None): Additional model configuration parameters
        tokenizer_kwargs (dict | None): Additional tokenizer configuration parameters

    Methods:
        encode(text: list[str], is_query: bool = False, show_progress_bar: bool = True) -> torch.Tensor:
            Encodes a list of texts into embeddings.
    """

    def __init__(
        self,
        text_emb_model_name: str,
        normalize: bool = False,
        batch_size: int = 32,
        query_instruct: str | None = None,
        passage_instruct: str | None = None,
        truncate_dim: int | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
    ) -> None:
        """
        Initialize the BaseTextEmbModel.

        Args:
            text_emb_model_name (str): Name or path of the SentenceTransformer model to use
            normalize (bool, optional): Whether to L2-normalize the embeddings. Defaults to False.
            batch_size (int, optional): Batch size for encoding. Defaults to 32.
            query_instruct (str | None, optional): Instruction/prompt to prepend to queries. Defaults to None.
            passage_instruct (str | None, optional): Instruction/prompt to prepend to passages. Defaults to None.
            truncate_dim (int | None, optional): Dimension to truncate the embeddings to. Defaults to None.
            model_kwargs (dict | None, optional): Additional keyword arguments for the model. Defaults to None.
            tokenizer_kwargs (dict | None, optional): Additional keyword arguments for the tokenizer. Defaults to None.
        """
        self.text_emb_model_name = text_emb_model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.query_instruct = query_instruct
        self.passage_instruct = passage_instruct
        self.truncate_dim = truncate_dim
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs

        self.text_emb_model = LLM(
            model=self.text_emb_model_name,
            task="embed",
            hf_overrides={"is_matryoshka": True},
        )

    def add_instruct(self, instruct: str | None, query: str) -> str:
        """Adds an instruction prefix to the query text if provided.

        Args:
            instruct (str | None): Instruction text to prepend to the query
            query (str): The query text to which the instruction will be added
        Returns:
            str: The query text with the instruction prepended, or just the query if no instruction is provided
        """

        if instruct is None:
            return query
        else:
            return f"{instruct}{query}"

    def encode(
        self, text: list[str], is_query: bool = False, show_progress_bar: bool = True
    ) -> torch.Tensor:
        """
        Encodes a list of text strings into embeddings using the text embedding model.

        Args:
            text (list[str]): List of text strings to encode
            is_query (bool, optional): Whether the text is a query (True) or passage (False).
                Determines which instruction prompt to use. Defaults to False.
            show_progress_bar (bool, optional): Whether to display progress bar during encoding.
                Defaults to True.

        Returns:
            torch.Tensor: Tensor containing the encoded embeddings for the input text

        Examples:
            >>> text_emb_model = Qwen3TextEmbModel("Qwen/Qwen3-Embedding-0.6B")
            >>> text = ["Hello, world!", "This is a test."]
            >>> embeddings = text_emb_model.encode(text)
        """
        text_with_instruct = [
            self.add_instruct(self.query_instruct, t)
            if is_query
            else self.add_instruct(self.passage_instruct, t)
            for t in text
        ]

        if self.truncate_dim is not None and self.truncate_dim > 0:
            output = self.text_emb_model.embed(
                text_with_instruct,
                pooling_params=PoolingParams(dimensions=self.truncate_dim),
                use_tqdm=show_progress_bar,
            )
        else:
            output = self.text_emb_model.embed(
                text_with_instruct, use_tqdm=show_progress_bar
            )

        return torch.tensor(
            [o.outputs.embedding for o in output],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def __del__(self) -> None:
        try:
            destroy_model_parallel()
        except Exception:
            pass
