from sentence_transformers import SentenceTransformer

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
        text_emb_model (SentenceTransformer): The underlying SentenceTransformer model
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

        self.text_emb_model = SentenceTransformer(
            self.text_emb_model_name,
            trust_remote_code=True,
            truncate_dim=self.truncate_dim,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
        )
