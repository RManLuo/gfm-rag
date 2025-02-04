import torch
from sentence_transformers import SentenceTransformer


class BaseTextEmbModel:
    """A base class for text embedding models using SentenceTransformer.

    This class provides functionality to encode text into embeddings using various
    SentenceTransformer models with configurable parameters.

    Args:
        text_emb_model_name (str): Name or path of the SentenceTransformer model to use
        normalize (bool, optional): Whether to L2-normalize the embeddings. Defaults to False.
        batch_size (int, optional): Batch size for encoding. Defaults to 32.
        query_instruct (str | None, optional): Instruction/prompt to prepend to queries. Defaults to None.
        passage_instruct (str | None, optional): Instruction/prompt to prepend to passages. Defaults to None.
        model_kwargs (dict | None, optional): Additional keyword arguments for the model. Defaults to None.

    Attributes:
        text_emb_model (SentenceTransformer): The underlying SentenceTransformer model
        text_emb_model_name (str): Name of the model being used
        normalize (bool): Whether embeddings are L2-normalized
        batch_size (int): Batch size used for encoding
        query_instruct (str | None): Instruction text for queries
        passage_instruct (str | None): Instruction text for passages
        model_kwargs (dict | None): Additional model configuration parameters

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
        model_kwargs: dict | None = None,
    ) -> None:
        """
        Initialize the BaseTextEmbModel.

        Args:
            text_emb_model_name (str): Name or path of the SentenceTransformer model to use
            normalize (bool, optional): Whether to L2-normalize the embeddings. Defaults to False.
            batch_size (int, optional): Batch size for encoding. Defaults to 32.
            query_instruct (str | None, optional): Instruction/prompt to prepend to queries. Defaults to None.
            passage_instruct (str | None, optional): Instruction/prompt to prepend to passages. Defaults to None.
            model_kwargs (dict | None, optional): Additional keyword arguments for the model. Defaults to None.
        """
        self.text_emb_model_name = text_emb_model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.query_instruct = query_instruct
        self.passage_instruct = passage_instruct
        self.model_kwargs = model_kwargs

        self.text_emb_model = SentenceTransformer(
            self.text_emb_model_name,
            trust_remote_code=True,
            model_kwargs=self.model_kwargs,
        )

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
            >>> text_emb_model = BaseTextEmbModel("sentence-transformers/all-mpnet-base-v2")
            >>> text = ["Hello, world!", "This is a test."]
            >>> embeddings = text_emb_model.encode(text)
        """

        return self.text_emb_model.encode(
            text,
            device="cuda" if torch.cuda.is_available() else "cpu",
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            prompt=self.query_instruct if is_query else self.passage_instruct,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
        )
