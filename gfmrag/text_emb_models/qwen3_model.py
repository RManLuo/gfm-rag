import os

import requests
import torch
from openai import NOT_GIVEN, OpenAI
from tqdm import tqdm
from vllm import LLM, PoolingParams

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
        api_base (str, optional): Base URL for the vLLM server. If no URL is provided, a local server will be started.
        api_key (str, optional): API key for authentication. Defaults to "EMPTY".
        vllm_timeout (int, optional): Timeout for vLLM requests in seconds. Defaults to 600.

    Attributes:
        client (OpenAI): The OpenAI client for making requests to vLLM server
        async_client (AsyncOpenAI): The async OpenAI client for concurrent requests
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
        api_base: str | None = None,
        api_key: str = "EMPTY",
        vllm_timeout: int = 600,
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
            api_base (str | None, optional): Base URL for the vLLM server. If no url is provided we would start a local server.
            api_key (str | None, optional): API key for authentication. Defaults to "EMPTY".
            vllm_timeout (int, optional): Timeout for vLLM requests in seconds. Defaults to 600.
        """
        self.text_emb_model_name = text_emb_model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.query_instruct = query_instruct
        self.passage_instruct = passage_instruct
        self.truncate_dim = truncate_dim
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.api_base = api_base
        self.api_key = api_key
        self.vllm_timeout = vllm_timeout

        if api_base is None:
            self.text_emb_model = self._start_vllm_server()
        else:
            # Check if API is available
            if not self._is_api_available():
                raise RuntimeError("vLLM API is not available")

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )

    def _is_api_available(self) -> bool:
        """Check if the vLLM API is available at the specified URL."""
        try:
            health_url = self.api_base.replace("/v1", "/health")  # type: ignore
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _start_vllm_server(self) -> LLM:
        """Start a vLLM server for embedding generation."""

        dist_keys = [
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_NAME",
            "OMP_NUM_THREADS",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_USE_AGENT_STORE",
            "TORCHELASTIC_MAX_RESTARTS",
            "TORCHELASTIC_RUN_ID",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "TORCHELASTIC_ERROR_FILE",
        ]
        old_env = {}
        for dist_key in dist_keys:
            if dist_key in os.environ:
                old_env[dist_key] = os.environ.pop(dist_key)
        os.environ["CUDA_VISIBLE_DEVICES"] = old_env.get("LOCAL_RANK", "0")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        text_emb_model = LLM(
            model=self.text_emb_model_name,
            enforce_eager=True,
            task="embed",
            hf_overrides={"is_matryoshka": True},
        )
        # Restore environment variables
        os.environ.pop("CUDA_VISIBLE_DEVICES")
        for dist_key, dist_value in old_env.items():
            if dist_value is not None:
                os.environ[dist_key] = dist_value
        return text_emb_model

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

    def _make_request(self, text: list[str], show_progress_bar: bool) -> torch.Tensor:
        """Makes a request to the vLLM API to get embeddings for the provided text.

        Args:
            text (list[str]): List of text strings to encode
            is_query (bool): Whether the text is a query (True) or passage (False).
            show_progress_bar (bool): Whether to display a progress bar during the request.

        Returns:
            torch.Tensor: Tensor containing the embeddings for the input text
        """
        # Make request to vLLM server
        dimensions = (
            self.truncate_dim
            if self.truncate_dim is not None and self.truncate_dim > 0
            else NOT_GIVEN
        )

        # Process in batches
        all_embeddings = []
        for i in tqdm(
            range(0, len(text), self.batch_size), disable=not show_progress_bar
        ):
            batch = text[i : min(i + self.batch_size, len(text))]

            response = self.client.embeddings.create(
                model=self.text_emb_model_name,
                input=batch,
                dimensions=dimensions,
                timeout=self.vllm_timeout,
            )

            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)

        return torch.tensor(all_embeddings, device="cpu", dtype=torch.float32)

    def embed(self, text: list[str], show_progress_bar: bool = True) -> torch.Tensor:
        """
        Embeds a list of text strings using the text embedding model.

        Args:
            text (list[str]): List of text strings to embed.
            show_progress_bar (bool, optional): Whether to display a progress bar during embedding.
                Defaults to True.

        Returns:
            torch.Tensor: Tensor containing the embeddings for the input text.
        """
        all_embeddings = []
        for i in tqdm(
            range(0, len(text), self.batch_size), disable=not show_progress_bar
        ):
            batch = text[i : min(i + self.batch_size, len(text))]
            if self.truncate_dim is not None and self.truncate_dim > 0:
                output = self.text_emb_model.embed(
                    batch,
                    pooling_params=PoolingParams(dimensions=self.truncate_dim),
                    use_tqdm=False,
                )
            else:
                output = self.text_emb_model.embed(batch, use_tqdm=False)

            # Move each batch to CPU immediately to avoid holding the full
            # embedding matrix on GPU when indexing large fact collections.
            batch_embeddings = torch.tensor(
                [o.outputs.embedding for o in output],
                device="cpu",
                dtype=torch.float32,
            )
            all_embeddings.append(batch_embeddings)

            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not all_embeddings:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.cat(all_embeddings, dim=0)

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
        if self.api_base:
            return self._make_request(text_with_instruct, show_progress_bar)
        else:
            return self.embed(text_with_instruct, show_progress_bar)
