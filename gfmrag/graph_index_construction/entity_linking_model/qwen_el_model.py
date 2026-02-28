import hashlib
import os
from typing import Any

import torch
from vllm import LLM

from .base_model import BaseELModel

# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# os.environ["VLLM_USE_DEEPSPEED_ZERO3"] = "1"


class QWENELModel(BaseELModel):
    """
    Entity Linking Model based on Qwen-embedding series model (0.6B, 4B, 8B).

    This class implements an entity linking model using VLLM
    for encoding entities and computing similarity scores between mentions and candidate entities.

    Args:
        model_name (str): Name or path of the SentenceTransformer model to use
        root (str, optional): Root directory for caching embeddings. Defaults to "tmp".
        use_cache (bool, optional): Whether to cache and reuse entity embeddings. Defaults to True.
        normalize (bool, optional): Whether to L2-normalize embeddings. Defaults to True.
        batch_size (int, optional): Batch size for encoding. Defaults to 32.
        query_instruct (str, optional): Instruction/prompt prefix for query encoding. Defaults to "".
        passage_instruct (str, optional): Instruction/prompt prefix for passage encoding. Defaults to "".
        model_kwargs (dict, optional): Additional kwargs to pass to SentenceTransformer. Defaults to None.

    Methods:
        index(entity_list): Indexes a list of entities by computing and caching their embeddings
        __call__(ner_entity_list, topk): Links named entities to indexed entities and returns top-k matches
    """

    def __init__(
        self,
        model_name: str,
        root: str = "tmp",
        use_cache: bool = True,
        normalize: bool = True,
        batch_size: int = 8,
        topk: int = 5,
        query_instruct: str = "",
        passage_instruct: str = "",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        model_kwargs: dict | None = None,
    ) -> None:
        """Initialize DPR Entity Linking Model.

        Args:
            model_name (str): Name or path of the pre-trained model to load.
            root (str, optional): Root directory for cache storage. Defaults to "tmp".
            use_cache (bool, optional): Whether to use cache for embeddings. Defaults to True.
            normalize (bool, optional): Whether to normalize the embeddings. Defaults to True.
            batch_size (int, optional): Batch size for encoding. Defaults to 32.
            query_instruct (str, optional): Instruction prefix for query encoding. Defaults to "".
            passage_instruct (str, optional): Instruction prefix for passage encoding. Defaults to "".
            model_kwargs (dict | None, optional): Additional arguments to pass to the model. Defaults to None.
        """

        self.model_name = model_name
        self.use_cache = use_cache
        self.normalize = normalize
        self.batch_size = batch_size
        self.topk = topk
        self.root = os.path.join(root, f"{self.model_name.replace('/', '_')}_emb_cache")
        if self.use_cache and not os.path.exists(self.root):
            os.makedirs(self.root)
        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            task="embed",
        )

        self.query_instruct = query_instruct
        self.passage_instruct = passage_instruct

    def batch_index(self, entity_list: list) -> torch.Tensor:
        # Get md5 fingerprint of the whole given entity list
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        cache_file = f"{self.root}/{fingerprint}.pt"
        if os.path.exists(cache_file):
            embeddings = torch.load(
                cache_file,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
                weights_only=True,
            )
        else:
            all_embeddings = []
            from tqdm import tqdm

            for i in tqdm(range(0, len(entity_list), self.batch_size)):
                batch = entity_list[i : i + self.batch_size]
                outputs = self.model.embed(batch, use_tqdm=False)
                batch_emb = torch.tensor(
                    [o.outputs.embedding for o in outputs], dtype=torch.float32
                )
                all_embeddings.append(batch_emb)

            embeddings = torch.cat(all_embeddings, dim=0)

            if self.use_cache:
                torch.save(embeddings, cache_file, _use_new_zipfile_serialization=False)

        return embeddings

    def batch_index_v1(self, entity_list: list) -> torch.Tensor:
        # Get md5 fingerprint of the whole given entity list
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        cache_file = f"{self.root}/{fingerprint}.pt"
        if os.path.exists(cache_file):
            embeddings = torch.load(
                cache_file,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
                weights_only=True,
            )
        else:
            outputs = self.model.embed(
                entity_list,
            )

            embeddings = torch.tensor([o.outputs.embedding for o in outputs])

            if self.use_cache:
                torch.save(embeddings, cache_file, _use_new_zipfile_serialization=False)

        return embeddings

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery:{query}"

    def index(self, texts: list[Any], instruction: str = "") -> torch.Tensor:  # type: ignore
        if isinstance(texts, str):
            texts = [texts]
        input_texts = [self.get_detailed_instruct(instruction, text) for text in texts]

        outputs = self.model.embed(
            input_texts,
        )
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        return embeddings

    def __call__(self, ner_entity_list: list, topk: int = 1) -> dict:
        """
        Performs entity linking by matching input entities with pre-encoded entity embeddings.

        This method takes a list of named entities (e.g., from NER), computes their embeddings,
        and finds the closest matching entities from the pre-encoded knowledge base using
        cosine similarity.

        Args:
            ner_entity_list (list): List of named entities to link
            topk (int, optional): Number of top matches to return for each entity. Defaults to 1.

        Returns:
            dict: Dictionary mapping each input entity to its linked candidates. For each candidate:
                - entity (str): The matched entity name from the knowledge base
                - score (float): Raw similarity score
                - norm_score (float): Normalized similarity score (relative to top match)
        """
        raise NotImplementedError(
            "The __call__ method for QwenELModel is not implemented yet."
        )
