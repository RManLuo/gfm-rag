import hashlib
import os
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from .base_model import BaseELModel


class DPRELModel(BaseELModel):
    """
    Entity Linking Model based on Dense Passage Retrieval (DPR).

    This class implements an entity linking model using DPR architecture and SentenceTransformer
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

    Examples:
        >>> model = DPRELModel('sentence-transformers/all-mpnet-base-v2')
        >>> model.index(['Paris', 'London', 'Berlin'])
        >>> results = model(['paris city'], topk=2)
        >>> print(results)
        {'paris city': [{'entity': 'Paris', 'score': 0.82, 'norm_score': 1.0},
                        {'entity': 'London', 'score': 0.35, 'norm_score': 0.43}]}
    """

    def __init__(
        self,
        model_name: str,
        root: str = "tmp",
        use_cache: bool = True,
        normalize: bool = True,
        batch_size: int = 32,
        query_instruct: str = "",
        passage_instruct: str = "",
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
        self.root = os.path.join(root, f"{self.model_name.replace('/', '_')}_dpr_cache")
        if self.use_cache and not os.path.exists(self.root):
            os.makedirs(self.root)
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, model_kwargs=model_kwargs
        )
        self.query_instruct = query_instruct
        self.passage_instruct = passage_instruct

    def index(self, entity_list: list) -> None:
        """
        Index a list of entities by encoding them into embeddings and optionally caching the results.

        This method processes a list of entity strings, converting them into dense vector representations
        using a pre-trained model. To avoid redundant computation, it implements a caching mechanism
        based on the MD5 hash of the input entity list.

        Args:
            entity_list (list): A list of strings representing entities to be indexed.

        Returns:
            None

        Notes:
            - The method stores the embeddings in self.entity_embeddings
            - If caching is enabled and a cache file exists for the given entity list,
              embeddings are loaded from cache instead of being recomputed
            - Cache files are stored using the MD5 hash of the concatenated entity list as filename
            - Embeddings are computed on GPU if available, otherwise on CPU
        """
        self.entity_list = entity_list
        # Get md5 fingerprint of the whole given entity list
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        cache_file = f"{self.root}/{fingerprint}.pt"
        if os.path.exists(cache_file):
            self.entity_embeddings = torch.load(
                cache_file,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
                weights_only=True,
            )
        else:
            self.entity_embeddings = self.model.encode(
                entity_list,
                device="cuda" if torch.cuda.is_available() else "cpu",
                convert_to_tensor=True,
                show_progress_bar=True,
                prompt=self.passage_instruct,
                normalize_embeddings=self.normalize,
                batch_size=self.batch_size,
            )
            if self.use_cache:
                torch.save(self.entity_embeddings, cache_file)

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
        ner_entity_embeddings = self.model.encode(
            ner_entity_list,
            device="cuda" if torch.cuda.is_available() else "cpu",
            convert_to_tensor=True,
            prompt=self.query_instruct,
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
        )
        scores = ner_entity_embeddings @ self.entity_embeddings.T
        top_k_scores, top_k_values = torch.topk(scores, topk, dim=-1)
        linked_entity_dict: dict[str, list] = {}
        for i in range(len(ner_entity_list)):
            linked_entity_dict[ner_entity_list[i]] = []

            sorted_score = top_k_scores[i]
            sorted_indices = top_k_values[i]
            max_score = sorted_score[0].item()

            for score, top_k_index in zip(sorted_score, sorted_indices):
                linked_entity_dict[ner_entity_list[i]].append(
                    {
                        "entity": self.entity_list[top_k_index],
                        "score": score.item(),
                        "norm_score": score.item() / max_score,
                    }
                )
        return linked_entity_dict


class NVEmbedV2ELModel(DPRELModel):
    """
    A DPR-based Entity Linking model specialized for NVEmbed V2 embeddings.

    This class extends DPRELModel with specific adaptations for handling NVEmbed V2 models,
    including increased sequence length and right-side padding.

    Attributes:
        model: The underlying model with max_seq_length of 32768 and right-side padding.

    Methods:
        add_eos(input_examples): Adds EOS token to input examples.
        __call__(ner_entity_list): Processes entity list with EOS tokens before linking.

    Examples:
        >>> model = NVEmbedV2ELModel('nvidia/NV-Embed-v2', query_instruct=\"Instruct: Given a entity, retrieve entities that are semantically equivalent to the given entity\\nQuery: \")
        >>> model.index(['Paris', 'London', 'Berlin'])
        >>> results = model(['paris city'], topk=2)
        >>> print(results)
        {'paris city': [{'entity': 'Paris', 'score': 0.82, 'norm_score': 1.0},
                        {'entity': 'London', 'score': 0.35, 'norm_score': 0.43}]}
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the DPR Entity Linking model.

        This initialization extends the base class initialization and sets specific model parameters
        for entity linking tasks. It configures the maximum sequence length to 32768 and sets
        the tokenizer padding side to "right".

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        super().__init__(
            *args,
            **kwargs,
        )
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"

    def add_eos(self, input_examples: list[str]) -> list[str]:
        """
        Appends EOS (End of Sequence) token to each input example in the list.

        Args:
            input_examples (list[str]): List of input text strings.

        Returns:
            list[str]: List of input texts with EOS token appended to each example.
        """
        input_examples = [
            input_example + self.model.tokenizer.eos_token
            for input_example in input_examples
        ]
        return input_examples

    def __call__(self, ner_entity_list: list, *args: Any, **kwargs: Any) -> dict:
        """
        Execute entity linking for a list of named entities.

        Args:
            ner_entity_list (list): List of named entities to be linked.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            dict: Entity linking results mapping entities to their linked entries.
        """
        ner_entity_list = self.add_eos(ner_entity_list)
        return super().__call__(ner_entity_list, *args, **kwargs)
