import hashlib
import os
import shutil

from colbert import Indexer, Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig
from huggingface_hub import snapshot_download

from gfmrag.graph_index_construction.utils import processing_phrases

from .base_model import BaseELModel


class ColbertELModel(BaseELModel):
    """ColBERT-based Entity Linking Model.

    This class implements an entity linking model using ColBERT, a neural information retrieval
    framework. It indexes a list of entities and performs entity linking by finding the most
    similar entities in the index for given named entities.

    Attributes:
        model_name_or_path (str): Path to the ColBERT checkpoint file
        root (str): Root directory for storing indices
        doc_index_name (str): Name of document index
        phrase_index_name (str): Name of phrase index
        force (bool): Whether to force reindex if index exists
        entity_list (list): List of entities to be indexed

    Raises:
        AttributeError: If entity linking is attempted before indexing.

    Examples:
        >>> model = ColbertELModel("colbert-ir/colbertv2.0")
        >>> model.index(["entity1", "entity2", "entity3"])
        >>> results = model(["query1", "query2"], topk=3)
        >>> print(results)
        {'paris city': [{'entity': 'entity1', 'score': 0.82, 'norm_score': 1.0},
                        {'entity': 'entity2', 'score': 0.35, 'norm_score': 0.43}]}
    """

    def __init__(
        self,
        model_name_or_path: str = "colbert-ir/colbertv2.0",
        root: str = "tmp",
        doc_index_name: str = "nbits_2",
        phrase_index_name: str = "nbits_2",
        force: bool = False,
        **_: str,
    ) -> None:
        """
        Initialize the ColBERT entity linking model.

        This initializes a ColBERT model for entity linking using pre-trained checkpoints and indices.

        Args:
            model_name_or_path (str, optional): Hugging Face model name or local checkpoint path. Defaults to "colbert-ir/colbertv2.0".
            root (str, optional): Root directory for storing indices. Defaults to "tmp".
            doc_index_name (str, optional): Name of the document index. Defaults to "nbits_2".
            phrase_index_name (str, optional): Name of the phrase index. Defaults to "nbits_2".
            force (bool, optional): Whether to force recomputation of existing indices. Defaults to False.

        Returns:
            None
        """
        self.model_name_or_path = model_name_or_path
        self.root = root
        self.doc_index_name = doc_index_name
        self.phrase_index_name = phrase_index_name
        self.force = force
        self.checkpoint_path = self._resolve_checkpoint_path(model_name_or_path)

    def _resolve_checkpoint_path(self, model_name_or_path: str) -> str:
        """Resolve a local ColBERT checkpoint path from a local path or HF repo id."""
        if os.path.exists(model_name_or_path):
            return model_name_or_path
        return snapshot_download(
            repo_id=model_name_or_path,
            local_dir=os.path.join(
                self.root, "hf_cache", model_name_or_path.replace("/", "__")
            ),
            local_dir_use_symlinks=False,
        )

    def index(self, entity_list: list) -> None:
        """
        Index a list of entities using ColBERT for efficient similarity search.

        This method processes and indexes a list of entities using the ColBERT model. It creates
        a unique index based on the MD5 hash of the entity list and stores it in the specified
        root directory.

        Args:
            entity_list (list): List of entity strings to be indexed.

        Returns:
            None

        Notes:
            - Creates a unique index directory based on MD5 hash of entities
            - If force=True, will delete existing index with same fingerprint
            - Processes entities into phrases before indexing
            - Sets up ColBERT indexer and searcher with specified configuration
            - Stores phrase_searcher as instance variable for later use
        """
        self.entity_list = entity_list
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        exp_name = f"Entity_index_{fingerprint}"
        if os.path.exists(f"{self.root}/colbert/{fingerprint}") and self.force:
            shutil.rmtree(f"{self.root}/colbert/{fingerprint}")
        phrases = [processing_phrases(p) for p in entity_list]
        colbert_root = f"{self.root}/colbert/{fingerprint}"
        with Run().context(RunConfig(nranks=1, experiment=exp_name, root=colbert_root)):
            config = ColBERTConfig(nbits=2, root=colbert_root)
            indexer = Indexer(checkpoint=self.checkpoint_path, config=config)
            indexer.index(
                name=self.phrase_index_name,
                collection=phrases,
                overwrite="reuse" if not self.force else True,
            )

        with Run().context(RunConfig(nranks=1, experiment=exp_name, root=colbert_root)):
            config = ColBERTConfig(root=colbert_root)
            self.phrase_searcher = Searcher(
                index=self.phrase_index_name,
                config=config,
                verbose=1,
            )

    def __call__(self, ner_entity_list: list, topk: int = 1) -> dict:
        """
        Link entities in the given text to the knowledge graph.

        Args:
            ner_entity_list (list): list of named entities
            topk (int): number of linked entities to return

        Returns:
            dict: dict of linked entities in the knowledge graph

                - key (str): named entity
                - value (list[dict]): list of linked entities

                    - entity: linked entity
                    - score: score of the entity
                    - norm_score: normalized score of the entity
        """

        try:
            self.__getattribute__("phrase_searcher")
        except AttributeError as e:
            raise AttributeError("Index the entities first using index method") from e

        queries = [processing_phrases(p) for p in ner_entity_list]
        query_data: dict[int, str] = {i: query for i, query in enumerate(queries)}
        ranking = self.phrase_searcher.search_all(
            Queries(path=None, data=query_data), k=topk
        )

        linked_entity_dict: dict[str, list] = {}
        for i, query in enumerate(queries):
            linked_entity_dict[query] = []
            rank = ranking.data[i]
            max_score = rank[0][2] if rank else 1.0
            for phrase_id, _rank, score in rank:
                linked_entity_dict[query].append(
                    {
                        "entity": self.entity_list[phrase_id],
                        "score": score,
                        "norm_score": score / max_score if max_score else 0.0,
                    }
                )

        return linked_entity_dict
