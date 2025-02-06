import hashlib
import os
import shutil

from colbert import Indexer, Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig

from gfmrag.kg_construction.utils import processing_phrases

from .base_model import BaseELModel


class ColbertELModel(BaseELModel):
    """ColBERT-based Entity Linking Model.

    This class implements an entity linking model using ColBERT, a neural information retrieval
    framework. It indexes a list of entities and performs entity linking by finding the most
    similar entities in the index for given named entities.

    Attributes:
        checkpoint_path (str): Path to the ColBERT checkpoint file
        root (str): Root directory for storing indices
        doc_index_name (str): Name of document index
        phrase_index_name (str): Name of phrase index
        force (bool): Whether to force reindex if index exists
        entity_list (list): List of entities to be indexed
        phrase_searcher: ColBERT phrase searcher object

    Raises:
        FileNotFoundError: If the checkpoint file is not found at the specified path.
        AttributeError: If entity linking is attempted before indexing.

    Notes:
        You need to download the checkpoint by running the following command:
        `wget -O https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz && tar -zxvf tmp/colbertv2.0.tar.gz -C tmp/`

    Examples:
        >>> model = ColbertELModel("path/to/checkpoint")
        >>> model.index(["entity1", "entity2", "entity3"])
        >>> results = model(["query1", "query2"], topk=3)
        >>> print(results)
        {'paris city': [{'entity': 'entity1', 'score': 0.82, 'norm_score': 1.0},
                        {'entity': 'entity2', 'score': 0.35, 'norm_score': 0.43}]}
    """

    def __init__(
        self,
        checkpint_path: str,
        root: str = "tmp",
        doc_index_name: str = "nbits_2",
        phrase_index_name: str = "nbits_2",
        force: bool = False,
    ) -> None:
        """
        Initialize the ColBERT entity linking model.

        This initializes a ColBERT model for entity linking using pre-trained checkpoints and indices.

        Args:
            checkpint_path (str): Path to the ColBERT checkpoint file. Model weights will be loaded from this path. Can be downloaded [here](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz)
            root (str, optional): Root directory for storing indices. Defaults to "tmp".
            doc_index_name (str, optional): Name of the document index. Defaults to "nbits_2".
            phrase_index_name (str, optional): Name of the phrase index. Defaults to "nbits_2".
            force (bool, optional): Whether to force recomputation of existing indices. Defaults to False.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist at the specified path.

        Returns:
            None
        """

        if not os.path.exists(checkpint_path):
            raise FileNotFoundError(
                "Checkpoint not found, download the checkpoint with: 'wget -O https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz && tar -zxvf tmp/colbertv2.0.tar.gz -C tmp/'"
            )
        self.checkpoint_path = checkpint_path
        self.root = root
        self.doc_index_name = doc_index_name
        self.phrase_index_name = phrase_index_name
        self.force = force

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
        # Get md5 fingerprint of the whole given entity list
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        exp_name = f"Entity_index_{fingerprint}"
        if os.path.exists(f"{self.root}/colbert/{fingerprint}") and self.force:
            shutil.rmtree(f"{self.root}/colbert/{fingerprint}")
        colbert_config = {
            "root": f"{self.root}/colbert/{fingerprint}",
            "doc_index_name": self.doc_index_name,
            "phrase_index_name": self.phrase_index_name,
        }
        phrases = [processing_phrases(p) for p in entity_list]
        with Run().context(
            RunConfig(nranks=1, experiment=exp_name, root=colbert_config["root"])
        ):
            config = ColBERTConfig(
                nbits=2,
                root=colbert_config["root"],
            )
            indexer = Indexer(checkpoint=self.checkpoint_path, config=config)
            indexer.index(
                name=self.phrase_index_name, collection=phrases, overwrite="reuse"
            )

        with Run().context(
            RunConfig(nranks=1, experiment=exp_name, root=colbert_config["root"])
        ):
            config = ColBERTConfig(
                root=colbert_config["root"],
            )
            phrase_searcher = Searcher(
                index=colbert_config["phrase_index_name"], config=config, verbose=1
            )
        self.phrase_searcher = phrase_searcher

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

        ner_entity_list = [processing_phrases(p) for p in ner_entity_list]
        query_data: dict[int, str] = {
            i: query for i, query in enumerate(ner_entity_list)
        }

        queries = Queries(path=None, data=query_data)
        ranking = self.phrase_searcher.search_all(queries, k=topk)

        linked_entity_dict: dict[str, list] = {}
        for i in range(len(queries)):
            query = queries[i]
            rank = ranking.data[i]
            linked_entity_dict[query] = []
            max_score = rank[0][2]

            for phrase_id, _rank, score in rank:
                linked_entity_dict[query].append(
                    {
                        "entity": self.entity_list[phrase_id],
                        "score": score,
                        "norm_score": score / max_score,
                    }
                )

        return linked_entity_dict
