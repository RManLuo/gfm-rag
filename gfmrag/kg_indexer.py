import json
import logging
import os

from omegaconf import DictConfig

from .kg_construction import BaseKGConstructor, BaseQAConstructor
from .kg_construction.utils import KG_DELIMITER

logger = logging.getLogger(__name__)


class KGIndexer:
    """
    A class for indexing and processing datasets by creating knowledge graph indices and preparing QA data.

    Attributes:
        DELIMITER (str): Delimiter used for separating elements in knowledge graph triples, default is `","`.
        kg_constructor (BaseKGConstructor): Constructor for building knowledge graphs
        qa_constructor (BaseQAConstructor): Constructor for preparing QA datasets
    """

    DELIMITER = KG_DELIMITER

    def __init__(
        self, kg_constructor: BaseKGConstructor, qa_constructor: BaseQAConstructor
    ) -> None:
        """
        Initializes the KGIndexer with the given knowledge graph and QA constructors.

        Args:
            kg_constructor (BaseKGConstructor): An instance of a knowledge graph constructor.
            qa_constructor (BaseQAConstructor): An instance of a QA constructor.

        Returns:
            None
        """
        self.kg_constructor = kg_constructor
        self.qa_constructor = qa_constructor

    def index_data(self, dataset_cfg: DictConfig) -> None:
        """Index and process dataset by creating knowledge graph (KG) indices and preparing QA data.

        This method performs two main tasks:
            1. Creates and saves knowledge graph related files (kg.txt and document2entities.json)
            2. Identify the query entities and supporting entities in training and testing data if available in the raw data directory

        Files created:
            - kg.txt: Contains knowledge graph triples
            - document2entities.json: Maps documents to their entities
            - train.json: Processed training data (if raw exists)
            - test.json: Processed test data (if raw exists)

            Directory structure:
            ```
                root/
                └── data_name/
                    ├── raw/
                    |   ├── dataset_corpus.json
                    │   ├── train.json (optional)
                    │   └── test.json (optional)
                    └── processed/
                        └── stage1/
                            ├── kg.txt
                            ├── document2entities.json
                            ├── train.json
                            └── test.json
            ```

        Args:
            dataset_cfg (DictConfig):
                - root (str): Root directory of the dataset
                - data_name (str): Name of the dataset

        Returns:
            None
        """

        root = dataset_cfg.root
        data_name = dataset_cfg.data_name
        raw_data_dir = os.path.join(root, data_name, "raw")
        prosessed_data_dir = os.path.join(root, data_name, "processed", "stage1")

        if not os.path.exists(prosessed_data_dir):
            os.makedirs(prosessed_data_dir)

        # Create KG index for each dataset
        if not os.path.exists(os.path.join(prosessed_data_dir, "kg.txt")):
            logger.info("Stage1 KG construction")
            kg = self.kg_constructor.create_kg(root, data_name)
            with open(os.path.join(prosessed_data_dir, "kg.txt"), "w") as f:
                for triple in kg:
                    f.write(self.DELIMITER.join(triple) + "\n")
        if not os.path.exists(
            os.path.join(prosessed_data_dir, "document2entities.json")
        ):
            logger.info("Stage1 Get document2entities")
            doc2entities = self.kg_constructor.get_document2entities(root, data_name)
            with open(
                os.path.join(prosessed_data_dir, "document2entities.json"), "w"
            ) as f:
                json.dump(doc2entities, f, indent=4)

        # Try to prepare training and testing data from dataset
        if os.path.exists(
            os.path.join(raw_data_dir, "train.json")
        ) and not os.path.exists(os.path.join(prosessed_data_dir, "train.json")):
            logger.info(f"Preparing {os.path.join(raw_data_dir, 'train.json')}")
            train_data = self.qa_constructor.prepare_data(root, data_name, "train.json")
            with open(os.path.join(prosessed_data_dir, "train.json"), "w") as f:
                json.dump(train_data, f, indent=4)

        if os.path.exists(
            os.path.join(raw_data_dir, "test.json")
        ) and not os.path.exists(os.path.join(prosessed_data_dir, "test.json")):
            logger.info(f"Preparing {os.path.join(raw_data_dir, 'test.json')}")
            test_data = self.qa_constructor.prepare_data(root, data_name, "test.json")
            with open(os.path.join(prosessed_data_dir, "test.json"), "w") as f:
                json.dump(test_data, f, indent=4)
