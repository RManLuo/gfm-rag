import json
import logging
import os

import pandas as pd
from omegaconf import DictConfig

from gfmrag.graph_index_construction import BaseGraphConstructor, BaseSFTConstructor
from gfmrag.graph_index_datasets.graph_index_dataset import GraphIndexDataset
from gfmrag.utils import check_all_files_exist

logger = logging.getLogger(__name__)


class GraphIndexer:
    """
    A class for indexing and processing datasets by creating graph indices and preparing SFT data.

    Attributes:
        graph_constructor (BaseKGConstructor): Constructor for building graph indices over documents.
        sft_constructor (BaseSFTConstructor): Constructor for preparing SFT datasets.
    """

    def __init__(
        self,
        graph_constructor: BaseGraphConstructor,
        sft_constructor: BaseSFTConstructor,
    ) -> None:
        """
        Initializes the GraphIndexer with the given graph and SFT constructors.

        Args:
            graph_constructor (BaseKGConstructor): An instance of a graph constructor.
            sft_constructor (BaseSFTConstructor): An instance of a SFT constructor.

        Returns:
            None
        """
        self.graph_constructor = graph_constructor
        self.sft_constructor = sft_constructor

    def index_data(self, dataset_cfg: DictConfig) -> None:
        """Index and process dataset by creating graph indices and preparing SFT data.

        This method performs two main tasks:
            1. Creates and saves graph related files (nodes.csv, relations.csv, edges.csv)
            2. Identify the query entities and supporting entities in training and testing data if available in the raw data directory

        Files created:
            - nodes.csv: Contains nodes of the graph
            - edges.csv: Contains edges of the graph
            - relations.csv: Contains relations of the graph
            - train.json: Processed training data (if raw exists)
            - test.json: Processed test data (if raw exists)

            Directory structure:
            ```
                root/
                └── data_name/
                    ├── raw/
                    |   ├── documents.json
                    │   ├── train.json (optional)
                    │   └── test.json (optional)
                    └── processed/
                        └── stage1/
                            ├── edges.csv
                            ├── nodes.csv
                            ├── relations.csv
                            ├── train.json (optional)
                            └── test.json (optional)
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
        force = dataset_cfg.get("force", False)
        raw_data_dir = os.path.join(root, data_name, "raw")
        prosessed_data_dir = os.path.join(root, data_name, "processed", "stage1")

        if not os.path.exists(prosessed_data_dir):
            os.makedirs(prosessed_data_dir)

        # Create graph index for each dataset
        raw_graph_files = [
            os.path.join(prosessed_data_dir, name)
            for name in GraphIndexDataset.RAW_GRAPH_NAMES
        ]
        if not check_all_files_exist(raw_graph_files) or force:
            logger.info("Stage1 Graph construction")
            graph = self.graph_constructor.build_graph(root, data_name)
            # Save nodes.csv, edges.csv, relations.csv
            nodes_df = pd.DataFrame(graph["nodes"])
            nodes_df.to_csv(os.path.join(prosessed_data_dir, "nodes.csv"), index=False)
            edges_df = pd.DataFrame(graph["edges"])
            edges_df.to_csv(os.path.join(prosessed_data_dir, "edges.csv"), index=False)
            relations_df = pd.DataFrame(graph["relations"])
            relations_df.to_csv(
                os.path.join(prosessed_data_dir, "relations.csv"), index=False
            )
            logger.info(
                f"Graph index files saved to {prosessed_data_dir}:\n"
                f"- nodes.csv\n- edges.csv\n- relations.csv"
            )

        # Try to prepare training and testing data from dataset
        if os.path.exists(os.path.join(raw_data_dir, "train.json")) and (
            not os.path.exists(os.path.join(prosessed_data_dir, "train.json")) or force
        ):
            logger.info(f"Preparing {os.path.join(raw_data_dir, 'train.json')}")
            train_data = self.sft_constructor.prepare_data(
                root, data_name, "train.json"
            )
            with open(os.path.join(prosessed_data_dir, "train.json"), "w") as f:
                json.dump(train_data, f, indent=4)

        if os.path.exists(os.path.join(raw_data_dir, "test.json")) and (
            not os.path.exists(os.path.join(prosessed_data_dir, "test.json")) or force
        ):
            logger.info(f"Preparing {os.path.join(raw_data_dir, 'test.json')}")
            test_data = self.sft_constructor.prepare_data(root, data_name, "test.json")
            with open(os.path.join(prosessed_data_dir, "test.json"), "w") as f:
                json.dump(test_data, f, indent=4)
