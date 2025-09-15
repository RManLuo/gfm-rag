import json
import logging
import os
from multiprocessing.dummy import Pool as ThreadPool

import pandas as pd
from tqdm import tqdm

from gfmrag.graph_index_datasets.graph_index_dataset import GraphIndexDataset
from gfmrag.utils.util import check_all_files_exist

from ..entity_linking_model import BaseELModel
from ..ner_model import BaseNERModel
from .base_sft_constructor import BaseSFTConstructor

logger = logging.getLogger(__name__)


class GFMRAGConstructor(BaseSFTConstructor):
    """SFT Constructor for building question-answer datasets with entity linking and named entity recognition used for GFM-RAG-v1.

    This class processes raw QA datasets by performing Named Entity Recognition (NER) on questions and Entity Linking (EL) to connect identified entities to a knowledge graph (KG) to create start_nodes.

    It extracts the entities from the supporting documents and links to the KGs to create target_nodes.

    Args:
        ner_model (BaseNERModel): Model for Named Entity Recognition
        el_model (BaseELModel): Model for Entity Linking
        root (str, optional): Root directory for temporary files. Defaults to "tmp/qa_construction"
        num_processes (int, optional): Number of processes for parallel processing. Defaults to 1
        force (bool, optional): Whether to force recomputation of cached results. Defaults to False

    Attributes:
        ner_model: The NER model instance
        el_model: The EL model instance
        root: Root directory path
        num_processes: Number of parallel processes
        data_name: Name of the current dataset being processed
        force: Whether to force recompute results
        DELIMITER: Delimiter used in knowledge graph files

    Methods:
        from_config: Creates a QAConstructor instance from a configuration
        prepare_data: Processes raw QA data to add entity information

    The class expects a knowledge graph and document-to-entities mapping to be pre-computed
    and stored in the processed/stage1 directory of the dataset.
    """

    def __init__(
        self,
        ner_model: BaseNERModel,
        el_model: BaseELModel,
        root: str = "tmp/qa_construction",
        num_processes: int = 1,
        force: bool = False,
    ) -> None:
        """Initialize the Question Answer Constructor.

        This constructor processes text data through Named Entity Recognition (NER) and Entity Linking (EL) models
        to generate question-answer pairs.

        Args:
            ner_model (BaseNERModel): Model for Named Entity Recognition.
            el_model (BaseELModel): Model for Entity Linking.
            root (str, optional): Root directory for saving processed data. Defaults to "tmp/qa_construction".
            num_processes (int, optional): Number of processes for parallel processing. Defaults to 1.
            force (bool, optional): If True, forces reprocessing of existing data. Defaults to False.

        Attributes:
            ner_model (BaseNERModel): Initialized NER model instance.
            el_model (BaseELModel): Initialized EL model instance.
            root (str): Root directory path.
            num_processes (int): Number of parallel processes.
            data_name (None): Name of the dataset, initialized as None.
            force (bool): Force reprocessing flag.
        """

        self.ner_model = ner_model
        self.el_model = el_model
        self.root = root
        self.num_processes = num_processes
        self.data_name = None
        self.force = force

    @property
    def tmp_dir(self) -> str:
        """
        Returns the temporary directory path for data processing.

        This property method creates and returns a directory path specific to the current
        data_name under the root directory. The directory is created if it doesn't exist.

        Returns:
            str: Path to the temporary directory.

        Raises:
            AssertionError: If data_name is not set before accessing this property.
        """
        assert (
            self.data_name is not None
        )  # data_name should be set before calling this property
        tmp_dir = os.path.join(self.root, self.data_name)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        return tmp_dir

    def prepare_data(self, data_root: str, data_name: str, file: str) -> list[dict]:
        """
        Prepares data for question answering by processing raw data, performing Named Entity Recognition (NER),
        and Entity Linking (EL).

        Args:
            data_root (str): Root directory path containing the dataset.
            data_name (str): Name of the dataset.
            file (str): Filename of the raw data.

        Returns:
            list[dict]: A list of processed data samples. Each sample is a dictionary containing:
                - Original sample fields
                - question_entities (list): Linked entities found in the question
                - supporting_entities (list): Entities from supporting facts

        Raises:
            FileNotFoundError: If the required KG file is not found in the processed directory.

        Notes:
            - Requires a pre-constructed knowledge graph (KG) file in the processed directory
            - Uses cached NER results if available, otherwise performs NER processing
            - Performs entity linking on identified entities
            - Combines question entities with supporting fact entities
        """
        # Get dataset information
        self.data_name = data_name  # type: ignore
        raw_path = os.path.join(data_root, data_name, "raw", file)
        processed_path = os.path.join(data_root, data_name, "processed", "stage1")

        if self.force:
            # Clear cache in tmp directory
            for tmp_file in os.listdir(self.tmp_dir):
                os.remove(os.path.join(self.tmp_dir, tmp_file))

        # Create graph index for each dataset
        raw_graph_files = [
            os.path.join(processed_path, name)
            for name in GraphIndexDataset.RAW_GRAPH_NAMES
        ]
        if not check_all_files_exist(raw_graph_files):
            raise FileNotFoundError(
                "Graph file not found. Please run KG construction first"
            )

        # Read nodes.csv to get entities
        nodes = pd.read_csv(
            os.path.join(processed_path, "nodes.csv"), keep_default_na=False
        )

        # Get nodes with type 'entity'
        entities = nodes[nodes["type"] == "entity"]["name"].tolist()

        # Read edges.csv
        edges = pd.read_csv(
            os.path.join(processed_path, "edges.csv"), keep_default_na=False
        )

        # Get document2entities mapping
        ent_doc_edges = edges[edges["relation"] == "is_mentioned_in"]
        doc2entities = ent_doc_edges.groupby("target")["source"].apply(list).to_dict()

        # Load data
        with open(raw_path) as f:
            data = json.load(f)

        ner_results = {}
        # Try to read ner results
        if os.path.exists(os.path.join(self.tmp_dir, "ner_results.jsonl")):
            with open(os.path.join(self.tmp_dir, "ner_results.jsonl")) as f:
                ner_logs = [json.loads(line) for line in f]
                ner_results = {log["id"]: log for log in ner_logs}

        unprocessed_data = [
            sample for sample in data if sample["id"] not in ner_results
        ]

        def _ner_process(sample: dict) -> dict:
            id = sample["id"]
            question = sample["question"]
            ner_ents = self.ner_model(question)
            return {
                "id": id,
                "question": question,
                "ner_ents": ner_ents,
            }

        # NER
        with ThreadPool(self.num_processes) as pool:
            with open(os.path.join(self.tmp_dir, "ner_results.jsonl"), "a") as f:
                for res in tqdm(
                    pool.imap(_ner_process, unprocessed_data),
                    total=len(unprocessed_data),
                ):
                    ner_results[res["id"]] = res
                    f.write(json.dumps(res) + "\n")

        # EL
        self.el_model.index(list(entities))

        ner_entities = []
        for res in ner_results.values():
            ner_entities.extend(res["ner_ents"])

        el_results = self.el_model(ner_entities, topk=1)

        # Prepare final data
        final_data = []
        for sample in data:
            id = sample["id"]
            ner_ents = ner_results[id]["ner_ents"]
            question_entities = []
            for ent in ner_ents:
                question_entities.append(el_results[ent][0]["entity"])

            supporting_documents = sample.get("supporting_documents", [])
            supporting_entities = []
            for item in list(set(supporting_documents)):
                supporting_entities.extend(doc2entities.get(item, []))

            final_data.append(
                {
                    **sample,
                    "start_nodes": {"entity": question_entities},
                    "target_nodes": {"entity": supporting_entities},
                }
            )

        return final_data
