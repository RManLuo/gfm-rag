import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from multiprocessing.dummy import Pool as ThreadPool

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gfmrag.kg_construction.utils import KG_DELIMITER

from .entity_linking_model import BaseELModel
from .ner_model import BaseNERModel

logger = logging.getLogger(__name__)


class BaseQAConstructor(ABC):
    """An abstract base class for constructing Question-Answering (QA) datasets.

    Attributes:
        None

    Methods:
        prepare_data:
            Abstract method that must be implemented by subclasses to prepare QA data.
            Takes data location parameters and returns processed data as a list of dictionaries.

    """

    @abstractmethod
    def prepare_data(self, data_root: str, data_name: str, file: str) -> list[dict]:
        """
        Prepare QA data for training and evaluation

        Args:
            data_root (str): path to the dataset
            data_name (str): name of the dataset
            file (str): file name to process
        Returns:
            list[dict]: list of processed data
        """
        pass


class QAConstructor(BaseQAConstructor):
    """QA Constructor for building question-answer datasets with entity linking and named entity recognition.

    This class processes raw QA datasets by performing Named Entity Recognition (NER) on questions
    and Entity Linking (EL) to connect identified entities to a knowledge graph (KG).

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

    DELIMITER = KG_DELIMITER

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

    @staticmethod
    def from_config(cfg: DictConfig) -> "QAConstructor":
        """Creates a QAConstructor instance from a configuration.

        This method initializes a QAConstructor using configuration parameters, creating a unique
        temporary directory based on the config fingerprint to store processing artifacts.

        Args:
            cfg (DictConfig): Configuration object containing:

                - root: Base directory path
                - ner_model: Named Entity Recognition model configuration
                - el_model: Entity Linking model configuration
                - num_processes: Number of processes to use
                - force: Force reprocessing flag (optional)

        Returns:
            QAConstructor: Initialized QAConstructor instance with specified configuration

        Note:
            The method creates a temporary directory using MD5 hash of the config as fingerprint,
            excluding the 'force' parameter. The full config is saved in this directory as
            'config.json'.
        """
        # create a fingerprint of config for tmp directory
        config = OmegaConf.to_container(cfg, resolve=True)
        if "force" in config:
            del config["force"]
        fingerprint = hashlib.md5(json.dumps(config).encode()).hexdigest()

        base_tmp_dir = os.path.join(cfg.root, fingerprint)
        if not os.path.exists(base_tmp_dir):
            os.makedirs(base_tmp_dir)
            json.dump(
                config,
                open(os.path.join(base_tmp_dir, "config.json"), "w"),
                indent=4,
            )
        return QAConstructor(
            root=base_tmp_dir,
            ner_model=instantiate(cfg.ner_model),
            el_model=instantiate(cfg.el_model),
            num_processes=cfg.num_processes,
            force=cfg.force,
        )

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

        if not os.path.exists(os.path.join(processed_path, "kg.txt")):
            raise FileNotFoundError(
                "KG file not found. Please run KG construction first"
            )

        # Read KG
        entities = set()
        with open(os.path.join(processed_path, "kg.txt")) as f:
            for line in f:
                try:
                    u, _, v = line.strip().split(self.DELIMITER)
                except Exception as e:
                    logger.error(f"Error in line: {line}, {e}, Skipping")
                    continue
                entities.add(u)
                entities.add(v)
        # Read document2entities
        with open(os.path.join(processed_path, "document2entities.json")) as f:
            doc2entities = json.load(f)

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

            supporting_facts = sample.get("supporting_facts", [])
            supporting_entities = []
            for item in list(set(supporting_facts)):
                supporting_entities.extend(doc2entities.get(item, []))

            final_data.append(
                {
                    **sample,
                    "question_entities": question_entities,
                    "supporting_entities": supporting_entities,
                }
            )

        return final_data
