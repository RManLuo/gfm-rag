import hashlib
import json
import logging
import os
import os.path as osp
import sys
import warnings

import datasets
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils import data as torch_data
from torch_geometric.data import InMemoryDataset, makedirs
from torch_geometric.data.dataset import _repr, files_exist

from gfmrag.datasets.kg_dataset import KGDataset
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils import get_rank, is_main_process, synchronize
from gfmrag.utils.qa_utils import entities_to_mask

logger = logging.getLogger(__name__)


class QADataset(InMemoryDataset):
    """A dataset class for Question-Answering tasks built on top of a Knowledge Graph.

    This dataset inherits from torch_geometric's InMemoryDataset and processes raw QA data
    into a format suitable for graph-based QA models. It handles both training and test splits.

    Args:
        root (str): Root directory where the dataset should be saved.
        data_name (str): Name of the dataset.
        text_emb_model_cfgs (DictConfig): Configuration for the text embedding model used to encode questions.
        force_rebuild (bool, optional): If True, forces the dataset to be reprocessed even if it exists. Defaults to False.

    Attributes:
        name (str): Name of the dataset.
        kg (KGDataset): The underlying knowledge graph dataset.
        rel_emb_dim (int): Dimension of relation embeddings.
        ent2id (dict): Mapping from entity names to IDs.
        rel2id (dict): Mapping from relation names to IDs.
        doc (dict): Corpus of documents.
        doc2entities (dict): Mapping from documents to contained entities.
        raw_train_data (list): Raw training data samples.
        raw_test_data (list): Raw test data samples.
        ent2docs (torch.Tensor): Sparse tensor mapping entities to documents.
        id2doc (dict): Mapping from document IDs to document names.

    Notes:
        The processed dataset contains:
        - Question embeddings
        - Question entity masks
        - Supporting entity masks
        - Supporting document masks
        - Sample IDs

        The dataset processes raw JSON files and creates PyTorch tensors for efficient training.
    """

    def __init__(
        self,
        root: str,
        data_name: str,
        text_emb_model_cfgs: DictConfig,
        force_rebuild: bool = False,
    ):
        self.name = data_name
        self.force_rebuild = force_rebuild
        self.text_emb_model_cfgs = text_emb_model_cfgs
        # Get fingerprint of the model configuration
        self.fingerprint = hashlib.md5(
            json.dumps(
                OmegaConf.to_container(text_emb_model_cfgs, resolve=True)
            ).encode()
        ).hexdigest()
        self.kg = KGDataset(root, data_name, text_emb_model_cfgs, force_rebuild)[0]
        self.rel_emb_dim = self.kg.rel_emb.shape[-1]
        super().__init__(root, None, None)
        self.data = torch.load(self.processed_paths[0], weights_only=False)
        self.load_property()

    def __repr__(self) -> str:
        return f"{self.name}()"

    @property
    def raw_file_names(self) -> list:
        return ["train.json", "test.json"]

    @property
    def raw_dir(self) -> str:
        return os.path.join(str(self.root), str(self.name), "processed", "stage1")

    @property
    def processed_dir(self) -> str:
        return os.path.join(
            str(self.root),
            str(self.name),
            "processed",
            "stage2",
            self.fingerprint,
        )

    @property
    def processed_file_names(self) -> str:
        return "qa_data.pt"

    def load_property(self) -> None:
        """
        Load necessary properties from the KG dataset.
        """
        with open(os.path.join(self.processed_dir, "ent2id.json")) as fin:
            self.ent2id = json.load(fin)
        with open(os.path.join(self.processed_dir, "rel2id.json")) as fin:
            self.rel2id = json.load(fin)
        with open(
            os.path.join(str(self.root), str(self.name), "raw", "dataset_corpus.json")
        ) as fin:
            self.doc = json.load(fin)
        with open(os.path.join(self.raw_dir, "document2entities.json")) as fin:
            self.doc2entities = json.load(fin)
        if os.path.exists(os.path.join(self.raw_dir, "train.json")):
            with open(os.path.join(self.raw_dir, "train.json")) as fin:
                self.raw_train_data = json.load(fin)
        else:
            self.raw_train_data = []
        if os.path.exists(os.path.join(self.raw_dir, "test.json")):
            with open(os.path.join(self.raw_dir, "test.json")) as fin:
                self.raw_test_data = json.load(fin)
        else:
            self.raw_test_data = []

        self.ent2docs = torch.load(
            os.path.join(self.processed_dir, "ent2doc.pt"), weights_only=False
        )  # (n_nodes, n_docs)
        self.id2doc = {i: doc for i, doc in enumerate(self.doc2entities)}

    def _process(self) -> None:
        if is_main_process():
            logger.info(f"Processing QA dataset {self.name} at rank {get_rank()}")
            f = osp.join(self.processed_dir, "pre_transform.pt")
            if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
                self.pre_transform
            ):
                warnings.warn(
                    f"The `pre_transform` argument differs from the one used in "
                    f"the pre-processed version of this dataset. If you want to "
                    f"make use of another pre-processing technique, make sure to "
                    f"delete '{self.processed_dir}' first",
                    stacklevel=1,
                )

            f = osp.join(self.processed_dir, "pre_filter.pt")
            if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
                self.pre_filter
            ):
                warnings.warn(
                    f"The `pre_filter` argument differs from the one used in "
                    f"the pre-processed version of this dataset. If you want to "
                    f"make use of another pre-fitering technique, make sure to "
                    f"delete '{self.processed_dir}' first",
                    stacklevel=1,
                )

            if self.force_rebuild or not files_exist(self.processed_paths):
                if self.log and "pytest" not in sys.modules:
                    print("Processing...", file=sys.stderr)

                makedirs(self.processed_dir)
                self.process()

                path = osp.join(self.processed_dir, "pre_transform.pt")
                torch.save(_repr(self.pre_transform), path)
                path = osp.join(self.processed_dir, "pre_filter.pt")
                torch.save(_repr(self.pre_filter), path)

                if self.log and "pytest" not in sys.modules:
                    print("Done!", file=sys.stderr)
        else:
            logger.info(
                f"Rank [{get_rank()}]: Waiting for main process to finish processing QA dataset {self.name}"
            )
        synchronize()

    def process(self) -> None:
        """Process and prepare the question-answering dataset.

        This method processes raw data files to create a structured dataset for question answering
        tasks. It performs the following main operations:

        1. Loads entity and relation mappings from processed files
        2. Creates entity-document mapping tensors
        3. Processes question samples to generate:
            - Question embeddings
            - Question entity masks
            - Supporting entity masks
            - Supporting document masks

        The processed dataset is saved as torch splits containing:

        - Question embeddings
        - Various mask tensors for entities and documents
        - Sample IDs

        Files created:

        - ent2doc.pt: Sparse tensor mapping entities to documents
        - qa_data.pt: Processed QA dataset
        - text_emb_model_cfgs.json: Text embedding model configuration

        The method also saves the text embedding model configuration.

        Returns:
            None
        """
        with open(os.path.join(self.processed_dir, "ent2id.json")) as fin:
            self.ent2id = json.load(fin)
        with open(os.path.join(self.processed_dir, "rel2id.json")) as fin:
            self.rel2id = json.load(fin)
        with open(os.path.join(self.raw_dir, "document2entities.json")) as fin:
            self.doc2entities = json.load(fin)

        num_nodes = self.kg.num_nodes
        doc2id = {doc: i for i, doc in enumerate(self.doc2entities)}
        # Convert document to entities to entity to document
        n_docs = len(self.doc2entities)
        # Create a sparse tensor for entity to document
        doc2ent = torch.zeros((n_docs, num_nodes))
        for doc, entities in self.doc2entities.items():
            entity_ids = [self.ent2id[ent] for ent in entities if ent in self.ent2id]
            doc2ent[doc2id[doc], entity_ids] = 1
        ent2doc = doc2ent.T.to_sparse()  # (n_nodes, n_docs)
        torch.save(ent2doc, os.path.join(self.processed_dir, "ent2doc.pt"))

        sample_id = []
        questions = []
        question_entities_masks = []  # Convert question entities to mask with number of nodes
        supporting_entities_masks = []
        supporting_docs_masks = []
        num_samples = []

        for path in self.raw_paths:
            if not os.path.exists(path):
                num_samples.append(0)
                continue  # Skip if the file does not exist
            num_sample = 0
            with open(path) as fin:
                data = json.load(fin)
                for index, item in enumerate(data):
                    question_entities = [
                        self.ent2id[x]
                        for x in item["question_entities"]
                        if x in self.ent2id
                    ]

                    supporting_entities = [
                        self.ent2id[x]
                        for x in item["supporting_entities"]
                        if x in self.ent2id
                    ]

                    supporting_docs = [
                        doc2id[doc] for doc in item["supporting_facts"] if doc in doc2id
                    ]

                    # Skip samples if any of the entities or documens are empty
                    if any(
                        len(x) == 0
                        for x in [
                            question_entities,
                            supporting_entities,
                            supporting_docs,
                        ]
                    ):
                        continue
                    num_sample += 1
                    sample_id.append(index)
                    question = item["question"]
                    questions.append(question)

                    question_entities_masks.append(
                        entities_to_mask(question_entities, num_nodes)
                    )

                    supporting_entities_masks.append(
                        entities_to_mask(supporting_entities, num_nodes)
                    )

                    supporting_docs_masks.append(
                        entities_to_mask(supporting_docs, n_docs)
                    )
                num_samples.append(num_sample)

        # Generate question embeddings
        logger.info("Generating question embeddings")
        text_emb_model: BaseTextEmbModel = instantiate(self.text_emb_model_cfgs)
        question_embeddings = text_emb_model.encode(
            questions,
            is_query=True,
        ).cpu()
        question_entities_masks = torch.stack(question_entities_masks)
        supporting_entities_masks = torch.stack(supporting_entities_masks)
        supporting_docs_masks = torch.stack(supporting_docs_masks)
        sample_id = torch.tensor(sample_id, dtype=torch.long)

        dataset = datasets.Dataset.from_dict(
            {
                "question_embeddings": question_embeddings,
                "question_entities_masks": question_entities_masks,
                "supporting_entities_masks": supporting_entities_masks,
                "supporting_docs_masks": supporting_docs_masks,
                "sample_id": sample_id,
            }
        ).with_format("torch")
        offset = 0
        splits = []
        for num_sample in num_samples:
            split = torch_data.Subset(dataset, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        torch.save(splits, self.processed_paths[0])

        # Save text embeddings model configuration
        with open(self.processed_dir + "/text_emb_model_cfgs.json", "w") as f:
            json.dump(OmegaConf.to_container(self.text_emb_model_cfgs), f, indent=4)
