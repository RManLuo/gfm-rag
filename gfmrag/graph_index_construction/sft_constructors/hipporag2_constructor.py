import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from gfmrag.graph_index_datasets.graph_index_dataset import GraphIndexDataset
from gfmrag.utils.util import check_all_files_exist

from ..entity_linking_model import BaseELModel
from ..ner_model import BaseNERModel
from .base_sft_constructor import BaseSFTConstructor
from .hipporag2.rerank import DSPyFilter

logger = logging.getLogger(__name__)


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val

    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x

    return (x - min_val) / range_val


def get_query_instruction(linking_method: str) -> str:
    instructions = {
        "ner_to_node": "Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.",
        "query_to_node": "Given a question, retrieve relevant phrases that are mentioned in this question.",
        "query_to_fact": "Given a question, retrieve relevant triplet facts that matches this question.",
        "query_to_sentence": "Given a question, retrieve relevant sentences that best answer the question.",
        "query_to_passage": "Given a question, retrieve relevant documents that best answer the question.",
    }
    default_instruction = (
        "Given a question, retrieve relevant documents that best answer the question."
    )
    return instructions.get(linking_method, default_instruction)


class HippoRAG2Constructor(BaseSFTConstructor):
    """SFT Constructor for building question-answer datasets with entity linking and named entity recognition used for HippoRAG 2.

    This class processes raw QA datasets by performing Named Entity Recognition (NER) on questions and Entity Linking (EL) to connect identified entities to a knowledge graph (KG) to create start_nodes.

    It uses the supporting documents and answers to create target_nodes.

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
        topk: int = 5,
        force: bool = False,
        start_type: list | None = None,
        target_type: list | None = None,
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
        self.topk = topk
        self.start_type = start_type
        self.target_type = target_type
        self.rerank_filter = DSPyFilter(self)

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

    def index(self) -> None:
        self.passage_embeddings = self.el_model.batch_index(self.docs)
        self.entities_embeddings = self.el_model.batch_index(self.entities)
        self.facts_id = [str(fact) for fact in self.facts]
        self.fact_embeddings = self.el_model.batch_index(self.facts_id)

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.el_model.index(
            [query],
            instruction=get_query_instruction("query_to_fact"),  # type: ignore
        )

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])

        try:
            query_embedding = query_embedding.to(self.fact_embeddings.device)  # type: ignore
            query_fact_scores = (
                self.fact_embeddings @ query_embedding.T
            )  # shape: (#facts, )
            query_fact_scores = query_fact_scores.cpu().numpy()
            query_fact_scores = (
                np.squeeze(query_fact_scores)
                if query_fact_scores.ndim == 2
                else query_fact_scores
            )
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            print(e)
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def rerank_facts(
        self, query: str, query_fact_scores: np.ndarray
    ) -> tuple[list[int], list[tuple], dict]:
        link_top_k: int = self.topk

        if len(query_fact_scores) == 0 or len(self.facts_id) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {"facts_before_rerank": [], "facts_after_rerank": []}

        try:
            if len(query_fact_scores) <= link_top_k:
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][
                    ::-1
                ].tolist()

            candidate_facts = [self.facts[idx] for idx in candidate_fact_indices]

            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(
                query,
                candidate_facts,
                candidate_fact_indices,
                len_after_rerank=link_top_k,
            )

            rerank_log = {
                "facts_before_rerank": candidate_facts,
                "facts_after_rerank": top_k_facts,
            }

            return top_k_fact_indices, top_k_facts, rerank_log

        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return (
                [],
                [],
                {"facts_before_rerank": [], "facts_after_rerank": [], "error": str(e)},
            )

    def dense_passage_retrieval(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.el_model.index(
            [query],
            instruction=get_query_instruction("query_to_passage"),  # type: ignore
        )
        query_embedding = query_embedding.to(self.passage_embeddings.device)  # type: ignore
        query_doc_scores = self.passage_embeddings @ query_embedding.T
        query_doc_scores = query_doc_scores.cpu().numpy()
        query_doc_scores = (
            np.squeeze(query_doc_scores)
            if query_doc_scores.ndim == 2
            else query_doc_scores
        )
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores

    def el_answer(self, ans: str) -> str:
        ans_embedding = self.el_model.index(
            [ans],
            instruction=get_query_instruction("query_to_node"),  # type: ignore
        )
        ans_embedding = ans_embedding.to(self.entities_embeddings.device)  # type: ignore
        ans_ent_scores = self.entities_embeddings @ ans_embedding.T
        ans_ent_scores = ans_ent_scores.cpu().numpy()
        ans_ent_scores = (
            np.squeeze(ans_ent_scores) if ans_ent_scores.ndim == 2 else ans_ent_scores
        )
        ans_ent_scores = min_max_normalize(ans_ent_scores)

        sorted_ent_ids = np.argsort(ans_ent_scores)[::-1]
        return self.entities[sorted_ent_ids[0]]

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
        corpus_path = os.path.join(
            data_root, data_name, "raw", GraphIndexDataset.RAW_DOCUMENT_NAME
        )
        processed_path = os.path.join(data_root, data_name, "processed", "stage1")

        # Load data
        with open(raw_path) as f:
            data = json.load(f)

        # corpus embeddings
        corpus = json.load(open(corpus_path))
        self.docs = [f"{title}\n{text}" for title, text in corpus.items()]

        # Read nodes.csv to get entities
        nodes = pd.read_csv(
            os.path.join(processed_path, "nodes.csv"), keep_default_na=False
        )
        # Get nodes with type 'entity'
        self.entities = nodes[nodes["type"] == "entity"]["name"].tolist()
        self.nodes = nodes["name"].tolist()

        # Read edges.csv to get triples
        edges = pd.read_csv(
            os.path.join(processed_path, "edges.csv"), keep_default_na=False
        )
        self.facts = edges[edges["relation"] != "is_mentioned_in"][
            ["source", "relation", "target"]
        ].values.tolist()

        self.ent_node_to_chunk_ids = defaultdict(set)
        mention_edges = edges[edges["relation"] == "is_mentioned_in"]
        for _, row in mention_edges.iterrows():
            source = row["source"]
            target = row["target"]
            self.ent_node_to_chunk_ids[source].add(target)

        # generate embeddings
        self.index()

        # Create graph index for each dataset
        raw_graph_files = [
            os.path.join(processed_path, name)
            for name in GraphIndexDataset.RAW_GRAPH_NAMES
        ]
        if not check_all_files_exist(raw_graph_files):
            raise FileNotFoundError(
                "Graph file not found. Please run KG construction first"
            )

        # # Prepare final data
        final_data = []
        for sample in data:
            query = sample["question"]
            answer = sample["answer"]

            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts, _ = self.rerank_facts(
                query, query_fact_scores
            )

            if len(top_k_facts) == 0:
                logger.info("No facts found after reranking, return DPR results")
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                top_k_docs = [self.docs[idx] for idx in sorted_doc_ids[: self.topk]]

                question_entities = [
                    doc.split("\n")[0] for doc in top_k_docs[: self.topk]
                ]
                starting_documents = question_entities

            else:
                linking_score_map = self.graph_search_with_fact_entities(
                    query=query,
                    link_top_k=self.topk,
                    query_fact_scores=query_fact_scores,
                    top_k_facts=top_k_facts,
                    top_k_fact_indices=top_k_fact_indices,
                    passage_node_weight=0.05,
                )

                question_entities = []
                starting_documents = []
                start_nodes = list(linking_score_map.keys())
                for k in start_nodes:
                    if "\n" in k:
                        doc = k.split("\n")[0]
                        starting_documents.append(doc)
                    else:
                        question_entities.append(k)

            answer_entities = self.el_answer(answer)
            supporting_documents = sample.get("supporting_documents", [])

            final_data.append(
                {
                    **sample,
                    "start_nodes": {
                        "entity": question_entities[: self.topk],
                        "document": starting_documents[: self.topk],
                    },
                    "target_nodes": {
                        "entity": answer_entities,
                        "document": supporting_documents,
                    },
                }
            )

        return final_data

    def graph_search_with_fact_entities(
        self,
        query: str,
        link_top_k: int,
        query_fact_scores: np.ndarray,
        top_k_facts: list[tuple],
        top_k_fact_indices: list[int],
        passage_node_weight: float = 0.05,
    ) -> dict:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """

        # Assigning phrase weights based on selected facts from previous steps.
        linking_score_map: dict[
            str, float
        ] = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores: dict[
            str, list[float]
        ] = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.nodes))
        np.zeros(len(self.nodes))
        number_of_occurs = np.zeros(len(self.nodes))

        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            object_phrase = f[2].lower()
            fact_score = (
                query_fact_scores[top_k_fact_indices[rank]]
                if query_fact_scores.ndim > 0
                else query_fact_scores
            )

            for phrase in [subject_phrase, object_phrase]:
                phrase_id = self.nodes.index(phrase)

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase])

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                phrases_and_ids.add((phrase, phrase_id))

        phrase_weights /= number_of_occurs

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(phrase_weights[phrase_id])

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            linking_score_map = dict(
                sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[
                    :link_top_k
                ]
            )

        # Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_text = self.docs[dpr_sorted_doc_id]
            linking_score_map[passage_node_text] = (
                passage_dpr_score * passage_node_weight
            )

        # Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(
                sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30]
            )

        return linking_score_map
