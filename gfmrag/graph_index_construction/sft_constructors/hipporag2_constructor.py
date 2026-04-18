import json
import logging
import os
from ast import literal_eval
from collections import defaultdict
from multiprocessing.dummy import Pool
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch

from gfmrag.graph_index_datasets.graph_index_dataset import GraphIndexDataset
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils.util import check_all_files_exist

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


class HippoRAG2Constructor(BaseSFTConstructor):
    """Construct SFT samples for HippoRAG 2 with embedding retrieval and fact reranking.

    The constructor embeds graph nodes, graph facts, questions, and answers with a
    shared text embedding model. It builds FAISS indices over the selected node
    types and optionally over facts, retrieves fact candidates for each question,
    reranks them with `DSPyFilter`, and then derives:

    - `start_nodes` from fact-linked entities and optional dense document retrieval
    - `target_nodes` from answer-to-entity retrieval and supporting documents

    The graph files under `processed/stage1` must already exist before this
    constructor runs.
    """

    def __init__(
        self,
        text_emb_model: BaseTextEmbModel,
        root: str = "tmp/qa_construction",
        enable_filtering: bool = True,
        num_processes: int = 1,
        topk: int = 5,
        force: bool = False,
        llm_for_filtering: str = "gpt-4o-mini",
        retry: int = 5,
        start_type: list | None = None,
        target_type: list | None = None,
    ) -> None:
        """Initialize the HippoRAG 2 SFT constructor.

        Args:
            text_emb_model: Embedding model used for nodes, facts, questions, and answers.
            root: Directory for temporary constructor outputs.
            enable_filtering: Whether to enable filtering of facts using LLM.
            num_processes: Worker count for fact reranking.
            topk: Number of start or target nodes to keep per sample.
            force: Reserved flag for compatibility with other constructors.
            llm_for_filtering: Model name used by the fact reranker.
            retry: Retry count for reranker calls.
            start_type: Node types to include in `start_nodes`.
            target_type: Node types to include in `target_nodes`.
        """
        self.text_emb_model = text_emb_model
        self.root = root
        self.num_processes = num_processes
        self.data_name = None
        self.topk = topk
        self.force = force
        self.llm_for_filtering = llm_for_filtering
        self.retry = retry
        self.enable_filtering = enable_filtering
        self.start_type = start_type
        self.target_type = target_type
        self.rerank_filter = (
            DSPyFilter(llm_for_filtering, retry) if enable_filtering else None
        )

        self.node_names: list[str] = []
        self.nodes_by_type: dict[str, list[str]] = {}
        self.node_texts_by_type: dict[str, list[str]] = {}
        self.node_embeddings_by_type: dict[str, np.ndarray] = {}
        self.node_indices_by_type: dict[str, faiss.IndexFlatIP] = {}

        self.document_nodes: list[str] = []

        self.facts: list[tuple[str, str, str]] = []
        self.fact_texts: list[str] = []
        self.fact_index: faiss.IndexFlatIP | None = None
        self.selected_start_types: list[str] = []
        self.selected_target_types: list[str] = []
        self.enable_fact_retrieval: bool = False

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

    def _encode_texts(self, text: list[str], is_query: bool = False) -> np.ndarray:
        if len(text) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        embeddings = self.text_emb_model.encode(
            text,
            is_query=is_query,
            show_progress_bar=False,
        )
        if isinstance(embeddings, torch.Tensor):
            emb = embeddings.detach().cpu().numpy().astype(np.float32)
        else:
            emb = np.asarray(embeddings, dtype=np.float32)

        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        # Normalize to make inner product equivalent to cosine similarity.
        faiss.normalize_L2(emb)
        return emb

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP | None:
        if embeddings.size == 0:
            return None
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)  # type: ignore[call-arg]
        return index

    def _safe_parse_attributes(self, attrs: str) -> dict:
        if not attrs:
            return {}
        try:
            parsed = literal_eval(attrs)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _node_text(self, row: pd.Series) -> str:
        name = str(row["name"])
        attrs = self._safe_parse_attributes(str(row.get("attributes", "")))
        content = attrs.get("content", "")
        content = str(content).strip() if content is not None else ""
        if content:
            return f"{name}\n{content}"
        return name

    def _resolve_selected_types(
        self, requested_types: list | None, available_types: list[str], node_group: str
    ) -> list[str]:
        """Resolve selected node types with fallback to all available types."""
        if requested_types is None or len(requested_types) == 0:
            return available_types

        requested = {str(node_type) for node_type in requested_types}
        selected = [
            node_type for node_type in available_types if node_type in requested
        ]

        if len(selected) == 0:
            logger.warning(
                f"No valid {node_group} types matched from {requested_types}. "
                f"Fallback to all available types: {available_types}."
            )
            return available_types

        return selected

    def index(self) -> None:
        self.node_embeddings_by_type = {}
        self.node_indices_by_type = {}
        for node_type, node_texts in self.node_texts_by_type.items():
            node_embeddings = self._encode_texts(node_texts, is_query=False)
            self.node_embeddings_by_type[node_type] = node_embeddings
            index = self._build_faiss_index(node_embeddings)
            if index is not None:
                self.node_indices_by_type[node_type] = index

        if self.enable_fact_retrieval:
            fact_embeddings = self._encode_texts(self.fact_texts, is_query=False)
            self.fact_index = self._build_faiss_index(fact_embeddings)
        else:
            self.fact_index = None

    def _search_by_type(
        self, node_type: str, query_embedding: np.ndarray, top_k: int
    ) -> tuple[list[str], np.ndarray]:
        index = self.node_indices_by_type.get(node_type)
        if index is None or top_k <= 0:
            return [], np.array([], dtype=np.float32)

        k = min(top_k, index.ntotal)
        if k <= 0:
            return [], np.array([], dtype=np.float32)

        scores, local_ids = index.search(query_embedding, k)  # type: ignore[call-arg]
        scores_1d = np.squeeze(scores).astype(np.float32)
        local_ids_1d = np.squeeze(local_ids)

        if scores_1d.ndim == 0:
            scores_1d = np.array([float(scores_1d)], dtype=np.float32)
            local_ids_1d = np.array([int(local_ids_1d)], dtype=np.int64)

        valid_pairs = [
            (int(local_id), float(score))
            for local_id, score in zip(local_ids_1d.tolist(), scores_1d.tolist())
            if local_id >= 0
        ]
        if not valid_pairs:
            return [], np.array([], dtype=np.float32)

        labels = self.nodes_by_type[node_type]
        retrieved_labels = [labels[local_id] for local_id, _ in valid_pairs]
        retrieved_scores = np.array(
            [score for _, score in valid_pairs], dtype=np.float32
        )
        retrieved_scores = min_max_normalize(retrieved_scores)
        return retrieved_labels, retrieved_scores

    def retrieve_fact_candidates(
        self, query_embedding: np.ndarray, top_k: int
    ) -> tuple[list[int], list[tuple[str, str, str]], dict[int, float]]:
        if self.fact_index is None or len(self.facts) == 0:
            logger.warning("No facts available for retrieval. Returning empty lists.")
            return [], [], {}

        k = min(top_k, self.fact_index.ntotal)
        if k <= 0:
            return [], [], {}

        scores, ids = self.fact_index.search(query_embedding, k)  # type: ignore[call-arg]
        scores_1d = np.squeeze(scores).astype(np.float32)
        ids_1d = np.squeeze(ids)

        if scores_1d.ndim == 0:
            scores_1d = np.array([float(scores_1d)], dtype=np.float32)
            ids_1d = np.array([int(ids_1d)], dtype=np.int64)

        valid_pairs = [
            (int(fact_id), float(score))
            for fact_id, score in zip(ids_1d.tolist(), scores_1d.tolist())
            if fact_id >= 0
        ]
        if not valid_pairs:
            return [], [], {}

        candidate_indices = [fact_id for fact_id, _ in valid_pairs]
        candidate_facts = [self.facts[idx] for idx in candidate_indices]
        normalized_scores = min_max_normalize(
            np.array([score for _, score in valid_pairs], dtype=np.float32)
        )
        score_map = {
            fact_id: float(score)
            for fact_id, score in zip(candidate_indices, normalized_scores.tolist())
        }
        return candidate_indices, candidate_facts, score_map

    def rerank_facts(
        self,
        query: str,
        candidate_fact_indices: list[int],
        candidate_facts: list[tuple[str, str, str]],
    ) -> tuple[list[int], list[tuple], dict]:
        link_top_k: int = self.topk

        if len(candidate_fact_indices) == 0 or len(candidate_facts) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {"facts_before_rerank": [], "facts_after_rerank": []}

        if self.rerank_filter is None:
            top_k_fact_indices = candidate_fact_indices[:link_top_k]
            top_k_facts = candidate_facts[:link_top_k]
            rerank_log = {
                "facts_before_rerank": candidate_facts,
                "facts_after_rerank": top_k_facts,
                "skipped_llm_filtering": True,
            }
            return top_k_fact_indices, top_k_facts, rerank_log

        try:
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

    def dense_passage_retrieval(
        self, query_embedding: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        """Retrieve top documents for an encoded query from the document index."""
        docs, scores = self._search_by_type("document", query_embedding, self.topk)
        return docs, scores

    def dense_entity_retrieval(
        self, query_embedding: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        entities, scores = self._search_by_type("entity", query_embedding, self.topk)
        return entities, scores

    def retrieve_answer_entity(self, answer_embedding: np.ndarray | None) -> str:
        if answer_embedding is None:
            return ""
        entities, _ = self._search_by_type("entity", answer_embedding, 1)
        if len(entities) == 0:
            return ""
        return entities[0]

    def prepare_data(self, data_root: str, data_name: str, file: str) -> list[dict]:
        """Build HippoRAG 2 training samples from a raw QA file.

        Args:
            data_root: Root directory containing the dataset.
            data_name: Dataset name.
            file: Raw QA JSON file name.

        Returns:
            A list of samples augmented with `start_type`, `target_type`,
            `start_nodes`, and `target_nodes`.

        Raises:
            FileNotFoundError: If the processed graph files are missing.
        """
        # Get dataset information
        self.data_name = data_name  # type: ignore
        raw_path = os.path.join(data_root, data_name, "raw", file)
        processed_path = os.path.join(data_root, data_name, "processed", "stage1")

        # Load data
        with open(raw_path) as f:
            data = json.load(f)

        # Read nodes.csv to get entities
        nodes = pd.read_csv(
            os.path.join(processed_path, "nodes.csv"), keep_default_na=False
        )
        nodes["name"] = nodes["name"].astype(str)
        nodes["type"] = nodes["type"].astype(str)
        self.node_names = nodes["name"].tolist()

        available_node_types = list(dict.fromkeys(nodes["type"].astype(str).tolist()))
        self.selected_start_types = self._resolve_selected_types(
            self.start_type,
            available_node_types,
            node_group="start",
        )
        self.selected_target_types = self._resolve_selected_types(
            self.target_type,
            ["entity", "document"],
            node_group="target",
        )

        emb_node_types = set(self.selected_start_types)
        if "entity" in self.selected_target_types:
            emb_node_types.add("entity")

        self.nodes_by_type = {}
        self.node_texts_by_type = {}
        for node_type, group_df in nodes.groupby("type", sort=False):
            node_type = str(node_type)
            if node_type not in emb_node_types:
                continue
            node_names = group_df["name"].astype(str).tolist()
            node_texts = [self._node_text(row) for _, row in group_df.iterrows()]
            self.nodes_by_type[node_type] = node_names
            self.node_texts_by_type[node_type] = node_texts

        self.document_nodes = self.nodes_by_type.get("document", [])
        self.enable_fact_retrieval = "entity" in self.selected_start_types

        # Read edges.csv to get triples
        edges = pd.read_csv(
            os.path.join(processed_path, "edges.csv"), keep_default_na=False
        )
        self.facts = [
            (str(source).lower(), str(relation), str(target).lower())
            for source, relation, target in edges[
                edges["relation"] != "is_mentioned_in"
            ][["source", "relation", "target"]].values.tolist()
        ]
        self.fact_texts = [
            f"{source} [SEP] {relation} [SEP] {target}"
            for source, relation, target in self.facts
        ]

        self.ent_node_to_chunk_ids = defaultdict(set)
        mention_edges = edges[edges["relation"] == "is_mentioned_in"]
        for _, row in mention_edges.iterrows():
            source = str(row["source"]).lower()
            target = row["target"]
            self.ent_node_to_chunk_ids[source].add(target)

        # generate embeddings
        self.index()

        queries = [sample["question"] for sample in data]
        query_embeddings = self._encode_texts(queries, is_query=True)
        answer_embeddings: np.ndarray | None = None
        if "entity" in self.selected_target_types:
            answers = [str(sample.get("answer", "")) for sample in data]
            answer_embeddings = self._encode_texts(answers, is_query=False)

        # Create graph index for each dataset
        raw_graph_files = [
            os.path.join(processed_path, name)
            for name in GraphIndexDataset.RAW_GRAPH_NAMES
        ]
        if not check_all_files_exist(raw_graph_files):
            raise FileNotFoundError(
                "Graph file not found. Please run KG construction first"
            )

        # Precompute query-fact scores sequentially to avoid concurrent embedding inference.
        prepared_samples: list[dict] = []
        for idx, sample in enumerate(data):
            query = sample["question"]
            query_embedding = query_embeddings[idx : idx + 1]
            answer_embedding = (
                answer_embeddings[idx : idx + 1]
                if answer_embeddings is not None
                else None
            )
            if self.enable_fact_retrieval:
                candidate_fact_indices, candidate_facts, fact_score_map = (
                    self.retrieve_fact_candidates(query_embedding, self.topk * 4)
                )
            else:
                candidate_fact_indices, candidate_facts, fact_score_map = ([], [], {})
            prepared_samples.append(
                {
                    "idx": idx,
                    "sample": sample,
                    "query": query,
                    "answer": sample["answer"],
                    "query_embedding": query_embedding,
                    "answer_embedding": answer_embedding,
                    "candidate_fact_indices": candidate_fact_indices,
                    "candidate_facts": candidate_facts,
                    "fact_score_map": fact_score_map,
                }
            )

        # Run optional reranking in parallel, then consume results sequentially.
        rerank_results: dict[int, tuple[list[int], list[tuple[str, str, str]]]] = {}
        max_workers = max(1, self.num_processes)

        def _rerank_item(
            item: dict,
        ) -> tuple[int, list[int], list[tuple[str, str, str]]]:
            idx = item["idx"]
            try:
                if self.enable_filtering:
                    top_k_fact_indices, top_k_facts, _ = self.rerank_facts(
                        item["query"],
                        item["candidate_fact_indices"],
                        item["candidate_facts"],
                    )
                else:
                    top_k_fact_indices = item["candidate_fact_indices"][: self.topk]
                    top_k_facts = item["candidate_facts"][: self.topk]
                return idx, top_k_fact_indices, top_k_facts
            except Exception as e:
                logger.error(f"Parallel rerank failed for sample index {idx}: {str(e)}")
                return idx, [], []

        if max_workers == 1:
            for item in prepared_samples:
                _, top_k_fact_indices, top_k_facts = _rerank_item(item)
                rerank_results[item["idx"]] = (top_k_fact_indices, top_k_facts)
        else:
            with Pool(processes=max_workers) as pool:
                for idx, top_k_fact_indices, top_k_facts in pool.map(
                    _rerank_item, prepared_samples
                ):
                    rerank_results[idx] = (top_k_fact_indices, top_k_facts)

        # # Prepare final data
        final_data = []
        for item in prepared_samples:
            sample = item["sample"]
            query_embedding = item["query_embedding"]
            answer_embedding = item["answer_embedding"]
            fact_score_map = item["fact_score_map"]
            top_k_fact_indices, top_k_facts = rerank_results.get(item["idx"], ([], []))
            start_entity_nodes: list[str] = []
            starting_documents: list[str] = []
            use_start_entity = "entity" in self.selected_start_types
            use_start_document = "document" in self.selected_start_types

            if len(top_k_facts) == 0:
                if use_start_document:
                    logger.info("No facts found after reranking, return DPR results")
                    top_k_docs, _ = self.dense_passage_retrieval(query_embedding)
                    starting_documents = top_k_docs[: self.topk]

                if use_start_entity:
                    top_k_entities, _ = self.dense_entity_retrieval(query_embedding)
                    start_entity_nodes = top_k_entities[: self.topk]

            else:
                linking_score_map = self.graph_search_with_fact_entities(
                    link_top_k=self.topk,
                    fact_score_map=fact_score_map,
                    top_k_facts=top_k_facts,
                    top_k_fact_indices=top_k_fact_indices,
                    query_embedding=query_embedding,
                    include_documents=use_start_document,
                    passage_node_weight=0.05,
                )

                document_node_set = set(self.document_nodes)
                start_nodes = list(linking_score_map.keys())
                for k in start_nodes:
                    if k in document_node_set and use_start_document:
                        starting_documents.append(k)
                    elif use_start_entity:
                        start_entity_nodes.append(k)

            answer_entity = self.retrieve_answer_entity(answer_embedding)
            supporting_documents = sample.get("supporting_documents", [])

            start_nodes_out: dict[str, Any] = {}
            if use_start_entity:
                start_nodes_out["entity"] = start_entity_nodes[: self.topk]
            if use_start_document:
                start_nodes_out["document"] = starting_documents[: self.topk]

            target_nodes_out: dict[str, Any] = {}
            if "entity" in self.selected_target_types:
                target_nodes_out["entity"] = answer_entity
            if "document" in self.selected_target_types:
                target_nodes_out["document"] = supporting_documents

            final_data.append(
                {
                    **sample,
                    "start_type": self.selected_start_types,
                    "target_type": self.selected_target_types,
                    "start_nodes": start_nodes_out,
                    "target_nodes": target_nodes_out,
                }
            )

        return final_data

    def graph_search_with_fact_entities(
        self,
        link_top_k: int,
        fact_score_map: dict[int, float],
        top_k_facts: list[tuple],
        top_k_fact_indices: list[int],
        query_embedding: np.ndarray,
        include_documents: bool,
        passage_node_weight: float = 0.05,
    ) -> dict:
        """Aggregate fact-linked entity scores and optional dense document scores."""

        # Assigning phrase weights based on selected facts from previous steps.
        linking_score_map: dict[
            str, float
        ] = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores: dict[
            str, list[float]
        ] = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        node_to_id = {name: idx for idx, name in enumerate(self.node_names)}
        phrase_weights = np.zeros(len(self.node_names))
        number_of_occurs = np.zeros(len(self.node_names))

        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            object_phrase = f[2].lower()
            fact_score = fact_score_map.get(top_k_fact_indices[rank], 0.0)

            for phrase in [subject_phrase, object_phrase]:
                phrase_id = node_to_id.get(phrase)

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase])

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                phrases_and_ids.add((phrase, phrase_id))

        valid_occurs = number_of_occurs > 0
        phrase_weights[valid_occurs] /= number_of_occurs[valid_occurs]

        for phrase, phrase_id in phrases_and_ids:
            if phrase_id is None:
                continue
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(float(phrase_weights[phrase_id]))

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
        if include_documents:
            dpr_sorted_docs, dpr_sorted_doc_scores = self.dense_passage_retrieval(
                query_embedding
            )

            for i, doc_name in enumerate(dpr_sorted_docs):
                passage_dpr_score = dpr_sorted_doc_scores[i]
                linking_score_map[doc_name] = passage_dpr_score * passage_node_weight

        # Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(
                sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30]
            )

        return linking_score_map
