import hashlib
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models

from gfmrag.graph_index_construction.utils import processing_phrases

from .base_model import BaseELModel


class ColbertELModel(BaseELModel):
    def __init__(
        self,
        model_name_or_path: str = "colbert-ir/colbertv2.0",
        root: str = "tmp",
        doc_index_name: str = "nbits_2",
        phrase_index_name: str = "nbits_2",
        force: bool = False,
        use_in_memory: bool = False,
        **_: Any,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.root = root
        self.doc_index_name = doc_index_name
        self.phrase_index_name = phrase_index_name
        self.force = force
        self.use_in_memory = use_in_memory
        self.embedding_model = LateInteractionTextEmbedding(
            model_name=model_name_or_path,
            lazy_load=True,
        )
        self.client = QdrantClient(":memory:") if use_in_memory else None
        self.collection_name = phrase_index_name

    def _get_index_root(self, fingerprint: str) -> Path:
        return Path(self.root) / "colbert" / fingerprint

    def _get_metadata_path(self, fingerprint: str) -> Path:
        return self._get_index_root(fingerprint) / "metadata.json"

    def _get_client(self, fingerprint: str) -> QdrantClient:
        if self.use_in_memory:
            if self.client is None:
                self.client = QdrantClient(":memory:")
            return self.client
        index_root = self._get_index_root(fingerprint)
        index_root.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(index_root))

    def _embedding_size(self) -> int:
        return int(self.embedding_model.get_embedding_size())

    def _to_multivector(
        self, embedding: np.ndarray | list[list[float]]
    ) -> list[list[float]]:
        if isinstance(embedding, np.ndarray):
            return embedding.astype(float).tolist()
        return [[float(value) for value in vector] for vector in embedding]

    def _read_metadata(self, metadata_path: Path) -> dict[str, Any] | None:
        if not metadata_path.exists():
            return None
        with metadata_path.open() as metadata_file:
            return json.load(metadata_file)

    def _write_metadata(
        self, metadata_path: Path, fingerprint: str, entity_list: list[str]
    ) -> None:
        metadata = {
            "fingerprint": fingerprint,
            "model_name_or_path": self.model_name_or_path,
            "doc_index_name": self.doc_index_name,
            "phrase_index_name": self.phrase_index_name,
            "entity_list": entity_list,
        }
        with metadata_path.open("w") as metadata_file:
            json.dump(metadata, metadata_file)

    def _build_points(
        self, entity_list: list[str], phrases: list[str]
    ) -> Iterable[models.PointStruct]:
        for idx, (entity, embedding) in enumerate(
            zip(entity_list, self.embedding_model.passage_embed(phrases), strict=False)
        ):
            yield models.PointStruct(
                id=idx,
                vector=self._to_multivector(embedding),
                payload={"entity": entity},
            )

    def index(self, entity_list: list) -> None:
        self.entity_list = entity_list
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        self.fingerprint = fingerprint
        metadata_path = self._get_metadata_path(fingerprint)
        index_root = self._get_index_root(fingerprint)

        if not self.use_in_memory and self.force and index_root.exists():
            shutil.rmtree(index_root)

        client = self._get_client(fingerprint)
        self.client = client
        metadata = None if self.use_in_memory else self._read_metadata(metadata_path)
        should_reuse = (
            not self.force
            and client.collection_exists(self.collection_name)
            and metadata is not None
            and metadata.get("fingerprint") == fingerprint
            and metadata.get("entity_list") == entity_list
            and metadata.get("model_name_or_path") == self.model_name_or_path
            and metadata.get("doc_index_name") == self.doc_index_name
            and metadata.get("phrase_index_name") == self.phrase_index_name
        )

        if should_reuse:
            return

        if client.collection_exists(self.collection_name):
            client.delete_collection(self.collection_name)

        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self._embedding_size(),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )

        phrases = [processing_phrases(p) for p in entity_list]
        client.upsert(
            collection_name=self.collection_name,
            points=list(self._build_points(entity_list, phrases)),
        )

        if not self.use_in_memory:
            self._write_metadata(metadata_path, fingerprint, entity_list)

    def __call__(self, ner_entity_list: list, topk: int = 1) -> dict:
        if self.client is None or not hasattr(self, "entity_list"):
            raise AttributeError("Index the entities first using index method")
        if not self.client.collection_exists(self.collection_name):
            raise AttributeError("Index the entities first using index method")

        queries = [processing_phrases(p) for p in ner_entity_list]
        linked_entity_dict: dict[str, list] = {}

        for query, embedding in zip(
            queries, self.embedding_model.query_embed(queries), strict=False
        ):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=self._to_multivector(embedding),
                limit=topk,
                with_payload=True,
            )
            points = response.points
            max_score = points[0].score if points else 1.0
            linked_entity_dict[query] = [
                {
                    "entity": point.payload["entity"],
                    "score": point.score,
                    "norm_score": point.score / max_score if max_score else 0.0,
                }
                for point in points
            ]

        return linked_entity_dict
