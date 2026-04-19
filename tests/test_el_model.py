import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from gfmrag.graph_index_construction.utils import processing_phrases


class FakeLateInteractionTextEmbedding:
    passage_embed_calls = 0
    last_embedding_size_model_name: str | None = None

    def __init__(self, model_name: str, **_: object) -> None:
        self.model_name = model_name

    def get_embedding_size(self, model_name: str) -> int:
        type(self).last_embedding_size_model_name = model_name
        return 2

    def passage_embed(self, texts: list[str]) -> Iterator[list[list[float]]]:
        type(self).passage_embed_calls += 1
        for text in texts:
            yield self._embed(text)

    def query_embed(self, texts: list[str]) -> Iterator[list[list[float]]]:
        for text in texts:
            yield self._embed(text)

    def _embed(self, text: str) -> list[list[float]]:
        lookup = {
            "south chicago community hospital": [[1.0, 0.0], [1.0, 0.0]],
            "july 13 14  1966": [[0.0, 1.0], [0.0, 1.0]],
            "trial of richard speck": [[0.8, 0.2], [0.8, 0.2]],
            "richard speck": [[0.2, 0.8], [0.2, 0.8]],
        }
        return lookup.get(text, [[0.5, 0.5], [0.5, 0.5]])


@dataclass
class FakeScoredPoint:
    payload: dict[str, str]
    score: float


@dataclass
class FakeQueryResponse:
    points: list[FakeScoredPoint]


class FakeQdrantClient:
    def __init__(self, *_: object, **__: object) -> None:
        self._collection_exists = False
        self.points: list[Any] = []
        self.upload_points_calls = 0
        self.query_batch_points_calls = 0
        self.upload_batch_size: int | None = None

    def collection_exists(self, _: str) -> bool:
        return self._collection_exists

    def create_collection(self, **_: object) -> None:
        self._collection_exists = True

    def delete_collection(self, _: str) -> None:
        self._collection_exists = False
        self.points = []

    def upload_points(
        self,
        collection_name: str,
        points: Iterator[Any],
        batch_size: int = 64,
        **_: object,
    ) -> None:
        assert collection_name
        if isinstance(points, list):
            raise AssertionError("points should be streamed, not materialized")
        self._collection_exists = True
        self.upload_points_calls += 1
        self.upload_batch_size = batch_size
        self.points = list(points)

    def query_points(self, **_: object) -> None:
        raise AssertionError("query_points should not be used")

    def query_batch_points(
        self, collection_name: str, requests: list[Any], **_: object
    ) -> list[FakeQueryResponse]:
        assert collection_name
        self.query_batch_points_calls += 1
        responses: list[FakeQueryResponse] = []
        for request in requests:
            query = request.query
            scored_points = [
                FakeScoredPoint(
                    payload=point.payload,
                    score=self._score(query, point.vector),
                )
                for point in self.points
            ]
            scored_points.sort(key=lambda point: point.score, reverse=True)
            limit = request.limit if request.limit is not None else len(scored_points)
            responses.append(FakeQueryResponse(points=scored_points[:limit]))
        return responses

    def close(self) -> None:
        return None

    def _score(self, query: list[list[float]], vector: list[list[float]]) -> float:
        score = 0.0
        for query_token in query:
            score += max(
                sum(
                    query_dim * vector_dim
                    for query_dim, vector_dim in zip(
                        query_token, vector_token, strict=False
                    )
                )
                for vector_token in vector
            )
        return score


def processed_phrases_hash(entity_list: list[str]) -> str:
    phrases = [processing_phrases(entity) for entity in entity_list]
    return hashlib.md5(json.dumps(phrases, separators=(",", ":")).encode()).hexdigest()


def test_colbert_el_model_supports_in_memory_mode(
    tmp_path: Path,
) -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.graph_index_construction.entity_linking_model.ColbertELModel",
            "model_name_or_path": "colbert-ir/colbertv2.0",
            "root": str(tmp_path),
            "force": False,
            "use_in_memory": True,
        }
    )

    el_model = instantiate(cfg)
    assert el_model.use_in_memory is True
    assert not (tmp_path / "colbert").exists()

    entity_list = [
        "trial of richard speck",
        "south chicago community hospital",
        "july 13 14 1966",
    ]
    with pytest.raises(
        AttributeError, match="Index the entities first using index method"
    ):
        el_model(["july 13 14 1966"], topk=1)

    el_model.index(entity_list)
    linked_entity_dict = el_model(["south chicago community hospital"], topk=1)
    assert linked_entity_dict["south chicago community hospital"][0]["entity"] == (
        "south chicago community hospital"
    )
    assert (
        linked_entity_dict["south chicago community hospital"][0]["norm_score"] == 1.0
    )
    assert not (tmp_path / "colbert").exists()


def test_colbert_el_model_uses_batched_qdrant_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from gfmrag.graph_index_construction.entity_linking_model import colbert_el_model

    monkeypatch.setattr(
        colbert_el_model,
        "LateInteractionTextEmbedding",
        FakeLateInteractionTextEmbedding,
    )
    monkeypatch.setattr(colbert_el_model, "QdrantClient", FakeQdrantClient)

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.graph_index_construction.entity_linking_model.ColbertELModel",
            "model_name_or_path": "colbert-ir/colbertv2.0",
            "root": str(tmp_path),
            "force": False,
            "use_in_memory": True,
        }
    )

    el_model = instantiate(cfg)
    entity_list = [
        "trial of richard speck",
        "south chicago community hospital",
        "july 13 14  1966",
    ]
    el_model.index(entity_list)
    linked_entity_dict = el_model(
        ["south chicago community hospital", "july 13 14  1966"], topk=2
    )

    assert el_model.client.upload_points_calls == 1
    assert el_model.client.upload_batch_size == 64
    assert el_model.client.query_batch_points_calls == 1
    assert linked_entity_dict["south chicago community hospital"][0]["entity"] == (
        "south chicago community hospital"
    )
    assert linked_entity_dict["july 13 14  1966"][0]["entity"] == "july 13 14  1966"


def test_colbert_el_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from gfmrag.graph_index_construction.entity_linking_model import colbert_el_model

    FakeLateInteractionTextEmbedding.passage_embed_calls = 0
    FakeLateInteractionTextEmbedding.last_embedding_size_model_name = None
    monkeypatch.setattr(
        colbert_el_model,
        "LateInteractionTextEmbedding",
        FakeLateInteractionTextEmbedding,
    )

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.graph_index_construction.entity_linking_model.ColbertELModel",
            "model_name_or_path": "colbert-ir/colbertv2.0",
            "root": str(tmp_path),
            "force": False,
            "use_in_memory": False,
        }
    )

    el_model = instantiate(cfg)
    ner_entity_list = ["south chicago community hospital", "july 13 14  1966"]

    entity_list = [
        "controversy surrounding chief illiniwek",
        "supervisor in the state s attorney s office",
        "may 31  2016",
        "trial of john wayne gacy",
        "june 4  1931",
        "former cook county judge",
        "louis b  garippo",
        "trial of richard speck",
        "richard speck",
        "december 5  1991",
        "eight student nurses",
        "july 13 14  1966",
        "american mass murderer",
        "south chicago community hospital",
        "december 6  1941",
        "beaulieu mine",
        "northwest territories",
        "930 g",
        "yellowknife",
        "7 troy ounces",
        "chaos and bankruptcy",
        "november",
        "world war ii",
        "30 troy ounces",
        "october 1947",
        "1948",
        "schumacher",
        "porcupine gold rush",
        "downtown timmins",
        "mcintyre mine",
        "abandoned underground gold mine",
        "canada",
        "ontario",
        "canadian mining history",
        "the nation s most important mines",
        "headframe",
        "considerable amount of copper",
    ]
    fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
    metadata_path = tmp_path / "colbert" / fingerprint / "metadata.json"

    with pytest.raises(
        AttributeError, match="Index the entities first using index method"
    ):
        el_model(ner_entity_list, topk=2)

    el_model.index(entity_list)
    linked_entity_dict = el_model(ner_entity_list, topk=2)
    assert linked_entity_dict["south chicago community hospital"][0]["entity"] == (
        "south chicago community hospital"
    )
    assert linked_entity_dict["july 13 14  1966"][0]["entity"] == "july 13 14  1966"
    assert (
        linked_entity_dict["south chicago community hospital"][0]["norm_score"] == 1.0
    )
    assert metadata_path.exists()
    assert FakeLateInteractionTextEmbedding.passage_embed_calls == 1
    assert (
        FakeLateInteractionTextEmbedding.last_embedding_size_model_name
        == "colbert-ir/colbertv2.0"
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["backend"] == "fastembed_qdrant_multivector"
    assert metadata["schema_version"] == 1
    assert metadata["embedding_dimension"] == 2
    assert metadata["processed_phrases_hash"] == processed_phrases_hash(entity_list)

    el_model.client.close()
    cached_el_model = instantiate(cfg)
    cached_el_model.index(entity_list)
    cached_linked_entity_dict = cached_el_model(ner_entity_list, topk=2)
    assert (
        cached_linked_entity_dict["south chicago community hospital"][0]["entity"]
        == "south chicago community hospital"
    )
    assert (
        cached_linked_entity_dict["july 13 14  1966"][0]["entity"] == "july 13 14  1966"
    )
    assert (
        cached_linked_entity_dict["south chicago community hospital"][0]["norm_score"]
        == 1.0
    )
    assert cached_linked_entity_dict["july 13 14  1966"][0]["norm_score"] == 1.0
    assert FakeLateInteractionTextEmbedding.passage_embed_calls == 1
    cached_el_model.client.close()


def test_colbert_el_model_rebuilds_when_metadata_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from gfmrag.graph_index_construction.entity_linking_model import colbert_el_model

    FakeLateInteractionTextEmbedding.passage_embed_calls = 0
    monkeypatch.setattr(
        colbert_el_model,
        "LateInteractionTextEmbedding",
        FakeLateInteractionTextEmbedding,
    )

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.graph_index_construction.entity_linking_model.ColbertELModel",
            "model_name_or_path": "colbert-ir/colbertv2.0",
            "root": str(tmp_path),
            "force": False,
            "use_in_memory": False,
        }
    )

    entity_list = [
        "trial of richard speck",
        "south chicago community hospital",
        "july 13 14  1966",
    ]
    fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
    metadata_path = tmp_path / "colbert" / fingerprint / "metadata.json"

    el_model = instantiate(cfg)
    el_model.index(entity_list)
    el_model.client.close()
    assert FakeLateInteractionTextEmbedding.passage_embed_calls == 1

    metadata = json.loads(metadata_path.read_text())
    metadata["processed_phrases_hash"] = "stale"
    metadata_path.write_text(json.dumps(metadata))

    cached_el_model = instantiate(cfg)
    cached_el_model.index(entity_list)
    assert FakeLateInteractionTextEmbedding.passage_embed_calls == 2
    cached_el_model.client.close()


def test_dpr_el_model() -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.graph_index_construction.entity_linking_model.DPRELModel",
            "model_name": "BAAI/bge-large-en-v1.5",
            "root": "tmp",
            "use_cache": True,
            "normalize": True,
        }
    )

    el_model = instantiate(cfg)
    ner_entity_list = ["south chicago community hospital", "july 13 14  1966"]

    entity_list = [
        "controversy surrounding chief illiniwek",
        "supervisor in the state s attorney s office",
        "may 31  2016",
        "trial of john wayne gacy",
        "june 4  1931",
        "former cook county judge",
        "louis b  garippo",
        "trial of richard speck",
        "richard speck",
        "december 5  1991",
        "eight student nurses",
        "july 13 14  1966",
        "american mass murderer",
        "south chicago community hospital",
        "december 6  1941",
        "beaulieu mine",
        "northwest territories",
        "930 g",
        "yellowknife",
        "7 troy ounces",
        "chaos and bankruptcy",
        "november",
        "world war ii",
        "30 troy ounces",
        "october 1947",
        "1948",
        "schumacher",
        "porcupine gold rush",
        "downtown timmins",
        "mcintyre mine",
        "abandoned underground gold mine",
        "canada",
        "ontario",
        "canadian mining history",
        "the nation s most important mines",
        "headframe",
        "considerable amount of copper",
    ]
    el_model.index(entity_list)
    linked_entity_list = el_model(ner_entity_list, topk=2)
    print(linked_entity_list)
    assert isinstance(linked_entity_list, dict)


def test_nv_el_model() -> None:
    import pytest
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.graph_index_construction.entity_linking_model.NVEmbedV2ELModel",
            "model_name": "nvidia/NV-Embed-v2",
            "query_instruct": "Instruct: Given a entity, retrieve entities that are semantically equivalent to the given entity\nQuery: ",
            "passage_instruct": None,
            "root": "tmp",
            "use_cache": True,
            "normalize": True,
            "topk": 5,
            "batch_size": 256,
        }
    )

    if not Path("data/hotpotqa/raw/documents.json").exists():
        pytest.skip("data/hotpotqa/raw/documents.json is required for NVEmbed EL test")

    el_model = instantiate(cfg)
    ner_entity_list = ["what is one of the stars of  The Newcomers known for"]
    docs = [
        "The Newcomers\nThe Newcomers is a British television series.",
        "Milo O'Shea\nMilo O'Shea was an Irish actor and one of the stars of The Newcomers.",
        "The Newcomers cast\nThe cast includes actors known for film and television roles.",
    ]
    el_model.index(docs)
    linked_entity_list = el_model(ner_entity_list, topk=5)
    print(linked_entity_list)
    assert isinstance(linked_entity_list, dict)
