import hashlib
import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fakes for PyLate ColBERT / PLAID / Retriever
# ---------------------------------------------------------------------------

_EMBED_LOOKUP: dict[str, np.ndarray] = {
    "south chicago community hospital": np.array([[1.0, 0.0], [1.0, 0.0]]),
    "july 13 14  1966": np.array([[0.0, 1.0], [0.0, 1.0]]),
    "trial of richard speck": np.array([[0.8, 0.2], [0.8, 0.2]]),
    "richard speck": np.array([[0.2, 0.8], [0.2, 0.8]]),
}


def _fake_embed(text: str) -> np.ndarray:
    return _EMBED_LOOKUP.get(text, np.array([[0.5, 0.5], [0.5, 0.5]]))


def _maxsim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    score = 0.0
    for q_token in query_emb:
        score += float(max(np.dot(q_token, d_token) for d_token in doc_emb))
    return score


# Class-level store so FakePLAID instances share state across override=False loads
_plaid_store: dict[str, list[tuple[str, np.ndarray]]] = {}


class FakeColBERTModel:
    encode_calls: int = 0

    def __init__(self, model_name_or_path: str, **_: Any) -> None:
        self.model_name_or_path = model_name_or_path

    def encode(
        self,
        sentences: list[str],
        is_query: bool = True,
        **_: Any,
    ) -> list[np.ndarray]:
        if not is_query:
            type(self).encode_calls += 1
        return [_fake_embed(s) for s in sentences]


class FakePLAIDIndex:
    def __init__(
        self,
        index_folder: str,
        index_name: str,
        override: bool = False,
        **_: Any,
    ) -> None:
        self.key = f"{index_folder}/{index_name}"
        if override:
            _plaid_store.pop(self.key, None)

    def add_documents(
        self,
        documents_ids: list[str],
        documents_embeddings: list[np.ndarray],
    ) -> None:
        _plaid_store[self.key] = list(zip(documents_ids, documents_embeddings))


class FakeRetrievalHit(TypedDict):
    id: str
    score: float


class FakeColBERTRetriever:
    def __init__(self, index: FakePLAIDIndex) -> None:
        self.index = index

    def retrieve(
        self,
        queries_embeddings: list[np.ndarray],
        k: int = 10,
        **_: Any,
    ) -> list[list[FakeRetrievalHit]]:
        docs = _plaid_store.get(self.index.key, [])
        results: list[list[FakeRetrievalHit]] = []
        for q_emb in queries_embeddings:
            scores: list[FakeRetrievalHit] = [
                {"id": doc_id, "score": _maxsim(q_emb, doc_emb)}
                for doc_id, doc_emb in docs
            ]
            scores.sort(key=lambda x: x["score"], reverse=True)
            results.append(scores[:k])
        return results


# ---------------------------------------------------------------------------
# ColBERT EL model tests (PyLate / PLAID backend)
# ---------------------------------------------------------------------------

ENTITY_LIST = [
    "trial of richard speck",
    "south chicago community hospital",
    "july 13 14  1966",
]


def _fingerprint(entity_list: list[str]) -> str:
    return hashlib.md5("".join(entity_list).encode()).hexdigest()


def _patch_pylate(monkeypatch: pytest.MonkeyPatch) -> None:
    from gfmrag.graph_index_construction.entity_linking_model import colbert_el_model

    monkeypatch.setattr(colbert_el_model, "ColBERTModel", FakeColBERTModel)
    monkeypatch.setattr(colbert_el_model, "PLAID", FakePLAIDIndex)
    monkeypatch.setattr(colbert_el_model, "ColBERTRetriever", FakeColBERTRetriever)
    FakeColBERTModel.encode_calls = 0
    _plaid_store.clear()


def test_colbert_el_model_raises_before_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    _patch_pylate(monkeypatch)

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.graph_index_construction.entity_linking_model.ColbertELModel",
            "model_name_or_path": "lightonai/GTE-ModernColBERT-v1",
            "root": str(tmp_path),
            "force": False,
        }
    )
    el_model = instantiate(cfg)
    with pytest.raises(AttributeError, match="Index the entities first"):
        el_model(["july 13 14  1966"], topk=1)


def test_colbert_el_model_indexes_and_retrieves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    _patch_pylate(monkeypatch)

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.graph_index_construction.entity_linking_model.ColbertELModel",
            "model_name_or_path": "lightonai/GTE-ModernColBERT-v1",
            "root": str(tmp_path),
            "force": False,
        }
    )
    el_model = instantiate(cfg)
    el_model.index(ENTITY_LIST)

    result = el_model(["south chicago community hospital"], topk=1)
    assert (
        result["south chicago community hospital"][0]["entity"]
        == "south chicago community hospital"
    )
    assert result["south chicago community hospital"][0]["norm_score"] == 1.0


def test_colbert_el_model_writes_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    _patch_pylate(monkeypatch)

    model_name = "lightonai/GTE-ModernColBERT-v1"
    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.graph_index_construction.entity_linking_model.ColbertELModel",
            "model_name_or_path": model_name,
            "root": str(tmp_path),
            "force": False,
        }
    )
    el_model = instantiate(cfg)
    el_model.index(ENTITY_LIST)

    fp = _fingerprint(ENTITY_LIST)
    model_slug = model_name.replace("/", "_")
    metadata_path = tmp_path / "pylate" / model_slug / fp / "metadata.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["fingerprint"] == fp
    assert metadata["model_name_or_path"] == model_name
    assert metadata["entity_list"] == ENTITY_LIST


def test_colbert_el_model_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    _patch_pylate(monkeypatch)

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.graph_index_construction.entity_linking_model.ColbertELModel",
            "model_name_or_path": "lightonai/GTE-ModernColBERT-v1",
            "root": str(tmp_path),
            "force": False,
        }
    )

    el_model = instantiate(cfg)
    el_model.index(ENTITY_LIST)
    assert FakeColBERTModel.encode_calls == 1

    el_model2 = instantiate(cfg)
    el_model2.index(ENTITY_LIST)
    assert FakeColBERTModel.encode_calls == 1  # no re-encoding on cache hit

    result = el_model2(["south chicago community hospital"], topk=1)
    assert (
        result["south chicago community hospital"][0]["entity"]
        == "south chicago community hospital"
    )


def test_colbert_el_model_rebuilds_when_metadata_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    _patch_pylate(monkeypatch)

    model_name = "lightonai/GTE-ModernColBERT-v1"
    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.graph_index_construction.entity_linking_model.ColbertELModel",
            "model_name_or_path": model_name,
            "root": str(tmp_path),
            "force": False,
        }
    )

    el_model = instantiate(cfg)
    el_model.index(ENTITY_LIST)
    assert FakeColBERTModel.encode_calls == 1

    fp = _fingerprint(ENTITY_LIST)
    model_slug = model_name.replace("/", "_")
    metadata_path = tmp_path / "pylate" / model_slug / fp / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["fingerprint"] = "stale"
    metadata_path.write_text(json.dumps(metadata))

    el_model2 = instantiate(cfg)
    el_model2.index(ENTITY_LIST)
    assert FakeColBERTModel.encode_calls == 2  # rebuilt due to stale metadata


def test_colbert_el_model_force_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    _patch_pylate(monkeypatch)

    base_cfg = {
        "_target_": "gfmrag.graph_index_construction.entity_linking_model.ColbertELModel",
        "model_name_or_path": "lightonai/GTE-ModernColBERT-v1",
        "root": str(tmp_path),
        "force": False,
    }
    el_model = instantiate(OmegaConf.create(base_cfg))
    el_model.index(ENTITY_LIST)
    assert FakeColBERTModel.encode_calls == 1

    forced_cfg = {**base_cfg, "force": True}
    el_model2 = instantiate(OmegaConf.create(forced_cfg))
    el_model2.index(ENTITY_LIST)
    assert FakeColBERTModel.encode_calls == 2  # force=True always rebuilds


def test_colbert_el_model_topk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    _patch_pylate(monkeypatch)

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.graph_index_construction.entity_linking_model.ColbertELModel",
            "model_name_or_path": "lightonai/GTE-ModernColBERT-v1",
            "root": str(tmp_path),
            "force": False,
        }
    )
    el_model = instantiate(cfg)
    el_model.index(ENTITY_LIST)

    result = el_model(["south chicago community hospital", "july 13 14  1966"], topk=2)
    assert len(result["south chicago community hospital"]) == 2
    assert (
        result["south chicago community hospital"][0]["entity"]
        == "south chicago community hospital"
    )
    assert result["july 13 14  1966"][0]["entity"] == "july 13 14  1966"


# ---------------------------------------------------------------------------
# DPR EL model tests (unchanged)
# ---------------------------------------------------------------------------


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
