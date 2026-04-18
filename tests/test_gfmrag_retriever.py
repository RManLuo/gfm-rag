# tests/test_gfmrag_retriever.py
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data


@pytest.fixture
def mock_graph() -> MagicMock:
    graph = MagicMock(spec=Data)
    graph.num_nodes = 4
    graph.nodes_by_type = {
        "document": torch.tensor([0, 1]),
        "entity": torch.tensor([2, 3]),
    }
    return graph


@pytest.fixture
def mock_qa_data(mock_graph: MagicMock) -> MagicMock:
    qa_data = MagicMock()
    qa_data.graph = mock_graph
    qa_data.node2id = {"DocA": 0, "DocB": 1, "EntA": 2, "EntB": 3}
    qa_data.id2node = {0: "DocA", 1: "DocB", 2: "EntA", 3: "EntB"}
    return qa_data


@pytest.fixture
def mock_node_info() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "type": ["document", "document", "entity", "entity"],
            "attributes": [
                {"content": "Content of DocA"},
                {"content": "Content of DocB"},
                {"description": "Entity A"},
                {"description": "Entity B"},
            ],
        },
        index=pd.Index(["DocA", "DocB", "EntA", "EntB"], name="name"),
    )


@pytest.fixture
def retriever(
    mock_qa_data: MagicMock, mock_node_info: pd.DataFrame, mock_graph: MagicMock
) -> Any:
    from gfmrag.gfmrag_retriever import GFMRetriever

    text_emb_model = MagicMock()
    text_emb_model.encode.return_value = torch.zeros(1, 128)

    ner_model = MagicMock()
    ner_model.return_value = ["DocA"]

    el_model = MagicMock()
    el_model.return_value = {"DocA": [{"entity": "DocA", "score": 1.0}]}

    graph_retriever = MagicMock()
    # scores: DocA=0.9, DocB=0.1, EntA=0.8, EntB=0.2
    graph_retriever.return_value = torch.tensor([[0.9, 0.1, 0.8, 0.2]])

    return GFMRetriever(  # type: ignore[call-arg]
        qa_data=mock_qa_data,
        text_emb_model=text_emb_model,
        ner_model=ner_model,
        el_model=el_model,
        graph_retriever=graph_retriever,
        node_info=mock_node_info,
        device=torch.device("cpu"),
    )


def test_retrieve_returns_dict(retriever: Any) -> None:
    result = retriever.retrieve("test query", top_k=2)
    assert isinstance(result, dict)
    assert "document" in result


def test_retrieve_top_k_document(retriever: Any) -> None:
    result = retriever.retrieve("test query", top_k=1)
    docs = result["document"]
    assert len(docs) == 1
    assert docs[0]["id"] == "DocA"
    assert docs[0]["type"] == "document"
    assert docs[0]["attributes"] == {"content": "Content of DocA"}
    assert docs[0]["score"] == pytest.approx(0.9, abs=1e-4)


def test_retrieve_multiple_types(retriever: Any) -> None:
    result = retriever.retrieve(
        "test query", top_k=1, target_types=["document", "entity"]
    )
    assert "document" in result
    assert "entity" in result
    assert result["document"][0]["id"] == "DocA"
    assert result["entity"][0]["id"] == "EntA"


def test_retrieve_unknown_type_raises(retriever: Any) -> None:
    with pytest.raises(KeyError):
        retriever.retrieve("test query", top_k=1, target_types=["unknown_type"])


def test_retrieve_default_target_type_is_document(retriever: Any) -> None:
    result = retriever.retrieve("test query", top_k=1)
    assert list(result.keys()) == ["document"]
