# GFMRetriever Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `GFMRetriever.from_config()` with `from_index()`, update `retrieve()` to return `dict[str, list[dict]]` keyed by node type, and remove `DocumentRetriever` dependency.

**Architecture:** `from_index()` auto-detects whether `processed/stage1/` exists and builds it if needed, then assembles `GFMRetriever` from plain Python arguments. `retrieve()` slices GNN predictions per `target_types` following the `sft_trainer.py` pattern, returning node info directly from `nodes.csv`.

**Tech Stack:** PyTorch, torch_geometric, pandas, pytest, unittest.mock

---

## File Map

- **Modify:** `gfmrag/gfmrag_retriever.py` — core class changes
- **Modify:** `gfmrag/workflow/qa_ircot_inference.py` — migrate from `from_config` to `from_index`, update dedup logic
- **Create:** `tests/test_gfmrag_retriever.py` — unit tests for new interface

---

### Task 1: Write failing tests for `__init__` and `retrieve()`

**Files:**
- Create: `tests/test_gfmrag_retriever.py`

- [ ] **Step 1: Create the test file**

```python
# tests/test_gfmrag_retriever.py
import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock
from torch_geometric.data import Data


@pytest.fixture
def mock_graph():
    graph = MagicMock(spec=Data)
    graph.num_nodes = 4
    graph.nodes_by_type = {
        "document": torch.tensor([0, 1]),
        "entity": torch.tensor([2, 3]),
    }
    return graph


@pytest.fixture
def mock_qa_data(mock_graph):
    qa_data = MagicMock()
    qa_data.graph = mock_graph
    qa_data.node2id = {"DocA": 0, "DocB": 1, "EntA": 2, "EntB": 3}
    qa_data.id2node = {0: "DocA", 1: "DocB", 2: "EntA", 3: "EntB"}
    return qa_data


@pytest.fixture
def mock_node_info():
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
def retriever(mock_qa_data, mock_node_info, mock_graph):
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

    return GFMRetriever(
        qa_data=mock_qa_data,
        text_emb_model=text_emb_model,
        ner_model=ner_model,
        el_model=el_model,
        graph_retriever=graph_retriever,
        node_info=mock_node_info,
        device=torch.device("cpu"),
    )


def test_retrieve_returns_dict(retriever):
    result = retriever.retrieve("test query", top_k=2)
    assert isinstance(result, dict)
    assert "document" in result


def test_retrieve_top_k_document(retriever):
    result = retriever.retrieve("test query", top_k=1)
    docs = result["document"]
    assert len(docs) == 1
    assert docs[0]["id"] == "DocA"
    assert docs[0]["type"] == "document"
    assert docs[0]["attributes"] == {"content": "Content of DocA"}
    assert docs[0]["score"] == pytest.approx(0.9, abs=1e-4)


def test_retrieve_multiple_types(retriever):
    result = retriever.retrieve(
        "test query", top_k=1, target_types=["document", "entity"]
    )
    assert "document" in result
    assert "entity" in result
    assert result["document"][0]["id"] == "DocA"
    assert result["entity"][0]["id"] == "EntA"


def test_retrieve_unknown_type_raises(retriever):
    with pytest.raises(KeyError):
        retriever.retrieve("test query", top_k=1, target_types=["unknown_type"])


def test_retrieve_default_target_type_is_document(retriever):
    result = retriever.retrieve("test query", top_k=1)
    assert list(result.keys()) == ["document"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/lluo/projects/gfm_rag
pytest tests/test_gfmrag_retriever.py -v 2>&1 | head -40
```

Expected: `ImportError` or `TypeError` (signature mismatch since `__init__` still has old params).

---

### Task 2: Update `__init__` and `retrieve()` in `gfmrag_retriever.py`

**Files:**
- Modify: `gfmrag/gfmrag_retriever.py`

- [ ] **Step 1: Replace the imports and `__init__`**

In `gfmrag/gfmrag_retriever.py`, replace the top section through the end of `__init__`:

```python
import ast
import logging
import os

import pandas as pd
import torch
from omegaconf import OmegaConf

from gfmrag import utils
from gfmrag.graph_index_construction.entity_linking_model import BaseELModel
from gfmrag.graph_index_construction.ner_model import BaseNERModel
from gfmrag.graph_index_datasets import GraphIndexDataset
from gfmrag.models.base_model import BaseGNNModel
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils.qa_utils import entities_to_mask

logger = logging.getLogger(__name__)


class GFMRetriever:
    """Graph Foundation Model (GFM) Retriever for document retrieval.

    Attributes:
        qa_data (GraphIndexDataset): Dataset containing the knowledge graph and mappings.
        graph: Knowledge graph structure.
        text_emb_model (BaseTextEmbModel): Model for text embedding.
        ner_model (BaseNERModel): Named Entity Recognition model.
        el_model (BaseELModel): Entity Linking model.
        graph_retriever (BaseGNNModel): GNN-based retriever (GNNRetriever or GraphReasoner).
        node_info (pd.DataFrame): Node attributes from nodes.csv, indexed by node name/uid.
        device (torch.device): Device to run computations on.
        num_nodes (int): Number of nodes in the knowledge graph.

    Examples:
        >>> retriever = GFMRetriever.from_index(
        ...     data_dir="./data",
        ...     data_name="my_dataset",
        ...     model_path="rmanluo/GFM-RAG-8M",
        ...     ner_model=ner_model,
        ...     el_model=el_model,
        ... )
        >>> results = retriever.retrieve("Who is the president of France?", top_k=5)
    """

    def __init__(
        self,
        qa_data: GraphIndexDataset,
        text_emb_model: BaseTextEmbModel,
        ner_model: BaseNERModel,
        el_model: BaseELModel,
        graph_retriever: BaseGNNModel,
        node_info: pd.DataFrame,
        device: torch.device,
    ) -> None:
        self.qa_data = qa_data
        self.graph = qa_data.graph
        self.text_emb_model = text_emb_model
        self.ner_model = ner_model
        self.el_model = el_model
        self.graph_retriever = graph_retriever
        self.node_info = node_info
        self.device = device
        self.num_nodes = self.graph.num_nodes
```

- [ ] **Step 2: Replace `retrieve()` method**

Replace the existing `retrieve()` method body with:

```python
@torch.no_grad()
def retrieve(
    self,
    query: str,
    top_k: int,
    target_types: list[str] = ["document"],
) -> dict[str, list[dict]]:
    """Retrieve nodes from the graph based on the given query.

    Args:
        query (str): Input query text.
        top_k (int): Number of results to return per target type.
        target_types (list[str]): Node types to retrieve. Each type must exist
            in graph.nodes_by_type. Defaults to ["document"].

    Returns:
        dict[str, list[dict]]: Results keyed by target type. Each entry contains
            dicts with keys: id, type, attributes, score.
    """
    from gfmrag.models.ultra import query_utils

    graph_retriever_input = self.prepare_input_for_graph_retriever(query)
    graph_retriever_input = query_utils.cuda(graph_retriever_input, device=self.device)

    pred = self.graph_retriever(self.graph, graph_retriever_input)  # 1 x num_nodes

    results: dict[str, list[dict]] = {}
    for target_type in target_types:
        node_ids = self.graph.nodes_by_type[target_type]  # raises KeyError if missing
        type_pred = pred[:, node_ids].squeeze(0)
        topk = torch.topk(type_pred, k=min(top_k, len(node_ids)))
        original_ids = node_ids[topk.indices]
        results[target_type] = [
            {
                "id": self.qa_data.id2node[nid.item()],
                "type": target_type,
                "attributes": self.node_info.loc[
                    self.qa_data.id2node[nid.item()], "attributes"
                ],
                "score": score.item(),
            }
            for nid, score in zip(original_ids, topk.values)
        ]
    return results
```

- [ ] **Step 3: Keep `prepare_input_for_graph_retriever` unchanged**

No changes needed — it still uses `self.ner_model`, `self.el_model`, `self.qa_data.node2id`, `self.text_emb_model`, `self.num_nodes`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_gfmrag_retriever.py -v
```

Expected: All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_gfmrag_retriever.py gfmrag/gfmrag_retriever.py
git commit -m "feat: update GFMRetriever __init__ and retrieve() for multi-type results"
```

---

### Task 3: Write failing tests for `from_index()`

**Files:**
- Modify: `tests/test_gfmrag_retriever.py`

- [ ] **Step 1: Add error-case tests to the test file**

Append to `tests/test_gfmrag_retriever.py`:

```python
def test_from_index_raises_without_stage1_and_constructor(tmp_path):
    from gfmrag.gfmrag_retriever import GFMRetriever

    (tmp_path / "my_data" / "raw").mkdir(parents=True)
    (tmp_path / "my_data" / "raw" / "documents.json").write_text("[]")

    with pytest.raises(ValueError, match="graph_constructor"):
        GFMRetriever.from_index(
            data_dir=str(tmp_path),
            data_name="my_data",
            model_path="fake/model",
            ner_model=MagicMock(),
            el_model=MagicMock(),
        )


def test_from_index_raises_without_raw_documents(tmp_path):
    from gfmrag.gfmrag_retriever import GFMRetriever

    (tmp_path / "my_data").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        GFMRetriever.from_index(
            data_dir=str(tmp_path),
            data_name="my_data",
            model_path="fake/model",
            ner_model=MagicMock(),
            el_model=MagicMock(),
            graph_constructor=MagicMock(),
        )


def test_from_index_with_existing_stage1(tmp_path):
    from unittest.mock import patch
    import json
    from gfmrag.gfmrag_retriever import GFMRetriever

    # Create stage1 CSV files
    stage1 = tmp_path / "my_data" / "processed" / "stage1"
    stage1.mkdir(parents=True)
    (stage1 / "nodes.csv").write_text('name,type,attributes\nDocA,document,"{}"\n')
    (stage1 / "relations.csv").write_text('name,attributes\nrel1,"{}"\n')
    (stage1 / "edges.csv").write_text(
        "source,target,relation,attributes\nDocA,DocA,rel1,{}\n"
    )
    # Also create raw/documents.json (required by GraphIndexDataset.load_qa_data)
    raw = tmp_path / "my_data" / "raw"
    raw.mkdir(parents=True)
    (raw / "documents.json").write_text("{}")

    mock_model = MagicMock()
    mock_qa_data = MagicMock()
    mock_qa_data.node2id = {"DocA": 0}
    mock_qa_data.id2node = {0: "DocA"}
    mock_qa_data.graph = MagicMock()
    mock_qa_data.graph.num_nodes = 1

    el_model = MagicMock()
    ner_model = MagicMock()

    with (
        patch(
            "gfmrag.gfmrag_retriever.utils.load_model_from_pretrained",
            return_value=(
                mock_model,
                {
                    "text_emb_model_config": {
                        "_target_": "gfmrag.text_emb_models.BGETextEmbModel"
                    }
                },
            ),
        ),
        patch(
            "gfmrag.gfmrag_retriever.GraphIndexDataset",
            return_value=mock_qa_data,
        ),
        patch("gfmrag.gfmrag_retriever.instantiate", return_value=MagicMock()),
    ):
        retriever = GFMRetriever.from_index(
            data_dir=str(tmp_path),
            data_name="my_data",
            model_path="fake/model",
            ner_model=ner_model,
            el_model=el_model,
        )

    assert isinstance(retriever, GFMRetriever)
    el_model.index.assert_called_once()


def test_from_index_calls_graph_constructor_when_no_stage1(tmp_path):
    from unittest.mock import patch
    from gfmrag.gfmrag_retriever import GFMRetriever

    raw = tmp_path / "my_data" / "raw"
    raw.mkdir(parents=True)
    (raw / "documents.json").write_text("{}")

    graph_constructor = MagicMock()
    graph_constructor.build_graph.return_value = {
        "nodes": [{"name": "DocA", "type": "document", "attributes": "{}"}],
        "relations": [{"name": "rel1", "attributes": "{}"}],
        "edges": [
            {"source": "DocA", "target": "DocA", "relation": "rel1", "attributes": "{}"}
        ],
    }

    mock_model = MagicMock()
    mock_qa_data = MagicMock()
    mock_qa_data.node2id = {"DocA": 0}
    mock_qa_data.id2node = {0: "DocA"}
    mock_qa_data.graph = MagicMock()
    mock_qa_data.graph.num_nodes = 1

    with (
        patch(
            "gfmrag.gfmrag_retriever.utils.load_model_from_pretrained",
            return_value=(
                mock_model,
                {
                    "text_emb_model_config": {
                        "_target_": "gfmrag.text_emb_models.BGETextEmbModel"
                    }
                },
            ),
        ),
        patch(
            "gfmrag.gfmrag_retriever.GraphIndexDataset",
            return_value=mock_qa_data,
        ),
        patch("gfmrag.gfmrag_retriever.instantiate", return_value=MagicMock()),
    ):
        GFMRetriever.from_index(
            data_dir=str(tmp_path),
            data_name="my_data",
            model_path="fake/model",
            ner_model=MagicMock(),
            el_model=MagicMock(),
            graph_constructor=graph_constructor,
        )

    graph_constructor.build_graph.assert_called_once_with(str(tmp_path), "my_data")
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
pytest tests/test_gfmrag_retriever.py::test_from_index_raises_without_stage1_and_constructor \
       tests/test_gfmrag_retriever.py::test_from_index_raises_without_raw_documents \
       tests/test_gfmrag_retriever.py::test_from_index_with_existing_stage1 \
       tests/test_gfmrag_retriever.py::test_from_index_calls_graph_constructor_when_no_stage1 \
       -v 2>&1 | head -30
```

Expected: `AttributeError: type object 'GFMRetriever' has no attribute 'from_index'`

---

### Task 4: Implement `from_index()`, remove `from_config()`

**Files:**
- Modify: `gfmrag/gfmrag_retriever.py`

- [ ] **Step 1: Add required imports at the top of `gfmrag_retriever.py`**

Ensure these are present in the imports section (add any missing):

```python
import ast
import logging
import os

import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gfmrag import utils
from gfmrag.graph_index_construction.entity_linking_model import BaseELModel
from gfmrag.graph_index_construction.graph_constructors import BaseGraphConstructor
from gfmrag.graph_index_construction.ner_model import BaseNERModel
from gfmrag.graph_index_datasets import GraphIndexDataset
from gfmrag.models.base_model import BaseGNNModel
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils.qa_utils import entities_to_mask
```

- [ ] **Step 2: Replace `from_config()` with `from_index()`**

Remove the entire `from_config` static method and replace with:

```python
@staticmethod
def from_index(
    data_dir: str,
    data_name: str,
    model_path: str,
    ner_model: BaseNERModel,
    el_model: BaseELModel,
    graph_constructor: "BaseGraphConstructor | None" = None,
    force_reindex: bool = False,
) -> "GFMRetriever":
    """Construct a GFMRetriever from a data directory.

    Detects whether processed/stage1/ exists. If not, uses graph_constructor
    to build it from raw/documents.json. Then loads GraphIndexDataset (stage2),
    indexes the entity linking model, and assembles the retriever.

    Args:
        data_dir: Root data directory (contains data_name/ subdirectory).
        data_name: Dataset subdirectory name.
        model_path: HuggingFace model ID or local path (e.g. "rmanluo/GFM-RAG-8M").
        ner_model: Instantiated NER model.
        el_model: Instantiated EL model. index() is called internally.
        graph_constructor: Required only when stage1/ does not exist.
        force_reindex: Force rebuild of stage2 processed files.

    Returns:
        Fully initialized GFMRetriever.

    Raises:
        FileNotFoundError: If raw/documents.json is missing.
        ValueError: If stage1/ is missing and graph_constructor is None.
    """
    stage1_dir = os.path.join(data_dir, data_name, "processed", "stage1")
    stage1_files = [
        os.path.join(stage1_dir, name) for name in GraphIndexDataset.RAW_GRAPH_NAMES
    ]

    if not utils.check_all_files_exist(stage1_files):
        raw_docs = os.path.join(data_dir, data_name, "raw", "documents.json")
        if not os.path.exists(raw_docs):
            raise FileNotFoundError(
                f"raw/documents.json not found at {raw_docs}. "
                "Provide documents.json or pre-built stage1/ CSV files."
            )
        if graph_constructor is None:
            raise ValueError(
                "processed/stage1/ not found. Provide a graph_constructor "
                "to build the graph from raw/documents.json."
            )
        # Build stage1 from raw documents
        logger.info(f"Building graph index for {data_name}")
        os.makedirs(stage1_dir, exist_ok=True)
        graph = graph_constructor.build_graph(data_dir, data_name)
        pd.DataFrame(graph["nodes"]).to_csv(
            os.path.join(stage1_dir, "nodes.csv"), index=False
        )
        pd.DataFrame(graph["edges"]).to_csv(
            os.path.join(stage1_dir, "edges.csv"), index=False
        )
        pd.DataFrame(graph["relations"]).to_csv(
            os.path.join(stage1_dir, "relations.csv"), index=False
        )
        logger.info(f"Stage1 graph files saved to {stage1_dir}")

    # Load pretrained GNN model
    graph_retriever, model_config = utils.load_model_from_pretrained(model_path)
    graph_retriever.eval()

    # Build stage2 dataset
    text_emb_model_cfgs = OmegaConf.create(model_config["text_emb_model_config"])
    qa_data = GraphIndexDataset(
        root=data_dir,
        data_name=data_name,
        text_emb_model_cfgs=text_emb_model_cfgs,
        force_reload=force_reindex,
    )

    device = utils.get_device()
    graph_retriever = graph_retriever.to(device)
    qa_data.graph = qa_data.graph.to(device)

    # Index entity linking model over all graph nodes
    el_model.index(list(qa_data.node2id.keys()))

    # Load node attributes from nodes.csv
    nodes_csv = os.path.join(stage1_dir, "nodes.csv")
    nodes_df = pd.read_csv(nodes_csv, keep_default_na=False)
    nodes_df["attributes"] = nodes_df["attributes"].apply(
        lambda x: {} if x == "" else ast.literal_eval(x)
    )
    id_col = "uid" if "uid" in nodes_df.columns else "name"
    nodes_df = nodes_df.set_index(id_col)

    text_emb_model = instantiate(text_emb_model_cfgs)

    return GFMRetriever(
        qa_data=qa_data,
        text_emb_model=text_emb_model,
        ner_model=ner_model,
        el_model=el_model,
        graph_retriever=graph_retriever,
        node_info=nodes_df,
        device=device,
    )
```

Note: Fix the `import pandas as pd as _pd` — that is invalid syntax. The `pandas` import is already at the top of the file, so just use `pd` directly. The full correct block for the CSV saving is:

```python
pd.DataFrame(graph["nodes"]).to_csv(os.path.join(stage1_dir, "nodes.csv"), index=False)
pd.DataFrame(graph["edges"]).to_csv(os.path.join(stage1_dir, "edges.csv"), index=False)
pd.DataFrame(graph["relations"]).to_csv(
    os.path.join(stage1_dir, "relations.csv"), index=False
)
```

- [ ] **Step 3: Run all tests**

```bash
pytest tests/test_gfmrag_retriever.py -v
```

Expected: All 9 tests pass.

- [ ] **Step 4: Commit**

```bash
git add gfmrag/gfmrag_retriever.py tests/test_gfmrag_retriever.py
git commit -m "feat: add GFMRetriever.from_index(), remove from_config()"
```

---

### Task 5: Update `qa_ircot_inference.py`

**Files:**
- Modify: `gfmrag/workflow/qa_ircot_inference.py`

- [ ] **Step 1: Update imports**

Remove the unused `OmegaConf` import if not used elsewhere. Add `instantiate` if not already present. The file already imports `instantiate` and `DictConfig`.

- [ ] **Step 2: Replace `from_config` call and update `agent_reasoning` dedup logic**

Replace the entire `agent_reasoning` function and the `main()` retriever construction line:

In `agent_reasoning`, replace lines 61-74 (the dedup block that uses `doc["title"]` and `doc["norm_score"]`):

```python
# Merge new_ret_docs into retrieved_docs, dedup by id, keep highest score
for target_type, new_docs in new_ret_docs.items():
    existing = {d["id"]: d for d in retrieved_docs.get(target_type, [])}
    for doc in new_docs:
        if doc["id"] not in existing or doc["score"] > existing[doc["id"]]["score"]:
            existing[doc["id"]] = doc
    retrieved_docs[target_type] = sorted(
        existing.values(), key=lambda x: x["score"], reverse=True
    )[: cfg.test.top_k]
```

In `main()`, replace:
```python
gfmrag_retriever = GFMRetriever.from_config(cfg)
```
with:
```python
ner_model = instantiate(cfg.graph_retriever.ner_model)
el_model = instantiate(cfg.graph_retriever.el_model)
gfmrag_retriever = GFMRetriever.from_index(
    data_dir=cfg.dataset.cfgs.root,
    data_name=cfg.dataset.cfgs.data_name,
    model_path=cfg.graph_retriever.model_path,
    ner_model=ner_model,
    el_model=el_model,
)
```

- [ ] **Step 3: Verify no remaining `from_config` references**

```bash
grep -rn "from_config" /home/lluo/projects/gfm_rag/gfmrag/ --include="*.py"
```

Expected: no output.

- [ ] **Step 4: Run existing tests to check for regressions**

```bash
pytest tests/test_gfmrag_retriever.py -v
```

Expected: All 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add gfmrag/workflow/qa_ircot_inference.py
git commit -m "feat: migrate qa_ircot_inference to GFMRetriever.from_index()"
```
