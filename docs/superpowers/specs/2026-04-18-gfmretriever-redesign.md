# GFMRetriever Interface Redesign

**Date:** 2026-04-18
**Status:** Approved

## Goal

Redesign `GFMRetriever` so users can point at a data directory, index once, and retrieve with a simple Python API — without Hydra configs.

## Data Directory Convention

Users provide either:

- `raw/documents.json` only → system auto-builds `processed/stage1/` via `GraphIndexer`
- `processed/stage1/{nodes,edges,relations}.csv` → skip graph construction, go straight to stage2

```
data_dir/
└── data_name/
    ├── raw/
    │   └── documents.json
    └── processed/
        └── stage1/
            ├── nodes.csv
            ├── edges.csv
            └── relations.csv
```

## Interface

### `GFMRetriever.__init__`

```python
def __init__(
    self,
    qa_data: GraphIndexDataset,
    text_emb_model: BaseTextEmbModel,
    ner_model: BaseNERModel,
    el_model: BaseELModel,
    graph_retriever: BaseGNNModel,
    node_info: pd.DataFrame,  # nodes.csv loaded and indexed by node name/uid
    device: torch.device,
) -> None: ...
```

Removed from previous design: `doc_retriever`, `target_type`.

### `GFMRetriever.from_index` (replaces `from_config`)

```python
@staticmethod
def from_index(
    data_dir: str,
    data_name: str,
    model_path: str,
    ner_model: BaseNERModel,
    el_model: BaseELModel,
    graph_constructor: BaseGraphConstructor | None = None,
    force_reindex: bool = False,
) -> "GFMRetriever": ...
```

**Internal flow:**

1. Check if `processed/stage1/{nodes,edges,relations}.csv` all exist
   - If not, require `graph_constructor` → run `GraphIndexer.index_data()` → raises `ValueError` if `graph_constructor=None`, `FileNotFoundError` if `raw/documents.json` missing
2. `load_model_from_pretrained(model_path)` → `graph_retriever: BaseGNNModel`, `model_config`
3. `GraphIndexDataset(data_dir, data_name, text_emb_model_cfgs)` → `qa_data`
4. `el_model.index(list(qa_data.node2id.keys()))`
5. Load `node_info = _read_csv_file(stage1/nodes.csv)` (DataFrame indexed by name/uid)
6. Instantiate `text_emb_model` from `model_config`
7. Return `GFMRetriever(...)`

Supported models: `rmanluo/GFM-RAG-8M`, `rmanluo/G-Reasoner-34M`, or any local checkpoint.

### `GFMRetriever.retrieve`

```python
def retrieve(
    self,
    query: str,
    top_k: int,
    target_types: list[str] = ["document"],
) -> dict[str, list[dict]]: ...
```

**Internal flow** (aligned with `sft_trainer.py`):

```python
pred = self.graph_retriever(self.graph, input)  # shape: 1 x num_nodes
results = {}
for target_type in target_types:
    node_ids = self.graph.nodes_by_type[target_type]  # LongTensor
    type_pred = pred[:, node_ids].squeeze(0)
    topk = torch.topk(type_pred, k=top_k)
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

**Return structure:**

```python
{
    "document": [
        {
            "id": "Chris Evans (actor)",
            "type": "document",
            "attributes": {"content": "..."},
            "score": 0.87,
        },
    ],
    "entity": [
        {
            "id": "Marvel",
            "type": "entity",
            "attributes": {"description": "..."},
            "score": 0.62,
        },
    ],
}
```

`attributes` maps directly to the `attributes` column in `nodes.csv` — no transformation.

## Error Handling

| Condition | Error |
|---|---|
| `stage1/` missing + `graph_constructor=None` | `ValueError` with descriptive message |
| `raw/documents.json` missing | `FileNotFoundError` |
| `target_types` entry not in `graph.nodes_by_type` | `KeyError` raised at retrieve time with clear message |
| `model_path` load failure | Propagate from `load_model_from_pretrained` unchanged |

## Migration

`from_config(cfg: DictConfig)` is removed. Call sites in `workflow/` (e.g., `qa_ircot_inference.py`) must be updated to use `from_index()`.

## Example Usage

```python
from gfmrag import GFMRetriever
from gfmrag.graph_index_construction.ner_model import LLMNERModel
from gfmrag.graph_index_construction.entity_linking_model import ColbertELModel

retriever = GFMRetriever.from_index(
    data_dir="./data",
    data_name="my_dataset",
    model_path="rmanluo/GFM-RAG-8M",
    ner_model=LLMNERModel(...),
    el_model=ColbertELModel(...),
)

results = retriever.retrieve("Who played Captain America?", top_k=5)
# results["document"] → list of top-5 document nodes

results = retriever.retrieve(
    "Who played Captain America?", top_k=5, target_types=["document", "entity"]
)
# results["document"] + results["entity"]
```
