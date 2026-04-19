# Quick Start

This page shows the shortest path from a raw dataset to document retrieval with the current API.

GFM-RAG provides a **unified graph interface**: if you already have a graph that conforms to the three-file format (`nodes.csv` / `relations.csv` / `edges.csv`), you can **skip the index-building step entirely** and use it directly for retrieval and reasoning — regardless of how the graph was constructed.

There are two starting points:

- **Path A — Start from raw documents** (steps 1–3 below): provide `raw/documents.json` and let `GFMRetriever.from_index(...)` build the graph automatically.
- **Path B — Bring your own graph** (step 1b below): place pre-built graph files under `processed/stage1/` and `GFMRetriever.from_index(...)` will load them directly without rebuilding.

See [Data Format](../workflow/data_format.md) for the full schema of both paths.

---

## Path A: Start From Raw Documents

## 1. Create A Minimal Dataset

```text
data/
└── toy_raw/
    └── raw/
        ├── documents.json
        └── test.json
```

`raw/documents.json` is required:

```json
{
  "France": "France is a country in Western Europe. Paris is its capital.",
  "Paris": "Paris is the capital and most populous city of France.",
  "Emmanuel Macron": "Emmanuel Macron has served as president of France since 2017."
}
```

`raw/test.json` is optional for plain retrieval, but useful for later QA and evaluation:

```json
[
  {
    "id": "toy-1",
    "question": "Who is the president of France?",
    "answer": "Emmanuel Macron",
    "answer_aliases": ["Macron"],
    "supporting_documents": ["France", "Emmanuel Macron"]
  }
]
```

## 2. Initialize `GFMRetriever`

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gfmrag import GFMRetriever

cfg = OmegaConf.load("gfmrag/workflow/config/gfm_rag/qa_ircot_inference.yaml")

retriever = GFMRetriever.from_index(
    data_dir="./data",
    data_name="toy_raw",
    model_path="rmanluo/G-reasoner-34M",
    ner_model=instantiate(cfg.ner_model),
    el_model=instantiate(cfg.el_model),
    graph_constructor=instantiate(cfg.graph_constructor),
)
```

On the first run, `GFMRetriever.from_index(...)` builds `processed/stage1/` automatically if the graph files do not already exist.

## 3. Retrieve Documents

```python
results = retriever.retrieve(
    "Who is the president of France?",
    top_k=5,
)

for item in results["document"]:
    print(item["id"], item["score"])
```

## 4. Know The Default Dependencies

The default configs used above rely on instantiated components from the workflow configs:

- `ner_model: llm_ner_model`
- `openie_model: llm_openie_model`
- `el_model: colbert_el_model`

The default NER and OpenIE path uses API-backed components, so make sure the required credentials and services are available before running the example.

---

## Path B: Bring Your Own Graph

If you already have a graph — for example, an existing Knowledge Graph, a graph produced by another pipeline, or a graph you built manually — you can use it directly **without running the index-building step**, as long as it conforms to the GFM-RAG graph format.

### 1. Place Pre-built Graph Files

Create the following directory structure and populate it with your graph files:

```text
data/
└── my_dataset/
    └── processed/
        └── stage1/
            ├── nodes.csv
            ├── relations.csv
            ├── edges.csv
            └── test.json   (optional)
```

The three CSV files define the graph:

| File | Description |
|------|-------------|
| `nodes.csv` | Node name, type (`entity` / `document` / `summary`), and optional attributes |
| `relations.csv` | Relation name and optional attributes |
| `edges.csv` | Edges as `(source, relation, target)` triples with optional attributes |

See [Data Format](../workflow/data_format.md) for the full schema and examples.

### 2. Initialize `GFMRetriever` and Retrieve

`GFMRetriever.from_index(...)` detects that `processed/stage1/` already exists and loads the graph directly — no rebuild occurs.

```python
from gfmrag import GFMRetriever

retriever = GFMRetriever.from_index(
    data_dir="./data",
    data_name="my_dataset",
    model_path="rmanluo/G-reasoner-34M",  # or rmanluo/GFM-RAG-8M
)

results = retriever.retrieve("Your query here", top_k=5)

for item in results["document"]:
    print(item["id"], item["score"])
```

> **Note:** When using a pre-built graph you do not need to pass `ner_model`, `el_model`, or `graph_constructor` — those are only required when building the graph from raw documents.

---

## Next Steps

- Read [Workflow: Data Format](../workflow/data_format.md) for the full dataset and stage1 schema.
- Read [Workflow: Retrieval and QA](../workflow/retrieval_and_qa.md) for batch QA and agent reasoning.
