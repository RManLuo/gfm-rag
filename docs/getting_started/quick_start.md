# Quick Start

This page shows the shortest path from a raw dataset to document retrieval with the current API.

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

## Next Steps

- Read [Workflow: Data Format](../workflow/data_format.md) for the full dataset and stage1 schema.
- Read [Workflow: Retrieval and QA](../workflow/retrieval_and_qa.md) for batch QA and agent reasoning.
