# Data Format

This page defines the current dataset layouts consumed by the repository.

## What This Step Does

It specifies the files you need before indexing, retrieval, QA, or training can run.

## When You Need It

Read this page before:

- building a new dataset from raw documents
- reusing pre-built stage1 graph files
- preparing training and evaluation examples

## Supported Layouts

### Raw Input Layout

```text
root/
└── data_name/
    └── raw/
        ├── documents.json
        ├── train.json
        └── test.json
```

- `raw/documents.json` is required.
- `raw/train.json` and `raw/test.json` are optional, depending on whether you need training or evaluation data.

### Pre-built Stage1 Layout

```text
root/
└── data_name/
    └── processed/
        └── stage1/
            ├── nodes.csv
            ├── relations.csv
            ├── edges.csv
            ├── train.json
            └── test.json
```

- `nodes.csv`, `relations.csv`, and `edges.csv` are the graph files consumed by the current indexing and retrieval path.
- Processed `train.json` and `test.json` can be supplied directly instead of being generated from raw files.

## Minimal Raw Example

### `raw/documents.json`

```json
{
  "France": "France is a country in Western Europe. Paris is its capital.",
  "Paris": "Paris is the capital and most populous city of France."
}
```

### `raw/test.json`

```json
[
  {
    "id": "toy-1",
    "question": "What is the capital of France?",
    "answer": "Paris",
    "answer_aliases": ["City of Paris"],
    "supporting_documents": ["France", "Paris"]
  }
]
```

## Minimal Processed Stage1 Example

### `processed/stage1/nodes.csv`

```csv
name,type,attributes
France,document,"{}"
Paris,document,"{}"
capital,entity,"{}"
```

### `processed/stage1/relations.csv`

```csv
name,attributes
mentions,"{}"
```

### `processed/stage1/edges.csv`

```csv
source,relation,target,attributes
France,mentions,capital,"{}"
Paris,mentions,capital,"{}"
```

### `processed/stage1/test.json`

```json
[
  {
    "id": "toy-1",
    "question": "What is the capital of France?",
    "answer": "Paris",
    "answer_aliases": ["City of Paris"],
    "supporting_documents": ["France", "Paris"],
    "start_nodes": {
      "entity": ["capital"]
    },
    "target_nodes": {
      "document": ["France", "Paris"]
    }
  }
]
```

## Field Guide

### `raw/documents.json`

- Required
- JSON object
- Keys are document ids or titles
- Values are raw document text

### Raw `train.json` / `test.json`

- `id`: required
- `question`: required
- `answer`: optional but recommended for evaluation
- `answer_aliases`: optional
- `supporting_documents`: optional at raw stage, but useful for supervision and evaluation
- additional metadata is preserved into processed outputs when possible

### Processed `train.json` / `test.json`

- `id`: required
- `question`: required
- `start_nodes`: required for processed QA data
- `target_nodes`: required for processed QA data
- `answer`, `answer_aliases`, `supporting_documents`: optional but commonly present

## Outputs Used By Later Steps

- [Index](index.md) consumes `raw/` or `processed/stage1/`
- [Retrieval and QA](retrieval_and_qa.md) consumes `processed/stage1/` plus retrieval outputs
- [Training](training.md) consumes processed stage1 data and dataset lists

## Common Pitfalls

- `raw/documents.json` must exist if you expect `GFMRetriever.from_index(...)` or `index_dataset` to build stage1 automatically.
- `nodes.csv`, `relations.csv`, and `edges.csv` must stay consistent with each other when you provide pre-built stage1 files.
- Downstream QA examples often assume `answer_aliases` and `supporting_documents` are present, even though they are not mandatory for plain retrieval.
