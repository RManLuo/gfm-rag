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

The framework supports two input layouts for each dataset:

1. Raw input files under `raw/`
2. Pre-built stage1 files under `processed/stage1/`

If only `raw/` is provided, the indexing workflow will construct the graph files and processed QA files under `processed/stage1/`.
If `processed/stage1/` is already provided, the framework can consume those files directly without rebuilding stage1.

### Option 1: Raw Input Layout

```text
root/
└── data_name/
    └── raw/
        ├── documents.json
        ├── train.json (optional)
        └── test.json (optional)
```

- `raw/documents.json` is the raw graph source used to build stage1 graph files.
- `raw/train.json` and `raw/test.json` are optional raw QA files.
- When these QA files are provided, the workflow will generate `processed/stage1/train.json` and `processed/stage1/test.json`.

### Option 2: Pre-built Stage1 Layout

```text
root/
└── data_name/
    └── processed/
        └── stage1/
            ├── nodes.csv
            ├── relations.csv
            ├── edges.csv
            ├── train.json (optional)
            └── test.json (optional)
```

- `processed/stage1/nodes.csv`, `relations.csv`, and `edges.csv` are the graph files consumed by the framework.
- `processed/stage1/train.json` and `processed/stage1/test.json` are the processed QA files consumed by the framework directly.

---

## Graph Index File Structure

When using pre-built stage1 data, graph data consists of three CSV files:

- `nodes.csv`: Defines nodes and their attributes.
- `relations.csv`: Defines relationships and their attributes.
- `edges.csv`: Defines edges between nodes and their attributes.

---

### `nodes.csv` File Format

| **Field**  | **Type** | **Description** |
| ---------- | -------- | --------------- |
| name       | str      | Node name |
| type       | str      | Node type, e.g., `entity`, `document`, or `summary` |
| attributes | dict     | (Optional) Additional node attributes, stored as a JSON string |

> The `attributes` field is a JSON-formatted string used to store arbitrary structured attributes.

**Example Content (`nodes.csv`):**

```csv
name,type,attributes
"Barack Obama","entity","{}"
"White House","entity","{}"
"Obama Biography","document","{'title': 'The Life of Barack Obama', 'published_year': 2020}"
```

**Text attributes for a document node:**

```
name: Obama Biography
type: document
title: The Life of Barack Obama
published_year: 2020
```

**Text attributes for an entity node:**

```
name: Barack Obama
type: entity
```

---

### `relations.csv` File Format

| **Field**  | **Type** | **Description** |
| ---------- | -------- | --------------- |
| name       | str      | Relation name |
| attributes | dict     | (Optional) Additional relation attributes, stored as a JSON string |

**Example Content (`relations.csv`):**

```csv
name,attributes
lived_in,"{'description': 'A person has a habitual presence in a specific location.'}"
mentioned_in,"{'description': 'An entity is mentioned in the document'}"
```

**Text attributes:**

```
name: lived_in
description: A person has a habitual presence in a specific location.
```

---

### `edges.csv` File Format

| **Field**  | **Type** | **Description** |
| ---------- | -------- | --------------- |
| source     | str      | The `name` field of the source node |
| relation   | str      | The `name` field of the relation |
| target     | str      | The `name` field of the target node |
| attributes | dict     | (Optional) Additional edge attributes, stored as a JSON string |

> `source` and `target` must appear in the `name` column of `nodes.csv`.
> `relation` must appear in the `name` column of `relations.csv`.

**Example Content (`edges.csv`):**

```csv
source,relation,target,attributes
"Barack Obama","lived_in","White House","{'start_year': 2009, 'end_year': 2017}"
"Barack Obama","mentioned_in","Obama Biography",{}
```

**Text attributes:**

```
start_year: 2009
end_year: 2017
```

---

## Complete Example Graph Structure

**Nodes (`nodes.csv`):**

| **name**        | **type**  | **attributes** |
| --------------- | --------- | -------------- |
| Barack Obama    | entity    | `{"birth_date": "1961-08-04", "nationality": "USA"}` |
| White House     | entity    | `{"location": "Washington, D.C."}` |
| Obama Biography | document  | `{"title": "The Life of Barack Obama", "published_year": 2020}` |
| Summary_node_1  | summary   | `{"summary": "...", "title": "..."}` |

**Relations (`relations.csv`):**

| **name**     | **attributes** |
| ------------ | -------------- |
| lived_in     | `{"description": "A person has a habitual presence in a specific location."}` |
| mentioned_in | `{"description": "An entity is mentioned in the document"}` |

**Edges (`edges.csv`):**

| **source**   | **relation** | **target**      | **attributes** |
| ------------ | ------------ | --------------- | -------------- |
| Barack Obama | lived_in     | White House     | `{"start_year": 2009, "end_year": 2017}` |
| Barack Obama | mentioned_in | Obama Biography | `{}` |

---

## Raw QA Files: `raw/train.json` and `raw/test.json`

Raw QA files are optional inputs under `raw/`. When provided, the workflow processes them into stage1 QA files.

| **Field**            | **Type**    | **Description** |
| -------------------- | ----------- | --------------- |
| id                   | str         | A unique identifier for the example |
| question             | str         | The question or query |
| answer               | str         | (Optional) The ground-truth answer; recommended for evaluation |
| answer_aliases       | list[str]   | (Optional) Alternative acceptable answers |
| supporting_documents | list[str]   | (Optional) Document names supporting the answer; useful for supervision |
| Additional fields    | Any         | Any extra fields are preserved into processed outputs when possible |

**Example (`raw/test.json`):**

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

---

## Processed QA Files: `processed/stage1/train.json` and `processed/stage1/test.json`

These are the stage1 QA files consumed by the framework directly. They can either:

- be generated automatically from `raw/train.json` and `raw/test.json`, or
- be provided directly under `processed/stage1/`

| **Field**         | **Type**         | **Description** |
| ----------------- | ---------------- | --------------- |
| id                | str              | A unique identifier for the example |
| question          | str              | The question or query |
| start_nodes       | dict[str, list]  | Starting nodes grouped by type. Key: node type, Value: list of node names |
| target_nodes      | dict[str, list]  | Target nodes grouped by type. Key: node type, Value: list of node names |
| Additional fields | Any              | Any extra fields copied from the raw data |

**Example (`processed/stage1/test.json`):**

```json
[
  {
    "id": "5abc553a554299700f9d7871",
    "question": "Kyle Ezell is a professor at what School of Architecture building at Ohio State?",
    "answer": "Knowlton Hall",
    "start_nodes": {
      "entity": [
        "kyle ezell",
        "architectural association school of architecture",
        "ohio state"
      ]
    },
    "target_nodes": {
      "document": [
        "Knowlton Hall",
        "Kyle Ezell"
      ],
      "entity": [
        "10 million donation",
        "2004",
        "architecture",
        "austin e  knowlton",
        "austin e  knowlton school of architecture",
        "bachelor s in architectural engineering",
        "city and regional planning",
        "columbus  ohio  united states",
        "ives hall",
        "july 2002",
        "knowlton hall",
        "ksa",
        "landscape architecture",
        "ohio",
        "replacement for ives hall",
        "the ohio state university"
      ]
    }
  }
]
```

---

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

---

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

---

## Outputs Used By Later Steps

- [Index](index.md) consumes `raw/` or `processed/stage1/`
- [Retrieval and QA](retrieval_and_qa.md) consumes `processed/stage1/` plus retrieval outputs
- [Training](training.md) consumes processed stage1 data and dataset lists

## Common Pitfalls

- `raw/documents.json` must exist if you expect `GFMRetriever.from_index(...)` or `index_dataset` to build stage1 automatically.
- `nodes.csv`, `relations.csv`, and `edges.csv` must stay consistent with each other when you provide pre-built stage1 files.
- `source` and `target` in `edges.csv` must match the `name` column in `nodes.csv`; `relation` must match the `name` column in `relations.csv`.
- Downstream QA examples often assume `answer_aliases` and `supporting_documents` are present, even though they are not mandatory for plain retrieval.
