# Index

This page covers the current indexing entrypoint.

## What This Step Does

`python -m gfmrag.workflow.index_dataset` builds the graph files and processed QA data needed by retrieval, QA, and training.

## When You Need It

Run indexing when:

- you only have `raw/` files
- you changed the graph-construction setup
- you want to regenerate stage1 files with a different constructor or model stack

If you already have a complete `processed/stage1/`, you can skip this step.

## Inputs

- dataset root and dataset name
- the current indexing config family under `gfmrag/workflow/config/gfm_rag/index_dataset.yaml`
- graph constructor and SFT constructor component configs

## Outputs

Indexing writes into `data/<data_name>/processed/stage1/`.

Typical outputs include:

- `nodes.csv`
- `relations.csv`
- `edges.csv`
- processed `train.json`
- processed `test.json`

Hydra also writes run metadata under `outputs/kg_construction/<date>/<time>/`.

## Minimal Example

```bash
python -m gfmrag.workflow.index_dataset \
  dataset.root=./data \
  dataset.data_name=toy_raw
```

## Current Config Family

The default indexing entrypoint is configured under:

- [`gfmrag/workflow/config/gfm_rag/index_dataset.yaml`](https://github.com/RManLuo/gfm-rag/blob/main/gfmrag/workflow/config/gfm_rag/index_dataset.yaml)

Related component config groups live under:

- `gfmrag/workflow/config/ner_model/`
- `gfmrag/workflow/config/openie_model/`
- `gfmrag/workflow/config/el_model/`
- `gfmrag/workflow/config/graph_constructor/`
- `gfmrag/workflow/config/sft_constructor/`

For parameter descriptions, use the [Config overview](../config/index.md) and the workflow-specific graph-index pages instead of copying the full YAML into this guide.

## When To Re-index

Re-index when:

- raw documents changed
- the graph constructor changed
- the entity-linking or OpenIE setup changed
- you want to rebuild processed QA files from updated raw train/test files

## Common Pitfalls

- If `raw/documents.json` is missing, automatic stage1 construction cannot run.
- The temporary constructor directories depend on the resolved config, so changing component configs will create new fingerprints.
- Training and QA assume the stage1 files and processed samples match the same dataset version.
