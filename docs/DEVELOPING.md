# GFM-RAG Development

This page is for contributors maintaining the repository and documentation site.

## Requirements

| Name | Installation | Purpose |
| --- | --- | --- |
| Python 3.12 | <https://www.python.org/downloads/> | Runtime and package development |
| Poetry | <https://python-poetry.org/docs/#installation> | Dependency management and packaging |
| CUDA toolkit | NVIDIA or conda packages | Builds the `rspmm` extension and supports GPU workflows |

## Local Setup

```bash
poetry install
pre-commit install
```

If CUDA is installed manually, make sure `CUDA_HOME` is set:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
```

## Repository Structure

```text
gfm-rag/
├── docs/
│   ├── experiment/          # Script-first paper reproduction guides
│   ├── getting_started/     # Minimal user entrypoints
│   ├── workflow/            # General user workflow guides
│   ├── config/              # Configuration reference
│   ├── api/                 # API reference pages
│   ├── DEVELOPING.md
│   └── CHANGELOG.md
├── gfmrag/
│   ├── gfmrag_retriever.py
│   ├── graph_index_construction/
│   ├── graph_index_datasets/
│   ├── models/
│   ├── trainers/
│   ├── workflow/
│   │   ├── config/
│   │   │   ├── gfm_rag/
│   │   │   └── gfm_reasoner/
│   │   ├── index_dataset.py
│   │   ├── sft_training.py
│   │   ├── qa.py
│   │   └── qa_ircot_inference.py
│   ├── evaluation/
│   ├── llms/
│   └── utils/
├── scripts/
│   ├── gfm-rag/
│   └── g-reasoner/
├── tests/
├── mkdocs.yml
├── pyproject.toml
└── poetry.lock
```

## Common Commands

Serve the docs locally:

```bash
mkdocs serve
```

Run repository checks:

```bash
pre-commit run --all-files
```

Build the package:

```bash
poetry build
```
