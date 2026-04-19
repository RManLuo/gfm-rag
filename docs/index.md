# Overview

`G-reasoner` is the current main model line in this repository. `GFM-RAG` remains available as the original graph retrieval baseline, shared codebase, and experiment reproduction target.

![](images/intro.png)

This repository supports four common workflows:

- build a graph index from raw documents
- retrieve supporting documents with `GFMRetriever.from_index(...)`
- run QA or agent reasoning on top of retrieval results
- train and evaluate retrievers on indexed datasets

## Choose Your Path

- [Quick Start](getting_started/quick_start.md): shortest runnable path for a new user
- [Workflow](workflow/data_format.md): general usage from data formatting through training
- [Experiment](experiment/overview.md): script-first paper reproduction for `GFM-RAG` and `G-reasoner`
- [API Reference](api/gfmrag_retriever.md): developer-facing classes and modules

## How GFM-RAG And G-reasoner Relate

- `GFM-RAG` is the original graph foundation model retriever and remains the baseline workflow and published checkpoint family.
- `G-reasoner` extends the same repository with a newer training and reproduction path.
- Both lines share the same indexing, dataset, QA, and documentation structure in this site.

## Fastest Path

If you want to run something with minimal setup:

1. Follow [Install](install.md).
2. Prepare a tiny dataset with `raw/documents.json`.
3. Start from [Quick Start](getting_started/quick_start.md).

For reproduction scripts instead of the general API, go directly to [Experiment](experiment/overview.md).
