# GFM-RAG

Welcome to the documentation for GFM-RAG project.

## Overview

The GFM-RAG is the first graph foundation model-powered RAG pipeline that combines the power of graph neural networks to reason over graphs and retrieve relevant documents for question answering.

![](images/g-reasoner.png)

We first build a graph-index from the documents to capture the relationships between knowledge. Then, we feed the query and constructed graph-index into the pre-trained graph foundation model (GFM) retriever to obtain relevant documents for LLM generation. The GFM retriever experiences large-scale training and can be directly applied to unseen datasets without fine-tuning.

GFM-RAG is designed to be efficient and generalizable. You can bring your own dataset and directly apply the pre-trained GFM retriever to obtain relevant documents for question answering. You can also fine-tune the GFM retriever on your own dataset to improve performance on specific domains.

For more details, please refer to our [project](https://github.com/RManLuo/gfm-rag) and papers: [GFM-RAG](https://www.arxiv.org/abs/2502.01113), [G-reasoner](https://arxiv.org/abs/2509.24276).




## Features

- **Graph Foundation Model (GFM)**: A graph neural network-based retriever that can reason over the graph-index.
- **Universal Graph Index**: A universal graph index that can represent various types of structural knowledge such as Knowledge Graphs, Document Graphs, and Hierarchical Graphs.
- **Efficiency**: The GFM-RAG pipeline is efficient in conducting multi-hop reasoning with single-step retrieval.
- **Generalizability**: The GFM-RAG can be directly applied to unseen datasets without fine-tuning.
- **Transferability**: The GFM-RAG can be fine-tuned on your own dataset to improve performance on specific domains.
- **Compatibility**: The GFM-RAG is compatible with arbitrary agent-based framework to conduct multi-step reasoning.
- **Interpretability**: The GFM-RAG can illustrate the captured reasoning paths for better understanding.


## Choose Your Path

- [Quick Start](getting_started/quick_start.md): shortest runnable path for a new user
- [Workflow](workflow/data_format.md): general usage from data formatting through training
- [Experiment](experiment/overview.md): script-first paper reproduction for `GFM-RAG` and `G-reasoner`
- [API Reference](api/gfmrag_retriever.md): developer-facing classes and modules

## How GFM-RAG And G-reasoner Relate

- [GFM-RAG](https://www.arxiv.org/abs/2502.01113) is the original graph foundation model retriever and remains the baseline workflow and published checkpoint family.
- [G-reasoner](https://arxiv.org/abs/2509.24276) is the latest version of the graph foundation model retriever, which has new architecture and better performance.
- Both lines share the same indexing, dataset, QA, and documentation structure in this site.

## Fastest Path

If you want to run something with minimal setup:

1. Follow [Install](install.md).
2. Prepare a tiny dataset with `raw/documents.json`.
3. Start from [Quick Start](getting_started/quick_start.md).
