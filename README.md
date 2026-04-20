# GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation
<div align="left">
   <p>
   <a href='https://rmanluo.github.io/gfm-rag/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
   <a href='https://www.arxiv.org/abs/2502.01113'><img src='https://img.shields.io/badge/arXiv-2502.01113-b31b1b'></a>
    <a href='https://www.arxiv.org/abs/2509.24276'><img src='https://img.shields.io/badge/arXiv-2509.24276-b31b1b'></a>
   <a href='https://huggingface.co/collections/rmanluo/gfm-rag-67a1ef7bfe097a938d8848dc'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GFM--RAG-blue'></a>
    <a href='https://huggingface.co/collections/rmanluo/g-reasoner'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-G--Reasoner-blue'></a>
  <a href="https://pypi.org/project/gfmrag/">
  </p>
  <p>
  <img src='https://img.shields.io/github/stars/RManLuo/gfm-rag?color=green&style=social' />
  <a href="https://pypi.org/project/gfmrag/">
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/gfmrag">
  </a>
  <a href="https://pypi.org/project/gfmrag/">
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/gfmrag">
  </a>
  <a href="https://github.com/RManLuo/gfm-rag/issues">
    <img alt="GitHub Issues" src="https://img.shields.io/github/issues/RManLuo/gfm-rag">
  </a>
  <a href="https://github.com/RManLuo/gfm-rag/discussions">
    <img alt="GitHub Discussions" src="https://img.shields.io/github/discussions/RManLuo/gfm-rag">
  </a>
  </p>
</div>

[\[中文解读\]](https://rman.top/2025/03/01/gfm-rag/)

The GFM-RAG is the first graph foundation model-powered RAG pipeline that combines the power of graph neural networks to reason over graphs and retrieve relevant documents for question answering.

![](docs/images/g-reasoner.png)

We first build a graph-index from the documents to capture the relationships between knowledge. Then, we feed the query and constructed graph-index into the pre-trained graph foundation model (GFM) retriever to obtain relevant documents for LLM generation. The GFM retriever experiences large-scale training and can be directly applied to unseen datasets without fine-tuning.

GFM-RAG is designed to be efficient and generalizable. You can bring your own dataset and directly apply the pre-trained GFM retriever to obtain relevant documents for question answering. You can also fine-tune the GFM retriever on your own dataset to improve performance on specific domains.

For more details, please refer to our [project](https://github.com/RManLuo/gfm-rag) and papers: [GFM-RAG](https://www.arxiv.org/abs/2502.01113), [G-reasoner](https://arxiv.org/abs/2509.24276).


## 🎉 News
- **[2026-04-20]** We have released the G-reasoner codebase and a [34M pre-trained model](https://huggingface.co/rmanluo/G-reasoner-34M). 🚀
- **[2026-01-27]** We are excited to share that [G-reasoner](https://arxiv.org/abs/2509.24276) has been accepted by [ICLR 2026](https://iclr.cc/Conferences/2026).
- **[2025-10-01]** Checkout our latest progress: [G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](https://arxiv.org/abs/2509.24276). Code and model will be updated soon.
- **[2025-09-19]** We are excited to share that [GFM-RAG](https://www.arxiv.org/abs/2502.01113) has been accepted by [NeurIPS 2025](https://neurips.cc/Conferences/2025).
- **[2025-06-03]** We have released a new version of [GFM-RAG (2025-06-03)](https://huggingface.co/rmanluo/GFM-RAG-8M/commit/62cf6398c5875af1c4e04bbb35e4c3b21904d4ac) which is pre-trained on 286 KGs. Performance comparison with the previous version can be found in [CHANGELOG](docs/CHANGELOG.md).
- **[2025-02-06]** We have released the GFM-RAG codebase and a [8M pre-trained model](https://huggingface.co/rmanluo/GFM-RAG-8M). 🚀

## Features

- **Graph Foundation Model (GFM)**: A graph neural network-based retriever that can reason over the graph-index.
- **Universal Graph Index**: A universal graph index that can represent various types of structural knowledge such as Knowledge Graphs, Document Graphs, and Hierarchical Graphs.
- **Efficiency**: The GFM-RAG pipeline is efficient in conducting multi-hop reasoning with single-step retrieval.
- **Generalizability**: The GFM-RAG can be directly applied to unseen datasets without fine-tuning.
- **Transferability**: The GFM-RAG can be fine-tuned on your own dataset to improve performance on specific domains.
- **Compatibility**: The GFM-RAG is compatible with arbitrary agent-based framework to conduct multi-step reasoning.
- **Interpretability**: The GFM-RAG can illustrate the captured reasoning paths for better understanding.

## Dependencies

- Python 3.12
- CUDA 12 and above (CUDA 12.6.3 is recommended)

## Installation

Conda provides an easy way to install the CUDA development toolkit which is required by GFM-RAG

Install packages
```bash
conda create -n gfmrag python=3.12
conda activate gfmrag
conda install cuda-toolkit -c nvidia/label/cuda-12.6.3 # Replace with your desired CUDA version
pip install gfmrag
```

## Quick Start

> [!NOTE]
> Read the full documentation at: https://rmanluo.github.io/gfm-rag/

GFM-RAG provides a **unified graph interface**: if you already have a graph that conforms to the three-file format (`nodes.csv` / `relations.csv` / `edges.csv`), you can **skip the index-building step entirely** and use it directly for retrieval and reasoning — regardless of how the graph was constructed.

There are two starting points:

- **Path A — Start from raw documents** (steps 1–3 below): provide `raw/documents.json` and let `GFMRetriever.from_index(...)` build the graph automatically.
- **Path B — Bring your own graph** (step 1b below): place pre-built graph files under `processed/stage1/` and `GFMRetriever.from_index(...)` will load them directly without rebuilding.

See [Data Format](docs/workflow/data_format.md) for the full schema of both paths.

---

### Path A: Start From Raw Documents

#### 1. Create A Minimal Dataset

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

#### 2. Initialize `GFMRetriever`

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

#### 3. Retrieve Documents

```python
results = retriever.retrieve(
    "Who is the president of France?",
    top_k=5,
)

for item in results["document"]:
    print(item["id"], item["score"])
```

---

### Path B: Bring Your Own Graph

If you already have a graph — for example, an existing Knowledge Graph, a graph produced by another pipeline, or a graph you built manually — you can use it directly **without running the index-building step**, as long as it conforms to the GFM-RAG graph format.

#### 1. Place Pre-built Graph Files

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

See [Data Format](docs/workflow/data_format.md) for the full schema and examples.

#### 2. Initialize `GFMRetriever` and Retrieve

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

## GFM Fine-tuning

During fine-tuning, the GFM model will be trained on the query-documents pairs `train.json` from the labeled dataset to learn complex relationships for retrieval.

It can be conducted on your own dataset to improve the performance of the model on your specific domain.

An example of the training data:

```json
[
  {
    "id": "5abc553a554299700f9d7871",
    "question": "Kyle Ezell is a professor at what School of Architecture building at Ohio State?",
    "answer": "Knowlton Hall",
    "supporting_documents": ["Knowlton Hall", "Kyle Ezell"],
    "start_nodes": {
      "entity": [
        "kyle ezell",
        "architectural association school of architecture",
        "ohio state"
      ]
    },
    "target_nodes": {
      "document": ["Knowlton Hall", "Kyle Ezell"],
      "entity": [
        "10 million donation",
        "2004",
        "architecture",
        "austin e  knowlton",
        "austin e  knowlton school of architecture",
        "bachelor s in architectural engineering"
      ]
    }
  },
    ...

```

You need to create a [configuration file](gfmrag/workflow/config/gfm_reasoner/sft_training.yaml) for fine-tuning.

> [!NOTE]
> We have already released the two pre-trained model checkpoint [GFM-RAG-8M](https://huggingface.co/rmanluo/GFM-RAG-8M) and [G-reasoner-34M](https://huggingface.co/rmanluo/G-reasoner-34M), which can be used for further finetuning. The model will be automatically downloaded by specifying it in the configuration.
> ```yaml
> load_model_from_pretrained: rmanluo/G-reasoner-34M # or rmanluo/GFM-RAG-8M
> ```

Details of the configuration parameters are explained in the [Training](docs/workflow/training.md) page.

You can fine-tune the pre-trained GFM-RAG model on your dataset using the following command:

```bash
python -m gfmrag.workflow.sft_training --config-path config/gfm_reasoner
# Multi-GPU training
torchrun --nproc_per_node=4 -m gfmrag.workflow.sft_training --config-path config/gfm_reasoner
# Multi-node Multi-GPU training
torchrun --nproc_per_node=4 --nnodes=2 -m gfmrag.workflow.sft_training --config-path config/gfm_reasoner
```

## Reproduce Results reported in the paper

Please refer to the [Experiment](docs/experiment/overview.md) section for detailed reproduction instructions for both [GFM-RAG](docs/experiment/gfm_rag.md) and [G-Reasoner](docs/experiment/g_reasoner.md).

## Acknowledgements

We greatly appreciate the following repositories for their help to this project:

* [DeepGraphLearning/ULTRA](https://github.com/DeepGraphLearning/ULTRA): The ULTRA model is used as the base GNN model for the GFM retriever.
* [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG): We get great inspiration from the KG construction process of HippoRAG.
* [microsoft/graphrag](https://github.com/microsoft/graphrag): We get great inspiration from the project design of GraphRAG.

## Citation

If you find this repository helpful, please consider citing our paper:

```bibtex
@inproceedings{
	luo2026greasoner,
	title={G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge},
	author={Linhao Luo and Zicheng Zhao and Junnan Liu and Zhangchi Qiu and Junnan Dong and Serge Panev and Chen Gong and Thuy-Trang Vu and Gholamreza Haffari and Dinh Phung and Alan Wee-Chung Liew and Shirui Pan},
	booktitle={The Fourteenth International Conference on Learning Representations},
	year={2026},
	url={https://openreview.net/forum?id=zJm9nmoahk}
}
```

```bibtex
@article{luo2025gfmrag,
  title={GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation},
  author={Luo, Linhao and Zhao, Zicheng and Haffari, Gholamreza and Phung, Dinh and Gong, Chen and Pan, Shirui},
  journal={NeurIPS 2025},
  year={2025}
}
```
