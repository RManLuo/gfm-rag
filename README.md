# GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation
<div align="left">
   <p>
   <a href='https://rmanluo.github.io/gfm-rag/'><img src='https://img.shields.io/badge/Project-Page-Green'>
    </a>
  <a href='https://www.arxiv.org/abs/2502.01113'><img src='https://img.shields.io/badge/arXiv-2502.01113-b31b1b'></a>
  <a href="https://pypi.org/project/gfmrag/">
  </p>
  <p>
  <img src='https://img.shields.io/github/stars/RManLuo/gfm-rag?color=green&style=social' />
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

The GFM-RAG is the first graph foundation powered RAG pipeline that combines the power of graph neural networks to reason over knowledge graphs and retrieve relevant documents for question answering.

![](docs/images/intro.png)

We first build a knowledge graph index (KG-index) from the documents to capture the relationships between knowledge. Then, we feed the query and constructed KG-index into the pre-trained graph foundation model (GFM) retriever to obtain relevant documents for LLM generation. The GFM retriever experiences large-scale training and can be directly applied to unseen datasets without fine-tuning.

For more details, please refer to our [project page](https://rmanluo.github.io/gfm-rag/) and [paper](https://www.arxiv.org/abs/2502.01113).

## 🎉 News
- **[2022-02-06]** We have released the GFM-RAG codebase and 8M pre-trained models. 🚀

## Features

- **Graph Foundation Model (GFM)**: A graph neural network-based retriever that can reason over the KG-index.
- **Knowledge Graph Index**: A knowledge graph index that captures the relationships between knowledge.
- **Efficiency**: The GFM-RAG pipeline is efficient in conduct multi-hop reasoning with single step retrieval.
- **Generalizability**: The GFM-RAG can be directly applied to unseen datasets without fine-tuning.
- **Transferability**: The GFM-RAG can be fine-tuned on your own dataset to improve performance on specific domains.
- **Compatibility**: The GFM-RAG is compatible with arbitrary agent-based framework to conduct multi-step reasoning.
- **Interpretability**: The GFM-RAG can illustrate the captured reasoning paths for better understanding.

## Dependencies

- Python 3.12
- CUDA 12 and above

## Installation

Conda provides an easy way to install the CUDA development toolkit which is required by GFM-RAG

Install packages
```bash
conda create -n gfmrag python=3.12
conda activate gfmrag
conda install cuda-toolkit -c nvidia/label/cuda-12.4.1 # Replace with your desired CUDA version
pip install gfmrag
```

Install other dependencies
```bash
TORCH=$(python -c "import torch; print(torch.__version__)")
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
```

## Quick Start

### Prepare Data

You need to prepare the following files:

- `dataset_corpus.json`: A JSON file containing the entire document corpus.
- `train.json` (optional): A JSON file containing the training data.
- `test.json` (optional): A JSON file containing the test data.

Place your files in the following structure:
```
data_name/
├── raw/
│   ├── dataset_corpus.json
│   ├── train.json # (optional)
│   └── test.json # (optional)
└── processed/ # Output directory
```

#### `dataset_corpus.json`

The `dataset_corpus.json` is a dictionary where each key is the title or unique id of a document and the value is the text of the document.

```json
{
    "Fred Gehrke":
        "Clarence Fred Gehrke (April 24, 1918 – February 9, 2002) was an American football player and executive.  He played in the National Football League (NFL) for the Cleveland / Los Angeles Rams, San Francisco 49ers and Chicago Cardinals from 1940 through 1950.  To boost team morale, Gehrke designed and painted the Los Angeles Rams logo in 1948, which was the first painted on the helmets of an NFL team.  He later served as the general manager of the Denver Broncos from 1977 through 1981.  He is the great-grandfather of Miami Marlin Christian Yelich"
    ,
    "Manny Machado":
        "Manuel Arturo Machado (] ; born July 6, 1992) is an American professional baseball third baseman and shortstop for the Baltimore Orioles of Major League Baseball (MLB).  He attended Brito High School in Miami and was drafted by the Orioles with the third overall pick in the 2010 Major League Baseball draft.  He bats and throws right-handed."
    ,
    ...
 }
```

#### `train.json` and `test.json` (optional)
If you want to train and evaluate the model, you need to provide training and test data in the form of a JSON file. Each entry in the JSON file should contain the following fields:

- `id`: A unique identifier for the example.
- `question`: The question or query.
- `supporting_facts`: A list of supporting facts for the question. Each supporting fact is a list containing the title of the document that can be found in the `dataset_corpus.json` file.

Each entry can also contain additional fields depending on the task. For example:

- `answer`: The answer to the question.

The additional fields will be copied during the following steps of the pipeline.

Example:
```json
[
	{
		"id": "5adf5e285542992d7e9f9323",
		"question": "When was the judge born who made notable contributions to the trial of the man who tortured, raped, and murdered eight student nurses from South Chicago Community Hospital on the night of July 13-14, 1966?",
		"answer": "June 4, 1931",
		"supporting_facts": [
			"Louis B. Garippo",
			"Richard Speck"
		]
	},
	{
		"id": "5a7f7b365542992097ad2f80",
		"question": "Did the Beaulieu Mine or the McIntyre Mines yield gold and copper?",
		"answer": "The McIntyre also yielded a considerable amount of copper",
		"supporting_facts": [
			"Beaulieu Mine",
			"McIntyre Mines"
		]
	}
    ...
]
```

### Index Dataset

You need to create a KG-index [configuration file](gfmrag/workflow/config/stage1_index_dataset.yaml).

Details of the configuration parameters are explained in the [KG-index Configuration](https://rmanluo.github.io/gfm-rag/config/kg_index_config/) page.

```bash
python -m gfmrag.workflow.stage1_index_dataset
```

This method performs two main tasks:

1. Creates and saves knowledge graph related files (`kg.txt` and `document2entities.json`) from the `dataset_corpus.json` file
2. Identify the query entities and supporting entities in training and testing data if available in the raw data directory.

Files created:

- `kg.txt`: Contains knowledge graph triples
- `document2entities.json`: Maps documents to their entities
- `train.json`: Processed training data (if raw exists)
- `test.json`: Processed test data (if raw exists)

Directory structure:
```
root/
└── data_name/
	├── raw/
	│   ├── dataset_corpus.json
	│   ├── train.json (optional)
	│   └── test.json (optional)
	└── processed/
		└── stage1/
			├── kg.txt
			├── document2entities.json
			├── train.json
			└── test.json
```

### GFM-RAG Retrieval

You need to create a [configuration file](gfmrag/workflow/config/stage3_qa_ircot_inference.yaml) for inference.

Details of the configuration parameters are explained in the [GFM-RAG Configuration](https://rmanluo.github.io/gfm-rag/config/gfmrag_retriever_config/) page.

#### Initialize GFMRetriever

You can initialize the GFMRetriever with the following code. It will load the pre-trained GFM-RAG model and the KG-index for retrieval.

```python
import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from gfmrag import GFMRetriever

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="config", config_name="stage3_qa_ircot_inference", version_base=None
)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")

    gfmrag_retriever = GFMRetriever.from_config(cfg)
```

#### Document Retrieval

You can use GFM-RAG retriever to reason over the KG-index and obtain documents for a given query.
```python
docs = retriever.retrieve("Who is the president of France?", top_k=5)
```

#### Question Answering

```python
from hydra.utils import instantiate
from gfmrag.llms import BaseLanguageModel
from gfmrag.prompt_builder import QAPromptBuilder

llm = instantiate(cfg.llm)
qa_prompt_builder = QAPromptBuilder(cfg.qa_prompt)

message = qa_prompt_builder.build_input_prompt(current_query, retrieved_docs)
answer = llm.generate_sentence(message)  # Answer: "Emmanuel Macron"
```

## GFM Fine-tuning

During fine-tuning, the GFM model will be trained on the query-documents pairs `train.json` from the labeled dataset to learn complex relationships for retrieval.

It can be conduced on your own dataset to improve the performance of the model on your specific domain.

A example of the training data:

```json
[
	{
		"id": "5abc553a554299700f9d7871",
		"question": "Kyle Ezell is a professor at what School of Architecture building at Ohio State?",
		"answer": "Knowlton Hall",
		"supporting_facts": [
			"Knowlton Hall",
			"Kyle Ezell"
		],
		"question_entities": [
			"kyle ezell",
			"architectural association school of architecture",
			"ohio state"
		],
		"supporting_entities": [
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
		]
	},
    ...
]
```

You need to create a [configuration file](gfmrag/workflow/config/stage2_qa_finetune.yaml) for fine-tuning.

Details of the configuration parameters are explained in the [GFM-RAG Fine-tuning Configuration](https://rmanluo.github.io/gfm-rag/config/gfmrag_finetune_config/) page.

You can fine-tune the pre-trained GFM-RAG model on your dataset using the following command:

```bash
python -m gfmrag.workflow.stage2_qa_finetune
# Multi-GPU training
torchrun --nproc_per_node=4 gfmrag.workflow.stage2_qa_finetune
# Multi-node Multi-GPU training
torchrun --nproc_per_node=4 --nnodes=2 gfmrag.workflow.stage2_qa_finetune
```

## Workflow

### Stage1: Index Dataset
1. Create KG index for corpus.
2. Prepare QA dataset for training and evaluation (Optional)

```bash
python workflow/stage1_index_dataset.py
```


### Stage2: Deep GraphRAG Training

Unsupervised training on the constructed KG.

```bash
python workflow/stage2_kg_pretrain.py
# Multi-GPU training
torchrun --nproc_per_node=4 workflow/stage2_kg_pretrain.py
```

Supervised training on the QA dataset.

```bash
python workflow/stage2_qa_finetune.py
# Multi-GPU training
torchrun --nproc_per_node=4 workflow/stage2_qa_finetune.py
```

Evaluate retrieval performance of the trained model on QA dataset.

```bash
python workflow/stage2_qa_finetune.py train.num_epoch=0 datasets.train_names=[] checkpoint=save_models/qa_ultra_512_train_60000_w_pre-train/model.pth
# Multi-GPU evaluation
torchrun --nproc_per_node=4 workflow/stage2_qa_finetune.py train.num_epoch=0 datasets.train_names=[] checkpoint=save_models/qa_ultra_512_train_60000_w_pre-train/model.pth
```

### Stage3: QA Reasoning

#### Single Step QA Reasoning
```bash
python workflow/stage3_qa_inference.py
# Multi-GPU retrieval
torchrun --nproc_per_node=4 workflow/stage3_qa_inference.py
```

hotpotqa
```bash
torchrun --nproc_per_node=4 workflow/stage3_qa_inference.py dataset.data_name=hotpotqa_test qa_prompt=hotpotqa qa_evaluator=hotpotqa
```

musique
```bash
torchrun --nproc_per_node=4 workflow/stage3_qa_inference.py dataset.data_name=musique_test qa_prompt=musique qa_evaluator=musique
```

2Wikimultihopqa
```bash
torchrun --nproc_per_node=4 workflow/stage3_qa_inference.py dataset.data_name=2wikimultihopqa_test qa_prompt=2wikimultihopqa qa_evaluator=2wikimultihopqa
```
#### Multi Step IRCOT QA Reasoning
```bash
python workflow/stage3_qa_ircot_inference.py
```

hotpotqa
```bash
python workflow/stage3_qa_ircot_inference.py qa_prompt=hotpotqa qa_evaluator=hotpotqa agent_prompt=hotpotqa_ircot dataset.data_name=hotpotqa_test test.max_steps=2
```

musique
```bash
python workflow/stage3_qa_ircot_inference.py qa_prompt=musique qa_evaluator=musique agent_prompt=musique_ircot dataset.data_name=musique_test test.max_steps=4
```

2Wikimultihopqa
```bash
python workflow/stage3_qa_ircot_inference.py qa_prompt=2wikimultihopqa qa_evaluator=2wikimultihopqa agent_prompt=2wikimultihopqa_ircot dataset.data_name=2wikimultihopqa_test test.max_steps=2
```

### Visualize Paths
```bash
python workflow/experiments/visualize_path.py dataset.data_name=hotpotqa_test
```
