# GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation
<div align="left">
   <p>
   <a href='https://rmanluo.github.io/gfm-rag/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
   <a href='https://www.arxiv.org/abs/2502.01113'><img src='https://img.shields.io/badge/arXiv-2502.01113-b31b1b'></a>
   <a href='https://huggingface.co/collections/rmanluo/gfm-rag-67a1ef7bfe097a938d8848dc'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GFM--RAG-blue'></a>
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

The GFM-RAG is the first graph foundation model-powered RAG pipeline that combines the power of graph neural networks to reason over knowledge graphs and retrieve relevant documents for question answering.

![](docs/images/intro.png)

We first build a knowledge graph index (KG-index) from the documents to capture the relationships between knowledge. Then, we feed the query and constructed KG-index into the pre-trained graph foundation model (GFM) retriever to obtain relevant documents for LLM generation. The GFM retriever experiences large-scale training and can be directly applied to unseen datasets without fine-tuning.

For more details, please refer to our [project page](https://rmanluo.github.io/gfm-rag/) and [paper](https://www.arxiv.org/abs/2502.01113).

## 🎉 News
- **[2025-06-03]** We have released a new version of [GFM-RAG (2025-06-03)](https://huggingface.co/rmanluo/GFM-RAG-8M/commit/62cf6398c5875af1c4e04bbb35e4c3b21904d4ac) which is pre-trained on 286 KGs. Performance comparison with the previous version can be found in [CHANGELOG](docs/CHANGELOG.md).
- **[2025-02-06]** We have released the GFM-RAG codebase and a [8M pre-trained model](https://huggingface.co/rmanluo/GFM-RAG-8M). 🚀

## Features

- **Graph Foundation Model (GFM)**: A graph neural network-based retriever that can reason over the KG-index.
- **Knowledge Graph Index**: A knowledge graph index that captures the relationships between knowledge.
- **Efficiency**: The GFM-RAG pipeline is efficient in conducting multi-hop reasoning with single-step retrieval.
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

## Quick Start

> [!NOTE]
> Read the full documentation at: https://rmanluo.github.io/gfm-rag/

### Prepare Data

We have provided the testing split and an example of the training data in [here](https://drive.google.com/drive/folders/11xuSKD20c1X0bJRZRVvRc8ocW7wgX7Rw?usp=sharing).

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
If you want to train and evaluate the model, you need to provide training and testing data in the form of a JSON file. Each entry in the JSON file should contain the following fields:

- `id`: A unique identifier for the example.
- `question`: The question or query.
- `supporting_facts`: A list of supporting facts relevant to the question. Each supporting fact is a list containing the title of the document that can be found in the `dataset_corpus.json` file.

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

1. Create and save knowledge graph related files (`kg.txt` and `document2entities.json`) from the `dataset_corpus.json` file
2. Identify the query entities and supporting entities in training and testing data if available in the raw data directory.

Files created:

- `kg.txt`: Contains knowledge graph triples
- `document2entities.json`: Maps documents to their entities
- `train.json`: Processed training data (if raw exists)
- `test.json`: Processed testing data (if raw exists)

Directory structure:
```
data_name/
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

> [!NOTE]
> We have already released the pre-trained model [here](https://huggingface.co/rmanluo/GFM-RAG-8M), which can be used directly for retrieval. The model will be automatically downloaded by specifying it in the configuration.
> ```yaml
> graph_retriever:
>     model_path: rmanluo/GFM-RAG-8M
> ```

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

It can be conducted on your own dataset to improve the performance of the model on your specific domain.

An example of the training data:

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

> [!NOTE]
> We have already released the pre-trained model checkpoint [here](https://huggingface.co/rmanluo/GFM-RAG-8M), which can be used for further finetuning. The model will be automatically downloaded by specifying it in the configuration.
> ```yaml
> checkpoint: rmanluo/GFM-RAG-8M
> ```

Details of the configuration parameters are explained in the [GFM-RAG Fine-tuning Configuration](https://rmanluo.github.io/gfm-rag/config/gfmrag_finetune_config/) page.

You can fine-tune the pre-trained GFM-RAG model on your dataset using the following command:

```bash
python -m gfmrag.workflow.stage2_qa_finetune
# Multi-GPU training
torchrun --nproc_per_node=4 -m gfmrag.workflow.stage2_qa_finetune
# Multi-node Multi-GPU training
torchrun --nproc_per_node=4 --nnodes=2 -m gfmrag.workflow.stage2_qa_finetune
```

## Reproduce Results reported in the paper

### Download datasets

We are working on releasing the full training datasets.

We have provided the testing split and an example of the training data in [here](https://drive.google.com/drive/folders/11xuSKD20c1X0bJRZRVvRc8ocW7wgX7Rw?usp=sharing).

Download the datasets and put them under the `data` directory.

```text
data/
├── 2wikimultihopqa_test
│   ├── processed
│   └── raw
├── hotpotqa_test
│   ├── processed
│   └── raw
├── hotpotqa_train_example
│   ├── processed
│   └── raw
└── musique_test
    ├── processed
    └── raw
```


### Index Dataset

We have provided the indexed testing datasets in the `data/*/processed/stage1` directory. You can build the index for the testing dataset with the following command:

```bash
# Build the index for testing dataset
N_GPU=1
DATA_ROOT="data"
DATA_NAME_LIST="hotpotqa_test 2wikimultihopqa_test musique_test"
for DATA_NAME in ${DATA_NAME_LIST}; do
   python -m gfmrag.workflow.stage1_index_dataset \
   dataset.root=${DATA_ROOT} \
   dataset.data_name=${DATA_NAME}
done
```

Full script is available at [scripts/stage1_data_index.sh](scripts/stage1_data_index.sh).

### GFM Training

Unsupervised training on the constructed KG.

```bash
python -m gfmrag.workflow.stage2_kg_pretrain
# Multi-GPU training
torchrun --nproc_per_node=4 -m gfmrag.workflow.stage2_kg_pretrain
```

Full script is available at [scripts/stage2_pretrain.sh](scripts/stage2_pretrain.sh).

Supervised training on the QA dataset.

```bash
python -m gfmrag.workflow.stage2_qa_finetune
# Multi-GPU training
torchrun --nproc_per_node=4 -m gfmrag.workflow.stage2_qa_finetune
```

Full script is available at [scripts/stage2_finetune.sh](scripts/stage2_finetune.sh).

### Retrieval Evaluation

```bash
N_GPU=4
DATA_ROOT="data"
checkpoints=rmanluo/GFM-RAG-8M # Or the path to your checkpoints
torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.stage2_qa_finetune \
    train.checkpoint=${checkpoints} \
    datasets.cfgs.root=${DATA_ROOT} \
    datasets.train_names=[] \
    train.num_epoch=0
```

### QA Reasoning

#### Single Step QA Reasoning
```bash
# Batch inference for QA on the test set.
N_GPU=4
DATA_ROOT="data"
DATA_NAME="hotpotqa" # hotpotqa musique 2wikimultihopqa
LLM="gpt-4o-mini"
DOC_TOP_K=5
N_THREAD=10
torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.stage3_qa_inference \
    dataset.root=${DATA_ROOT} \
    qa_prompt=${DATA_NAME} \
    qa_evaluator=${DATA_NAME} \
    llm.model_name_or_path=${LLM} \
    test.n_threads=${N_THREAD} \
    test.top_k=${DOC_TOP_K} \
    dataset.data_name=${DATA_NAME}_test
```

hotpotqa
```bash
torchrun --nproc_per_node=4 -m gfmrag.workflow.stage3_qa_inference dataset.data_name=hotpotqa_test qa_prompt=hotpotqa qa_evaluator=hotpotqa
```

musique
```bash
torchrun --nproc_per_node=4 -m gfmrag.workflow.stage3_qa_inference dataset.data_name=musique_test qa_prompt=musique qa_evaluator=musique
```

2Wikimultihopqa
```bash
torchrun --nproc_per_node=4 -m gfmrag.workflow.stage3_qa_inference dataset.data_name=2wikimultihopqa_test qa_prompt=2wikimultihopqa qa_evaluator=2wikimultihopqa
```
#### Multi Step IRCOT QA Reasoning
```bash
# IRCoT + GFM-RAG inference on QA tasks
N_GPU=1
DATA_ROOT="data"
DATA_NAME="hotpotqa" # hotpotqa musique 2wikimultihopqa
LLM="gpt-4o-mini"
MAX_STEPS=3
MAX_SAMPLE=10
python -m gfmrag.workflow.stage3_qa_ircot_inference \
    dataset.root=${DATA_ROOT} \
    llm.model_name_or_path=${LLM} \
    qa_prompt=${DATA_NAME} \
    qa_evaluator=${DATA_NAME} \
    agent_prompt=${DATA_NAME}_ircot \
    test.max_steps=${MAX_STEPS} \
    test.max_test_samples=${MAX_SAMPLE} \
    dataset.data_name=${DATA_NAME}_test
```

hotpotqa
```bash
python -m gfmrag.workflow.stage3_qa_ircot_inference qa_prompt=hotpotqa qa_evaluator=hotpotqa agent_prompt=hotpotqa_ircot dataset.data_name=hotpotqa_test test.max_steps=2
```

musique
```bash
python -m gfmrag.workflow.stage3_qa_ircot_inference qa_prompt=musique qa_evaluator=musique agent_prompt=musique_ircot dataset.data_name=musique_test test.max_steps=4
```

2Wikimultihopqa
```bash
python -m gfmrag.workflow.stage3_qa_ircot_inference qa_prompt=2wikimultihopqa qa_evaluator=2wikimultihopqa agent_prompt=2wikimultihopqa_ircot dataset.data_name=2wikimultihopqa_test test.max_steps=2
```

### Path Interpretations
```bash
python -m gfmrag.workflow.experiments.visualize_path dataset.data_name=hotpotqa_test
```

## Acknowledgements

We greatly appreciate the following repositories for their help to this project:

* [DeepGraphLearning/ULTRA](https://github.com/DeepGraphLearning/ULTRA): The ULTRA model is used as the base GNN model for the GFM retriever.
* [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG): We get great inspiration from the KG construction process of HippoRAG.
* [microsoft/graphrag](https://github.com/microsoft/graphrag): We get great inspiration from the project design of GraphRAG.

## Citation

If you find this repository helpful, please consider citing our paper:

```bibtex
@article{luo2025gfmrag,
  title={GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation},
  author={Luo, Linhao and Zhao, Zicheng and Haffari, Gholamreza and Phung, Dinh and Gong, Chen and Pan, Shirui},
  journal={arXiv preprint arXiv:2502.01113},
  year={2025}
}
```
