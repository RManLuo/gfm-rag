# GFM-RAG Training

You can further fine-tune the pre-trained GFM-RAG model on your own dataset to improve the performance of the model on your specific domain.

## Data Preparation
Please follow the instructions in the [Data Preparation](data_preparation.md) to prepare your dataset in the following structure:
Make sure to have the `train.json` to perform the fine-tuning.

```
data_name/
├── raw/
│   ├── dataset_corpus.json
│   ├── train.json
│   └── test.json # (optional)
└── processed/
    └── stage1/
        ├── kg.txt
        ├── document2entities.json
        ├── train.json
        └── test.json # (optional)
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

You need to create a configuration file for fine-tuning.

??? example "gfmrag/workflow/config/stage2_qa_finetune.yaml"

    ```yaml title="gfmrag/workflow/config/stage2_qa_finetune.yaml"
    --8<-- "gfmrag/workflow/config/stage2_qa_finetune.yaml"
    ```

Details of the configuration parameters are explained in the [GFM-RAG Fine-tuning Configuration][gfm-rag-fine-tuning-configuration] page.


You can fine-tune the pre-trained GFM-RAG model on your dataset using the following command:

??? example "gfmrag/workflow/stage2_qa_finetune.py"

    ```python title="gfmrag/workflow/stage2_qa_finetune.py"
    --8 < --"gfmrag/workflow/stage2_qa_finetune.py"
    ```

```bash
python -m gfmrag.workflow.stage2_qa_finetune
# Multi-GPU training
torchrun --nproc_per_node=4 gfmrag.workflow.stage2_qa_finetune
# Multi-node Multi-GPU training
torchrun --nproc_per_node=4 --nnodes=2 gfmrag.workflow.stage2_qa_finetune
```

You can overwrite the configuration like this:

```bash
python -m gfmrag.workflow.stage2_qa_finetune train.batch_size=4
```

## GFM Pre-training

During pre-training, the GFM model will sample triples from the KG-index `kg.txt` to construct synthetic queries and target entities for training.

!!! tip
	It is only recommended to conduct pre-training when you want to train the model from scratch or when you have a large amount of unlabeled data.

!!! tip
    It is recommended to conduct [fine-tuning][gfm-fine-tuning] after the pre-training to empower the model with the ability to understand user queries and retrieve relevant documents.

A example of the KG-index:

```txt
fred gehrke,was,american football player
fred gehrke,was,executive
fred gehrke,played for,cleveland   los angeles rams
```

You need to create a configuration file for pre-training.

??? example "gfmrag/workflow/config/stage2_kg_pretrain.yaml"

    ```yaml title="gfmrag/workflow/config/stage2_kg_pretrain.yaml"
    --8<-- "gfmrag/workflow/config/stage2_kg_pretrain.yaml"
    ```

Details of the configuration parameters are explained in the [GFM-RAG Pre-training Config][gfm-rag-pre-training-configuration] page.

You can pre-train the GFM-RAG model on your dataset using the following command:

??? example "gfmrag/workflow/stage2_kg_pretrain.py"

    ```python title="gfmrag/workflow/stage2_kg_pretrain.py"
    --8 < --"gfmrag/workflow/stage2_kg_pretrain.py"
    ```

```bash
python -m gfmrag.workflow.stage2_kg_pretrain
# Multi-GPU training
torchrun --nproc_per_node=4 gfmrag.workflow.stage2_kg_pretrain
# Multi-node Multi-GPU training
torchrun --nproc_per_node=4 --nnodes=2 gfmrag.workflow.stage2_kg_pretrain
```

You can overwrite the configuration like this:

```bash
python -m gfmrag.workflow.stage2_kg_pretrain train.batch_size=4
```
