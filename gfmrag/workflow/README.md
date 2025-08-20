# MAJOR CHANGE

## Download the datasets in new graph format

Dataset Example: https://drive.google.com/file/d/1Hd_3-0DzEoYounRrU_0OBEfS_cZ014eA/view?usp=sharing

Full dataset in new format: https://drive.google.com/file/d/1sWx7TC3S9XpVKMcJRfDWe_NxRfvtYeRf/view?usp=drive_link

## Explanation of the new graph format

## **📂 Graph Index File Structure**

Graph data consists of three CSV files:

- `nodes.csv`: Defines nodes and their attributes.
- `relations.csv`: Defines relationships and their attributes.
- `edges.csv`: Defines edges between nodes and their attributes.

---

### **✅ `nodes.csv` File Format**

| **Field**   | **Type** | **Description** |
| ----------- | -------- | --------------- |
| name        | str      | Node name       |
| type        | str      | Node type, e.g., entity or document |
| attributes  | dict     | (Optional) Additional node attributes, stored as a JSON string |

> The `attributes` field is a JSON-formatted string used to store arbitrary structured attributes.

### **Example Content (`nodes.csv`):**
```
name,type,attributes
"Barack Obama","entity","{}"
"White House","entity","{}"
"Obama Biography","document","{'title': 'The Life of Barack Obama', 'published_year': 2020}"
```

### Text attributes
```
name: Obama Biography
type: document
title: The Life of Barack Obama
published_year: 2020
```

```
name: Barack Obama
type: entity
```

---

### **✅ `relations.csv` File Format**

| **Field**   | **Type** | **Description** |
| ----------- | -------- | --------------- |
| name        | str      | Relation name   |
| attributes  | dict     | (Optional) Additional relation attributes, stored as a JSON string |

**Example Content (`relations.csv`):**
```
name,attributes
lived_in,"{'description': 'A person has a habitual presence in a specific location.'}"
mentioned_in,"{'description': 'An entity is mentioned in the document'}"
```

### Text attributes
```
name: lived_in
description: A person has a habitual presence in a specific location.
```

---

### **✅ `edges.csv` File Format**

| **Field**   | **Type** | **Description** |
| ----------- | -------- | --------------- |
| source      | str      | The `name` field of the source node |
| relation    | str      | The `name` field of the relation |
| target      | str      | The `name` field of the target node |
| attributes  | dict     | (Optional) Additional edge attributes, stored as a JSON string |

> `source` and `target` must appear in the `name` column of `nodes.csv`.
> `relation` must appear in the `name` column of `relations.csv`.

### **Example Content (`edges.csv`):**
```
source,relation,target,attributes
"Barack Obama","lived_in","White House","{'start_year': 2009, 'end_year': 2017}"
"Barack Obama","mentioned_in","Obama Biography",{}
```

### Text attributes
```python
start_year: 2009
end_year: 2017
```

---

## **✅ Complete Example Graph Structure**

**Nodes (`nodes.csv`):**

| **name**         | **type**   | **attributes** |
| ---------------- | ---------- | --------------- |
| Barack Obama     | entity     | {"birth_date": "1961-08-04", "nationality": "USA"} |
| White House      | entity     | {"location": "Washington, D.C."} |
| Obama Biography  | document   | {"title": "The Life of Barack Obama", "published_year": 2020} |
| Summary_node_1   | summary    | {"summary": xxx, "title": xxx} |

**Relations (`relations.csv`):**

| **name**       | **attributes** |
| -------------- | --------------- |
| lived_in       | {"description": "A person has a habitual presence in a specific location."} |
| mentioned_in   | {"description": "An entity is mentioned in the document"} |

**Edges (`edges.csv`):**

| **source**     | **relation**  | **target**       | **attributes** |
| -------------- | ------------- | ---------------- | --------------- |
| Barack Obama   | lived_in      | White House      | {"start_year": 2009, "end_year": 2017} |
| Barack Obama   | mentioned_in  | Obama Biography  | {} |

---

## Training and Test Data: `train.json` and `test.json`

The `train.json` and `test.json` files contain processed training and test data in the following format:

| **Field**       | **Type**         | **Description** |
| --------------- | ---------------- | --------------- |
| id              | str              | A unique identifier for the example |
| question        | str              | The question or query |
| start_nodes     | dict[type][list] | A dictionary of starting nodes grouped by type. Key: node type, Value: list of node names |
| target_nodes    | dict[type][list] | A dictionary of target nodes grouped by type. Key: node type, Value: list of node names |
| Additional fields | Any            | Any extra fields copied from the raw data |

### Example
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
        "the ohio state university",
        "the ohio state university in 1931",
        "american urban planning practitioner",
        "expressing local culture",
        "knowlton school",
        "kyle ezell",
        "lawrenceburg  tennessee",
        "professor",
        "the ohio state university",
        "theorist",
        "undergraduate planning program",
        "vibrant downtowns",
        "writer"
      ]
    }
  }
]
```

## GFM-RAG
The old model.


### Retrieval

Download the model with modified configuration. I have changed the pre-train config into new format and upload to `save_models/GFM-RAG-8M`

```bash
N_GPU=4
DATA_ROOT="data_full/new_graph_interface"
checkpoints=save_models/GFM-RAG-8M # I have changed the pre-train config into new format
torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.sft_training --config-path config/gfm_rag \
    load_model_from_pretrained=${checkpoints} \
    datasets.cfgs.root=${DATA_ROOT} \
    datasets.train_names=[] \
    trainer.args.do_train=false \
    +trainer.args.eval_batch_size=1
```

### Index Datasets
```bash
python -m gfmrag.workflow.index_dataset --config-path config/gfm_rag
```

### KGC Training
```bash
N_GPU=4
DATA_ROOT="data_full/new_graph_interface"
N_EPOCH=1
BATCH_PER_EPOCH=30000
START_N=0
END_N=1
BATCH_SIZE=4
DATA_NAME_LIST="hotpotqa_train musique_train 2wikimultihopqa_train"
TRAIN_DATA_NAME_LIST=""
for DATA_NAME in ${DATA_NAME_LIST}; do
    for i in $(seq ${START_N} ${END_N}); do
        TRAIN_DATA_NAME_LIST="${TRAIN_DATA_NAME_LIST},${DATA_NAME}${i}"
    done
done
torchrun --nproc-per-node=${N_GPU} -m gfmrag.workflow.kgc_training --config-path config/gfm_rag \
    datasets.train_names=[${TRAIN_DATA_NAME_LIST}] \
    datasets.cfgs.root=${DATA_ROOT} \
    train.fast_test=5000 \
    trainer.args.num_epoch=${N_EPOCH} \
    trainer.args.max_steps_per_epoch=${BATCH_PER_EPOCH} \
    trainer.args.train_batch_size=${BATCH_SIZE}
```

### SFT Finetune

``` bash
DATA_ROOT="data_full/new_graph_interface"
N_GPU=4
N_EPOCH=10
START_N=0
END_N=1
BATCH_SIZE=4
DATA_NAME_LIST="hotpotqa_train musique_train 2wikimultihopqa_train"
TRAIN_DATA_NAME_LIST=""
for DATA_NAME in ${DATA_NAME_LIST}; do
    for i in $(seq ${START_N} ${END_N}); do
        TRAIN_DATA_NAME_LIST="${TRAIN_DATA_NAME_LIST},${DATA_NAME}${i}"
    done
done
TRAIN_DATA_NAME_LIST=${TRAIN_DATA_NAME_LIST:1}
echo "TRAIN_DATA_NAME_LIST: [${TRAIN_DATA_NAME_LIST}]"
torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.sft_training --config-path config/gfm_rag \
    datasets.train_names=[${TRAIN_DATA_NAME_LIST}] \
    datasets.cfgs.root=${DATA_ROOT} \
    trainer.args.num_epoch=${N_EPOCH} \
    trainer.args.train_batch_size=${BATCH_SIZE}
```


### QA inference

WIP

### IRCOT QA inference

WIP

## GFM-Reasoner

The new model.

### Index Datasets
```bash
python -m gfmrag.workflow.index_dataset --config-path config/gfm_reasoner
```

### KGC Training
```bash
N_GPU=4
DATA_ROOT="data_full/new_graph_interface"
N_EPOCH=1
BATCH_PER_EPOCH=30000
START_N=0
END_N=1
BATCH_SIZE=4
DATA_NAME_LIST="hotpotqa_train musique_train 2wikimultihopqa_train"
TRAIN_DATA_NAME_LIST=""
for DATA_NAME in ${DATA_NAME_LIST}; do
    for i in $(seq ${START_N} ${END_N}); do
        TRAIN_DATA_NAME_LIST="${TRAIN_DATA_NAME_LIST},${DATA_NAME}${i}"
    done
done
torchrun --nproc-per-node=${N_GPU} -m gfmrag.workflow.kgc_training --config-path config/gfm_reasoner \
    datasets.train_names=[${TRAIN_DATA_NAME_LIST}] \
    datasets.cfgs.root=${DATA_ROOT} \
    train.fast_test=5000 \
    trainer.args.num_epoch=${N_EPOCH} \
    trainer.args.max_steps_per_epoch=${BATCH_PER_EPOCH} \
    trainer.args.train_batch_size=${BATCH_SIZE}
```

### SFT Finetune

``` bash
DATA_ROOT="data_full/new_graph_interface"
N_GPU=4
N_EPOCH=10
START_N=0
END_N=1
BATCH_SIZE=4
DATA_NAME_LIST="hotpotqa_train musique_train 2wikimultihopqa_train"
TRAIN_DATA_NAME_LIST=""
for DATA_NAME in ${DATA_NAME_LIST}; do
    for i in $(seq ${START_N} ${END_N}); do
        TRAIN_DATA_NAME_LIST="${TRAIN_DATA_NAME_LIST},${DATA_NAME}${i}"
    done
done
TRAIN_DATA_NAME_LIST=${TRAIN_DATA_NAME_LIST:1}
echo "TRAIN_DATA_NAME_LIST: [${TRAIN_DATA_NAME_LIST}]"
torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.sft_training --config-path config/gfm_reasoner \
    datasets.train_names=[${TRAIN_DATA_NAME_LIST}] \
    datasets.cfgs.root=${DATA_ROOT} \
    trainer.args.num_epoch=${N_EPOCH} \
    trainer.args.train_batch_size=${BATCH_SIZE}
