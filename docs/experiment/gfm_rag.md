# GFM-RAG Reproduction

!!! note "Compatibility Notice"
    The original GFM-RAG paper results were produced with the `v1.0.0` codebase. While we have made every effort to maintain backward compatibility, if you need exact reproduction, please use the [gfm-rag branch](https://github.com/RManLuo/gfm-rag/tree/gfm-rag).

## Goal

Reproduce the paper results for [GFM-RAG](https://www.arxiv.org/abs/2502.01113) using the maintained shell scripts under `scripts/gfm-rag/`. The workflow consists of three stages: graph indexing, GFM training (pre-training + fine-tuning), and QA evaluation.

## Prerequisites

- Environment setup from [Install](../install.md)
- OpenAI API key set in the environment (`OPENAI_API_KEY`) — used by the NER and OpenIE models during indexing
- Datasets placed under `data/` (see [Data Format](../workflow/data_format.md))
- Sufficient GPU memory: Stage 2 scripts default to 8 GPUs

## Dataset Download

Download the testing split and full training data from [OneDrive](https://1drv.ms/f/c/cb4bbdfe5951d1a1/IgDTnyNJiiAPTJKqY1KizEVMAQ1jX5wAf94YMlF-VyLvscI?e=bgp0Yv) and place them under the `data/` directory:

!!! note "Dataset Format"
     If you are on the `v2.0.0` branch, you need to download the `GFM-RAG_training_new_graph_format.zip` file.

```text
data/
├── 2wikimultihopqa_test/
│   ├── processed/stage1/    # Pre-built graph index (provided)
│   └── raw/
├── hotpotqa_test/
│   ├── processed/stage1/    # Pre-built graph index (provided)
│   └── raw/
├── hotpotqa_train_example/
│   ├── processed/stage1/
│   └── raw/
└── musique_test/
    ├── processed/stage1/    # Pre-built graph index (provided)
    └── raw/
```

> The `processed/stage1/` directories for the test sets are pre-built and provided in the download. You only need to run Stage 1 if you are indexing training data or a custom dataset.

## Scripts To Run

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `scripts/gfm-rag/stage1_data_index.sh` | Build graph index from raw documents |
| 2a | `scripts/gfm-rag/stage2_pretrain.sh` | Unsupervised KG pre-training (optional) |
| 2b | `scripts/gfm-rag/stage2_finetune.sh` | Supervised fine-tuning + retrieval evaluation |
| 3a | `scripts/gfm-rag/stage3_qa_inference.sh` | Batch QA from saved retrieval results |
| 3b | `scripts/gfm-rag/stage3_qa_ircot_inference.sh` | Multi-step IRCoT QA reasoning |

---

## Stage 1: Build Graph Index

This stage extracts entities and relations from raw documents using LLM-based NER and OpenIE, then constructs a knowledge graph.

**Output**: `data/<data_name>/processed/stage1/` containing `nodes.csv`, `relations.csv`, `edges.csv`, `train.json`, `test.json`.

### Index Test Datasets

```bash
bash scripts/gfm-rag/stage1_data_index.sh
```

The script runs indexing for all three test sets and all training data shards (20 shards × 3 datasets):

```bash
# Test datasets (single GPU, no training data needed)
N_GPU=1
DATA_ROOT="data"
DATA_NAME_LIST="hotpotqa_test 2wikimultihopqa_test musique_test"
for DATA_NAME in ${DATA_NAME_LIST}; do
   HYDRA_FULL_ERROR=1 python -m gfmrag.workflow.index_dataset \
   dataset.root=${DATA_ROOT} \
   dataset.data_name=${DATA_NAME}
done

# Training data shards (hotpotqa_train0 … hotpotqa_train19, etc.)
DATA_NAME_LIST="hotpotqa_train musique_train 2wikimultihopqa_train"
START_N=0
END_N=19
for i in $(seq ${START_N} ${END_N}); do
   for DATA_NAME in ${DATA_NAME_LIST}; do
      HYDRA_FULL_ERROR=1 python -m gfmrag.workflow.index_dataset \
      dataset.root=${DATA_ROOT} \
      dataset.data_name=${DATA_NAME}${i}
   done
done
```

> The pre-built test set indexes are included in the download — you can skip indexing test sets and go directly to Stage 2 fine-tuning or Stage 3 evaluation.

### Default Indexing Components

The default config uses:

- **NER**: LLM-based (`llm_ner_model`, GPT-4o-mini via OpenAI API)
- **Entity Linking**: ColBERT with Qdrant vector DB backend (`colbert_el_model`)
- **OpenIE**: LLM-based (`llm_openie_model`, GPT-4o-mini)
- **Graph Constructor**: KG constructor (`kg_constructor`)

To override a component:

```bash
python -m gfmrag.workflow.index_dataset \
    dataset.root=data \
    dataset.data_name=hotpotqa_test \
    ner_model=llm_ner_model \
    el_model=colbert_el_model
```

---

## Stage 2a: KG Pre-training (Optional)

Unsupervised pre-training on the constructed knowledge graphs. Skip this step if you are loading the released checkpoint (`rmanluo/GFM-RAG-8M`).

```bash
bash scripts/gfm-rag/stage2_pretrain.sh
```

Equivalent command (8 GPUs, 1 epoch, 30,000 steps per epoch):

```bash
N_GPU=8
DATA_ROOT="data"
# Builds a comma-separated list: hotpotqa_train0,...,2wikimultihopqa_train19
HYDRA_FULL_ERROR=1 torchrun --nproc-per-node=${N_GPU} -m gfmrag.workflow.kgc_training \
    datasets.train_names=[${TRAIN_DATA_NAME_LIST}] \
    datasets.cfgs.root=${DATA_ROOT} \
    trainer.fast_test=5000 \
    trainer.args.num_epoch=1 \
    trainer.args.max_steps_per_epoch=30000 \
    trainer.args.train_batch_size=4
```

**Output**: Model checkpoint under `outputs/kg_pretrain/<date>/<time>/`.

---

## Stage 2b: Supervised Fine-tuning + Retrieval Evaluation

### Fine-tuning

Fine-tune on the QA datasets. This is required to reproduce the reported retrieval numbers.

```bash
bash scripts/gfm-rag/stage2_finetune.sh
```

Equivalent command (8 GPUs, 15 epochs):

```bash
N_GPU=8
DATA_ROOT="data"
HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.sft_training \
    datasets.train_names=[${TRAIN_DATA_NAME_LIST}] \
    datasets.cfgs.root=${DATA_ROOT} \
    trainer.args.num_epoch=15 \
    trainer.args.train_batch_size=4
```

**Output**: Checkpoint and per-dataset retrieval predictions under `outputs/qa_finetune/<date>/<time>/`.

### Retrieval Evaluation Only

To evaluate a pre-trained checkpoint (e.g., `rmanluo/GFM-RAG-8M`) without training:

```bash
N_GPU=4
DATA_ROOT="data"
CHECKPOINT="rmanluo/GFM-RAG-8M"  # or path to local checkpoint
HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.sft_training \
    load_model_from_pretrained=${CHECKPOINT} \
    datasets.cfgs.root=${DATA_ROOT} \
    datasets.train_names=[] \
    +trainer.args.eval_batch_size=1 \
    trainer.args.do_train=false \
    trainer.args.do_eval=true \
    trainer.args.do_predict=true
```

**Output**: `predictions_<data_name>.json` files under `outputs/qa_finetune/<date>/<time>/`.

---

## Stage 3a: Single-Step QA Reasoning

Reads the retrieval predictions from Stage 2 and generates answers with an LLM in one shot.

```bash
bash scripts/gfm-rag/stage3_qa_inference.sh
```

The script expects two paths produced by Stage 2:

```bash
DATA_NAME="hotpotqa"   # hotpotqa | musique | 2wikimultihopqa
LLM="gpt-4o-mini"
RETRIEVED_RESULT_PATH="outputs/qa_finetune/latest/predictions_${DATA_NAME}_test.json"
NODE_PATH="data/${DATA_NAME}_test/processed/stage1/nodes.csv"

HYDRA_FULL_ERROR=1 python -m gfmrag.workflow.qa \
    qa_prompt=${DATA_NAME} \
    qa_evaluator=${DATA_NAME} \
    llm.model_name_or_path=${LLM} \
    test.n_threads=10 \
    test.top_k=5 \
    test.target_types=[document] \
    test.retrieved_result_path=${RETRIEVED_RESULT_PATH} \
    test.node_path=${NODE_PATH}
```

### Per-dataset Commands

**HotpotQA**
```bash
python -m gfmrag.workflow.qa \
    qa_prompt=hotpotqa qa_evaluator=hotpotqa \
    test.retrieved_result_path=outputs/qa_finetune/latest/predictions_hotpotqa_test.json \
    test.node_path=data/hotpotqa_test/processed/stage1/nodes.csv
```

**MuSiQue**
```bash
python -m gfmrag.workflow.qa \
    qa_prompt=musique qa_evaluator=musique \
    test.retrieved_result_path=outputs/qa_finetune/latest/predictions_musique_test.json \
    test.node_path=data/musique_test/processed/stage1/nodes.csv
```

**2WikiMultihopQA**
```bash
python -m gfmrag.workflow.qa \
    qa_prompt=2wikimultihopqa qa_evaluator=2wikimultihopqa \
    test.retrieved_result_path=outputs/qa_finetune/latest/predictions_2wikimultihopqa_test.json \
    test.node_path=data/2wikimultihopqa_test/processed/stage1/nodes.csv
```

**Output**: `outputs/qa_inference/<date>/<time>/prediction.jsonl`

---

## Stage 3b: Multi-Step IRCoT QA Reasoning

Runs iterative retrieval and reasoning (IRCoT) using the GFM-RAG retriever and an LLM agent. Unlike Stage 3a, this stage performs retrieval online and does not require pre-computed retrieval results.

```bash
bash scripts/gfm-rag/stage3_qa_ircot_inference.sh
```

Equivalent command:

```bash
DATA_NAME="hotpotqa"   # hotpotqa | musique | 2wikimultihopqa
LLM="gpt-4o-mini"
MODEL_PATH="rmanluo/GFM-RAG-8M"
MAX_STEPS=3

python -m gfmrag.workflow.qa_ircot_inference \
    dataset.root=data \
    dataset.data_name=${DATA_NAME}_test \
    graph_retriever.model_path=${MODEL_PATH} \
    llm.model_name_or_path=${LLM} \
    qa_prompt=${DATA_NAME} \
    qa_evaluator=${DATA_NAME} \
    agent_prompt=${DATA_NAME}_ircot \
    test.max_steps=${MAX_STEPS} \
    test.max_test_samples=-1
```

### Per-dataset Commands

**HotpotQA** (2 reasoning steps)
```bash
python -m gfmrag.workflow.qa_ircot_inference \
    qa_prompt=hotpotqa qa_evaluator=hotpotqa \
    agent_prompt=hotpotqa_ircot \
    dataset.data_name=hotpotqa_test \
    test.max_steps=2
```

**MuSiQue** (4 reasoning steps)
```bash
python -m gfmrag.workflow.qa_ircot_inference \
    qa_prompt=musique qa_evaluator=musique \
    agent_prompt=musique_ircot \
    dataset.data_name=musique_test \
    test.max_steps=4
```

**2WikiMultihopQA** (2 reasoning steps)
```bash
python -m gfmrag.workflow.qa_ircot_inference \
    qa_prompt=2wikimultihopqa qa_evaluator=2wikimultihopqa \
    agent_prompt=2wikimultihopqa_ircot \
    dataset.data_name=2wikimultihopqa_test \
    test.max_steps=2
```

**Output**: `outputs/qa_agent_inference/<data_name>/<date>/<time>/prediction.jsonl`

---

## Path Interpretation / Visualization

Visualize the reasoning paths found by the GFM retriever over the knowledge graph:

```bash
python -m gfmrag.workflow.experiments.visualize_path \
    dataset.data_name=hotpotqa_test
```

---

## Expected Outputs Summary

| Stage | Output Location | Contents |
|-------|----------------|---------|
| Stage 1 | `data/<data_name>/processed/stage1/` | `nodes.csv`, `relations.csv`, `edges.csv`, `train.json`, `test.json` |
| Stage 2 (train) | `outputs/qa_finetune/<date>/<time>/` | Model checkpoints, training logs |
| Stage 2 (eval) | `outputs/qa_finetune/<date>/<time>/` | `predictions_<data_name>.json` per dataset |
| Stage 3a | `outputs/qa_inference/<date>/<time>/` | `prediction.jsonl` with answers and scores |
| Stage 3b | `outputs/qa_agent_inference/<data_name>/<date>/<time>/` | `prediction.jsonl` with answers and scores |

---

## Notes

- **Pre-built indexes**: The test set `processed/stage1/` directories are included in the provided download. You do not need to run Stage 1 for test sets.
- **Released checkpoint**: `rmanluo/GFM-RAG-8M` can be loaded directly for retrieval evaluation and IRCoT inference, skipping Stage 2 training.
- **Stage 2 produces Stage 3a inputs**: The QA script requires `predictions_<data_name>.json` from Stage 2 and a matching `nodes.csv` from `processed/stage1/`. The path `outputs/qa_finetune/latest/` is a symlink to the most recent run.
- **HYDRA_FULL_ERROR=1**: Set this variable to get full Python tracebacks instead of truncated Hydra error messages, which is useful for debugging.
- **Multi-GPU training**: Stage 2 scripts use `torchrun`. Adjust `--nproc_per_node` to match your GPU count. Stage 3 scripts use a single process.
- **LLM API costs**: Stage 1 and Stage 3 call the OpenAI API (GPT-4o-mini by default). Set `llm.model_name_or_path` to change the model.
- For general usage outside these reproduction scripts, see the [Workflow](../workflow/data_format.md) pages.
