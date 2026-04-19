# G-Reasoner Reproduction

## Goal

Reproduce the paper results for [G-Reasoner](https://arxiv.org/abs/2509.24276) using the maintained shell scripts under `scripts/g-reasoner/`. The workflow consists of three stages: text-embedding-based graph indexing, supervised fine-tuning, and QA evaluation.

## Prerequisites

- Environment setup from [Install](../install.md)
- OpenAI API key set in the environment (`OPENAI_API_KEY`) — used only in Stage 3 QA inference
- Datasets placed under `data/` (see [Data Format](../workflow/data_format.md))
- A text embedding model available locally or via API (set as `TEXT_EMBEDDING_MODEL`)
- Sufficient GPU memory: Stage 2 scripts default to 4 GPUs


## Dataset Download

Download the testing split and full training data from [OneDrive](https://1drv.ms/f/c/cb4bbdfe5951d1a1/IgDTnyNJiiAPTJKqY1KizEVMAQ1jX5wAf94YMlF-VyLvscI?e=bgp0Yv) and place them under the `data/` directory:

```text
data/
├── 2wikimultihopqa_test/
│   ├── processed/stage1/    # Pre-built graph index (provided)
│   └── raw/
├── hotpotqa_test_v2/
│   ├── processed/stage1/    # Pre-built graph index (provided)
│   └── raw/
├── hotpotqa_train_example/
│   ├── processed/stage1/
│   └── raw/
└── musique_test/
    ├── processed/stage1/    # Pre-built graph index (provided)
    └── raw/
```

> G-Reasoner uses `hotpotqa_test_v2` (not `hotpotqa_test`) as the validation split. The `processed/stage1/` directories for the test sets are pre-built and provided in the download.

## Scripts Overview

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `scripts/g-reasoner/stage1_data_index.sh` | Build text-embedding-based graph index |
| 2a | `scripts/g-reasoner/stage2_finetune.sh` | Supervised fine-tuning |
| 2b | `scripts/g-reasoner/stage2_evaluate.sh` | Retrieval evaluation with a pre-trained checkpoint |
| 3a | `scripts/g-reasoner/stage3_qa_inference.sh` | Batch QA from saved retrieval results |
| 3b | `scripts/g-reasoner/stage3_qa_ircot_inference.sh` | Multi-step IRCoT QA reasoning |

---

## Stage 1: Build Graph Index

This stage uses a text embedding model (instead of LLM-based NER/OpenIE) to extract entities and construct a graph, then builds the SFT training pairs with the `hipporag2_sft_constructor`.

**Output**: `data/<data_name>/processed/stage1/` containing `nodes.csv`, `relations.csv`, `edges.csv`, `train.json`, `test.json`.

```bash
bash scripts/g-reasoner/stage1_data_index.sh
```

### What the script does

**Index test datasets** (no SFT constructor needed):

```bash
DATA_ROOT="data"
DATA_NAME_LIST="hotpotqa_test_v2 musique_test 2wikimultihopqa_test"
for DATA_NAME in ${DATA_NAME_LIST}; do
    python -m gfmrag.workflow.index_dataset \
        --config-path config/gfm_reasoner \
        dataset.root=${DATA_ROOT} \
        text_emb_model=${TEXT_EMBEDDING_MODEL} \
        dataset.data_name=${DATA_NAME}
done
```

**Index training datasets** (20 shards × 3 datasets, with SFT filtering):

```bash
DATA_ROOT="data"
DATA_NAME_LIST="hotpotqa_train musique_train 2wikimultihopqa_train"
START_N=0
END_N=19
for i in $(seq ${START_N} ${END_N}); do
    for DATA_NAME in ${DATA_NAME_LIST}; do
        python -m gfmrag.workflow.index_dataset \
            --config-path config/gfm_reasoner \
            dataset.root=${DATA_ROOT} \
            text_emb_model=${TEXT_EMBEDDING_MODEL} \
            sft_constructor.enable_filtering=${ENABLE_FILTERING} \
            dataset.data_name=${DATA_NAME}${i}
    done
done
```

### Default Indexing Components

The `gfm_reasoner` config (`gfmrag/workflow/config/gfm_reasoner/index_dataset.yaml`) uses:

- **NER**: LLM-based (`llm_ner_model`)
- **Entity Linking**: ColBERT (`colbert_el_model`)
- **OpenIE**: LLM-based (`llm_openie_model`)
- **Graph Constructor**: KG constructor (`kg_constructor`)
- **Text Embedding**: `qwen3_8b` (override with `text_emb_model=<name>`)
- **SFT Constructor**: `hipporag2_sft_constructor` with optional `enable_filtering`

> The pre-built test set indexes are included in the download — you can skip Stage 1 for test sets and go directly to Stage 2 evaluation.

---

## Stage 2a: Supervised Fine-tuning

Fine-tune the G-Reasoner model on the QA datasets. The model uses a 6-layer `QueryNBFNet` with 1024-dimensional embeddings by default.

```bash
bash scripts/g-reasoner/stage2_finetune.sh
```

### What the script does

```bash
N_GPU=4
N_EPOCH=10
BATCH_SIZE=4
N_DIM=1024
N_LAYERS="[${N_DIM},${N_DIM},${N_DIM},${N_DIM},${N_DIM},${N_DIM}]"
DATA_ROOT="data"

# Builds comma-separated list: musique_train0,...,2wikimultihopqa_train19
HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.sft_training \
    --config-path config/gfm_reasoner \
    model.entity_model.input_dim=${N_DIM} \
    model.entity_model.hidden_dims=${N_LAYERS} \
    datasets.cfgs.root=${DATA_ROOT} \
    datasets.train_names=[${TRAIN_DATA_NAME_LIST}] \
    datasets.valid_names=[hotpotqa_test_v2,musique_test,2wikimultihopqa_test] \
    trainer.args.num_epoch=${N_EPOCH} \
    trainer.args.train_batch_size=${BATCH_SIZE} \
    +trainer.training_mode=${TRAIN_MODE} \
    trainer.args.split_graph_training=${SPLIT_GRAPH_TRAINING} \
    trainer.args.split_graph_inference=${SPLIT_GRAPH_INFERENCE} \
    trainer.args.split_graph_partition=${SPLIT_GRAPH_METHOD}
```

Key training arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `N_GPU` | 4 | Number of GPUs |
| `N_EPOCH` | 10 | Training epochs |
| `BATCH_SIZE` | 4 | Batch size per GPU |
| `N_DIM` | 1024 | Entity embedding dimension |
| `TRAIN_MODE` | — | Training mode (set via env var) |
| `SPLIT_GRAPH_TRAINING` | false | Split large graphs during training |
| `SPLIT_GRAPH_INFERENCE` | false | Split large graphs during inference |
| `SPLIT_GRAPH_METHOD` | `metis` | Partition algorithm (`metis` or `contiguous`) |

**Output**: Checkpoint and per-dataset retrieval predictions under `outputs/qa_finetune/<date>/<time>/`.

---

## Stage 2b: Retrieval Evaluation

Evaluate the pre-trained `rmanluo/G-reasoner-34M` checkpoint (or a locally fine-tuned model) without re-training.

```bash
bash scripts/g-reasoner/stage2_evaluate.sh
```

### What the script does

```bash
N_GPU=2
DATA_ROOT="data"
CHECKPOINT="rmanluo/G-reasoner-34M"

HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.sft_training \
    --config-path config/gfm_reasoner \
    --config-name sft_training \
    load_model_from_pretrained=${CHECKPOINT} \
    +datasets.cfgs.skip_empty_target=true \
    datasets.cfgs.root=${DATA_ROOT} \
    datasets.train_names=[] \
    datasets.valid_names=[hotpotqa_test_v2,musique_test,2wikimultihopqa_test] \
    +trainer.args.eval_batch_size=1 \
    trainer.metrics=[hits@2,hits@5,recall@2,recall@5,mrr] \
    trainer.args.do_train=false \
    trainer.args.do_eval=true \
    trainer.args.do_predict=true \
    trainer.args.split_graph_inference=false \
    trainer.args.split_graph_partition=metis
```

**Output**: `predictions_<data_name>.json` files under `outputs/qa_finetune/<date>/<time>/`.

---

## Stage 3a: Single-Step QA Reasoning

Reads the retrieval predictions from Stage 2b and generates answers with an LLM in one shot.

```bash
bash scripts/g-reasoner/stage3_qa_inference.sh
```

### What the script does

The script defaults to `2wikimultihopqa`; change `DATA_NAME` to run other datasets:

```bash
DATA_ROOT="data"
DATA_NAME="2wikimultihopqa"   # hotpotqa | musique | 2wikimultihopqa
LLM="gpt-4o-mini"
DOC_TOP_K=5
N_THREAD=10
RETRIEVED_RESULT_PATH="outputs/qa_finetune/latest/predictions_${DATA_NAME}_test.json"
NODE_PATH="${DATA_ROOT}/${DATA_NAME}_test/processed/stage1/nodes.csv"

python -m gfmrag.workflow.qa \
    --config-path config/gfm_reasoner \
    qa_prompt=${DATA_NAME} \
    qa_evaluator=${DATA_NAME} \
    llm.model_name_or_path=${LLM} \
    test.n_threads=${N_THREAD} \
    test.top_k=${DOC_TOP_K} \
    test.retrieved_result_path=${RETRIEVED_RESULT_PATH} \
    test.target_types=[document] \
    test.node_path=${NODE_PATH}
```

### Per-dataset Commands

**HotpotQA**
```bash
python -m gfmrag.workflow.qa \
    --config-path config/gfm_reasoner \
    qa_prompt=hotpotqa qa_evaluator=hotpotqa \
    test.retrieved_result_path=outputs/qa_finetune/latest/predictions_hotpotqa_test_v2.json \
    test.node_path=data/hotpotqa_test_v2/processed/stage1/nodes.csv
```

**MuSiQue**
```bash
python -m gfmrag.workflow.qa \
    --config-path config/gfm_reasoner \
    qa_prompt=musique qa_evaluator=musique \
    test.retrieved_result_path=outputs/qa_finetune/latest/predictions_musique_test.json \
    test.node_path=data/musique_test/processed/stage1/nodes.csv
```

**2WikiMultihopQA**
```bash
python -m gfmrag.workflow.qa \
    --config-path config/gfm_reasoner \
    qa_prompt=2wikimultihopqa qa_evaluator=2wikimultihopqa \
    test.retrieved_result_path=outputs/qa_finetune/latest/predictions_2wikimultihopqa_test.json \
    test.node_path=data/2wikimultihopqa_test/processed/stage1/nodes.csv
```

**Output**: `outputs/qa_inference/<date>/<time>/prediction.jsonl`

---

## Stage 3b: Multi-Step IRCoT QA Reasoning

Runs iterative retrieval and reasoning (IRCoT) using the G-Reasoner retriever and an LLM agent. Unlike Stage 3a, this stage performs retrieval online and does not require pre-computed retrieval results.

```bash
bash scripts/g-reasoner/stage3_qa_ircot_inference.sh
```

### What the script does

The script defaults to `2wikimultihopqa` and 5 test samples for a quick sanity check:

```bash
DATA_ROOT="data"
DATA_NAME="2wikimultihopqa"   # hotpotqa | musique | 2wikimultihopqa
LLM="gpt-4o-mini"
MAX_STEPS=3
MAX_SAMPLE=5
MODEL_PATH="save_models/G-reasoner-34M"   # or rmanluo/G-reasoner-34M

HYDRA_FULL_ERROR=1 python -m gfmrag.workflow.qa_ircot_inference \
    --config-path config/gfm_reasoner \
    --config-name stage3_qa_ircot_inference \
    dataset.root=${DATA_ROOT} \
    llm.model_name_or_path=${LLM} \
    qa_prompt=${DATA_NAME} \
    qa_evaluator=${DATA_NAME} \
    agent_prompt=${DATA_NAME}_ircot \
    test.max_steps=${MAX_STEPS} \
    test.max_test_samples=${MAX_SAMPLE} \
    dataset.data_name=${DATA_NAME}_test \
    graph_retriever.model_path=${MODEL_PATH}
```

Set `test.max_test_samples=-1` to run on the full test set.

### Per-dataset Commands

**HotpotQA** (2 reasoning steps)
```bash
python -m gfmrag.workflow.qa_ircot_inference \
    --config-path config/gfm_reasoner \
    --config-name stage3_qa_ircot_inference \
    qa_prompt=hotpotqa qa_evaluator=hotpotqa \
    agent_prompt=hotpotqa_ircot \
    dataset.data_name=hotpotqa_test_v2 \
    graph_retriever.model_path=rmanluo/G-reasoner-34M \
    test.max_steps=2 test.max_test_samples=-1
```

**MuSiQue** (4 reasoning steps)
```bash
python -m gfmrag.workflow.qa_ircot_inference \
    --config-path config/gfm_reasoner \
    --config-name stage3_qa_ircot_inference \
    qa_prompt=musique qa_evaluator=musique \
    agent_prompt=musique_ircot \
    dataset.data_name=musique_test \
    graph_retriever.model_path=rmanluo/G-reasoner-34M \
    test.max_steps=4 test.max_test_samples=-1
```

**2WikiMultihopQA** (3 reasoning steps)
```bash
python -m gfmrag.workflow.qa_ircot_inference \
    --config-path config/gfm_reasoner \
    --config-name stage3_qa_ircot_inference \
    qa_prompt=2wikimultihopqa qa_evaluator=2wikimultihopqa \
    agent_prompt=2wikimultihopqa_ircot \
    dataset.data_name=2wikimultihopqa_test \
    graph_retriever.model_path=rmanluo/G-reasoner-34M \
    test.max_steps=3 test.max_test_samples=-1
```

**Output**: `outputs/qa_agent_inference/<data_name>/<date>/<time>/prediction.jsonl`

---

## Expected Outputs Summary

| Stage | Output Location | Contents |
|-------|----------------|---------|
| Stage 1 | `data/<data_name>/processed/stage1/` | `nodes.csv`, `relations.csv`, `edges.csv`, `train.json`, `test.json` |
| Stage 2a (train) | `outputs/qa_finetune/<date>/<time>/` | Model checkpoints, training logs |
| Stage 2b (eval) | `outputs/qa_finetune/<date>/<time>/` | `predictions_<data_name>.json` per dataset |
| Stage 3a | `outputs/qa_inference/<date>/<time>/` | `prediction.jsonl` with answers and scores |
| Stage 3b | `outputs/qa_agent_inference/<data_name>/<date>/<time>/` | `prediction.jsonl` with answers and scores |

---

## Notes

- **Config path**: All G-Reasoner scripts pass `--config-path config/gfm_reasoner` to select the G-Reasoner config family instead of the default GFM-RAG config. This is a relative path resolved from the working directory.
- **Pre-built indexes**: The test set `processed/stage1/` directories are included in the provided download. You do not need to run Stage 1 for test sets.
- **Released checkpoint**: `rmanluo/G-reasoner-34M` can be loaded directly for Stage 2b retrieval evaluation and Stage 3b IRCoT inference, skipping Stage 2a training.
- **hotpotqa_test_v2**: G-Reasoner validates on `hotpotqa_test_v2` (not `hotpotqa_test`). Ensure you download and index this split.
- **TEXT_EMBEDDING_MODEL**: Stage 1 requires a text embedding model. Set this env variable to a config name under `gfmrag/workflow/config/text_emb_model/` (default: `qwen3_8b`). Unlike GFM-RAG, this replaces LLM NER/OpenIE, so no OpenAI API key is needed in Stage 1.
- **TRAIN_MODE**: Required for Stage 2a fine-tuning. Check the `gfm_reasoner` config or paper appendix for valid values.
- **Graph splitting**: For very large graphs, enable `split_graph_training=true` and/or `split_graph_inference=true`. Use `split_graph_partition=metis` (requires METIS library) or `contiguous`.
- **Stage 2b produces Stage 3a inputs**: The QA script requires `predictions_<data_name>.json` from Stage 2b and the matching `nodes.csv`. The path `outputs/qa_finetune/latest/` is a symlink to the most recent run.
- **HYDRA_FULL_ERROR=1**: Set this variable to get full Python tracebacks instead of truncated Hydra error messages.
- **LLM API costs**: Stage 3 calls the OpenAI API (GPT-4o-mini by default). Set `llm.model_name_or_path` to change the model.
- For general usage outside these reproduction scripts, see the [Workflow](../workflow/data_format.md) pages.
