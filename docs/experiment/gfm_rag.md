# GFM-RAG Reproduction

## Goal

Reproduce the script-based `GFM-RAG` indexing, fine-tuning, and QA evaluation flow using the maintained shell scripts under `scripts/gfm-rag/`.

## Prerequisites

- Environment setup from [Install](../install.md)
- Datasets placed under `data/`
- Access to the required external LLM or API-backed components referenced by the configs
- Enough GPUs for the script defaults when using the multi-GPU commands

## Scripts To Run

- `scripts/gfm-rag/stage1_data_index.sh`
- `scripts/gfm-rag/stage2_finetune.sh`
- `scripts/gfm-rag/stage3_qa_inference.sh`

## Run Order

### 1. Build stage1 files

```bash
bash scripts/gfm-rag/stage1_data_index.sh
```

### 2. Fine-tune or run retrieval evaluation

```bash
bash scripts/gfm-rag/stage2_finetune.sh
```

### 3. Run QA from saved retrieval outputs

```bash
bash scripts/gfm-rag/stage3_qa_inference.sh
```

## Expected Outputs

- indexed datasets under `data/<dataset_name>/processed/stage1/`
- training/evaluation runs under `outputs/qa_finetune/<date>/<time>/`
- retrieval predictions such as `predictions_<data_name>.json`
- QA predictions under `outputs/qa_inference/<date>/<time>/prediction.jsonl`

## Notes

- The stage2 script contains both a training block and a retrieval-evaluation block.
- The QA script expects retrieval outputs produced by the training workflow and a matching `nodes.csv` file under `processed/stage1/`.
- For general usage without these reproduction scripts, use the [Workflow](../workflow/data_format.md) pages instead.
