# G-reasoner Reproduction

## Goal

Reproduce the script-based `G-reasoner` pipeline with the current maintained shell scripts under `scripts/g-reasoner/`.

## Prerequisites

- Environment setup from [Install](../install.md)
- Datasets placed under `data/`
- The `gfm_reasoner` config family available under `gfmrag/workflow/config/gfm_reasoner/`
- Required text embedding and API-backed components configured in your environment

## Scripts To Run

- `scripts/g-reasoner/stage1_data_index.sh`
- `scripts/g-reasoner/stage2_finetune.sh`
- `scripts/g-reasoner/stage2_evaluate.sh`
- `scripts/g-reasoner/stage3_qa_inference.sh`

## Run Order

### 1. Build stage1 files

```bash
bash scripts/g-reasoner/stage1_data_index.sh
```

### 2. Fine-tune the retriever

```bash
bash scripts/g-reasoner/stage2_finetune.sh
```

### 3. Run retrieval evaluation

```bash
bash scripts/g-reasoner/stage2_evaluate.sh
```

### 4. Run QA from retrieved documents

```bash
bash scripts/g-reasoner/stage3_qa_inference.sh
```

## Expected Outputs

- indexed datasets under `data/<dataset_name>/processed/stage1/`
- training and retrieval-evaluation runs under `outputs/qa_finetune/<date>/<time>/`
- retrieval prediction files written by `sft_training`
- QA outputs under `outputs/qa_inference/<date>/<time>/prediction.jsonl`

## Notes

- These scripts intentionally mirror the direct runnable style under `scripts/gfm-rag/` and avoid job-submission wrappers.
- The stage2 evaluation script uses the current `sft_training` entrypoint with `do_train=false`, `do_eval=true`, and `do_predict=true`.
- The QA script reads saved retrieval outputs and requires the matching dataset `nodes.csv`.
