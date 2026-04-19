# Training

This page covers the current supervised training path for graph retrievers.

## What This Step Does

`gfmrag.workflow.sft_training` trains or evaluates a retriever on graph-index datasets that already have stage1 files and processed QA data. It can also load an existing checkpoint and emit retrieval predictions for downstream QA.

## When You Need It

Use this page when you want to:

- fine-tune on your own datasets
- run retrieval evaluation from a pre-trained checkpoint
- generate `predictions_<data_name>.json` files for later QA

If you have not prepared the data yet, start with [Data Format](data_format.md) and [Index](index.md).

## Inputs

- A dataset root under `datasets.cfgs.root`
- One or more indexed datasets with `processed/stage1/`
- Training names in `datasets.train_names`
- Validation names in `datasets.valid_names`
- A training config, usually [`gfmrag/workflow/config/gfm_rag/sft_training.yaml`](https://github.com/RManLuo/gfm-rag/blob/main/gfmrag/workflow/config/gfm_rag/sft_training.yaml)

## Outputs

Hydra writes runs under `outputs/qa_finetune/<date>/<time>/`.

Common outputs include:

- checkpoints managed by the trainer
- `pretrained/` when `save_pretrained=true`
- `predictions_<data_name>.json` when `trainer.args.do_predict=true`

## Minimal Example

Single-node fine-tuning:

```bash
python -m gfmrag.workflow.sft_training \
  datasets.cfgs.root=./data \
  datasets.train_names=[hotpotqa_train_example] \
  datasets.valid_names=[hotpotqa_test]
```

Multi-GPU fine-tuning:

```bash
torchrun --nproc_per_node=4 -m gfmrag.workflow.sft_training \
  datasets.cfgs.root=./data \
  datasets.train_names=[hotpotqa_train0,hotpotqa_train1] \
  datasets.valid_names=[hotpotqa_test,musique_test,2wikimultihopqa_test]
```

Retrieval evaluation from a pre-trained checkpoint:

```bash
torchrun --nproc_per_node=4 -m gfmrag.workflow.sft_training \
  load_model_from_pretrained=rmanluo/GFM-RAG-8M \
  datasets.cfgs.root=./data \
  datasets.train_names=[] \
  trainer.args.do_train=false \
  trainer.args.do_eval=true \
  trainer.args.do_predict=true \
  +trainer.args.eval_batch_size=1
```

## Key Configs

- [`gfmrag/workflow/config/gfm_rag/sft_training.yaml`](https://github.com/RManLuo/gfm-rag/blob/main/gfmrag/workflow/config/gfm_rag/sft_training.yaml)
- [GFM-RAG Fine-tuning Config](../config/gfmrag_finetune_config.md)
- [Text Embedding Config](../config/text_embedding_config.md)
- [Document Ranker Config](../config/doc_ranker_config.md)
- [Wandb Config](../config/wandb_config.md)

## Common Pitfalls

- `datasets.init_datasets=false` requires `datasets.feat_dim` to be set.
- `load_model_from_pretrained` overwrites the model configuration with the checkpoint config.
- Prediction files are only written when `trainer.args.do_predict=true`.
- The downstream QA script expects both the retrieval output file and the dataset `nodes.csv`.

## Related Legacy Workflows

The repository still contains older documentation pages discussing legacy stage-named training modules. Those are no longer the primary training path and are intentionally not used as the main tutorial flow.
