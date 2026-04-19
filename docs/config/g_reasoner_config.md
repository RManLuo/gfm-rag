# G-reasoner Configs

This page groups the task presets under `gfmrag/workflow/config/gfm_reasoner/`.

## Directory Layout

| File | Purpose | Typical entrypoint |
| --- | --- | --- |
| `index_dataset.yaml` | Build `processed/stage1/` from raw data | `python -m gfmrag.workflow.index_dataset --config-name gfm_reasoner/index_dataset` |
| `kgc_trianing.yaml` | Run KGC pretraining for the `G-reasoner` model family | `python -m gfmrag.workflow.kgc_training --config-name gfm_reasoner/kgc_trianing` |
| `qa_inference.yaml` | Run QA from saved retrieval outputs | `python -m gfmrag.workflow.qa --config-name gfm_reasoner/qa_inference` |
| `sft_training.yaml` | Run supervised fine-tuning | `python -m gfmrag.workflow.sft_training --config-name gfm_reasoner/sft_training` |
| `sft_training_w_answer.yaml` | Run SFT with additional answer supervision | `python -m gfmrag.workflow.sft_training --config-name gfm_reasoner/sft_training_w_answer` |
| `stage3_qa_ircot_inference.yaml` | Run retrieval plus IRCOT-style reasoning | `python -m gfmrag.workflow.qa_ircot_inference --config-name gfm_reasoner/stage3_qa_ircot_inference` |
| `visualize_path.yaml` | Visualize reasoning paths on dataset examples | visualization workflow |

## How To Read This Folder

The `gfm_reasoner` presets follow a stable pattern:

- `hydra.run.dir` controls the output root.
- `defaults` pulls in shared component groups such as `ner_model`, `openie_model`, `el_model`, `text_emb_model`, and `wandb`.
- task-specific sections such as `dataset`, `datasets`, `graph_retriever`, `model`, `trainer`, `llm`, and `test` then override the shared pieces.

## Main Pages In This Docs Section

- [G-reasoner Graph Index Config](g_reasoner_graph_index_config.md)
- [G-reasoner Retrieval and QA Config](g_reasoner_retrieval_config.md)
- [G-reasoner SFT Training Config](g_reasoner_sft_config.md)

## Shared Component Pages Used By This Folder

- [NER Model Config](ner_model_config.md)
- [OpenIE Model Config](openie_model_config.md)
- [Entity Linking Model Config](el_model_config.md)
- [Text Embedding Config](text_embedding_config.md)
- [Wandb Config](wandb_config.md)
