# GFM-RAG Configs

This page groups the task presets under `gfmrag/workflow/config/gfm_rag/`.

## Directory Layout

| File | Purpose | Typical entrypoint |
| --- | --- | --- |
| `index_dataset.yaml` | Build `processed/stage1/` from raw data | `python -m gfmrag.workflow.index_dataset` |
| `kgc_training.yaml` | Run KGC pretraining for the original `GFM-RAG` model family | `python -m gfmrag.workflow.kgc_training` |
| `qa_inference.yaml` | Run QA from saved retrieval outputs | `python -m gfmrag.workflow.qa` |
| `qa_ircot_inference.yaml` | Run retrieval plus IRCOT-style reasoning | `python -m gfmrag.workflow.qa_ircot_inference` |
| `sft_training.yaml` | Run supervised fine-tuning and retrieval evaluation | `python -m gfmrag.workflow.sft_training` |
| `visualize_path.yaml` | Visualize reasoning paths on dataset examples | visualization workflow |
| `exp_visualize_path.yaml` | Experimental visualization preset with retrieval controls | visualization workflow |

## How To Read This Folder

The `gfm_rag` presets follow a stable pattern:

- `hydra.run.dir` controls the output root.
- `defaults` pulls in shared component groups such as `ner_model`, `openie_model`, `el_model`, `text_emb_model`, `doc_ranker`, and `wandb`.
- task-specific sections such as `dataset`, `datasets`, `graph_retriever`, `model`, `trainer`, `llm`, and `test` then override the shared pieces.

## Main Pages In This Docs Section

- [GFM-RAG Graph Index Config](gfm_rag_graph_index_config.md)
- [GFM-RAG Retrieval and QA Config](gfmrag_retriever_config.md)
- [GFM-RAG KGC Training Config](gfmrag_pretrain_config.md)
- [GFM-RAG SFT Training Config](gfmrag_finetune_config.md)

## Shared Component Pages Used By This Folder

- [NER Model Config](ner_model_config.md)
- [OpenIE Model Config](openie_model_config.md)
- [Entity Linking Model Config](el_model_config.md)
- [Document Ranker Config](doc_ranker_config.md)
- [Text Embedding Config](text_embedding_config.md)
- [Wandb Config](wandb_config.md)
