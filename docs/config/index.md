# Config Overview

The configuration system is split into two layers:

- **Shared components**: reusable model and component configs such as graph constructor, SFT constructor, NER, OpenIE, EL, text embedding, document ranker, and wandb.
- **Workflow presets**: task-level YAML files under `gfmrag/workflow/config/gfm_rag/` and `gfmrag/workflow/config/gfm_reasoner/`.

Use this section in the following order:

1. Start with the workflow family you want to run.
2. Read the corresponding preset page to understand the top-level sections in that YAML file.
3. Jump to the shared component pages when you want to swap graph construction, SFT construction, NER, OpenIE, EL, text embedding, ranker, or logging backends.

## Workflow Families

### `gfmrag/workflow/config/gfm_rag/`

Use this directory for the original `GFM-RAG` workflow family. It contains presets for:

- indexing datasets
- KGC training
- retrieval and QA inference
- SFT training
- path visualization

See [GFM-RAG Configs](gfm_rag_config.md).

### `gfmrag/workflow/config/gfm_reasoner/`

Use this directory for the `G-reasoner` workflow family. It contains presets for:

- indexing datasets
- KGC training
- retrieval and QA inference
- SFT training
- SFT training with answer supervision
- path visualization

See:

- [G-reasoner Graph Index Config](g_reasoner_graph_index_config.md)
- [G-reasoner Retrieval and QA Config](g_reasoner_retrieval_config.md)
- [G-reasoner SFT Training Config](g_reasoner_sft_config.md)

## Shared Components

These pages document reusable config groups referenced by both workflow families:

- [Graph Constructor Config](graph_constructor_config.md)
- [SFT Constructor Config](sft_constructor_config.md)
- [NER Model Config](ner_model_config.md)
- [OpenIE Model Config](openie_model_config.md)
- [Entity Linking Model Config](el_model_config.md)
- [Document Ranker Config](doc_ranker_config.md)
- [Text Embedding Config](text_embedding_config.md)
- [Wandb Config](wandb_config.md)
