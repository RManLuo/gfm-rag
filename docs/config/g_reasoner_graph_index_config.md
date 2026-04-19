# G-reasoner Graph Index Configuration

This page documents the graph-index preset used by the `G-reasoner` workflow family.

## `index_dataset.yaml`

This preset is used by `python -m gfmrag.workflow.index_dataset --config-name gfm_reasoner/index_dataset`.

!!! example "gfmrag/workflow/config/gfm_reasoner/index_dataset.yaml"

    ```yaml title="gfmrag/workflow/config/gfm_reasoner/index_dataset.yaml"
    --8<-- "gfmrag/workflow/config/gfm_reasoner/index_dataset.yaml"
    ```

Compared with the `GFM-RAG` preset, this file additionally selects a `text_emb_model` because the default `hipporag2_sft_constructor` uses text embeddings during supervision-data construction.

### Top-level Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `hydra.run.dir` | `outputs/kg_construction/${now:%Y-%m-%d}/${now:%H-%M-%S}` | Directory used by Hydra for runtime logs and outputs. |
| `hydra.searchpath` | `pkg://gfmrag.workflow.config` | Adds the packaged workflow config directory to Hydra's search path. |
| `defaults` | List of config groups | Selects the shared component presets used by indexing. |
| `dataset` | Mapping | Controls the dataset root, dataset name, and whether to force recomputation. |

### `defaults` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_self_` | Current file | Loads the local values in this preset. |
| `ner_model` | `llm_ner_model` by default | Named entity recognition preset used by the SFT constructor. |
| `openie_model` | `llm_openie_model` by default | OpenIE preset used by the graph constructor. |
| `el_model` | `colbert_el_model` by default | Entity-linking preset used by both graph construction and supervision-data construction. |
| `text_emb_model` | `qwen3_8b` by default | Text embedding preset used by `hipporag2_sft_constructor`. |
| `graph_constructor` | `kg_constructor` by default | Graph construction preset that builds the stage1 graph files. |
| `sft_constructor` | `hipporag2_sft_constructor` by default | SFT constructor preset used to build G-reasoner supervision data. |

### `dataset` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `root` | Any valid data root | Root directory that contains the dataset folder. |
| `data_name` | Any dataset name | Dataset name under `root`. |
| `force` | `True`, `False` | Whether to rebuild the processed outputs even if cached files already exist. |

## Related Configurations

- [Graph Constructor Config](graph_constructor_config.md)
- [SFT Constructor Config](sft_constructor_config.md)
- [NER Model Config](ner_model_config.md)
- [OpenIE Model Config](openie_model_config.md)
- [Entity Linking Model Config](el_model_config.md)
- [Text Embedding Config](text_embedding_config.md)
