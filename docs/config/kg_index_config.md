# KG-index Config

This page documents the top-level indexing presets that reference the graph constructor and SFT constructor config groups.

## Files Covered

- `gfmrag/workflow/config/gfm_rag/index_dataset.yaml`
- `gfmrag/workflow/config/gfm_reasoner/index_dataset.yaml`

## Purpose

These presets are used by `python -m gfmrag.workflow.index_dataset` to:

- select shared construction components
- choose the graph constructor preset
- choose the SFT constructor preset
- specify the dataset root and dataset name

## Shared Top-level Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `hydra.run.dir` | `outputs/kg_construction/<date>/<time>/` | Directory used by Hydra for runtime logs and outputs. |
| `defaults` | List of config groups | Pulls in component config groups. |
| `dataset` | Mapping | Chooses the dataset root, dataset name, and force flag. |

## Defaults in `gfm_rag/index_dataset.yaml`

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `ner_model` | `llm_ner_model` by default | Named entity recognition preset. |
| `openie_model` | `llm_openie_model` by default | OpenIE preset used by the graph constructor. |
| `el_model` | `colbert_el_model` by default | Entity-linking preset. |
| `graph_constructor` | `kg_constructor` by default | Graph construction preset. |
| `sft_constructor` | `gfm_rag_sft_constructor` by default | SFT constructor preset. |

## Defaults in `gfm_reasoner/index_dataset.yaml`

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `ner_model` | `llm_ner_model` by default | Named entity recognition preset. |
| `openie_model` | `llm_openie_model` by default | OpenIE preset used by the graph constructor. |
| `el_model` | `colbert_el_model` by default | Entity-linking preset. |
| `text_emb_model` | `qwen3_8b` by default | Text embedding preset used by `hipporag2_sft_constructor`. |
| `graph_constructor` | `kg_constructor` by default | Graph construction preset. |
| `sft_constructor` | `hipporag2_sft_constructor` by default | SFT constructor preset. |

## `dataset` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `dataset.root` | Any valid data root | Root directory that contains the dataset folder. |
| `dataset.data_name` | Any dataset name | Dataset name under `root`. |
| `dataset.force` | `True`, `False` | Whether to force recomputation even if cached files exist. |

## Constructor Config Groups

The two constructor groups are documented separately:

- [Graph Constructor Config](graph_constructor_config.md)
- [SFT Constructor Config](sft_constructor_config.md)

Use those pages when you want to change constructor-specific fields such as `num_processes`, `threshold`, `topk`, `enable_filtering`, or temporary working directories.

## Related Shared Configs

- [NER Model Config](ner_model_config.md)
- [OpenIE Model Config](openie_model_config.md)
- [Entity Linking Model Config](el_model_config.md)
- [Text Embedding Config](text_embedding_config.md)
