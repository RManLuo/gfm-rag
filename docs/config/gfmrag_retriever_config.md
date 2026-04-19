# GFM-RAG Retrieval And QA Config

This page documents the retrieval and inference presets in `gfmrag/workflow/config/gfm_rag/`.

## Files Covered

- `qa_inference.yaml`
- `qa_ircot_inference.yaml`
- `visualize_path.yaml`
- `exp_visualize_path.yaml`

## `qa_inference.yaml`

This preset is used by `python -m gfmrag.workflow.qa` to turn saved retrieval outputs into final QA predictions.

!!! example "gfmrag/workflow/config/gfm_rag/qa_inference.yaml"

    ```yaml title="gfmrag/workflow/config/gfm_rag/qa_inference.yaml"
    --8<-- "gfmrag/workflow/config/gfm_rag/qa_inference.yaml"
    ```

### Top-level Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `hydra.run.dir` | `outputs/qa_inference/<date>/<time>/` | Directory used by Hydra for QA inference outputs. |
| `defaults` | List of config groups | Pulls in `qa_prompt` and `qa_evaluator`. |
| `llm` | Mapping | Configures the answer-generation model. |
| `test` | Mapping | Points to the retrieval result file, node table, and decoding options. |

### `test` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `llm.model_name_or_path` | Model name or path | LLM used to generate final answers. |
| `test.retrieved_result_path` | File path | Path to retrieval predictions, typically `predictions_<data_name>.json`. |
| `test.node_path` | File path | Path to `processed/stage1/nodes.csv`. |
| `test.top_k` | Positive integer | Number of retrieved nodes used to build the QA prompt. |
| `test.target_types` | List of node types | Target node types to read from the retrieval output. |

## `qa_ircot_inference.yaml`

This preset is used by `python -m gfmrag.workflow.qa_ircot_inference` to run retrieval and IRCOT-style reasoning in one workflow.

### `defaults` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `agent_prompt` | `hotpotqa_ircot` by default | Prompt template for iterative reasoning. |
| `qa_prompt` | `hotpotqa` by default | Prompt template used for final answer generation. |
| `ner_model` | `llm_ner_model` by default | NER preset used during retrieval-time processing. |
| `openie_model` | `llm_openie_model` by default | OpenIE preset used when graph construction is needed. |
| `el_model` | `colbert_el_model` by default | Entity-linking preset used during retrieval-time processing. |
| `qa_evaluator` | `hotpotqa` by default | Evaluator used to score predicted answers. |
| `graph_constructor` | `kg_constructor` by default | Graph constructor preset used if stage1 data must be built. |

### Key Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `dataset` | Mapping | Selects the dataset root and test split. |
| `llm` | Mapping | Chooses the reasoning and answer-generation model. |
| `graph_retriever` | Mapping | Selects the checkpoint and graph-side components. |
| `test` | Mapping | Controls `top_k`, `max_steps`, resume path, and target types. |
| `graph_retriever.model_path` | `rmanluo/GFM-RAG-8M` by default | Checkpoint path of the pretrained model. |
| `graph_retriever.graph_constructor` | `${graph_constructor}` | Constructor used when stage1 needs to be built. |
| `test.max_steps` | Positive integer | Maximum IRCOT reasoning steps. |
| `test.resume` | File path or `null` | Resume from a partially written prediction file. |

## Visualization Presets

### `visualize_path.yaml`

Use this preset for path visualization on a single `GraphIndexDatasetV1` dataset. It loads the dataset directly rather than using `dataset.root` and `dataset.data_name` as a separate pair.

### `exp_visualize_path.yaml`

This experimental preset adds retrieval-oriented controls such as:

- `test.retrieval_batch_size`
- `test.save_retrieval`
- `test.save_top_k_entity`
- `test.max_sample`

## Related Shared Configs

- [Document Ranker Config](doc_ranker_config.md)
- [NER Model Config](ner_model_config.md)
- [OpenIE Model Config](openie_model_config.md)
- [Entity Linking Model Config](el_model_config.md)
