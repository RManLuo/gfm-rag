# G-reasoner Retrieval And QA Configuration

This page documents the retrieval, QA, and path-visualization presets in `gfmrag/workflow/config/gfm_reasoner/`.

## `qa_inference.yaml`

This preset is used by `python -m gfmrag.workflow.qa --config-name gfm_reasoner/qa_inference`.

!!! example "gfmrag/workflow/config/gfm_reasoner/qa_inference.yaml"

    ```yaml title="gfmrag/workflow/config/gfm_reasoner/qa_inference.yaml"
    --8<-- "gfmrag/workflow/config/gfm_reasoner/qa_inference.yaml"
    ```

### Top-level Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `hydra.run.dir` | `outputs/qa_inference/${now:%Y-%m-%d}/${now:%H-%M-%S}` | Directory used by Hydra for QA inference outputs. |
| `hydra.searchpath` | `pkg://gfmrag.workflow.config` | Adds the packaged workflow config directory to Hydra's search path. |
| `defaults` | List of config groups | Selects the QA prompt and evaluator presets. |
| `seed` | Integer | Random seed used during inference. |
| `llm` | Mapping | Configures the LLM that turns retrieved evidence into final answers. |
| `test` | Mapping | Controls evaluation size, retrieval inputs, and output paths. |

### `defaults` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_self_` | Current file | Loads the local values in this preset. |
| `qa_prompt` | `hotpotqa` by default | Prompt template used for answer generation. |
| `qa_evaluator` | `hotpotqa` by default | Evaluator used to score predicted answers. |

### `llm` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_target_` | `gfmrag.llms.ChatGPT` by default | LLM wrapper class used for answer generation. |
| `model_name_or_path` | Any supported model name | Model used to answer from retrieved evidence. |
| `retry` | Integer | Maximum number of retry attempts when the LLM call fails. |

### `test` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `n_sample` | `-1` or positive integer | Number of samples to run. `-1` means all samples. |
| `top_k` | Positive integer | Number of retrieved nodes used to build the QA prompt. |
| `n_threads` | Positive integer | Number of worker threads used during evaluation. |
| `target_types` | List such as `[document]` | Node types consumed from the retrieval results. |
| `retrieved_result_path` | File path or `null` | Path to the saved retrieval predictions. |
| `node_path` | File path or `null` | Path to `processed/stage1/nodes.csv`. |
| `prediction_result_path` | File path or `null` | Optional output path for QA predictions. |

## `stage3_qa_ircot_inference.yaml`

This preset is used by `python -m gfmrag.workflow.qa_ircot_inference --config-name gfm_reasoner/stage3_qa_ircot_inference`.

!!! example "gfmrag/workflow/config/gfm_reasoner/stage3_qa_ircot_inference.yaml"

    ```yaml title="gfmrag/workflow/config/gfm_reasoner/stage3_qa_ircot_inference.yaml"
    --8<-- "gfmrag/workflow/config/gfm_reasoner/stage3_qa_ircot_inference.yaml"
    ```

### Top-level Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `hydra.run.dir` | `outputs/qa_agent_inference/${dataset.data_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}` | Directory used by Hydra for agent-style QA outputs. |
| `hydra.searchpath` | `pkg://gfmrag.workflow.config` | Adds the packaged workflow config directory to Hydra's search path. |
| `defaults` | List of config groups | Selects prompts, graph-side components, and evaluator presets. |
| `seed` | Integer | Random seed used during inference. |
| `dataset` | Mapping | Selects the dataset root and dataset split. |
| `llm` | Mapping | Configures the reasoning LLM. |
| `graph_retriever` | Mapping | Controls the graph retriever checkpoint and graph-side dependencies. |
| `test` | Mapping | Controls retrieval depth, max reasoning steps, and resume behavior. |

### `defaults` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_self_` | Current file | Loads the local values in this preset. |
| `agent_prompt` | `hotpotqa_ircot` by default | Prompt template for iterative reasoning. |
| `qa_prompt` | `hotpotqa` by default | Prompt template used for final answer generation. |
| `ner_model` | `llm_ner_model` by default | NER preset used during retrieval-time processing. |
| `openie_model` | `llm_openie_model` by default | OpenIE preset used when graph construction is needed. |
| `el_model` | `colbert_el_model` by default | Entity-linking preset used during retrieval-time processing. |
| `qa_evaluator` | `hotpotqa` by default | Evaluator used to score predicted answers. |
| `graph_constructor` | `kg_constructor` by default | Graph constructor preset used when stage1 needs to be built or refreshed. |

### `dataset` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `root` | Any valid data root | Root directory that contains the dataset folder. |
| `data_name` | Any dataset name | Dataset split used for reasoning-time inference. |

### `llm` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_target_` | `gfmrag.llms.ChatGPT` by default | LLM wrapper class used for iterative reasoning. |
| `model_name_or_path` | Any supported model name | Model used by the reasoning agent. |
| `retry` | Integer | Maximum number of retry attempts when the LLM call fails. |

### `graph_retriever` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `model_path` | Local path or HF model id | Checkpoint of the pretrained `G-reasoner` model. |
| `ner_model` | `${ner_model}` by default | NER preset passed into the graph retriever. |
| `el_model` | `${el_model}` by default | EL preset passed into the graph retriever. |
| `qa_evaluator` | `${qa_evaluator}` by default | QA evaluator preset used by the graph retriever. |
| `target_type` | `document` or other node type | Target node type retrieved by the graph retriever. |
| `graph_constructor` | `${graph_constructor}` by default | Graph constructor preset used if stage1 data must be built. |

### `test` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `top_k` | Positive integer | Number of nodes retrieved per reasoning step. |
| `max_steps` | Positive integer | Maximum number of IRCOT reasoning steps. |
| `max_test_samples` | `-1` or positive integer | Number of examples to run. `-1` means all samples. |
| `resume` | File path or `null` | Resume from a partially written prediction file. |
| `target_types` | List such as `[document]` | Target node types used during evaluation. |

## `visualize_path.yaml`

This preset is used for path-visualization experiments on `GraphIndexDataset`.

!!! example "gfmrag/workflow/config/gfm_reasoner/visualize_path.yaml"

    ```yaml title="gfmrag/workflow/config/gfm_reasoner/visualize_path.yaml"
    --8<-- "gfmrag/workflow/config/gfm_reasoner/visualize_path.yaml"
    ```

### Top-level Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `hydra.run.dir` | `outputs/experiments/visualize/${dataset.data_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}` | Directory used by Hydra for visualization outputs. |
| `hydra.searchpath` | `pkg://gfmrag.workflow.config` | Adds the packaged workflow config directory to Hydra's search path. |
| `defaults` | List | Loads the local values in this preset. |
| `timeout` | Positive integer | Timeout in minutes for multi-GPU execution. |
| `seed` | Integer | Random seed used during the experiment. |
| `load_model_from_pretrained` | File path or `null` | Optional pretrained checkpoint that overrides the model definition. |
| `dataset` | Mapping | Dataset configuration for the visualization run. |
| `test_max_sample` | Positive integer | Maximum number of samples used for visualization. |

### `dataset` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_target_` | `gfmrag.graph_index_datasets.GraphIndexDataset` | Dataset class used by the visualization script. |
| `data_name` | Any dataset name | Dataset split used for visualization. |
| `root` | Any valid data root | Root directory that contains the dataset folder. |
| `force_reload` | `True`, `False` | Whether to rebuild the dataset cache before visualization. |

## Related Configurations

- [Graph Constructor Config](graph_constructor_config.md)
- [NER Model Config](ner_model_config.md)
- [OpenIE Model Config](openie_model_config.md)
- [Entity Linking Model Config](el_model_config.md)
