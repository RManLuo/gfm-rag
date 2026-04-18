# GFM-RAG Configuration
An example configuration file for GFM-RAG is shown below:

!!! example

    ```yaml title="gfmrag/workflow/config/gfm_rag/qa_ircot_inference.yaml"
    --8<-- "gfmrag/workflow/config/gfm_rag/qa_ircot_inference.yaml"
    ```

## General Configuration

| Parameter | Options |              Note               |
| :-------: | :-----: | :-----------------------------: |
| `run.dir` |  None   | The output directory of the log |

## Defaults

|   Parameter    | Options |                                         Note                                          |
| :------------: | :-----: | :-----------------------------------------------------------------------------------: |
|  `ner_model`   |  None   |                  The config of the [ner_model](ner_model_config.md)                   |
|   `el_model`   |  None   |                   The config of the [el_model](el_model_config.md)                    |
|  `doc_ranker`  |  None   |                 The config of the [doc_ranker](doc_ranker_config.md)                  |
| `qa_evaluator` |  None   |                  The config of the [qa_evaluator][gfmrag.evaluation]                  |
| `agent_prompt` |  None   | The config of the [PromptBuilder][gfmrag.prompt_builder] used for IRCOT |
|  `qa_prompt`   |  None   |        The config of the [PromptBuilder][gfmrag.prompt_builder]         |


## Dataset

|  Parameter  | Options |          Note           |
| :---------: | :-----: | :---------------------: |
|   `root`    |  None   | The data root directory |
| `data_name` |  None   |      The data name      |


## LLM

|       Parameter       | Options |                           Note                           |
| :-------------------: | :-----: | :------------------------------------------------------: |
|      `_target_`       |  None   |         The [language model][gfmrag.llms] to use         |
| `model_name_or_path`  |  None   |                  The model name or path                  |
| Additional parameters |  None   | Parameters to initialize a [language model][gfmrag.llms] |

Please refer to the [LLMs][gfmrag.llms] page for more details.

## Graph Retriever

|       Parameter        |    Options     |                                Note                                |
| :--------------------: | :------------: | :----------------------------------------------------------------: |
|       `_target_`       |      None      |        The [graph retriever][gfmrag.GFMRetriever] to use        |
|      `model_path`      |      None      |          Checkpoint path of the pre-trained GFM-RAG model          |
|      `doc_ranker`      |      None      |          The [document ranker][gfmrag.models.gfm_rag_v1.rankers] to use          |
|      `ner_model`       |      None      |      The [NER model][gfmrag.graph_index_construction.ner_model] to use      |
|       `el_model`       |      None      | The [EL model][gfmrag.graph_index_construction.entity_linking_model] to use |
|     `qa_evaluator`     |      None      |                      The QA evaluator to use                       |
| `init_entities_weight` | `True`,`False` |             Whether to initialize the entities weight              |


## Test

|     Parameter      | Options |                          Note                          |
| :----------------: | :-----: | :----------------------------------------------------: |
|      `top_k`       |  None   |            Number of documents to retrieve             |
|    `max_steps`     |  None   |      Maximum number of steps, `1` for single step      |
| `max_test_samples` |  None   | Maximum number of samples to test (`-1` for all samples) |
|      `resume`      |  None   |            Resume from previous prediction             |
