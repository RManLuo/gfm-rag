## LLM OpenIE Model Configuration

An example of a LLM OpenIE model configuration file is shown below:

!!! example

    ```yaml title="gfmrag/workflow/config/openie_model/llm_openie_model.yaml"
    --8<-- "gfmrag/workflow/config/openie_model/llm_openie_model.yaml"
    ```


|      Parameter       |                             Options                             |                                           Note                                           |
| :------------------: | :-------------------------------------------------------------: | :--------------------------------------------------------------------------------------: |
|      `_target_`      |                              None                               | The class name of [LLM OpenIE model][gfmrag.graph_index_construction.openie_model.LLMOPENIEModel] |
|      `llm_api`       | ``openai``, ``nvidia``, ``together``, ``ollama``, ``llama.cpp`` |                            The API to use for the LLM model.                             |
|     `model_name`     |                              None                               |               The name of the LLM model to use. For example, `gpt-4o-mini`               |
|   `max_ner_tokens`   |                              None                               |                  The maximum number of tokens to use for the NER model.                  |
| `max_triples_tokens` |                              None                               |                The maximum number of tokens to use for the triples model.                |

Please refer to [LLMOPENIEModel][gfmrag.graph_index_construction.openie_model.LLMOPENIEModel] for more details of the other parameters.
