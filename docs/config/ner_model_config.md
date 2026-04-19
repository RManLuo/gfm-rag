# NER Model Configuration

## LLM NER Model Configuration

An example of a LLM NER model configuration file is shown below:

!!! example

    ```yaml title="gfmrag/workflow/config/ner_model/llm_ner_model.yaml"
    --8<-- "gfmrag/workflow/config/ner_model/llm_ner_model.yaml"
    ```

|  Parameter   |                             Options                             |                                      Note                                       |
| :----------: | :-------------------------------------------------------------: | :-----------------------------------------------------------------------------: |
|  `_target_`  |                              None                               | The class name of [LLM NER model][gfmrag.graph_index_construction.ner_model.LLMNERModel] |
|  `llm_api`   | ``openai``, ``nvidia``, ``together``, ``ollama``, ``llama.cpp`` |                        The API to use for the LLM model.                        |
| `model_name` |                              None                               |          The name of the LLM model to use. For example, `gpt-4o-mini`           |
| `max_tokens` |                              None                               |             The maximum number of tokens to use for the LLM model.              |

Please refer to [LLM NER model][gfmrag.graph_index_construction.ner_model.LLMNERModel] for details on the other parameters.
