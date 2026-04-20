# Text Embedding Model Configuration

## Pre-train Text Embedding Model Configuration

This configuration supports most pre-trained text embedding models from [SentenceTransformer](https://huggingface.co/sentence-transformers). Example configuration files are shown below:


!!! example "all-mpnet-base-v2"

    ```yaml title="gfmrag/workflow/config/text_emb_model/mpnet.yaml"
    --8<-- "gfmrag/workflow/config/text_emb_model/mpnet.yaml"
    ```

!!! example "BAAI/bge-large-en"

    ```yaml title="gfmrag/workflow/config/text_emb_model/bge_large_en.yaml"
    --8<-- "gfmrag/workflow/config/text_emb_model/bge_large_en.yaml"
    ```
  |       Parameter       |                  Options                  |                                       Note                                        |
  | :-------------------: | :---------------------------------------: | :-------------------------------------------------------------------------------: |
  |      `_target_`       | `gfmrag.text_emb_models.BaseTextEmbModel` | The class name of [Text Embedding model][gfmrag.text_emb_models.BaseTextEmbModel] |
  | `text_emb_model_name` |                   None                    |                  The name of the pre-train text embedding model.                  |
  |      `normalize`      |              `True`, `False`              |                       Whether to normalize the embeddings.                        |
  |   `query_instruct`    |                   None                    |                          The instruction for the query.                           |
  |  `passage_instruct`   |                   None                    |                         The instruction for the passage.                          |
  |    `model_kwargs`     |                   `{}`                    |                          The additional model arguments.                          |

## Qwen3 Embedding Model Configuration

This configuration supports Qwen3 embedding models through [gfmrag.text_emb_models.Qwen3TextEmbModel][gfmrag.text_emb_models.Qwen3TextEmbModel]. Example configuration files are shown below:

!!! example "Qwen/Qwen3-Embedding-0.6B"

    ```yaml title="gfmrag/workflow/config/text_emb_model/qwen3.yaml"
    --8<-- "gfmrag/workflow/config/text_emb_model/qwen3.yaml"
    ```

!!! example "Qwen/Qwen3-Embedding-8B"

    ```yaml title="gfmrag/workflow/config/text_emb_model/qwen3_8b.yaml"
    --8<-- "gfmrag/workflow/config/text_emb_model/qwen3_8b.yaml"
    ```

|       Parameter       |                                Options                                 |                                                      Note                                                       |
| :-------------------: | :--------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: |
|      `_target_`       |             `gfmrag.text_emb_models.Qwen3TextEmbModel`                 |                            The class name of `Qwen3TextEmbModel`.                            |
| `text_emb_model_name` | `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Embedding-8B`, or local path  |                                   The name or local path of the Qwen3 embedding model.                                   |
|      `normalize`      |                           `True`, `False`                              |                                          Whether to normalize the embeddings.                                          |
|      `batch_size`     |                          Positive integer                              |                                             Batch size used for encoding.                                             |
|   `query_instruct`    |                              String, `null`                            |                                     The instruction prepended to each query.                                      |
|  `passage_instruct`   |                              String, `null`                            |                                    The instruction prepended to each passage.                                     |
|    `truncate_dim`     |                          Positive integer, `null`                      | Optional output embedding dimension for Matryoshka truncation, such as `1024` for `0.6B` or `4096` for `8B`. |
|      `api_base`       |                              URL, `null`                               |               Base URL of an existing vLLM embedding server. If `null`, gfmrag starts a local vLLM instance.               |
|       `api_key`       |                                String                                  |                                  API key used when `api_base` points to a protected server.                                  |
|    `vllm_timeout`     |                          Positive integer                              |                                         Timeout in seconds for vLLM embedding requests.                                          |

## Nvidia Embedding Model Configuration

This configuration supports the [Nvidia embedding models](https://huggingface.co/nvidia/NV-Embed-v2). An example of a Nvidia embedding model configuration file is shown below:

!!! example "nvidia/NV-Embed-v2"

    ```yaml title="gfmrag/workflow/config/text_emb_model/nv_embed_v2.yaml"
    --8<-- "gfmrag/workflow/config/text_emb_model/nv_embed_v2.yaml"
    ```

|       Parameter       |                                                   Options                                                   |                                                   Note                                                   |
| :-------------------: | :---------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------: |
|      `_target_`       |                       `gfmrag.text_emb_models.NVEmbedV2`                        | The class name of `NVEmbedV2` |
| `text_emb_model_name` |                                            `nvidia/NV-Embed-v2`                                             |                                 The name of the Nvidia embedding model.                                  |
|      `normalize`      |                                               `True`, `False`                                               |                                   Whether to normalize the embeddings.                                   |
|   `query_instruct`    | `Instruct: Given an entity, retrieve entities that are semantically equivalent to the given entity\nQuery: ` |                                      The instruction for the query.                                      |
|  `passage_instruct`   |                                                    None                                                     |                                     The instruction for the passage.                                     |
|    `model_kwargs`     |                                                    `{}`                                                     |                                     The additional model arguments.                                      |
