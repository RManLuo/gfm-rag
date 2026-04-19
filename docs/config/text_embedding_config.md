# Text Embedding Model Configuration

## Pre-train Text Embedding Model Configuration

This configuration supports most of the pre-train text embedding models of [SentenceTransformer](https://huggingface.co/sentence-transformers). Examples of DPR text embedding model configuration files are shown below:


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
