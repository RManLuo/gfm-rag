
## Colbert EL Model Configuration

`ColbertELModel` now uses a FastEmbed-backed Qdrant index. The supported remote model path should be a FastEmbed model id such as `colbert-ir/colbertv2.0`; local model paths are not the default path for this backend.

An example Colbert EL model configuration file is shown below:

!!! example "colbertv2.0"

    ```yaml title="gfmrag/workflow/config/el_model/colbert_el_model.yaml"
    --8<-- "gfmrag/workflow/config/el_model/colbert_el_model.yaml"
    ```

```yaml
_target_: gfmrag.graph_index_construction.entity_linking_model.ColbertELModel
model_name_or_path: colbert-ir/colbertv2.0
root: tmp
phrase_index_name: nbits_2
force: false
use_in_memory: true
```

`use_in_memory: true` switches the backend to `QdrantClient(":memory:")`. This is now the default mode. In that mode, no persistent cache files are written, and the index is only reused within the current process.

|      Parameter      |                           Options                            |                                               Note                                               |
| :-----------------: | :----------------------------------------------------------: | :----------------------------------------------------------------------------------------------: |
|     `_target_`      | `gfmrag.graph_index_construction.entity_linking_model.ColbertELModel` | The class name of [Colbert EL model][gfmrag.graph_index_construction.entity_linking_model.ColbertELModel] |
|  `model_name_or_path`  |                             None                             |                          A supported FastEmbed model id such as `colbert-ir/colbertv2.0`.                           |
|       `root`        |                             None                             |                                 The root directory of the model.                                 |
| `force` |                     `True`, `False`                          | Whether to force re-indexing the entities. If set to `True`, it will delete the existing index and re-index the entities. |
| `use_in_memory` |                     `True`, `False`                          | Whether to use `QdrantClient(":memory:")` instead of the persistent cache under `root/colbert/{fingerprint}`. |

Please refer to [ColbertELModel][gfmrag.graph_index_construction.entity_linking_model.ColbertELModel] for details on the other parameters.

## Dense Pre-train Text Embedding Model Configuration

This configuration supports most of the dense pre-train text embedding models of [SentenceTransformer](https://huggingface.co/sentence-transformers). An example of a dense pre-train text embedding model configuration file is shown below:

!!! example "DPR EL Model"

    ```yaml title="gfmrag/workflow/config/el_model/dpr_el_model.yaml"
    --8<-- "gfmrag/workflow/config/el_model/dpr_el_model.yaml"
    ```


|     Parameter      |                         Options                          |                                                       Note                                                       |
| :----------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
|     `_target_`     | `gfmrag.graph_index_construction.entity_linking_model.DPRELModel` | The class name of [Dense Pre-train Text Embedding model][gfmrag.graph_index_construction.entity_linking_model.DPRELModel] |
|    `model_name`    |                           None                           |                              The name of the dense pre-train text embedding model.                               |
|       `root`       |                           None                           |                                         The root directory of the model.                                         |
|    `use_cache`     |                     `True`, `False`                      |                                              Whether to use cache.                                               |
|    `normalize`     |                     `True`, `False`                      |                                       Whether to normalize the embeddings.                                       |
|  `query_instruct`  |                           None                           |                                          The instruction for the query.                                          |
| `passage_instruct` |                           None                           |                                         The instruction for the passage.                                         |
|   `model_kwargs`   |                           None                           |                                         The additional model arguments.                                          |

Please refer to [DPR EL Model][gfmrag.graph_index_construction.entity_linking_model.DPRELModel] for details on the other parameters.

## Nvidia Embedding Model Configuration

This configuration supports most of the [Nvidia embedding models](https://huggingface.co/nvidia/NV-Embed-v2). An example of a Nvidia embedding model configuration file is shown below:

!!! example "nvidia/NV-Embed-v2"

    ```yaml title="gfmrag/workflow/config/el_model/nv_embed_v2.yaml"
    --8<-- "gfmrag/workflow/config/el_model/nv_embed_v2.yaml"
    ```

|     Parameter      |                                                   Options                                                   |                                                   Note                                                   |
| :----------------: | :---------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------: |
|     `_target_`     |                       `gfmrag.graph_index_construction.entity_linking_model.NVEmbedV2ELModel`                        | The class name of [Nvidia Embedding model][gfmrag.graph_index_construction.entity_linking_model.NVEmbedV2ELModel] |
|    `model_name`    |                                            `nvidia/NV-Embed-v2`                                             |                                 The name of the Nvidia embedding model.                                  |
|       `root`       |                                                    None                                                     |                                     The root directory of the model.                                     |
|    `use_cache`     |                                               `True`, `False`                                               |                                          Whether to use cache.                                           |
|    `normalize`     |                                               `True`, `False`                                               |                                   Whether to normalize the embeddings.                                   |
|  `query_instruct`  | `Instruct: Given a entity, retrieve entities that are semantically equivalent to the given entity\nQuery: ` |                                      The instruction for the query.                                      |
| `passage_instruct` |                                                    None                                                     |                                     The instruction for the passage.                                     |
|   `model_kwargs`   |                                                    `{}`                                                     |                                     The additional model arguments.                                      |


Please refer to [NVEmbedV2 EL Model][gfmrag.graph_index_construction.entity_linking_model.NVEmbedV2ELModel] for details on the other parameters.
