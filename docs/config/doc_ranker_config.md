# Document Ranker Configuration

## Simple Ranker

An example Simple Ranker configuration file is shown below:

!!! example "simple_ranker"

    ```yaml title="gfmrag/workflow/config/doc_ranker/simple_ranker.yaml"
    --8<-- "gfmrag/workflow/config/doc_ranker/simple_ranker.yaml"
    ```

| Parameter  |              Options              |                               Note                                |
| :--------: | :-------------------------------: | :---------------------------------------------------------------: |
| `_target_` | `gfmrag.models.gfm_rag_v1.rankers.SimpleRanker` | The class name of [SimpleRanker][gfmrag.models.gfm_rag_v1.rankers.SimpleRanker] |

## IDF Ranker

An example IDF Ranker configuration file is shown below:

!!! example "idf_ranker"

    ```yaml title="gfmrag/workflow/config/doc_ranker/idf_ranker.yaml"
    --8<-- "gfmrag/workflow/config/doc_ranker/idf_ranker.yaml"
    ```

| Parameter  |                Options                 |                                     Note                                     |
| :--------: | :------------------------------------: | :--------------------------------------------------------------------------: |
| `_target_` | `gfmrag.models.gfm_rag_v1.rankers.IDFWeightedRanker` | The class name of [IDFWeightedRanker ][gfmrag.models.gfm_rag_v1.rankers.IDFWeightedRanker] |

## Top-k Ranker

An example Top-k Ranker configuration file is shown below:

!!! example "topk_ranker"

    ```yaml title="gfmrag/workflow/config/doc_ranker/topk_ranker.yaml"
    --8<-- "gfmrag/workflow/config/doc_ranker/topk_ranker.yaml"
    ```

| Parameter  |             Options             |                             Note                              |
| :--------: | :-----------------------------: | :-----------------------------------------------------------: |
| `_target_` | `gfmrag.models.gfm_rag_v1.rankers.TopKRanker` | The class name of [TopKRanker][gfmrag.models.gfm_rag_v1.rankers.TopKRanker] |
|  `top_k`   |             Integer             |        The top-k entities used for document retrieval         |

## IDF Top-k Ranker

An example IDF Top-k Ranker configuration file is shown below:

!!! example "idf_topk_ranker"

    ```yaml title="gfmrag/workflow/config/doc_ranker/idf_topk_ranker.yaml"
    --8<-- "gfmrag/workflow/config/doc_ranker/idf_topk_ranker.yaml"
    ```

| Parameter  |                  Options                   |                                        Note                                         |
| :--------: | :----------------------------------------: | :---------------------------------------------------------------------------------: |
| `_target_` | `gfmrag.models.gfm_rag_v1.rankers.IDFWeightedTopKRanker` | The class name of [IDFWeightedTopKRanker][gfmrag.models.gfm_rag_v1.rankers.IDFWeightedTopKRanker] |
|  `top_k`   |                  Integer                   |                   The top-k entities used for document retrieval                    |
