# SFT Constructor Configuration

SFT constructors turn raw QA files into processed supervision data for downstream training and evaluation.

## GFM-RAG SFT Constructor

An example GFM-RAG SFT constructor configuration file is shown below:

!!! example "gfm_rag_sft_constructor"

    ```yaml title="gfmrag/workflow/config/sft_constructor/gfm_rag_sft_constructor.yaml"
    --8<-- "gfmrag/workflow/config/sft_constructor/gfm_rag_sft_constructor.yaml"
    ```

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_target_` | `gfmrag.graph_index_construction.sft_constructors.GFMRAGConstructor` | The class name of `GFMRAGConstructor`. |
| `root` | `tmp/qa_construction` | Temporary directory for processed intermediate files. |
| `ner_model` | `${ner_model}` | NER config used to identify start entities. |
| `el_model` | `${el_model}` | EL config used to map entities into the graph. |
| `num_processes` | Positive integer | Number of worker processes used during preprocessing. |
| `force` | `True`, `False` | Whether to recompute processed outputs. |

## GFM-Reasoner SFT Constructor

An example GFM-Reasoner SFT constructor configuration file is shown below:

!!! example "gfm_reasoner_sft_constructor"

    ```yaml title="gfmrag/workflow/config/sft_constructor/gfm_reasoner_sft_constructor.yaml"
    --8<-- "gfmrag/workflow/config/sft_constructor/gfm_reasoner_sft_constructor.yaml"
    ```

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_target_` | `gfmrag.graph_index_construction.sft_constructors.GFMReasonerConstructor` | The class name of `GFMReasonerConstructor`. |
| `root` | `tmp/qa_construction` | Temporary directory for processed intermediate files. |
| `ner_model` | `${ner_model}` | NER config used to identify start entities. |
| `el_model` | `${el_model}` | EL config used to map entities into the graph. |
| `num_processes` | Positive integer | Number of worker processes used during preprocessing. |
| `force` | `True`, `False` | Whether to recompute processed outputs. |

## HippoRAG2 SFT Constructor

An example HippoRAG2 SFT constructor configuration file is shown below:

!!! example "hipporag2_sft_constructor"

    ```yaml title="gfmrag/workflow/config/sft_constructor/hipporag2_sft_constructor.yaml"
    --8<-- "gfmrag/workflow/config/sft_constructor/hipporag2_sft_constructor.yaml"
    ```

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_target_` | `gfmrag.graph_index_construction.sft_constructors.HippoRAG2Constructor` | The class name of `HippoRAG2Constructor`. |
| `root` | `tmp/qa_construction` | Temporary directory for processed intermediate files. |
| `text_emb_model` | `${text_emb_model}` | Text embedding model used for candidate generation. |
| `enable_filtering` | `True`, `False` | Whether to run LLM-based filtering over candidate facts. |
| `num_processes` | Positive integer | Number of worker processes. |
| `topk` | Positive integer | Number of candidate nodes selected per question. |
| `llm_for_filtering` | Model name | LLM used for fact filtering (e.g. `gpt-4o-mini`). |
| `retry` | Positive integer | Retry count for filtering calls. |
| `force` | `True`, `False` | Whether to recompute processed outputs. |
| `start_type` | List of node types | Node types allowed in start nodes. |
| `target_type` | List of node types | Node types allowed in target nodes. |
