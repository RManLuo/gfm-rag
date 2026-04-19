# Graph Constructor Configuration

## KG Constructor Configuration

This configuration controls how raw documents are converted into the stage1 graph files (`nodes.csv`, `relations.csv`, `edges.csv`). An example KG constructor configuration file is shown below:

!!! example "kg_constructor"

    ```yaml title="gfmrag/workflow/config/graph_constructor/kg_constructor.yaml"
    --8<-- "gfmrag/workflow/config/graph_constructor/kg_constructor.yaml"
    ```

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_target_` | `gfmrag.graph_index_construction.graph_constructors.KGConstructor` | The class name of `KGConstructor`. |
| `open_ie_model` | `${openie_model}` | OpenIE config used to extract triples from documents. |
| `el_model` | `${el_model}` | Entity-linking config used during graph construction. |
| `root` | `tmp/kg_construction` | Temporary working directory for intermediate files. |
| `num_processes` | Positive integer | Number of worker processes. |
| `cosine_sim_edges` | `True`, `False` | Whether to add similarity-based edges between entities. |
| `threshold` | Float in `[0, 1]` | Cosine similarity threshold for edge creation. |
| `max_sim_neighbors` | Positive integer | Maximum number of similar neighbors to keep. |
| `add_title` | `True`, `False` | Whether to prepend the document title during OpenIE extraction. |
| `force` | `True`, `False` | Whether to rebuild intermediate graph-construction artifacts. |
