# GFM-RAG KGC Training Config

This page documents `gfmrag/workflow/config/gfm_rag/kgc_training.yaml`.

## Purpose

This preset is used by `python -m gfmrag.workflow.kgc_training` to train the original `GFM-RAG` query-side graph model on graph construction data.

!!! example "gfmrag/workflow/config/gfm_rag/kgc_training.yaml"

    ```yaml title="gfmrag/workflow/config/gfm_rag/kgc_training.yaml"
    --8<-- "gfmrag/workflow/config/gfm_rag/kgc_training.yaml"
    ```

## Top-level Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `hydra.run.dir` | `outputs/kg_pretrain/<date>/<time>/` | Directory used by Hydra for runtime logs and outputs. |
| `defaults` | List of config groups | Pulls in `text_emb_model` and `wandb`. |
| `datasets` | Mapping | Defines the training and validation dataset loader. |
| `model` | Mapping | Configures `gfmrag.models.gfm_rag_v1.QueryGNN`. |
| `optimizer` | Mapping | Sets optimizer type and learning rate. |
| `trainer` | Mapping | Configures `KGCTrainer` and negative-sampling behavior. |

## `datasets` Fields

The default dataset class is `GraphIndexDatasetV1`.

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `datasets.cfgs.root` | Any valid data root | Root directory containing datasets. |
| `datasets.cfgs.text_emb_model_cfgs` | `${text_emb_model}` | Shared text embedding model config. |
| `datasets.cfgs.target_type` | Node type string | Node type used as the graph target. |
| `datasets.train_names` | List of dataset names | Training dataset list. |
| `datasets.valid_names` | List of dataset names | Validation dataset list. |
| `datasets.init_datasets` | `True`, `False` | Whether to preprocess all listed datasets before training. |
| `datasets.feat_dim` | Positive integer | Required when `init_datasets=false`. |

## `model` Fields

The default model target is `gfmrag.models.gfm_rag_v1.QueryGNN`.

The nested `entity_model` block configures `QueryNBFNet`, including:

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `entity_model.input_dim` | Positive integer | Input embedding dimension. |
| `entity_model.hidden_dims` | List of integers | Hidden dimensions of each layer. |
| `entity_model.message_func` | `distmult` and others | Message function used by the graph encoder. |
| `entity_model.aggregate_func` | `sum` and others | Aggregation function used by the graph encoder. |
| `entity_model.short_cut` | `yes`, `no` | Whether to enable shortcut connections. |
| `entity_model.layer_norm` | `yes`, `no` | Whether to enable layer normalization. |

## `trainer` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `trainer.args.train_batch_size` | Positive integer | Training batch size. |
| `trainer.args.num_epoch` | Positive integer | Number of training epochs. |
| `trainer.args.logging_steps` | Positive integer | Logging interval in steps. |
| `trainer.num_negative` | Positive integer | Number of negative samples per query. |
| `trainer.strict_negative` | `True`, `False` | Whether to sample strict negatives. |
| `trainer.adversarial_temperature` | Float | Negative-sampling temperature. |
| `trainer.metrics` | List of metric names | Evaluation metrics such as `mr`, `mrr`, `hits@k`. |
| `trainer.fast_test` | Positive integer | Number of samples used in fast evaluation mode. |

## Related Shared Configs

- [Text Embedding Config](text_embedding_config.md)
- [Wandb Config](wandb_config.md)
