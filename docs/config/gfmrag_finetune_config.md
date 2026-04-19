# GFM-RAG SFT Training Config

This page documents `gfmrag/workflow/config/gfm_rag/sft_training.yaml`.

## Purpose

This preset is used by `python -m gfmrag.workflow.sft_training` for supervised fine-tuning and retrieval evaluation in the original `GFM-RAG` model family.

!!! example "gfmrag/workflow/config/gfm_rag/sft_training.yaml"

    ```yaml title="gfmrag/workflow/config/gfm_rag/sft_training.yaml"
    --8<-- "gfmrag/workflow/config/gfm_rag/sft_training.yaml"
    ```

## Top-level Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `hydra.run.dir` | `outputs/qa_finetune/<date>/<time>/` | Directory used by Hydra for runtime logs and outputs. |
| `defaults` | List of config groups | Pulls in `doc_ranker`, `text_emb_model`, and `wandb`. |
| `datasets` | Mapping | Selects indexed datasets and feature settings. |
| `model` | Mapping | Configures `gfmrag.models.gfm_rag_v1.GNNRetriever`. |
| `losses` | List | Defines one or more supervised losses. |
| `optimizer` | Mapping | Sets optimizer type and learning rate. |
| `trainer` | Mapping | Configures `SFTTrainer`, evaluation metrics, and prediction behavior. |

## `datasets` Fields

The default dataset class is `GraphIndexDatasetV1`.

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `datasets.cfgs.root` | Any valid data root | Root directory containing indexed datasets. |
| `datasets.cfgs.target_type` | Node type string | Graph target node type. |
| `datasets.cfgs.use_node_feat` | `True`, `False` | Whether to use node features. |
| `datasets.cfgs.use_edge_feat` | `True`, `False` | Whether to use edge features. |
| `datasets.cfgs.use_relation_feat` | `True`, `False` | Whether to use relation features. |
| `datasets.train_names` | List of dataset names | Training split list. |
| `datasets.valid_names` | List of dataset names | Validation split list. |
| `datasets.max_datasets_in_memory` | Positive integer | Maximum number of datasets kept in memory. |
| `datasets.data_loading_workers` | Positive integer | Number of background loading workers. |

## `model` Fields

The default model target is `gfmrag.models.gfm_rag_v1.GNNRetriever`.

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `model.init_nodes_weight` | `True`, `False` | Whether to initialize node weights at input. |
| `model.init_nodes_type` | Node type string | Node type used for initialization. |
| `model.ranker` | Mapping | Shared document ranker config. |
| `model.entity_model` | Mapping | Nested `QueryNBFNet` settings. |

## `losses` Fields

The default preset includes two losses: `bce_loss` and `pcr_loss`. Each loss block defines:

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `losses[].name` | Any string | Human-readable loss name. |
| `losses[].loss._target_` | Loss class path | Concrete loss implementation. |
| `losses[].weight` | Float | Loss weight in the total objective. |
| `losses[].target_node_type` | Node type string | Target node type for that loss. |
| `losses[].is_distillation_loss` | `True`, `False` | Optional distillation flag. |

## `trainer` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `trainer.args.train_batch_size` | Positive integer | Training batch size. |
| `trainer.args.num_epoch` | Positive integer | Number of epochs. |
| `trainer.args.do_train` | `True`, `False` | Whether to run training. |
| `trainer.args.do_eval` | `True`, `False` | Whether to run evaluation. |
| `trainer.args.save_best_only` | `True`, `False` | Whether to save only the best checkpoint. |
| `trainer.args.metric_for_best_model` | Metric name | Metric used to select the best checkpoint. |
| `trainer.metrics` | List of metric names | Evaluation metrics. |
| `trainer.target_types` | List of node types | Node types used in evaluation. |

## Related Shared Configs

- [Document Ranker Config](doc_ranker_config.md)
- [Text Embedding Config](text_embedding_config.md)
- [Wandb Config](wandb_config.md)
