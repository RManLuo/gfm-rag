# G-reasoner SFT Training Configuration

This page documents the supervised fine-tuning presets in `gfmrag/workflow/config/gfm_reasoner/`.

## `sft_training.yaml`

This preset is used by `python -m gfmrag.workflow.sft_training --config-name gfm_reasoner/sft_training`.

!!! example "gfmrag/workflow/config/gfm_reasoner/sft_training.yaml"

    ```yaml title="gfmrag/workflow/config/gfm_reasoner/sft_training.yaml"
    --8<-- "gfmrag/workflow/config/gfm_reasoner/sft_training.yaml"
    ```

### Top-level Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `hydra.run.dir` | `outputs/qa_finetune/${now:%Y-%m-%d}/${now:%H-%M-%S}` | Directory used by Hydra for SFT outputs. |
| `hydra.searchpath` | `pkg://gfmrag.workflow.config` | Adds the packaged workflow config directory to Hydra's search path. |
| `defaults` | List of config groups | Selects the text embedding and wandb presets. |
| `seed` | Integer | Random seed used during training. |
| `timeout` | Positive integer | Timeout in minutes for multi-GPU training. |
| `save_pretrained` | `yes`, `no` | Whether to save the trained model in pretrained format. |
| `load_model_from_pretrained` | File path or `null` | Optional pretrained checkpoint that overrides the model definition. |
| `datasets` | Mapping | Dataset construction and loading options. |
| `model` | Mapping | `GraphReasoner` model configuration. |
| `losses` | List | Loss definitions used during fine-tuning. |
| `optimizer` | Mapping | Optimizer type and hyperparameters. |
| `trainer` | Mapping | Trainer arguments, evaluation metrics, and target types. |

### `defaults` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_self_` | Current file | Loads the local values in this preset. |
| `text_emb_model` | `qwen3` by default | Text embedding preset used by the dataset loader. |
| `wandb` | `default` by default | Weights and Biases logging preset. |

### `datasets` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_target_` | `gfmrag.graph_index_datasets.GraphIndexDataset` | Dataset class used for graph-index supervision data. |
| `cfgs.root` | Any valid data root | Root directory that contains the indexed datasets. |
| `cfgs.force_reload` | `True`, `False` | Whether to rebuild the dataset cache before loading. |
| `cfgs.text_emb_model_cfgs` | `${text_emb_model}` by default | Text embedding config passed to the dataset loader. |
| `train_names` | List of dataset names | Training dataset splits. |
| `valid_names` | List of dataset names | Validation dataset splits. |
| `init_datasets` | `True`, `False` | Whether to preprocess all listed datasets before training starts. |
| `feat_dim` | Positive integer | Embedding feature dimension used when datasets are not initialized up front. |
| `max_datasets_in_memory` | Positive integer | Maximum number of datasets kept in memory at once. |
| `data_loading_workers` | Positive integer | Number of worker processes for data loading. |

### `model` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_target_` | `gfmrag.models.gfm_reasoner.GraphReasoner` | Model class used for G-reasoner SFT. |
| `use_ent_emb` | `early-late-fusion` and other supported modes | Entity embedding integration mode. |
| `dtype` | `float32`, `float16`, `bfloat16`, `auto` | Precision mode for the model. |
| `entity_model._target_` | `gfmrag.models.ultra.models.QueryNBFNet` | Graph encoder used inside `GraphReasoner`. |
| `entity_model.input_dim` | Positive integer | Input embedding dimension of the entity model. |
| `entity_model.hidden_dims` | List of integers | Hidden dimensions of each entity-model layer. |
| `entity_model.message_func` | Supported message functions such as `distmult` | Message function used by the graph encoder. |
| `entity_model.aggregate_func` | Supported aggregation functions such as `sum` | Aggregation function used by the graph encoder. |
| `entity_model.short_cut` | `yes`, `no` | Whether to enable shortcut connections. |
| `entity_model.layer_norm` | `yes`, `no` | Whether to enable layer normalization. |
| `entity_model.return_hidden` | `True`, `False` | Whether to return hidden states for downstream losses or distillation. |

### `losses` Fields

Each entry in `losses` follows the same schema:

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `name` | Any loss name | Logical name used to identify the loss block. |
| `loss._target_` | Loss class path | Concrete loss implementation, such as `BCELoss`, `ListCELoss`, or `MSELoss`. |
| `loss.adversarial_temperature` | Float | Optional temperature used by adversarial BCE loss. |
| `weight` | Float | Weight assigned to this loss in the total objective. |
| `target_node_type` | `document`, `entity`, or another node type | Node type supervised by this loss. |
| `is_distillation_loss` | `True`, `False` | Marks a loss as distillation-style supervision. |

### `optimizer` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_target_` | `torch.optim.AdamW` by default | Optimizer class used for training. |
| `lr` | Positive float | Learning rate. |

### `trainer` Fields

| Parameter | Options | Note |
| :--: | :--: | :-- |
| `_target_` | `gfmrag.trainers.SFTTrainer` | Trainer class used for fine-tuning. |
| `args._target_` | `gfmrag.trainers.TrainingArguments` | Training-argument class used by the trainer. |
| `args.train_batch_size` | Positive integer | Training batch size. |
| `args.num_epoch` | Positive integer | Number of epochs. |
| `args.logging_steps` | Positive integer | Logging interval in steps. |
| `args.max_steps_per_epoch` | Positive integer or `null` | Optional cap on steps per epoch. |
| `args.resume_from_checkpoint` | File path or `null` | Resume training from a saved checkpoint. |
| `args.do_train` | `true`, `false` | Whether to run training. |
| `args.do_eval` | `true`, `false` | Whether to run evaluation. |
| `args.save_best_only` | `yes`, `no` | Whether to only keep the best checkpoint. |
| `args.metric_for_best_model` | Metric name | Metric used to select the best checkpoint. |
| `args.dtype` | `${model.dtype}` by default | Trainer-side precision mode. |
| `args.split_graph_inference` | `true`, `false` | Whether to enable split-graph inference. |
| `args.split_graph_training` | `true`, `false` | Whether to enable split-graph training. |
| `args.split_graph_partition` | `contiguous`, `metis`, or supported methods | Partition strategy used for split-graph execution. |
| `metrics` | List of metric names | Ranking metrics computed during evaluation. |
| `target_types` | List of node types | Node types included in metric computation. |

## `sft_training_w_answer.yaml`

This preset extends the base SFT configuration with additional answer supervision.

!!! example "gfmrag/workflow/config/gfm_reasoner/sft_training_w_answer.yaml"

    ```yaml title="gfmrag/workflow/config/gfm_reasoner/sft_training_w_answer.yaml"
    --8<-- "gfmrag/workflow/config/gfm_reasoner/sft_training_w_answer.yaml"
    ```

### Additional Differences

Compared with `sft_training.yaml`, this variant:

- adds `entity`-targeted `bce_loss`, `pcr_loss`, and `mse_loss` blocks
- changes `trainer.target_types` from `[document]` to `[document, entity]`
- keeps the same dataset, model, optimizer, and trainer structure

## Related Configurations

- [Text Embedding Config](text_embedding_config.md)
- [Wandb Config](wandb_config.md)
