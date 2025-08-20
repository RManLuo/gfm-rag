# GFM-RAG Fine-tuning Configuration
An example configuration file for GFM fine-tuning is shown below:

!!! example

    ```yaml title="gfmrag/workflow/config/stage2_qa_finetune.yaml"
    --8<-- "gfmrag/workflow/config/stage2_qa_finetune.yaml"
    ```

## General Configuration

| Parameter | Options |              Note               |
| :-------: | :-----: | :-----------------------------: |
| `run.dir` |  None   | The output directory of the log |

## Defaults

|    Parameter     | Options |                                 Note                                  |
| :--------------: | :-----: | :-------------------------------------------------------------------: |
| `text_emb_model` |  None   | The [text embedding model][text-embedding-model-configuration] to use |


## Training datasets

|         Parameter          | Options |                                  Note                                   |
| :------------------------: | :-----: | :---------------------------------------------------------------------: |
|        `_target_`        |  None   |                 [QADataset][gfmrag.graph_index_datasets.QADataset]                  |
|        `cfgs.root`         |  None   |               root dictionary of the datasets saving path               |
|    `cfgs.force_rebuild`    |  None   |                  whether to force rebuild the dataset                   |
| `cfgs.text_emb_model_cfgs` |  None   | [text embedding model][text-embedding-model-configuration]configuration |
|       `train_names`        |  `[]`   |                     List of training dataset names                      |
|       `valid_names`        |  `[]`   |                    List of validation dataset names    |

## GFM model configuration

|    Parameter     |            Options             |                          Note                          |
| :--------------: | :----------------------------: | :----------------------------------------------------: |
|   `_target_`   |              None              |        [QueryGNN][gfmrag.models.QueryGNN] model        |
|  `entity_model`  |              None              | [EntityNBFNet][gfmrag.ultra.models.QueryNBFNet] model |
|   `input_dim`    |              None              |              input dimension of the model              |
|  `hidden_dims`   |              `[]`              |             hidden dimensions of the model             |
|  `message_func`  |  `transe`,`rotate`,`distmult`  |             message function of the model              |
| `aggregate_func` | `pna`,`min`,`max`,`mean`,`sum` |            aggregate function of the model             |
|   `short_cut`    |        `True`, `False`         |                whether to use short cut                |
|   `layer_norm`   |        `True`, `False`         |               whether to use layer norm                |


## Loss configuration

|         Parameter         | Options |                     Note                      |
| :-----------------------: | :-----: | :-------------------------------------------: |
```
|     `strict_negative`     |  None   |    whether to use strict negative sampling    |
| `metric` |  None   |    evaluation metrics to use    |
| `losses` |  None   |    list of losses to use    |
| `losses[].name` |  None   |    name of the loss    |
| `losses[]._target_` |  None   |    [loss function][gfmrag.losses.BaseLoss] to use    |
| `losses[].cfg` |  None   |    configuration of the loss    |
| `losses[].cfg.weight` |  None   |    weight of the loss    |
| `losses[].cfg.is_doc_loss` |  None   |    whether the loss is for document    |

## Optimizer configuration

|   Parameter   | Options |                        Note                        |
| :-----------: | :-----: | :------------------------------------------------: |
| `optimizer._target_` |  None   |         torch optimizer for the model         |
| `optimizer.lr` |  None   |        learning rate for the optimizer        |

## Training configuration

|     Parameter     | Options |                        Note                        |
| :---------------: | :-----: | :------------------------------------------------: |
|   `batch_size`    |  None   |            batch size for the training             |
|    `num_epoch`    |  None   |           number of epochs for training            |
|  `log_interval`   |  None   |         logging interval for the training          |
|    `fast_test`    |  None   |          number of samples for fast test           |
| `save_best_only`  |  None   | whether to save the best model based on the metric |
| `save_pretrained` |  None   |     whether to save the model for QA inference     |
| `batch_per_epoch` |  None   |      number of batches per epoch for training      |
|     `timeout`     |  None   |       timeout minutes for multi-gpu training       |
|   `checkpoint`    |  None   |          checkpoint path for the training          |
