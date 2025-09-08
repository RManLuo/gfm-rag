import json
import logging
import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist

from gfmrag import utils
from gfmrag.graph_index_datasets import GraphDatasetLoader
from gfmrag.trainers.sft_trainer import SFTLoss
from gfmrag.utils.wandb_utils import (
    finish_wandb,
    init_wandb,
    watch_model,
)

# A logger for this file
logger = logging.getLogger(__name__)

separator = ">" * 30
line = "-" * 30


@hydra.main(config_path="config/gfm_rag", config_name="sft_training", version_base=None)
def main(cfg: DictConfig) -> None:
    utils.init_distributed_mode(cfg.timeout)
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        output_dir = HydraConfig.get().runtime.output_dir
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")
        output_dir_list = [output_dir]
    else:
        output_dir_list = [None]
    if utils.get_world_size() > 1:
        dist.broadcast_object_list(
            output_dir_list, src=0
        )  # Use the output dir from rank 0
    output_dir = output_dir_list[0]

    # Initialize the datasets in the each process, make sure they are processed
    if cfg.datasets.init_datasets:
        feat_dim_list = utils.init_multi_dataset(
            cfg, utils.get_world_size(), utils.get_rank()
        )
        feat_dim = set(feat_dim_list)
        assert len(feat_dim) == 1, (
            "All datasets should have the same feature embedding dimension"
        )
    else:
        assert cfg.datasets.feat_dim is not None, (
            "If datasets.init_datasets is False, cfg.datasets.feat_dim must be set"
        )
        feat_dim = {cfg.datasets.feat_dim}
    if utils.get_rank() == 0:
        logger.info(
            f"Datasets {cfg.datasets.train_names} and {cfg.datasets.valid_names} initialized"
        )

    # Load model from pre-trained format, which would overwrite the model configuration
    if cfg.load_model_from_pretrained:
        model, _ = utils.load_model_from_pretrained(cfg.load_model_from_pretrained)
        logger.info(f"Loaded pre-trained model from {cfg.load_model_from_pretrained}")
    else:
        model = instantiate(cfg.model, feat_dim=feat_dim.pop())

    # Initialize wandb logging (only on rank 0)
    if utils.get_rank() == 0:
        init_wandb(cfg, project_name="gfm-rag")
        watch_model(model, log_freq=cfg.trainer.args.get("logging_steps", 1000))

    if utils.get_rank() == 0:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(line)
        logger.info(f"Number of parameters: {num_params}")

    train_graph_dataset_loader = GraphDatasetLoader(
        cfg.datasets,
        cfg.datasets.train_names,
        max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
        data_loading_workers=cfg.datasets.data_loading_workers,
    )
    valid_graph_dataset_loader = GraphDatasetLoader(
        cfg.datasets,
        cfg.datasets.valid_names,
        shuffle=False,
        max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
        data_loading_workers=cfg.datasets.data_loading_workers,
    )

    optimizer = instantiate(cfg.optimizer, model.parameters())

    # Initialize loss
    loss_functions: list[SFTLoss] = []
    for loss_cfg in cfg.losses:
        loss_fn = instantiate(loss_cfg.loss)
        loss_functions.append(
            SFTLoss(
                name=loss_cfg.name,
                loss_fn=loss_fn,
                weight=loss_cfg.weight,
                target_node_type=loss_cfg.target_node_type,
                is_distillation_loss=loss_cfg.get("is_distillation_loss", False),
            )
        )

    trainer = instantiate(
        cfg.trainer,
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        loss_functions=loss_functions,
        train_graph_dataset_loader=train_graph_dataset_loader,
        eval_graph_dataset_loader=valid_graph_dataset_loader,
    )

    trainer.train()

    # Save the model into the format for QA inference
    if utils.is_main_process() and cfg.save_pretrained:
        pre_trained_dir = os.path.join(output_dir, "pretrained")
        utils.save_model_to_pretrained(model, cfg, pre_trained_dir)

    if trainer.args.do_predict:
        predictions = trainer.predict()
        if utils.is_main_process():
            for data_name, preds in predictions.items():
                pred_path = os.path.join(output_dir, f"predictions_{data_name}.json")
                with open(pred_path, "w") as f:
                    json.dump(preds, f, indent=4)
                logger.info(f"Predictions saved to {pred_path}")

    # Shutdown the dataset loaders
    train_graph_dataset_loader.shutdown()
    valid_graph_dataset_loader.shutdown()

    utils.synchronize()
    utils.cleanup()

    # Finish wandb logging
    if utils.get_rank() == 0:
        finish_wandb()


if __name__ == "__main__":
    main()
