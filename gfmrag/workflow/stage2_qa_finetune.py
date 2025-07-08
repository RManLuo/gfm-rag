import logging
import os
from itertools import islice

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F  # noqa:N812
from torch.utils import data as torch_data
from tqdm import tqdm

from gfmrag import utils
from gfmrag.datasets import GraphIndexDataset
from gfmrag.ultra import query_utils
from gfmrag.utils import GraphDatasetLoader
from gfmrag.utils.wandb_utils import (
    finish_wandb,
    init_wandb,
    log_metrics,
    log_model_checkpoint,
    watch_model,
)

# A logger for this file
logger = logging.getLogger(__name__)

separator = ">" * 30
line = "-" * 30


def create_qa_dataloader(
    dataset: dict[str, GraphIndexDataset],
    batch_size: int,
    world_size: int,
    rank: int,
    is_train: bool = True,
    shuffle: bool = True,
) -> dict:
    """
    Create a dataloader for the QA dataset.
    """
    data_name = dataset["data_name"]
    qa_dataset = dataset["data"]
    data = qa_dataset.train_data if is_train else qa_dataset.test_data

    sampler = torch_data.DistributedSampler(
        data,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )
    data_loader = torch_data.DataLoader(
        data,
        batch_size=batch_size,
        sampler=sampler,
    )

    # Return data
    return {
        "data_name": data_name,
        "data_loader": data_loader,
        "graph": qa_dataset.graph,
    }


def train_and_validate(
    cfg: DictConfig,
    output_dir: str,
    model: nn.Module,
    train_dataset_loader: GraphDatasetLoader,
    valid_dataset_loader: GraphDatasetLoader,
    device: torch.device,
    batch_per_epoch: int | None = None,
) -> None:
    if cfg.train.num_epoch == 0:
        return

    world_size = utils.get_world_size()
    rank = utils.get_rank()

    optimizer = instantiate(cfg.optimizer, model.parameters())
    start_epoch = 0
    # Load optimizer state and epoch if exists
    if "checkpoint" in cfg.train and cfg.train.checkpoint is not None:
        if os.path.exists(cfg.train.checkpoint):
            state = torch.load(
                cfg.train.checkpoint, map_location="cpu", weights_only=True
            )
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            else:
                logger.warning(
                    f"Optimizer state not found in {cfg.train.checkpoint}, using default optimizer."
                )
            if "epoch" in state:
                start_epoch = state["epoch"]
                logger.warning(f"Resuming training from epoch {start_epoch}.")
        else:
            logger.warning(
                f"Checkpoint {cfg.train.checkpoint} does not exist, using default optimizer."
            )

    # Initialize Losses
    loss_fn_list = []
    distillation_list = set()
    for loss_cfg in cfg.task.losses:
        loss_fn = instantiate(loss_cfg.loss)
        target_node_type = loss_cfg.cfg["target_node_type"]
        if loss_cfg.cfg.get("is_distillation_loss", False):
            distillation_list.add(target_node_type)
        loss_fn_list.append(
            {
                "name": f"{target_node_type}_{loss_cfg.name}",
                "loss_fn": loss_fn,
                **loss_cfg.cfg,
            }
        )

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(start_epoch, cfg.train.num_epoch):
        epoch = i + 1
        parallel_model.train()

        if utils.get_rank() == 0:
            logger.info(separator)
            logger.info(f"Epoch {epoch} begin")

        losses: dict[str, list] = {loss_dict["name"]: [] for loss_dict in loss_fn_list}
        losses["loss"] = []
        train_dataset_loader.set_epoch(
            epoch
        )  # Make sure the datasets order is the same across all processes
        for train_dataset in train_dataset_loader:
            train_dataset = create_qa_dataloader(
                train_dataset,
                cfg.train.batch_size,
                world_size,
                rank,
                is_train=True,
                shuffle=True,
            )
            train_loader = train_dataset["data_loader"]
            train_loader.sampler.set_epoch(epoch)
            data_name = train_dataset["data_name"]
            graph = train_dataset["graph"].to(device)
            batch_per_epoch = batch_per_epoch or len(train_loader)
            for batch in tqdm(
                islice(train_loader, batch_per_epoch),
                desc=f"Training Batches: {data_name}: {epoch}",
                total=batch_per_epoch,
                disable=not utils.is_main_process(),
            ):
                batch = query_utils.cuda(batch, device=device)
                pred = parallel_model(graph, batch)
                target = batch["target_nodes_mask"]  # target_nodes_mask

                # Get the distillation targets
                distillation_target_list = {}
                for target_node_type in distillation_list:
                    question_emb = batch["question_embeddings"]
                    target_node_ids = graph.nodes_by_type[target_node_type]
                    target_node_emb = graph.x[target_node_ids]
                    distillation_target = question_emb @ target_node_emb.T
                    distillation_target_list[target_node_type] = distillation_target

                loss = 0
                tmp_losses = {}
                for loss_dict in loss_fn_list:
                    loss_fn = loss_dict["loss_fn"]
                    weight = loss_dict["weight"]
                    target_node_type = loss_dict["target_node_type"]
                    target_node_ids = graph.nodes_by_type[target_node_type]
                    # Get the predictions and targets for the current target node type
                    target_node_pred = pred[:, target_node_ids]
                    target_node_label = target[:, target_node_ids]
                    if loss_dict.get("is_distillation_loss", False):
                        single_loss = loss_fn(
                            target_node_pred, distillation_target_list[target_node_type]
                        )
                    else:
                        single_loss = loss_fn(target_node_pred, target_node_label)
                    tmp_losses[loss_dict["name"]] = single_loss.item()
                    loss += weight * single_loss
                tmp_losses["loss"] = loss.item()  # type: ignore

                loss.backward()  # type: ignore
                optimizer.step()
                optimizer.zero_grad()

                for loss_log in tmp_losses:
                    losses[loss_log].append(tmp_losses[loss_log])

                if utils.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.info(separator)
                    for loss_log in tmp_losses:
                        logger.info(f"{loss_log}: {tmp_losses[loss_log]:g}")
                    # Log training losses to wandb
                    train_metrics = {f"train/{k}": v for k, v in tmp_losses.items()}
                    log_metrics(train_metrics, step=batch_id)
                batch_id += 1

        if utils.get_rank() == 0:
            logger.info(separator)
            logger.info(f"Epoch {epoch} end")
            logger.info(line)
            # Calculate and log epoch averages
            epoch_metrics = {}
            for loss_log in losses:
                avg_loss = sum(losses[loss_log]) / len(losses[loss_log])
                logger.info(f"Avg: {loss_log}: {avg_loss:g}")
                epoch_metrics[f"train/epoch_{loss_log}"] = avg_loss
            epoch_metrics["train/epoch"] = epoch
            log_metrics(epoch_metrics)

        utils.synchronize()

        if cfg.train.do_eval:
            if rank == 0:
                logger.info(separator)
                logger.info("Evaluate on valid")
            result = test(cfg, model, valid_dataset_loader, device=device)
        else:
            result = float("inf")
            best_result = float("-inf")
        if rank == 0:
            if result > best_result:
                best_result = result
                best_epoch = epoch
                logger.info("Save checkpoint to model_best.pth")
                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                checkpoint_path = os.path.join(output_dir, "model_best.pth")
                torch.save(state, checkpoint_path)
                # Log best model to wandb
                log_model_checkpoint(
                    checkpoint_path,
                    name=f"best_model_epoch_{epoch}",
                    metadata={
                        "epoch": epoch,
                        "metric": float(result)
                        if isinstance(result, (int, float))
                        else str(result),
                        "best": True,
                    },
                )
            if not cfg.train.save_best_only:
                logger.info(f"Save checkpoint to model_epoch_{epoch}.pth")
                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
                torch.save(state, checkpoint_path)
                # Log epoch model to wandb
                log_model_checkpoint(
                    checkpoint_path,
                    name=f"model_epoch_{epoch}",
                    metadata={
                        "epoch": epoch,
                        "metric": float(result)
                        if isinstance(result, (int, float))
                        else str(result),
                    },
                )
            logger.info(
                f"Best {cfg.train.watched_metric}: {best_result:g} at epoch {best_epoch}"
            )

    if rank == 0:
        logger.info("Load checkpoint from model_best.pth")
    utils.synchronize()
    state = torch.load(
        os.path.join(output_dir, "model_best.pth"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state["model"])


@torch.no_grad()
def test(
    cfg: DictConfig,
    model: nn.Module,
    test_dataset_loader: GraphDatasetLoader,
    device: torch.device,
    return_metrics: bool = False,
) -> float | dict:
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    # process sequentially of test datasets
    watched_metric = cfg.train.get("watched_metric", "document_mrr")
    all_metrics = {}
    all_watched_metric = []
    # Avoid using set() to keep the order of target node types for each process
    target_type_list = []
    for loss in cfg.task.losses:
        if loss.cfg.target_node_type not in target_type_list:
            target_type_list.append(loss.cfg.target_node_type)

    for dataset in test_dataset_loader:
        dataset = create_qa_dataloader(
            dataset,
            cfg.train.batch_size,
            world_size,
            rank,
            is_train=False,
            shuffle=False,
        )
        test_loader = dataset["data_loader"]
        test_loader.sampler.set_epoch(0)
        data_name = dataset["data_name"]
        graph = dataset["graph"].to(device)

        model.eval()

        # Initialize the predictions and targets lists
        preds_list: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
            target_type: [] for target_type in target_type_list
        }
        targets_list: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
            target_type: [] for target_type in target_type_list
        }

        for batch in tqdm(
            test_loader,
            desc=f"Testing {data_name}",
            disable=not utils.is_main_process(),
        ):
            batch = query_utils.cuda(batch, device=device)
            pred = model(graph, batch)
            target = batch["target_nodes_mask"].bool()  # target_nodes_mask
            for node_type in target_type_list:
                target_node_ids = graph.nodes_by_type[node_type]
                pred_by_type = pred[:, target_node_ids]
                target_by_type = target[:, target_node_ids]
                node_ranking, target_node_ranking = utils.batch_evaluate(
                    pred_by_type, target_by_type
                )

                # Answer set cardinality prediction
                node_prob = F.sigmoid(pred_by_type)
                num_pred = (node_prob * (node_prob > 0.5)).sum(dim=-1)
                num_target = target_by_type.sum(dim=-1)
                preds_list[node_type].append((node_ranking, num_pred))
                targets_list[node_type].append((target_node_ranking, num_target))

        metrics_by_type = {}
        for node_type in target_type_list:
            # Concatenate the predictions and targets for the current node type
            preds = preds_list[node_type]
            targets = targets_list[node_type]
            if len(preds) == 0 or len(targets) == 0:
                continue

            # Gather results across all processes
            node_pred, node_target = query_utils.cat(preds), query_utils.cat(targets)
            node_pred, node_target = utils.gather_results(
                node_pred, node_target, rank, world_size, device
            )

            # Evaluate the metrics for the current node type
            metrics_by_type[node_type] = utils.evaluate(
                node_pred, node_target, cfg.task.metric
            )

        metrics = {}
        for node_type, metric in metrics_by_type.items():
            for key, value in metric.items():
                metrics[f"{node_type}_{key}"] = value

        if rank == 0:
            logger.info(f"{'-' * 15} Test on {data_name} {'-' * 15}")
            query_utils.print_metrics(metrics, logger)

        all_metrics[data_name] = metrics
        all_watched_metric.append(metrics[watched_metric])
        # Log evaluation metrics to wandb
        if utils.get_rank() == 0:
            eval_metrics = {f"test/{data_name}/{k}": v for k, v in metrics.items()}
            log_metrics(eval_metrics)
    utils.synchronize()
    all_avg_watched_metric = np.mean(all_watched_metric)
    return all_avg_watched_metric if not return_metrics else metrics


@hydra.main(config_path="config", config_name="stage2_qa_finetune", version_base=None)
def main(cfg: DictConfig) -> None:
    utils.init_distributed_mode(cfg.train.timeout)
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

    device = utils.get_device()
    model = instantiate(cfg.model, feat_dim=feat_dim.pop())

    # Initialize wandb logging (only on rank 0)
    if utils.get_rank() == 0:
        init_wandb(cfg, project_name="gfm-rag-finetune")
        watch_model(model, log_freq=cfg.train.log_interval)

    if "checkpoint" in cfg.train and cfg.train.checkpoint is not None:
        if os.path.exists(cfg.train.checkpoint):
            state = torch.load(
                cfg.train.checkpoint, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state["model"])
        # Try to load the model from the remote dictionary
        else:
            model, _ = utils.load_model_from_pretrained(cfg.train.checkpoint)

    model = model.to(device)
    if utils.get_rank() == 0:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(line)
        logger.info(f"Number of parameters: {num_params}")

    train_dataset_loader = GraphDatasetLoader(
        cfg.datasets,
        cfg.datasets.train_names,
        max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
        data_loading_workers=cfg.datasets.data_loading_workers,
    )
    valid_dataset_loader = GraphDatasetLoader(
        cfg.datasets,
        cfg.datasets.valid_names,
        shuffle=False,
        max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
        data_loading_workers=cfg.datasets.data_loading_workers,
    )

    train_and_validate(
        cfg,
        output_dir,
        model,
        train_dataset_loader,
        valid_dataset_loader,
        device=device,
        batch_per_epoch=cfg.train.batch_per_epoch,
    )

    if cfg.train.do_eval:
        if utils.get_rank() == 0:
            logger.info(separator)
            logger.info("Evaluate on valid")
        test(cfg, model, valid_dataset_loader, device=device)

    # Save the model into the format for QA inference
    if (
        utils.is_main_process()
        and cfg.train.save_pretrained
        and cfg.train.num_epoch > 0
    ):
        pre_trained_dir = os.path.join(output_dir, "pretrained")
        utils.save_model_to_pretrained(model, cfg, pre_trained_dir)

    # Shutdown the dataset loaders
    train_dataset_loader.shutdown()
    valid_dataset_loader.shutdown()

    utils.synchronize()
    utils.cleanup()

    # Finish wandb logging
    if utils.get_rank() == 0:
        finish_wandb()


if __name__ == "__main__":
    main()
