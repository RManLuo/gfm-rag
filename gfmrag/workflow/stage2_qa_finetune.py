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
from gfmrag.datasets import QADataset
from gfmrag.ultra import query_utils
from gfmrag.utils import GraphDatasetLoader

# A logger for this file
logger = logging.getLogger(__name__)

separator = ">" * 30
line = "-" * 30


def create_qa_dataloader(
    dataset: dict[str, QADataset],
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
    qa_data = dataset["data"]
    train_data, valid_data = qa_data._data
    data = train_data if is_train else valid_data

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
        "graph": qa_data.kg,
        "ent2docs": qa_data.ent2docs,
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
    has_doc_loss = False
    for loss_cfg in cfg.task.losses:
        loss_fn = instantiate(loss_cfg.loss)
        if loss_cfg.cfg.is_doc_loss:
            has_doc_loss = True
        loss_fn_list.append(
            {
                "name": loss_cfg.name,
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
            ent2docs = train_dataset["ent2docs"].to(device)
            entities_weight = None
            if cfg.train.init_entities_weight:
                entities_weight = utils.get_entities_weight(ent2docs)
            batch_per_epoch = batch_per_epoch or len(train_loader)
            for batch in tqdm(
                islice(train_loader, batch_per_epoch),
                desc=f"Training Batches: {data_name}: {epoch}",
                total=batch_per_epoch,
                disable=not utils.is_main_process(),
            ):
                batch = query_utils.cuda(batch, device=device)
                pred = parallel_model(graph, batch, entities_weight=entities_weight)
                target = batch["supporting_entities_masks"]  # supporting_entities_mask

                if has_doc_loss:
                    doc_pred = torch.sparse.mm(pred, ent2docs)
                    doc_target = batch["supporting_docs_masks"]  # supporting_docs_mask

                loss = 0
                tmp_losses = {}
                for loss_dict in loss_fn_list:
                    loss_fn = loss_dict["loss_fn"]
                    weight = loss_dict["weight"]
                    if loss_dict["is_doc_loss"]:
                        single_loss = loss_fn(doc_pred, doc_target)
                    else:
                        single_loss = loss_fn(pred, target)
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
                batch_id += 1

        if utils.get_rank() == 0:
            logger.info(separator)
            logger.info(f"Epoch {epoch} end")
            logger.info(line)
            for loss_log in losses:
                logger.info(
                    f"Avg: {loss_log}: {sum(losses[loss_log]) / len(losses[loss_log]):g}"
                )

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
                torch.save(state, os.path.join(output_dir, "model_best.pth"))
            if not cfg.train.save_best_only:
                logger.info(f"Save checkpoint to model_epoch_{epoch}.pth")
                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, os.path.join(output_dir, f"model_epoch_{epoch}.pth"))
            logger.info(f"Best mrr: {best_result:g} at epoch {best_epoch}")

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
    all_metrics = {}
    all_mrr = []
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
        ent2docs = dataset["ent2docs"].to(device)

        model.eval()
        ent_preds = []
        ent_targets = []
        doc_preds = []
        doc_targets = []

        # Create doc retriever
        doc_ranker = instantiate(
            cfg.doc_ranker,
            ent2doc=ent2docs,
        )

        entities_weight = None
        if cfg.train.init_entities_weight:
            entities_weight = utils.get_entities_weight(ent2docs)

        for batch in tqdm(
            test_loader,
            desc=f"Testing {data_name}",
            disable=not utils.is_main_process(),
        ):
            batch = query_utils.cuda(batch, device=device)
            ent_pred = model(graph, batch, entities_weight=entities_weight)
            doc_pred = doc_ranker(ent_pred)  # Ent2docs mapping
            target_entities_mask = batch[
                "supporting_entities_masks"
            ]  # supporting_entities_mask
            target_docs_mask = batch["supporting_docs_masks"]  # supporting_docs_mask
            target_entities = target_entities_mask.bool()
            target_docs = target_docs_mask.bool()
            ent_ranking, target_ent_ranking = utils.batch_evaluate(
                ent_pred, target_entities
            )
            doc_ranking, target_doc_ranking = utils.batch_evaluate(
                doc_pred, target_docs
            )

            # answer set cardinality prediction
            ent_prob = F.sigmoid(ent_pred)
            num_pred = (ent_prob * (ent_prob > 0.5)).sum(dim=-1)
            num_target = target_entities_mask.sum(dim=-1)
            ent_preds.append((ent_ranking, num_pred))
            ent_targets.append((target_ent_ranking, num_target))

            # document set cardinality prediction
            doc_prob = F.sigmoid(doc_pred)
            num_pred = (doc_prob * (doc_prob > 0.5)).sum(dim=-1)
            num_target = target_docs_mask.sum(dim=-1)
            doc_preds.append((doc_ranking, num_pred))
            doc_targets.append((target_doc_ranking, num_target))

        ent_pred = query_utils.cat(ent_preds)
        ent_target = query_utils.cat(ent_targets)
        doc_pred = query_utils.cat(doc_preds)
        doc_target = query_utils.cat(doc_targets)

        ent_pred, ent_target = utils.gather_results(
            ent_pred, ent_target, rank, world_size, device
        )
        doc_pred, doc_target = utils.gather_results(
            doc_pred, doc_target, rank, world_size, device
        )
        ent_metrics = utils.evaluate(ent_pred, ent_target, cfg.task.metric)
        metrics = {}
        if rank == 0:
            doc_metrics = utils.evaluate(doc_pred, doc_target, cfg.task.metric)
            for key, value in ent_metrics.items():
                metrics[f"ent_{key}"] = value
            for key, value in doc_metrics.items():
                metrics[f"doc_{key}"] = value
            metrics["mrr"] = ent_metrics["mrr"]
            logger.info(f"{'-' * 15} Test on {data_name} {'-' * 15}")
            query_utils.print_metrics(metrics, logger)
        else:
            metrics["mrr"] = ent_metrics["mrr"]
        all_metrics[data_name] = metrics
        all_mrr.append(metrics["mrr"])
    utils.synchronize()
    all_avg_mrr = np.mean(all_mrr)
    return all_avg_mrr if not return_metrics else metrics


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
        rel_emb_dim_list = utils.init_multi_dataset(
            cfg, utils.get_world_size(), utils.get_rank()
        )
        rel_emb_dim = set(rel_emb_dim_list)
        assert len(rel_emb_dim) == 1, (
            "All datasets should have the same relation embedding dimension"
        )
    else:
        assert cfg.datasets.feat_dim is not None, (
            "If datasets.init_datasets is False, cfg.datasets.feat_dim must be set"
        )
        rel_emb_dim = {cfg.datasets.feat_dim}
    if utils.get_rank() == 0:
        logger.info(
            f"Datasets {cfg.datasets.train_names} and {cfg.datasets.valid_names} initialized"
        )

    device = utils.get_device()
    model = instantiate(cfg.model, rel_emb_dim=rel_emb_dim.pop())

    if "checkpoint" in cfg.train and cfg.train.checkpoint is not None:
        if os.path.exists(cfg.train.checkpoint):
            state = torch.load(
                cfg.train.checkpoint, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state["model"], strict=False)
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


if __name__ == "__main__":
    main()
