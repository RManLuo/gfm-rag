import copy
import logging
import math
import os
from functools import partial
from itertools import islice

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F  # noqa:N812
from torch.utils import data as torch_data
from torch_geometric.data import Data
from tqdm import tqdm

from gfmrag import utils
from gfmrag.ultra import tasks

# A logger for this file
logger = logging.getLogger(__name__)

separator = ">" * 30
line = "-" * 30


def multigraph_collator(batch: list, train_graphs: list) -> tuple[Data, torch.Tensor]:
    probs = torch.tensor([graph.edge_index.shape[1] for graph in train_graphs]).float()
    probs /= probs.sum()
    graph_id = torch.multinomial(probs, 1, replacement=False).item()

    graph = train_graphs[graph_id]
    bs = len(batch)
    edge_mask = torch.randperm(graph.target_edge_index.shape[1])[:bs]

    colleted_batch = torch.cat(
        [
            graph.target_edge_index[:, edge_mask],
            graph.target_edge_type[edge_mask].unsqueeze(0),
        ]
    ).t()
    return graph, colleted_batch


def train_and_validate(
    cfg: DictConfig,
    output_dir: str,
    model: nn.Module,
    kg_data_list: list[Data],
    valid_data_list: dict[str, Data],
    device: torch.device,
    filtered_data_list: list[Data],
    batch_per_epoch: int | None = None,
) -> None:
    if cfg.train.num_epoch == 0:
        return

    world_size = utils.get_world_size()
    rank = utils.get_rank()

    train_triplets = torch.cat(
        [
            torch.cat([g.target_edge_index, g.target_edge_type.unsqueeze(0)]).t()
            for g in kg_data_list
        ]
    )
    if utils.is_main_process():
        logger.info(
            f"Number of training KGs: {len(kg_data_list)} Number of training triplets: {train_triplets.shape[0]}"
        )
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(
        train_triplets,
        cfg.train.batch_size,
        sampler=sampler,
        collate_fn=partial(multigraph_collator, train_graphs=kg_data_list),
    )

    batch_per_epoch = batch_per_epoch or len(train_loader)

    optimizer = instantiate(cfg.optimizer, model.parameters())

    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch):
        epoch = i + 1
        parallel_model.train()

        if utils.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Epoch %d begin" % epoch)

        losses = []
        sampler.set_epoch(epoch)
        for batch in tqdm(
            islice(train_loader, batch_per_epoch),
            desc=f"Training Batches: {epoch}",
            total=batch_per_epoch,
        ):
            train_graph, batch = batch
            batch = tasks.negative_sampling(
                train_graph,
                batch,
                cfg.task.num_negative,
                strict=cfg.task.strict_negative,
            )
            pred = parallel_model(train_graph, batch)
            target = torch.zeros_like(pred)
            target[:, 0] = 1
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            neg_weight = torch.ones_like(pred)
            if cfg.task.adversarial_temperature > 0:
                with torch.no_grad():
                    neg_weight[:, 1:] = F.softmax(
                        pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1
                    )
            else:
                neg_weight[:, 1:] = 1 / cfg.task.num_negative
            loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if utils.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                logger.warning(separator)
                logger.warning(f"binary cross entropy: {loss:g}")
            losses.append(loss.item())
            batch_id += 1

        if utils.get_rank() == 0:
            avg_loss = sum(losses) / len(losses)
            logger.warning(separator)
            logger.warning("Epoch %d end" % epoch)
            logger.warning(line)
            logger.warning(f"average binary cross entropy: {avg_loss:g}")

        utils.synchronize()
        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")

        result = test(
            cfg,
            model,
            valid_data_list,
            filtered_data_list=filtered_data_list,
            device=device,
        )
        if rank == 0:
            if result > best_result:
                best_result = result
                best_epoch = epoch
                logger.warning("Save checkpoint to model_best.pth")
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, os.path.join(output_dir, "model_best.pth"))
            if not cfg.train.save_best_only:
                logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(
                    state, os.path.join(output_dir, "model_epoch_%d.pth" % epoch)
                )
            logger.warning(f"Best mrr: {best_result:g} at epoch {best_epoch}")
    utils.synchronize()
    if rank == 0:
        logger.warning("Load checkpoint from model_best.pth")
    state = torch.load(
        os.path.join(output_dir, "model_best.pth"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state["model"])
    utils.synchronize()


@torch.no_grad()
def test(
    cfg: DictConfig,
    model: nn.Module,
    test_data_list: dict[str, Data],
    device: torch.device,
    filtered_data_list: list[Data],
    return_metrics: bool = False,
) -> float | dict:
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    # test_data is a tuple of validation/test datasets
    # process sequentially
    all_metrics = {}
    all_mrr = []
    for test_data_tuple, filtered_data in zip(
        test_data_list.items(), filtered_data_list
    ):
        test_data_name, test_data = test_data_tuple
        test_triplets = torch.cat(
            [test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]
        ).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        test_loader = torch_data.DataLoader(
            test_triplets, cfg.train.batch_size, sampler=sampler
        )

        model.eval()
        rankings = []
        num_negatives = []
        tail_rankings, num_tail_negs = (
            [],
            [],
        )  # for explicit tail-only evaluation needed for 5 datasets
        for batch in tqdm(test_loader):
            t_batch, h_batch = tasks.all_negative(test_data, batch)
            t_pred = model(test_data, t_batch)
            h_pred = model(test_data, h_batch)

            if filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
            else:
                t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
            pos_h_index, pos_t_index, pos_r_index = batch.t()
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            num_t_negative = t_mask.sum(dim=-1)
            num_h_negative = h_mask.sum(dim=-1)

            rankings += [t_ranking, h_ranking]
            num_negatives += [num_t_negative, num_h_negative]

            tail_rankings += [t_ranking]
            num_tail_negs += [num_t_negative]

        ranking = torch.cat(rankings)
        num_negative = torch.cat(num_negatives)
        all_size = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size[rank] = len(ranking)

        # ugly repetitive code for tail-only ranks processing
        tail_ranking = torch.cat(tail_rankings)
        num_tail_neg = torch.cat(num_tail_negs)
        all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size_t[rank] = len(tail_ranking)
        if world_size > 1:
            dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

        # obtaining all ranks
        cum_size = all_size.cumsum(0)
        all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_ranking[cum_size[rank] - all_size[rank] : cum_size[rank]] = ranking
        all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_num_negative[cum_size[rank] - all_size[rank] : cum_size[rank]] = (
            num_negative
        )

        # the same for tails-only ranks
        cum_size_t = all_size_t.cumsum(0)
        all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
        all_ranking_t[cum_size_t[rank] - all_size_t[rank] : cum_size_t[rank]] = (
            tail_ranking
        )
        all_num_negative_t = torch.zeros(
            all_size_t.sum(), dtype=torch.long, device=device
        )
        all_num_negative_t[cum_size_t[rank] - all_size_t[rank] : cum_size_t[rank]] = (
            num_tail_neg
        )
        if world_size > 1:
            dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)

        metrics = {}
        if rank == 0:
            logger.warning(f"{'-' * 15} Test on {test_data_name} {'-' * 15}")
            for metric in cfg.task.metric:
                if "-tail" in metric:
                    _metric_name, direction = metric.split("-")
                    if direction != "tail":
                        raise ValueError("Only tail metric is supported in this mode")
                    _ranking = all_ranking_t
                    _num_neg = all_num_negative_t
                else:
                    _ranking = all_ranking
                    _num_neg = all_num_negative
                    _metric_name = metric

                if _metric_name == "mr":
                    score = _ranking.float().mean()
                elif _metric_name == "mrr":
                    score = (1 / _ranking.float()).mean()
                elif _metric_name.startswith("hits@"):
                    values = _metric_name[5:].split("_")
                    threshold = int(values[0])
                    if len(values) > 1:
                        num_sample = int(values[1])
                        # unbiased estimation
                        fp_rate = (_ranking - 1).float() / _num_neg
                        score = 0
                        for i in range(threshold):
                            # choose i false positive from num_sample - 1 negatives
                            num_comb = (
                                math.factorial(num_sample - 1)
                                / math.factorial(i)
                                / math.factorial(num_sample - i - 1)
                            )
                            score += (
                                num_comb
                                * (fp_rate**i)
                                * ((1 - fp_rate) ** (num_sample - i - 1))
                            )
                        score = score.mean()
                    else:
                        score = (_ranking <= threshold).float().mean()
                logger.warning(f"{metric}: {score:g}")
                metrics[metric] = score
        mrr = (1 / all_ranking.float()).mean()
        all_mrr.append(mrr)
        all_metrics[test_data_name] = metrics
        if rank == 0:
            logger.warning(separator)
    avg_mrr = sum(all_mrr) / len(all_mrr)
    return avg_mrr if not return_metrics else all_metrics


@hydra.main(config_path="config", config_name="stage2_kg_pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    utils.init_distributed_mode(cfg.train.timeout)
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")

    datasets = utils.get_multi_dataset(cfg)
    kg_datasets = {
        k: v[0] for k, v in datasets.items()
    }  # Only use the first element (KG) for training

    if utils.is_main_process():
        for name, g in kg_datasets.items():
            # Show the number of entities, relations, and triples in the dataset
            # The number of relations is divided by 2 because the dataset stores both the forward and backward relations
            logger.info(
                f"Dataset {name}: #Entities: {g.num_nodes}, #Relations: {g.num_relations // 2}, #Triples: {len(g.target_edge_type)}"
            )

    device = utils.get_device()
    kg_data_list = [g.to(device) for g in kg_datasets.values()]

    rel_emb_dim = {kg.rel_emb.shape[-1] for kg in kg_data_list}
    assert (
        len(rel_emb_dim) == 1
    ), "All datasets should have the same relation embedding dimension"

    model = instantiate(cfg.model, rel_emb_dim=rel_emb_dim.pop())

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        if os.path.exists(cfg.checkpoint):
            state = torch.load(cfg.checkpoint, map_location="cpu")
            model.load_state_dict(state["model"])
        # Try to load the model from the remote dictionary
        else:
            model, _ = utils.load_model_from_pretrained(cfg.checkpoint)

    model = model.to(device)

    val_filtered_data = [
        Data(
            edge_index=g.target_edge_index,
            edge_type=g.target_edge_type,
            num_nodes=g.num_nodes,
        ).to(device)
        for g in kg_data_list
    ]

    # By default, we use the full validation set
    if "fast_test" in cfg.train:
        num_val_edges = cfg.train.fast_test
        if utils.is_main_process():
            logger.info(f"Fast evaluation on {num_val_edges} samples in validation")
        short_valid = {vn: copy.deepcopy(vd) for vn, vd in kg_datasets.items()}
        for graph in short_valid.values():
            mask = torch.randperm(graph.target_edge_index.shape[1])[:num_val_edges]
            graph.target_edge_index = graph.target_edge_index[:, mask]
            graph.target_edge_type = graph.target_edge_type[mask]

        valid_data_list = {sn: sv.to(device) for sn, sv in short_valid.items()}
    else:
        valid_data_list = {vn: vd.to(device) for vn, vd in kg_datasets.items()}

    train_and_validate(
        cfg,
        output_dir,
        model,
        kg_data_list=kg_data_list,
        valid_data_list=valid_data_list,
        filtered_data_list=val_filtered_data,
        device=device,
        batch_per_epoch=cfg.train.batch_per_epoch,
    )

    if utils.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(
        cfg, model, valid_data_list, filtered_data_list=val_filtered_data, device=device
    )

    # Save the model into the format for QA inference
    if utils.is_main_process() and cfg.train.save_pretrained:
        pre_trained_dir = os.path.join(output_dir, "pretrained")
        utils.save_model_to_pretrained(model, cfg, pre_trained_dir)

    utils.synchronize()
    utils.cleanup()


if __name__ == "__main__":
    main()
