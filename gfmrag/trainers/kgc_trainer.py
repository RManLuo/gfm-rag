import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa:N812
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from gfmrag import utils
from gfmrag.graph_index_datasets.graph_dataset_loader import (
    GraphDataset,
    GraphDatasetLoader,
)
from gfmrag.models.ultra import tasks

from .base_trainer import BaseTrainer, TaskDataset
from .training_args import TrainingArguments


@dataclass(kw_only=True)
class PretrainTaskDataset(TaskDataset):
    val_filtered_graph: Data  # The filtered validation graph


class KGCTrainer(BaseTrainer):
    """
    Trainer for Knowledge Graph Completion pretraining
    """

    def __init__(
        self,
        output_dir: str,
        args: TrainingArguments,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_graph_dataset_loader: GraphDatasetLoader | None = None,
        eval_graph_dataset_loader: GraphDatasetLoader | None = None,
        # KG-specific parameters
        num_negative: int = 256,
        strict_negative: bool = True,
        adversarial_temperature: float = 1.0,
        fast_test: int | None = 500,
        # KG-specific parameters
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # Set default metric for best model if not specified
        if args.metric_for_best_model is None:
            args.metric_for_best_model = "mrr"

        super().__init__(
            output_dir=output_dir,
            model=model,
            args=args,
            train_graph_dataset_loader=train_graph_dataset_loader,
            eval_graph_dataset_loader=eval_graph_dataset_loader,
            optimizer=optimizer,
            **kwargs,
        )

        # KG-specific parameters
        self.num_negative = num_negative
        self.strict_negative = strict_negative
        self.adversarial_temperature = adversarial_temperature
        self.fast_test = fast_test
        self.metrics = metrics or ["mr", "mrr", "hits@1", "hits@3", "hits@10"]

    def _create_task_dataset(
        self, graph_dataset: GraphDataset, is_train: bool = True
    ) -> PretrainTaskDataset:
        """Create KGC dataset for training/evaluation."""
        data_name = graph_dataset.name
        graph = graph_dataset.data.graph

        # The original triples for ranking evaluation
        val_filtered_data = Data(
            edge_index=graph.target_edge_index,
            edge_type=graph.target_edge_type,
            num_nodes=graph.num_nodes,
        )

        # Create DataLoader for triples
        if not is_train and self.fast_test is not None:
            # Sample for fast testing
            mask = torch.randperm(graph.target_edge_index.shape[1])[: self.fast_test]
            sampled_target_edge_index = graph.target_edge_index[:, mask]
            sampled_target_edge_type = graph.target_edge_type[mask]
            triples = torch.cat(
                [sampled_target_edge_index, sampled_target_edge_type.unsqueeze(0)]
            ).t()
        else:
            triples = torch.cat(
                [graph.target_edge_index, graph.target_edge_type.unsqueeze(0)]
            ).t()

        sampler = torch.utils.data.DistributedSampler(
            triples,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=is_train,
        )

        batch_size = (
            self.args.train_batch_size if is_train else self.args.eval_batch_size
        )
        data_loader = DataLoader(
            triples,
            batch_size=batch_size,
            sampler=sampler,
        )

        return PretrainTaskDataset(
            name=data_name,
            graph=graph.to(self.device),
            val_filtered_graph=val_filtered_data.to(self.device),
            data_loader=data_loader,
        )

    def train_step(
        self,
        batch: Any,
        task_dataset: PretrainTaskDataset,  # type: ignore[override]
    ) -> dict[str, float | torch.Tensor]:
        """Perform a single training step for KG pretraining."""
        graph = task_dataset.graph

        batch = batch.to(self.device)

        # Negative sampling
        batch = tasks.negative_sampling(
            graph,
            batch,
            self.num_negative,
            strict=self.strict_negative,
        )

        # Forward pass
        pred = self.parallel_model(graph, batch)

        # Create targets (first column should be 1 for positive samples)
        target = torch.zeros_like(pred)
        target[:, 0] = 1

        # Compute BCE loss
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # Apply adversarial weighting
        neg_weight = torch.ones_like(pred)
        if self.adversarial_temperature > 0:
            with torch.no_grad():
                neg_weight[:, 1:] = F.softmax(
                    pred[:, 1:] / self.adversarial_temperature, dim=-1
                )
        else:
            neg_weight[:, 1:] = 1 / self.num_negative

        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()

        return {"loss": loss}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate the model on validation datasets."""
        if self.eval_graph_dataset_loader is None:
            return {}

        self.model.eval()
        all_metrics = {}
        all_mrr = []

        # Set epoch for evaluation
        if hasattr(self.eval_graph_dataset_loader, "set_epoch"):
            self.eval_graph_dataset_loader.set_epoch(0)

        for graph_dataset in self.eval_graph_dataset_loader:
            test_dataset = self._create_task_dataset(graph_dataset, is_train=False)

            data_name = test_dataset.name
            graph = test_dataset.graph
            data_loader = test_dataset.data_loader
            filtered_data = test_dataset.val_filtered_graph

            # Set epoch for sampler
            if hasattr(data_loader.sampler, "set_epoch"):
                data_loader.sampler.set_epoch(0)

            rankings = []
            num_negatives = []
            tail_rankings, num_tail_negs = [], []

            for batch in tqdm(
                data_loader,
                disable=not utils.is_main_process(),
                desc=f"Evaluating {data_name}",
            ):
                with torch.amp.autocast(
                    device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp
                ):
                    batch = batch.to(self.device)
                    t_batch, h_batch = tasks.all_negative(graph, batch)
                    t_pred = self.parallel_model(graph, t_batch)
                    h_pred = self.parallel_model(graph, h_batch)

                    if filtered_data is None:
                        t_mask, h_mask = tasks.strict_negative_mask(graph, batch)
                    else:
                        t_mask, h_mask = tasks.strict_negative_mask(
                            filtered_data, batch
                        )

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
            all_size = torch.zeros(
                self.world_size, dtype=torch.long, device=self.device
            )
            all_size[self.rank] = len(ranking)

            # ugly repetitive code for tail-only ranks processing
            tail_ranking = torch.cat(tail_rankings)
            num_tail_neg = torch.cat(num_tail_negs)
            all_size_t = torch.zeros(
                self.world_size, dtype=torch.long, device=self.device
            )
            all_size_t[self.rank] = len(tail_ranking)
            if self.world_size > 1:
                dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
                dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

            # obtaining all ranks
            cum_size = all_size.cumsum(0)
            all_ranking = torch.zeros(
                all_size.sum(), dtype=torch.long, device=self.device
            )
            all_ranking[
                cum_size[self.rank] - all_size[self.rank] : cum_size[self.rank]
            ] = ranking
            all_num_negative = torch.zeros(
                all_size.sum(), dtype=torch.long, device=self.device
            )
            all_num_negative[
                cum_size[self.rank] - all_size[self.rank] : cum_size[self.rank]
            ] = num_negative

            # the same for tails-only ranks
            cum_size_t = all_size_t.cumsum(0)
            all_ranking_t = torch.zeros(
                all_size_t.sum(), dtype=torch.long, device=self.device
            )
            all_ranking_t[
                cum_size_t[self.rank] - all_size_t[self.rank] : cum_size_t[self.rank]
            ] = tail_ranking
            all_num_negative_t = torch.zeros(
                all_size_t.sum(), dtype=torch.long, device=self.device
            )
            all_num_negative_t[
                cum_size_t[self.rank] - all_size_t[self.rank] : cum_size_t[self.rank]
            ] = num_tail_neg
            if self.world_size > 1:
                dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
                dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
                dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)

            metrics = {}
            if self.rank == 0:
                for metric in self.metrics:
                    if "-tail" in metric:
                        _metric_name, direction = metric.split("-")
                        if direction != "tail":
                            raise ValueError(
                                "Only tail metric is supported in this mode"
                            )
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
                    metrics[metric] = score
                # Log evaluation metrics to wandb
                if self.rank == 0:
                    eval_metrics = {f"{data_name}/{k}": v for k, v in metrics.items()}
                    all_metrics.update(eval_metrics)
            mrr = (1 / all_ranking.float()).mean()
            all_mrr.append(mrr)
        avg_mrr = sum(all_mrr) / len(all_mrr)
        all_metrics["mrr"] = avg_mrr
        return all_metrics
