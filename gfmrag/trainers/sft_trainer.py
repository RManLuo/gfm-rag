import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa:N812
from torch.utils.data import DataLoader
from tqdm import tqdm

from gfmrag import utils
from gfmrag.graph_index_datasets.graph_dataset_loader import (
    GraphDataset,
    GraphDatasetLoader,
)
from gfmrag.losses import BaseLoss
from gfmrag.models.ultra import query_utils

from .base_trainer import BaseTrainer, TaskDataset
from .training_args import TrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class SFTLoss:
    name: str
    loss_fn: BaseLoss
    weight: float
    target_node_type: str
    is_distillation_loss: bool | None = False


class SFTTrainer(BaseTrainer):
    """
    Trainer for Supervised Fine-Tuning (SFT)
    """

    def __init__(
        self,
        output_dir: str,
        args: TrainingArguments,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_functions: list[SFTLoss],
        train_graph_dataset_loader: GraphDatasetLoader | None = None,
        eval_graph_dataset_loader: GraphDatasetLoader | None = None,
        # SFT-specific parameters
        target_types: list[str] | None = None,
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # Set default metric for best model if not specified
        if args.metric_for_best_model is None:
            args.metric_for_best_model = "document_mrr"

        super().__init__(
            output_dir=output_dir,
            model=model,
            args=args,
            train_graph_dataset_loader=train_graph_dataset_loader,
            eval_graph_dataset_loader=eval_graph_dataset_loader,
            optimizer=optimizer,
            **kwargs,
        )

        # SFT-specific parameters
        self.target_types = target_types or ["entity", "document"]
        self.metrics = metrics or [
            "mrr",
            "hits@1",
            "hits@2",
            "hits@3",
            "hits@5",
            "hits@10",
            "hits@20",
            "hits@50",
            "hits@100",
        ]

        # Initialize loss functions
        self.loss_functions = loss_functions
        self.distillation_types = {
            sft_loss.target_node_type
            for sft_loss in self.loss_functions
            if sft_loss.is_distillation_loss
        }

    def _create_task_dataset(
        self, dataset: GraphDataset, is_train: bool = True
    ) -> TaskDataset:
        """Create a SFT dataset from graph dataset."""
        data_name = dataset.name
        sft_dataset = dataset.data
        data = sft_dataset.train_data if is_train else sft_dataset.test_data

        sampler = torch.utils.data.DistributedSampler(
            data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=is_train,
        )

        batch_size = (
            self.args.train_batch_size if is_train else self.args.eval_batch_size
        )
        data_loader = DataLoader(
            data,
            batch_size=batch_size,
            sampler=sampler,
        )

        return TaskDataset(
            name=data_name,
            graph=sft_dataset.graph.to(self.device),
            data_loader=data_loader,
        )

    def train_step(
        self, batch: Any, task_dataset: TaskDataset
    ) -> dict[str, float | torch.Tensor]:
        """Perform a single training step for QA fine-tuning."""
        graph = task_dataset.graph.to(self.device)
        batch = query_utils.cuda(batch, device=self.device)

        # Forward pass
        pred = self.parallel_model(graph, batch)
        target = batch["target_nodes_mask"]  # type: ignore

        # Get distillation targets if needed
        distillation_targets = {}
        for target_node_type in self.distillation_types:
            question_emb = batch["question_embeddings"]  # type: ignore
            target_node_ids = graph.nodes_by_type[target_node_type]  # type: ignore
            target_node_emb = graph.x[target_node_ids]  # type: ignore
            distillation_target = question_emb @ target_node_emb.T
            distillation_targets[target_node_type] = distillation_target

        # Compute losses
        total_loss: torch.Tensor = torch.tensor(
            0.0, device=self.device, requires_grad=True
        )
        step_metrics = {}

        for sft_loss in self.loss_functions:
            loss_fn = sft_loss.loss_fn
            weight = sft_loss.weight
            target_node_type = sft_loss.target_node_type
            loss_name = sft_loss.name

            # Get predictions and targets for current target node type
            target_node_ids = graph.nodes_by_type[target_node_type]  # type: ignore
            target_node_pred = pred[:, target_node_ids]  # type: ignore
            target_node_label = target[:, target_node_ids]  # type: ignore

            # Compute loss
            if sft_loss.is_distillation_loss:
                single_loss = loss_fn(
                    target_node_pred, distillation_targets[target_node_type]
                )
            else:
                single_loss = loss_fn(target_node_pred, target_node_label)

            step_metrics[loss_name] = single_loss.item()
            total_loss = total_loss + weight * single_loss

        step_metrics["loss"] = total_loss

        return step_metrics

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate the model on validation datasets."""
        if self.eval_graph_dataset_loader is None:
            return {}

        self.model.eval()
        all_metrics = {}
        all_watched_metric = []

        for dataset in self.eval_graph_dataset_loader:
            sft_dataset = self._create_task_dataset(dataset, is_train=False)

            data_name = sft_dataset.name
            graph = sft_dataset.graph
            data_loader = sft_dataset.data_loader

            # Set epoch for sampler
            if hasattr(data_loader.sampler, "set_epoch"):
                data_loader.sampler.set_epoch(0)

            # Initialize predictions and targets for each target type
            preds_list: dict[str, list[tuple]] = {
                target_type: [] for target_type in self.target_types
            }
            targets_list: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
                target_type: [] for target_type in self.target_types
            }

            for batch in tqdm(
                data_loader,
                desc=f"Evaluating {data_name}",
                disable=not utils.is_main_process(),
            ):
                # Eval step
                with torch.amp.autocast(
                    device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp
                ):
                    batch = query_utils.cuda(batch, device=self.device)
                    # Forward pass
                    pred = self.model(graph, batch)
                target = batch["target_nodes_mask"].bool()  # type: ignore

                # Collect predictions and targets for each target type
                for target_type in self.target_types:
                    target_node_ids = graph.nodes_by_type[target_type]  # type: ignore
                    target_node_pred = pred[:, target_node_ids]  # type: ignore
                    target_node_label = target[:, target_node_ids]  # type: ignore
                    node_ranking, target_node_ranking = utils.batch_evaluate(
                        target_node_pred, target_node_label
                    )
                    # Answer set cardinality prediction
                    node_prob = F.sigmoid(target_node_pred)
                    num_pred = (node_prob * (node_prob > 0.5)).sum(dim=-1)
                    num_target = target_node_label.sum(dim=-1)
                    preds_list[target_type].append((node_ranking, num_pred))
                    targets_list[target_type].append((target_node_ranking, num_target))

            # Compute metrics for each target type
            metrics_by_type = {}
            for node_type in self.target_types:
                # Concatenate the predictions and targets for the current node type
                preds = preds_list[node_type]
                targets = targets_list[node_type]
                if len(preds) == 0 or len(targets) == 0:
                    continue

                # Gather results across all processes
                node_pred, node_target = (
                    query_utils.cat(preds),
                    query_utils.cat(targets),
                )
                node_pred, node_target = utils.gather_results(
                    node_pred, node_target, self.rank, self.world_size, self.device
                )

                # Evaluate the metrics for the current node type
                metrics_by_type[node_type] = utils.evaluate(
                    node_pred, node_target, self.metrics
                )

            metrics = {}
            for node_type, metric in metrics_by_type.items():
                for key, value in metric.items():
                    metrics[f"{node_type}_{key}"] = value

            all_metrics[data_name] = metrics

        # Synchronize across processes
        utils.synchronize()

        eval_metrics = {}
        # Compute average watched metric
        for data_name, metrics in all_metrics.items():
            all_watched_metric.append(metrics.get(self.args.metric_for_best_model, 0.0))  # type: ignore
            for metric_name, metric_value in metrics.items():
                eval_metrics[f"{data_name}/{metric_name}"] = metric_value

        all_avg_watched_metric = np.mean(all_watched_metric)
        eval_metrics[self.args.metric_for_best_model] = all_avg_watched_metric  # type: ignore

        return eval_metrics

    @torch.no_grad()
    def predict(self) -> dict[str, Any]:
        """Perform the prediction

        Returns:
            List of predicted outputs of target node types
        """
        if self.eval_graph_dataset_loader is None:
            return {}

        self.model.eval()
        predictions_by_dataset = {}

        for test_dataset in self.eval_graph_dataset_loader:
            sft_dataset = self._create_task_dataset(test_dataset, is_train=False)

            data_name = sft_dataset.name
            graph = sft_dataset.graph
            data_loader = sft_dataset.data_loader

            # Set epoch for sampler
            if hasattr(data_loader.sampler, "set_epoch"):
                data_loader.sampler.set_epoch(0)

            # Initialize predictions and targets for each target type
            preds_list: list[dict] = []

            for batch in tqdm(
                data_loader,
                desc=f"Predicting {data_name}",
                disable=not utils.is_main_process(),
            ):
                batch = query_utils.cuda(batch, device=self.device)

                # Forward pass
                pred = self.model(graph, batch)
                idx = batch["id"]
                preds_by_type: dict[str, torch.Tensor] = {
                    target_type: [] for target_type in self.target_types
                }
                # Collect predictions and targets for each target type
                for target_type in self.target_types:
                    target_node_ids = graph.nodes_by_type[target_type]  # type: ignore
                    target_node_pred = pred[:, target_node_ids]  # type: ignore
                    # Get top-k predictions
                    top_k = torch.topk(
                        target_node_pred, k=self.args.predict_top_k, dim=-1
                    )
                    top_k_indices = top_k.indices
                    top_k_scores = top_k.values
                    # Convert to original node ids and names
                    original_node_ids = target_node_ids[top_k_indices]
                    node_name = [
                        [
                            (test_dataset.data.id2node[node_id.item()], score.item())
                            for node_id, score in zip(
                                original_node_ids[batch_idx], top_k_scores[batch_idx]
                            )
                        ]
                        for batch_idx in range(len(original_node_ids))
                    ]
                    preds_by_type[target_type] = node_name

                for i in range(len(idx)):
                    preds_list.append(
                        {
                            "id": idx[i].item()
                            if isinstance(idx[i], torch.Tensor)
                            else idx[i],
                            "predictions": {
                                target_type: p[i]
                                for target_type, p in preds_by_type.items()
                            },
                        }
                    )

            # Gather the predictions across all processes
            if utils.get_world_size() > 1:
                gathered_predictions = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(gathered_predictions, preds_list)
            else:
                gathered_predictions = [preds_list]  # type: ignore

            flatten_predictions = [
                item
                for sublist in gathered_predictions
                for item in sublist  # type: ignore
            ]

            id_to_raw_data = {
                sample["id"]: sample for sample in test_dataset.data.raw_test_data
            }
            # Map the predictions to the dataset name
            retrieval_results = []
            for pred in flatten_predictions:
                raw_sample = id_to_raw_data[pred["id"]]
                raw_sample.update({"predictions": pred["predictions"]})
                retrieval_results.append(raw_sample)

            predictions_by_dataset[data_name] = retrieval_results

        utils.synchronize()
        return predictions_by_dataset
