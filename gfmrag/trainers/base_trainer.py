import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import islice
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from gfmrag import utils
from gfmrag.graph_index_datasets.graph_dataset_loader import (
    GraphDataset,
    GraphDatasetLoader,
)
from gfmrag.utils.wandb_utils import (
    log_metrics,
    log_model_checkpoint,
)

from .training_args import TrainingArguments

logger = logging.getLogger(__name__)
# Disable on non-master processes
if utils.get_rank() != 0:
    logger.setLevel(logging.CRITICAL + 1)


@dataclass
class TaskDataset:
    """Task-specific dataset for GFM-RAG models."""

    name: str
    graph: Any
    data_loader: DataLoader


class BaseTrainer(ABC):
    """
    Base trainer class for GFM-RAG models, similar to HuggingFace Trainer.
    """

    separator = ">" * 30
    line = "-" * 30

    def __init__(
        self,
        output_dir: str,
        args: TrainingArguments,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_graph_dataset_loader: GraphDatasetLoader | None = None,
        eval_graph_dataset_loader: GraphDatasetLoader | None = None,
        **kwargs: Any,
    ) -> None:
        self.output_dir = output_dir
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.train_graph_dataset_loader = train_graph_dataset_loader
        self.eval_graph_dataset_loader = eval_graph_dataset_loader

        # Evaluation strategy
        self.eval_strategy = args.eval_strategy
        self.eval_steps = args.eval_steps

        # Set up distributed training
        self.device = utils.get_device()
        self.world_size = utils.get_world_size()
        self.rank = utils.get_rank()

        # Training state
        self.state: dict[str, Any] = {
            "epoch": 0,
            "global_step": 0,
            "best_metric": float("-inf") if args.greater_is_better else float("inf"),
            "best_epoch": -1,
        }

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up model for training
        self._setup_model()

    def _setup_model(self) -> None:
        """Set up the model for training."""
        # Load checkpoint if specified
        if self.args.resume_from_checkpoint:
            self._load_checkpoint(self.args.resume_from_checkpoint)

        self.model = self.model.to(self.device)
        # Configure model precision based on config
        self.model, self.dtype = utils.configure_model_precision(
            self.model, self.device, self.args.dtype
        )

        self.use_amp = self.dtype != torch.float32
        self.enable_grad_scaler = self.dtype not in [torch.float32, torch.bfloat16]
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.enable_grad_scaler
        )

        if self.world_size > 1 and self.args.training_mode == "ddp" and not self.args.split_graph_training:
            self.parallel_model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.device]
            )
        else:
            self.parallel_model = self.model

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint."""
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            state = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )

            # Load model state
            if "model" in state:
                self.model.load_state_dict(state["model"], strict=False)

            # Load optimizer state
            if "optimizer" in state and hasattr(self, "optimizer"):
                try:
                    self.optimizer.load_state_dict(state["optimizer"])
                    logger.info("Loaded optimizer state from checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}")

            # Load training state
            if "epoch" in state:
                self.state["epoch"] = state["epoch"]
            if "global_step" in state:
                self.state["global_step"] = state["global_step"]
            if "best_metric" in state:
                self.state["best_metric"] = state["best_metric"]
            if "best_epoch" in state:
                self.state["best_epoch"] = state["best_epoch"]
            if "scaler" in state:
                self.scaler.load_state_dict(state["scaler"])

            logger.info(
                f"Resumed from epoch {self.state['epoch']}, global step {self.state['global_step']}"
            )
        else:
            logger.warning(f"Checkpoint {checkpoint_path} does not exist")

    def _save_checkpoint(self, output_dir: str, is_best: bool = False) -> None:
        """Save a checkpoint."""
        if utils.get_rank() == 0:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": self.state["epoch"],
                "global_step": self.state["global_step"],
                "best_metric": self.state["best_metric"],
                "best_epoch": self.state["best_epoch"],
            }

            # Save best checkpoint
            if is_best:
                best_path = os.path.join(output_dir, "model_best.pth")
                torch.save(state, best_path)
                logger.info(f"Saved best model to {best_path}")

                # Log model checkpoint to wandb
                log_model_checkpoint(
                    best_path,
                    f"best-epoch-{self.state['epoch']}-step-{self.state['global_step']}",
                    metadata={
                        "epoch": self.state["epoch"],
                        "best_metric": self.state["best_metric"],
                        "best": True,
                    },
                )
            # Save regular checkpoint
            elif not self.args.save_best_only:
                checkpoint_path = os.path.join(
                    output_dir,
                    f"checkpoint-epoch-{self.state['epoch']}-step-{self.state['global_step']}.pth",
                )
                torch.save(state, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                log_model_checkpoint(
                    checkpoint_path,
                    f"checkpoint-epoch-{self.state['epoch']}-step-{self.state['global_step']}",
                    metadata={
                        "epoch": self.state["epoch"],
                        "global_step": self.state["global_step"],
                    },
                )

    def _log_metrics(
        self, logs: dict[str, Any], step: int | None = None, prefix: str = ""
    ) -> None:
        """Log metrics."""
        if utils.get_rank() == 0:
            if step is None:
                step = self.state["global_step"]  # type: ignore

            # Log to console
            order = sorted(list(logs.keys()))
            for key in order:
                logger.info(f"{key}: {logs[key]:.4f}")

            # Add Prefix to the logs
            if prefix:
                log_with_prefix = {f"{prefix}/{k}": v for k, v in logs.items()}
            else:
                log_with_prefix = logs

            # Add step to logs
            logs_with_step = {**log_with_prefix, "step": step}

            # Log to wandb
            log_metrics(logs_with_step)

    @abstractmethod
    def _create_task_dataset(
        self, graph_dataset: GraphDataset, is_train: bool = True
    ) -> TaskDataset:
        """
        Create a task-specific dataset from the graph dataset

        Args:
            graph_dataset (GraphDataset): The graph dataset to use.
            is_train (bool): Whether this is for training.

        Returns:
            TaskDataset
        """
        pass

    @abstractmethod
    def train_step(
        self, batch: Any, task_dataset: TaskDataset
    ) -> dict[str, float | torch.Tensor]:
        """
        Perform a single training step.

        Args:
            batch: Training batch from dataloader
            task_dataset: Information about the current dataset

        Returns:
            Dictionary containing loss and other metrics
        """
        pass

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """
        Perform the evaluation.

        Returns:
            Dictionary containing evaluation metrics
        """
        pass

    def train(self) -> None:
        """Main training loop."""
        if self.args.do_train:
            logger.info("***** Running training *****")
            logger.info(f"  Num epochs = {self.args.num_epoch}")
            logger.info(
                f"  Instantaneous batch size per device = {self.args.train_batch_size}"
            )
            logger.info(
                f"  Total train batch size (w. parallel & distributed) = {self.args.train_batch_size * self.world_size}"
            )

            start_epoch = self.state["epoch"]

            for epoch in range(start_epoch, self.args.num_epoch):  # type: ignore
                self.state["epoch"] = epoch + 1

                if utils.get_rank() == 0:
                    logger.info(f"{'=' * 50}")
                    logger.info(f"Epoch {self.state['epoch']} / {self.args.num_epoch}")
                    logger.info(f"{'=' * 50}")

                # Training
                self._train_epoch()
                utils.synchronize()

                # Evaluation by epoch
                if (
                    self.eval_strategy == "epoch"
                    and self.eval_graph_dataset_loader is not None
                ):
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics, prefix="eval")
                    self._maybe_save_best_model(eval_metrics)

                # Save checkpoint
                if not self.args.save_best_only:
                    self._save_checkpoint(self.output_dir)

            utils.synchronize()
            # Load best model at the end
            if self.args.load_best_model_at_end:
                best_model_path = os.path.join(self.output_dir, "model_best.pth")
                if os.path.exists(best_model_path):
                    logger.info("Loading best model for final evaluation")
                    self._load_checkpoint(best_model_path)

            logger.info("Training completed!")

        # Final evaluation
        if self.eval_graph_dataset_loader is not None and self.args.do_eval:
            logger.info("***** Running final evaluation *****")
            final_metrics = self.evaluate()
            self._log_metrics(final_metrics, prefix="final")

    def _train_epoch(self) -> None:
        """Train for one epoch."""
        if self.train_graph_dataset_loader is None:
            logger.warning("No training dataset loader provided")
            return

        self.parallel_model.train()

        epoch_losses = []
        epoch_metrics: dict[str, list[float]] = {}

        # Set epoch for data loader
        if hasattr(self.train_graph_dataset_loader, "set_epoch"):
            self.train_graph_dataset_loader.set_epoch(self.state["epoch"])  # type: ignore

        for graph_dataset in self.train_graph_dataset_loader:
            dataset_name = graph_dataset.name
            task_dataset = self._create_task_dataset(graph_dataset, is_train=True)
            data_loader = task_dataset.data_loader

            # Set epoch for sampler
            if hasattr(data_loader, "sampler") and hasattr(
                data_loader.sampler, "set_epoch"
            ):
                data_loader.sampler.set_epoch(self.state["epoch"])  # type: ignore

            # Limit steps per epoch if specified
            if self.args.max_steps_per_epoch:
                data_iterator = islice(data_loader, self.args.max_steps_per_epoch)
                total_steps = self.args.max_steps_per_epoch
            else:
                data_iterator = data_loader
                total_steps = len(data_loader)

            progress_bar = tqdm(
                data_iterator,
                desc=f"Training {dataset_name} - Epoch {self.state['epoch']}",
                total=total_steps,
                disable=not utils.is_main_process(),
            )

            for batch in progress_bar:
                # Training step
                with torch.amp.autocast(
                    device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp
                ):
                    step_metrics = self.train_step(batch, task_dataset)

                    assert "loss" in step_metrics, (
                        "Training step must return 'loss' in metrics"
                    )

                    # Backward pass
                    loss = step_metrics["loss"]
                    self.scaler.scale(loss).backward()

                    # Split-graph training: manual gradient sync (no DDP wrapper)
                    if self.args.split_graph_training and self.world_size > 1:
                        self.scaler.unscale_(self.optimizer)
                        for param in self.model.parameters():
                            if param.grad is not None:
                                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                epoch_losses.append(loss.item())  # type: ignore

                # Convert step metrics to float for logging
                step_metrics = {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in step_metrics.items()
                }

                # Accumulate metrics
                for key, value in step_metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)

                self.state["global_step"] += 1

                # Log step metrics
                if self.state["global_step"] % self.args.logging_steps == 0:
                    self._log_metrics(step_metrics, prefix="train")

                # Evaluation by step
                if (
                    self.eval_strategy == "step"
                    and self.eval_graph_dataset_loader is not None
                    and self.eval_steps is not None
                    and self.state["global_step"] % self.eval_steps == 0
                ):
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics, prefix="eval")
                    self._maybe_save_best_model(eval_metrics)

                    # Save checkpoint
                    if not self.args.save_best_only:
                        self._save_checkpoint(self.output_dir)

                # Update progress bar
                progress_bar.set_postfix(loss=step_metrics.get("loss", 0.0))

        # Log epoch averages
        if utils.get_rank() == 0:
            epoch_avg_metrics = {
                f"epoch_{k}": np.mean(v) for k, v in epoch_metrics.items()
            }
            epoch_avg_metrics["epoch"] = self.state["epoch"]
            self._log_metrics(epoch_avg_metrics, prefix="train")

            logger.info(
                f"Epoch {self.state['epoch']} completed - Average loss: {np.mean(epoch_losses):.4f}"
            )

    def _maybe_save_best_model(self, eval_metrics: dict[str, float]) -> None:
        """Save model if it's the best so far."""
        if self.args.metric_for_best_model is None:
            return

        metric_value = eval_metrics.get(self.args.metric_for_best_model)
        if metric_value is None:
            logger.warning(
                f"Metric {self.args.metric_for_best_model} not found in eval metrics"
            )
            return

        is_best = (
            self.args.greater_is_better and metric_value > self.state["best_metric"]
        ) or (
            not self.args.greater_is_better and metric_value < self.state["best_metric"]
        )

        if is_best:
            self.state["best_metric"] = metric_value
            self.state["best_epoch"] = self.state["epoch"]
            if utils.get_rank() == 0:
                logger.info(
                    f"New best model! {self.args.metric_for_best_model}: {metric_value:.4f} at epoch {self.state['epoch']}"
                )
            self._save_checkpoint(self.output_dir, is_best=True)

        else:
            if utils.get_rank() == 0:
                logger.info(
                    f"Current best {self.args.metric_for_best_model}: {self.state['best_metric']:.4f} at epoch {self.state['best_epoch']}, not updated"
                )
