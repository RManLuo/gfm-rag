"""
Weights & Biases logging utilities for GFM-RAG training.
"""

import logging
import os
from typing import Any

import torch
import wandb
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def init_wandb(
    cfg: DictConfig,
    project_name: str = "gfm-rag",
    run_name: str | None = None,
    tags: list | None = None,
) -> None:
    """
    Initialize Weights & Biases logging.

    Args:
        cfg: Configuration object
        project_name: W&B project name
        run_name: Optional run name
        tags: Optional list of tags
    """
    # Check if wandb is disabled
    if hasattr(cfg, "wandb") and not cfg.wandb.get("enabled", True):
        logger.info("W&B logging is disabled")
        return

    # Get wandb config if it exists
    wandb_cfg = getattr(cfg, "wandb", {})

    # Set up wandb config
    wandb_config = {
        "project": wandb_cfg.get("project", project_name),
        "name": run_name or wandb_cfg.get("name", None),
        "tags": tags or wandb_cfg.get("tags", []),
        "notes": wandb_cfg.get("notes", ""),
        "config": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    }

    # Add entity if specified
    if "entity" in wandb_cfg:
        wandb_config["entity"] = wandb_cfg["entity"]

    # Add group if specified
    if "group" in wandb_cfg:
        wandb_config["group"] = wandb_cfg["group"]

    # Initialize wandb
    try:
        wandb.init(**wandb_config)
        logger.info(f"Initialized W&B logging for project: {wandb_config['project']}")
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")


def log_metrics(
    metrics: dict[str, Any], step: int | None = None, prefix: str = ""
) -> None:
    """
    Log metrics to wandb.

    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number
        prefix: Optional prefix for metric names
    """
    if not wandb.run:
        return

    # Add prefix to metric names if specified
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    try:
        wandb.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"Failed to log metrics to W&B: {e}")


def log_model_checkpoint(
    checkpoint_path: str,
    name: str = "model",
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Log model checkpoint as wandb artifact.

    Args:
        checkpoint_path: Path to the checkpoint file
        name: Name for the artifact
        metadata: Optional metadata for the artifact
    """
    if not wandb.run or not os.path.exists(checkpoint_path):
        return

    try:
        artifact = wandb.Artifact(
            name=name,
            type="model",
            metadata=metadata or {},
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        logger.info(f"Logged model checkpoint: {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Failed to log model checkpoint to W&B: {e}")


def finish_wandb() -> None:
    """
    Finish wandb run.
    """
    if wandb.run:
        try:
            wandb.finish()
            logger.info("Finished W&B logging")
        except Exception as e:
            logger.warning(f"Failed to finish W&B run: {e}")


def watch_model(model: torch.nn.Module, log_freq: int = 100) -> None:
    """
    Watch model with wandb.

    Args:
        model: PyTorch model to watch
        log_freq: Frequency of logging
    """
    if not wandb.run:
        return

    try:
        wandb.watch(model, log_freq=log_freq)
        logger.info("Started watching model with W&B")
    except Exception as e:
        logger.warning(f"Failed to watch model with W&B: {e}")
