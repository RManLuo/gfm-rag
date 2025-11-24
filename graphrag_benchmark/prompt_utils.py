from __future__ import annotations

import os
from typing import Dict

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from gfmrag.prompt_builder import QAPromptBuilder

# Base location for prompt configs so we rely on the same files as the main
# workflow (Hydra search path also points here).
PROMPT_CONFIG_ROOT = os.path.join("gfmrag", "workflow", "config")


def load_prompt_builders(prompt_map: Dict[str, str]) -> dict[str, QAPromptBuilder]:
    """Create a prompt builder per task type as described in the config map."""
    builders: dict[str, QAPromptBuilder] = {}
    for task_type, prompt_path in prompt_map.items():
        config_path = (
            prompt_path if prompt_path.endswith(".yaml") else f"{prompt_path}.yaml"
        )
        abs_config_path = to_absolute_path(
            os.path.join(PROMPT_CONFIG_ROOT, config_path)
        )
        if not os.path.exists(abs_config_path):
            raise FileNotFoundError(
                f"Prompt config not found for task '{task_type}' at {abs_config_path}"
            )
        builders[task_type] = QAPromptBuilder(OmegaConf.load(abs_config_path))
    return builders


def resolve_task_type(sample: dict, task_cfg: DictConfig) -> str:
    """Resolve a task type from a sample using configuration rules."""
    source = task_cfg.get("source", "field")
    if source == "id_prefix":
        delimiter = task_cfg.get("delimiter", "-")
        index = task_cfg.get("index", 0)
        try:
            return sample["id"].split(delimiter)[index]
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Failed to parse task type from id with delimiter '{delimiter}' and index {index}"
            ) from exc
    if source == "field":
        field_name = task_cfg.get("field")
        if not field_name:
            raise ValueError("task_type.field must be provided when source='field'")
        if field_name not in sample:
            raise ValueError(
                f"Field '{field_name}' not found in sample for task resolution."
            )
        return sample[field_name]
    raise ValueError(f"Unsupported task type source: {source}")
