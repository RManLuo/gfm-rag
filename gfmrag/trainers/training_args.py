from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TrainingArguments:
    """Training arguments for GFM-RAG trainers"""

    # Training hyperparameters
    num_epoch: int = field(
        default=10, metadata={"help": "Total number of training epochs to perform."}
    )
    train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/CPU for training."}
    )
    eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/CPU for evaluation."}
    )
    max_steps_per_epoch: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of steps per epoch. If None, will use all data."
        },
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run evaluation."})
    # Checkpoint resuming
    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={"help": "The path to a checkpoint to resume training from."},
    )

    # Logging and evaluation
    logging_steps: int = field(
        default=100, metadata={"help": "Log every X updates steps."}
    )
    eval_strategy: Literal["epoch", "step"] = field(
        default="epoch", metadata={"help": "Evaluation strategy: 'epoch' or 'step'."}
    )
    eval_steps: int | None = field(
        default=None,
        metadata={"help": "Run evaluation every X steps if eval_strategy == 'step'."},
    )

    # Checkpointing
    save_best_only: bool = field(
        default=True, metadata={"help": "Whether to save only the best model."}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={
            "help": "Whether to load the best model found during training at the end of training."
        },
    )
    metric_for_best_model: str | None = field(
        default=None,
        metadata={"help": "The metric to use to compare two different models."},
    )
    greater_is_better: bool = field(
        default=True,
        metadata={
            "help": "Whether the `metric_for_best_model` should be maximized or not."
        },
    )

    # Training mode
    training_mode: Literal["ddp", "spmd"] = field(
        default="ddp", metadata={"help": "The distributed training mode."}
    )
    dtype: Literal["float32", "float16", "bfloat16", "auto"] = field(
        default="float32", metadata={"help": "The dtype to use for mixed precision training."}
    )