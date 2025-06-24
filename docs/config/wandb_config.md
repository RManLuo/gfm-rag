# Weights & Biases Integration

This document explains how to use the Weights & Biases (wandb) integration in GFM-RAG for tracking training experiments.

## Overview

GFM-RAG now includes comprehensive wandb integration that automatically logs:

- **Training Loss**: Real-time loss values during training
- **Evaluation Metrics**: Validation and test results
- **Model Configuration**: All hyperparameters and settings
- **Model Checkpoints**: Best model and epoch-wise checkpoints as artifacts
- **System Information**: Hardware and environment details

## Quick Start

### 1. Install and Setup Wandb

```bash
# wandb is already included in the dependencies
# Log in to your wandb account
wandb login
```

### 2. Basic Usage

The integration is enabled by default. Simply run your training as usual:

```bash
# Pre-training with wandb logging
python -m gfmrag.workflow.stage2_kg_pretrain

# Fine-tuning with wandb logging
python -m gfmrag.workflow.stage2_qa_finetune
```

### 3. Configure Wandb Settings

You can customize wandb settings in the configuration files:

#### Using Command Line Overrides

```bash
# Set custom project name
python -m gfmrag.workflow.stage2_kg_pretrain wandb.project="my-experiment"

# Add custom tags
python -m gfmrag.workflow.stage2_kg_pretrain wandb.tags=["experiment1","baseline"]

# Set run name
python -m gfmrag.workflow.stage2_kg_pretrain wandb.name="baseline-run-1"

# Disable wandb logging
python -m gfmrag.workflow.stage2_kg_pretrain wandb.enabled=false
```

#### Editing Configuration Files

Edit `gfmrag/workflow/config/wandb/default.yaml`:

```yaml
enabled: true
project: "my-gfm-rag-project"
entity: "my-team"  # Your team name or username
name: null  # Will auto-generate
group: "experiment-1"  # Group related runs
tags: ["baseline", "v1"]
notes: "Baseline experiment with default settings"
```

## Configuration Options

### Core Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enabled` | Enable/disable wandb logging | `true` |
| `project` | Project name in wandb | varies by script |
| `entity` | Team/username for the project | `null` (uses default) |
| `name` | Run name | `null` (auto-generated) |
| `group` | Group name for organizing runs | `null` |
| `tags` | List of tags for the run | `[]` |
| `notes` | Description of the experiment | `""` |

### Advanced Settings

You can also configure additional wandb settings by editing the default configuration:

```yaml
# Directory to save wandb files
dir: "./wandb_logs"

# Mode: "online", "offline", or "disabled"
mode: "online"

# Save source code with the run
save_code: true

# Log frequency for model watching
log_frequency: 100
```

## Logged Metrics

### Pre-training (`stage2_kg_pretrain.py`)

**Training Metrics:**

- `train/loss` - Binary cross-entropy loss (per batch)
- `train/epoch_loss` - Average loss per epoch
- `train/epoch` - Current epoch number

**Evaluation Metrics:**

- `eval/{dataset_name}/mr` - Mean rank
- `eval/{dataset_name}/mrr` - Mean reciprocal rank
- `eval/{dataset_name}/hits@{k}` - Hits at k (k=1,3,10)
- `eval/mrr` - Overall MRR
- `eval/best_mrr` - Best MRR achieved

**Model Artifacts:**

- Best model checkpoints with metadata
- Epoch-wise checkpoints (if `save_best_only=false`)

### Fine-tuning (`stage2_qa_finetune.py`)

**Training Metrics:**

- `train/{loss_name}` - Individual loss components (per batch)
- `train/epoch_{loss_name}` - Average losses per epoch
- `train/epoch` - Current epoch number

**Evaluation Metrics:**

- `eval/{watched_metric}` - Primary evaluation metric
- `eval/best_{watched_metric}` - Best value achieved
- `test/{dataset_name}/{metric}` - Final test results

**Model Artifacts:**

- Best model checkpoints with metadata
- Epoch-wise checkpoints (if `save_best_only=false`)

## Multi-GPU Training

The wandb integration is multi-GPU aware:
- Only rank 0 process logs to wandb (prevents duplicate logs)
- All metrics are properly synchronized across processes
- Model checkpoints are saved only on rank 0

```bash
# Multi-GPU training with wandb
torchrun --nproc_per_node=4 gfmrag.workflow.stage2_kg_pretrain
```

## Best Practices

### 1. Organize Experiments

Use groups and tags to organize related experiments:

```bash
# Hyperparameter sweep
python -m gfmrag.workflow.stage2_qa_finetune wandb.group="hp_sweep" wandb.tags=["lr_0.001"]
python -m gfmrag.workflow.stage2_qa_finetune wandb.group="hp_sweep" wandb.tags=["lr_0.0005"]
```

### 2. Meaningful Run Names

Set descriptive run names for important experiments:

```bash
python -m gfmrag.workflow.stage2_qa_finetune wandb.name="baseline_hotpot_qa_v1"
```

### 3. Add Experiment Notes

Include context about your experiments:

```bash
python -m gfmrag.workflow.stage2_qa_finetune wandb.notes="Testing new loss function combination"
```

## Troubleshooting

### Common Issues

**1. Login Required**
```bash
wandb login
```

**2. Disable Wandb for Debugging**
```bash
python -m gfmrag.workflow.stage2_kg_pretrain wandb.enabled=false
```

**3. Offline Mode**
If you want to log locally without uploading:
```bash
export WANDB_MODE=offline
python -m gfmrag.workflow.stage2_kg_pretrain
```

**4. Network Issues**
For environments with restricted internet access:
```bash
export WANDB_MODE=disabled
python -m gfmrag.workflow.stage2_kg_pretrain
```

### Verify Integration

Check that wandb is working correctly:

```python
import wandb

print(f"Wandb version: {wandb.__version__}")
print(f"Wandb is available: {wandb.api.api_key is not None}")
```

## Example Workflow

```bash
# 1. Start pre-training with custom settings
python -m gfmrag.workflow.stage2_kg_pretrain \\
    wandb.project="gfm-rag-experiments" \\
    wandb.group="pretraining" \\
    wandb.tags=["baseline","hotpot"] \\
    wandb.name="pretrain_baseline_v1"

# 2. Fine-tune using the pre-trained model
python -m gfmrag.workflow.stage2_qa_finetune \\
    wandb.project="gfm-rag-experiments" \\
    wandb.group="finetuning" \\
    wandb.tags=["finetune","hotpot"] \\
    wandb.name="finetune_baseline_v1" \\
    train.checkpoint="path/to/pretrained/model"
```

This will create a complete experiment tracking workflow where you can easily compare different runs, track improvements, and share results with your team.
