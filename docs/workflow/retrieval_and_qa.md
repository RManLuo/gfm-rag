# Retrieval and QA

This page covers the current post-index usage path.

## What This Step Does

Once `processed/stage1/` exists, you can:

- retrieve graph-grounded results directly with `GFMRetriever.from_index(...)`
- run batch QA with `python -m gfmrag.workflow.qa`
- run agent reasoning with `python -m gfmrag.workflow.qa_ircot_inference`

## When You Need It

Use this page after indexing when you want to retrieve documents, answer questions from saved retrieval outputs, or run iterative reasoning.

## Direct Retrieval With `GFMRetriever`

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gfmrag import GFMRetriever

cfg = OmegaConf.load("gfmrag/workflow/config/gfm_rag/qa_ircot_inference.yaml")

retriever = GFMRetriever.from_index(
    data_dir="./data",
    data_name="toy_raw",
    model_path="rmanluo/G-reasoner-34M",
    ner_model=instantiate(cfg.ner_model),
    el_model=instantiate(cfg.el_model),
    graph_constructor=instantiate(cfg.graph_constructor),
)

results = retriever.retrieve(
    "Who is the president of France?",
    top_k=5,
    target_types=["document"],
)
```

The returned structure is keyed by target type, for example `results["document"]`.

## Batch QA From Retrieved Results

`gfmrag.workflow.qa` takes a saved retrieval file plus the dataset node table.

```bash
python -m gfmrag.workflow.qa \
  qa_prompt=hotpotqa \
  qa_evaluator=hotpotqa \
  llm.model_name_or_path=gpt-4o-mini \
  test.top_k=5 \
  test.target_types=[document] \
  test.retrieved_result_path=outputs/qa_finetune/latest/predictions_hotpotqa_test.json \
  test.node_path=./data/hotpotqa_test/processed/stage1/nodes.csv
```

This writes `prediction.jsonl` under `outputs/qa_inference/<date>/<time>/`.

### Required Inputs For QA

- `test.retrieved_result_path`: retrieval results, usually produced by `sft_training`
- `test.node_path`: path to `processed/stage1/nodes.csv`
- `qa_prompt` and `qa_evaluator`: prompt/evaluation config groups

## Agent Reasoning

`gfmrag.workflow.qa_ircot_inference` combines retrieval and reasoning in a single workflow.

```bash
python -m gfmrag.workflow.qa_ircot_inference \
  dataset.root=./data \
  dataset.data_name=hotpotqa_test \
  graph_retriever.model_path=rmanluo/G-reasoner-34M \
  test.top_k=10 \
  test.max_steps=2
```

This writes `prediction.jsonl` under `outputs/qa_agent_inference/<data_name>/<date>/<time>/`.

## How The Pieces Fit Together

- `GFMRetriever.from_index(...)` loads or builds stage1 files, restores the dataset view from the checkpoint config, and assembles the retriever.
- `sft_training` can emit `predictions_<data_name>.json` for retrieval evaluation and later QA.
- `qa.py` reads those retrieval outputs, looks up node metadata from `nodes.csv`, builds prompts, and evaluates the final answers.
- `qa_ircot_inference.py` performs retrieval and multi-step reasoning together before writing final predictions.

## Common Pitfalls

- `qa.py` fails fast if `test.retrieved_result_path` is missing.
- `qa.py` also requires `test.node_path`; retrieval outputs alone are not enough.
- `target_types` used during QA must exist in the retrieval predictions.
- Agent reasoning requires the prompt and evaluator configs to match the dataset/task.
