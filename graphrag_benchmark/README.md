# GFM-RAG Evaluation on GraphRAG Benchmarks

Evaluate GFM-RAG on two GraphRAG benchmarks. Start with data, build the KG index, run retrieval + QA, then score with the official scripts.

> Install and set up the environment following the top-level `README.md` before running the commands here.

## Benchmarks at a glance

Two GraphRAG benchmarks uses similar name, we will use the following names to avoid confusion.

- `graphrag_bench` → GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation (aka **G-Bench CS**). [Repo](https://github.com/jeremycp3/GraphRAG-Bench)
- `graphrag_benchmark_medical` / `graphrag_benchmark_novel` → When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation (aka **G-Bench Medical / Novel**). [Repo](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)

## 1) Prepare data

Download our preprocessed data from [here](https://huggingface.co/datasets/graphrag/graphrag-benchmark-data) and place it under `data/`.

```text
data/
├── graphrag_bench/
│   └── raw/
│       ├── documents.json
│       └── test.json
├── graphrag_benchmark_medical/
│   └── raw/
│       ├── documents.json
│       └── test.json
└── graphrag_benchmark_novel/
    └── raw/
        ├── documents.json
        └── test.json
```

## 2) Build the KG index

You can run indexing yourself or use our prebuilt KG indices. Download our prebuilt KG indices from [here](https://huggingface.co/datasets/graphrag/graphrag-benchmark-indices).

To build KG indices locally, create `nodes.csv`, `edges.csv`, `relations.csv`, and processed `test.json` for each dataset running the following script.

```bash
bash graphrag_benchmark/scripts/stage1_data_index.sh
```

Expected layout after indexing:

```text
data/<dataset>/processed/stage1/
├── edges.csv
├── nodes.csv
├── relations.csv
└── test.json
```

## 3) Generate retrieval results

The QA scripts need retrieval results per dataset with top documents and entities for each question.

You can either run retrieval yourself or use our precomputed results. Download our precomputed retrieval results from [here](https://huggingface.co/datasets/graphrag/graphrag-benchmark-retrieval).

To generate retrieval results locally using GFM-RAG running the following script:

```bash
bash graphrag_benchmark/scripts/stage2_retrieval.sh
```

## 4) Run QA

Use the provided scripts to load the retrieval outputs, build prompts, and call the LLM. Yon can check the prompts in `gfmrag/workflow/config/qa_prompt/`.

### GraphRAG-Bench (CS)

```bash
bash graphrag_benchmark/scripts/stage3_qa_inference_graphrag_bench.sh
```

Outputs: one JSON per task type (FB, MC, MS, OE, TF) named for the official evaluator.

### GraphRAG-Benchmark (Novel / Medical)

```bash
bash graphrag_benchmark/scripts/stage3_qa_inference_graphrag_benchmark.sh
```

Outputs: one `prediction.jsonl`.

## 5) Evaluate with the official scripts

We use the official evaluation scripts from corresponding repos with minimal modifications to fit our data and outputs.

### GraphRAG-Bench (CS)

1) Clone the [repo](https://github.com/icedpanda/GraphRAG-Bench) and download their [data](https://huggingface.co/datasets/Awesome-GraphRAG/GraphRAG-Bench).
2) Copy the five JSON outputs from step 4 into `GraphRAG-Bench/Datasets/output/g-reasoner/`.
3) Run the evaluator inside that repo, e.g. `python evaluator.py`.

### GraphRAG-Benchmark (Novel / Medical)

1) Clone the [repo](https://github.com/icedpanda/GraphRAG-Benchmark) and download their [data](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench).
2) Copy `prediction.jsonl` from step 4 into `GraphRAG-Benchmark/Datasets/output/g-reasoner/<domain>/prediction.jsonl`.
3) Run the evaluation entry point, e.g. `bash run_retreival_evaluation.sh` and `bash run_gen_evaluation.sh` for retrieval and generation evaluation respectively.
