#!/bin/bash

# Batch inference for GraphRAG-Benchmark (Novel + Medical)
LLM="gpt-4o-mini"
DOC_TOP_K=10
N_THREAD=10

# Medical split
torchrun --nproc_per_node=1 -m graphrag_benchmark.graphrag_benchmark_qa \
  --config-name=graphrag_benchmark_qa_inference \
  dataset_name=graphrag_benchmark_medical \
  prompt_root=qa_prompt/graphrag_benchmark/medical \
  llm.model_name_or_path=${LLM} \
  test.n_threads=${N_THREAD} \
  test.top_k=${DOC_TOP_K} \
  test.retrieved_result_path=outputs/qa_finetune/documents_only/predictions_graphrag_benchmark_medical.json

# Novel split
torchrun --nproc_per_node=1 -m graphrag_benchmark.graphrag_benchmark_qa \
  --config-name=graphrag_benchmark_qa_inference \
  dataset_name=graphrag_benchmark_novel \
  prompt_root=qa_prompt/graphrag_benchmark/novel \
  llm.model_name_or_path=${LLM} \
  test.n_threads=${N_THREAD} \
  test.top_k=${DOC_TOP_K} \
  test.retrieved_result_path=outputs/qa_finetune/documents_only/predictions_graphrag_benchmark_novel.json
