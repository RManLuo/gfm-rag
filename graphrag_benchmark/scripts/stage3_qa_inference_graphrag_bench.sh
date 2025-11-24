#!/bin/bash

# GraphRAG-Bench (CS)
LLM="gpt-4o-mini"
DOC_TOP_K=10
N_THREAD=10
RETRIEVED_RESULT_PATH="outputs/qa_finetune/documents_only/predictions_graphrag_bench.json"
DOCUMENT_PATH="data/graphrag_bench/raw/documents.json"
torchrun --nproc_per_node=1 -m graphrag_benchmark.graphrag_bench_qa \
  --config-name=graphrag_bench_qa_inference \
  llm.model_name_or_path=${LLM} \
  test.n_threads=${N_THREAD} \
  test.top_k=${DOC_TOP_K} \
  test.retrieved_result_path=${RETRIEVED_RESULT_PATH} \
  test.document_path=${DOCUMENT_PATH}
