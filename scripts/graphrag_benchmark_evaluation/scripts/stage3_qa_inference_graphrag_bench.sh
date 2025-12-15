#!/bin/bash

# GraphRAG-Bench (CS)
LLM="gpt-4o-mini"
DOC_TOP_K=10
N_THREAD=10
RETRIEVED_RESULT_PATH="outputs/qa_finetune/documents_only/predictions_graphrag_bench_cs.json" # retrieval results from stage2_retrieval.sh
DOCUMENT_PATH="data/graphrag_bench_cs/raw/documents.json"
python -m graphrag_benchmark.graphrag_bench_qa \
  --config-name=graphrag_bench_cs_qa_inference \
  llm.model_name_or_path=${LLM} \
  test.n_threads=${N_THREAD} \
  test.top_k=${DOC_TOP_K} \
  test.retrieved_result_path=${RETRIEVED_RESULT_PATH} \
  test.document_path=${DOCUMENT_PATH}
