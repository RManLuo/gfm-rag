# Batch inference for QA on the test set.
N_GPU=4
DATA_ROOT="data"
DATA_NAME="hotpotqa" # hotpotqa musique 2wikimultihopqa
LLM="gpt-4o-mini"
DOC_TOP_K=5
N_THREAD=10
torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.stage3_qa_inference \
    dataset.root=${DATA_ROOT} \
    qa_prompt=${DATA_NAME} \
    qa_evaluator=${DATA_NAME} \
    llm.model_name_or_path=${LLM} \
    test.n_threads=${N_THREAD} \
    test.top_k=${DOC_TOP_K} \
    dataset.data_name=${DATA_NAME}_test
