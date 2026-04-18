# QA inference from retrieved documents
DATA_ROOT="data"
DATA_NAME="2wikimultihopqa"
LLM="gpt-4o-mini"
DOC_TOP_K=5
N_THREAD=10
RETRIEVED_RESULT_PATH="outputs/qa_finetune/latest/predictions_${DATA_NAME}_test.json"
NODE_PATH="${DATA_ROOT}/${DATA_NAME}_test/processed/stage1/nodes.csv"

python -m gfmrag.workflow.qa \
    --config-path config/gfm_reasoner \
    qa_prompt=${DATA_NAME} \
    qa_evaluator=${DATA_NAME} \
    llm.model_name_or_path=${LLM} \
    test.n_threads=${N_THREAD} \
    test.top_k=${DOC_TOP_K} \
    test.retrieved_result_path=${RETRIEVED_RESULT_PATH} \
    test.target_types=[document] \
    test.node_path=${NODE_PATH}
