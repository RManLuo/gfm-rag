# !/bin/bash
N_GPU=1
DATA_ROOT="data"
DATA_NAME="2wikimultihopqa" # hotpotqa musique 2wikimultihopqa
LLM="gpt-4o-mini"
MAX_STEPS=3
MAX_SAMPLE=5
MODEL_PATH=save_models/G-reasoner-34M
HYDRA_FULL_ERROR=1 python -m gfmrag.workflow.qa_ircot_inference \
    --config-path config/gfm_reasoner \
    --config-name stage3_qa_ircot_inference \
    dataset.root=${DATA_ROOT} \
    llm.model_name_or_path=${LLM} \
    qa_prompt=${DATA_NAME} \
    qa_evaluator=${DATA_NAME} \
    agent_prompt=${DATA_NAME}_ircot \
    test.max_steps=${MAX_STEPS} \
    test.max_test_samples=${MAX_SAMPLE} \
    dataset.data_name=${DATA_NAME}_test \
    graph_retriever.model_path=${MODEL_PATH}
