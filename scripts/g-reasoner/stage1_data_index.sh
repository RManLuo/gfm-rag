# Build the index for evaluation datasets
DATA_ROOT="data"
DATA_NAME_LIST="hotpotqa_test_v2 musique_test 2wikimultihopqa_test"
for DATA_NAME in ${DATA_NAME_LIST}; do
    python -m gfmrag.workflow.index_dataset \
        --config-path config/gfm_reasoner \
        dataset.root=${DATA_ROOT} \
        text_emb_model=${TEXT_EMBEDDING_MODEL} \
        dataset.data_name=${DATA_NAME}
done


# Build the index for training datasets
N_GPU=1
DATA_ROOT="data"
DATA_NAME_LIST="hotpotqa_train musique_train 2wikimultihopqa_train" #
START_N=0
END_N=19
for i in $(seq ${START_N} ${END_N}); do
    for DATA_NAME in ${DATA_NAME_LIST}; do
        python -m gfmrag.workflow.index_dataset \
            --config-path config/gfm_reasoner \
            dataset.root=${DATA_ROOT} \
            text_emb_model=${TEXT_EMBEDDING_MODEL} \
            sft_constructor.enable_filtering=${ENABLE_FILTERING} \
            dataset.data_name=${DATA_NAME}${i}
    done
done
