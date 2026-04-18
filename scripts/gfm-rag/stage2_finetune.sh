# Large scale fine-tuning
DATA_ROOT="data"
N_GPU=8
N_EPOCH=15
START_N=0
END_N=19
DATA_NAME_LIST="hotpotqa_train musique_train 2wikimultihopqa_train"
TRAIN_DATA_NAME_LIST=""
for DATA_NAME in ${DATA_NAME_LIST}; do
    for i in $(seq ${START_N} ${END_N}); do
        TRAIN_DATA_NAME_LIST="${TRAIN_DATA_NAME_LIST},${DATA_NAME}${i}"
    done
done
TRAIN_DATA_NAME_LIST=${TRAIN_DATA_NAME_LIST:1}
echo "TRAIN_DATA_NAME_LIST: [${TRAIN_DATA_NAME_LIST}]"
torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.stage2_qa_finetune \
    datasets.train_names=[${TRAIN_DATA_NAME_LIST}] \
    datasets.cfgs.root=${DATA_ROOT} \
    train.num_epoch=${N_EPOCH}

# Retrieval evaluation
N_GPU=4
DATA_ROOT="data"
checkpoints=rmanluo/GFM-RAG-8M # Or the path to your checkpoints
torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.stage2_qa_finetune \
    train.checkpoint=${checkpoints} \
    datasets.cfgs.root=${DATA_ROOT} \
    datasets.train_names=[] \
    train.num_epoch=0
