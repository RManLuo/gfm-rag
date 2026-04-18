# Large scale pretraining on the KG-index
N_GPU=8
DATA_ROOT="data"
N_EPOCH=1
BATCH_PER_EPOCH=30000
START_N=0
END_N=19
BATCH_SIZE=4
DATA_NAME_LIST="hotpotqa_train musique_train 2wikimultihopqa_train"
TRAIN_DATA_NAME_LIST=""
for DATA_NAME in ${DATA_NAME_LIST}; do
    for i in $(seq ${START_N} ${END_N}); do
        TRAIN_DATA_NAME_LIST="${TRAIN_DATA_NAME_LIST},${DATA_NAME}${i}"
    done
done
TRAIN_DATA_NAME_LIST=${TRAIN_DATA_NAME_LIST:1}
echo "TRAIN_DATA_NAME_LIST: [${TRAIN_DATA_NAME_LIST}]"
torchrun --nproc-per-node=${N_GPU} -m gfmrag.workflow.stage2_kg_pretrain \
    datasets.train_names=[${TRAIN_DATA_NAME_LIST}] \
    datasets.cfgs.root=${DATA_ROOT} \
    train.fast_test=5000 \
    train.num_epoch=${N_EPOCH} \
    train.batch_per_epoch=${BATCH_PER_EPOCH} \
    train.batch_size=${BATCH_SIZE}
