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
HYDRA_FULL_ERROR=1 torchrun --nproc-per-node=${N_GPU} -m gfmrag.workflow.kgc_training \
    datasets.train_names=[${TRAIN_DATA_NAME_LIST}] \
    datasets.cfgs.root=${DATA_ROOT} \
    trainer.fast_test=5000 \
    trainer.args.num_epoch=${N_EPOCH} \
    trainer.args.max_steps_per_epoch=${BATCH_PER_EPOCH} \
    trainer.args.train_batch_size=${BATCH_SIZE}
