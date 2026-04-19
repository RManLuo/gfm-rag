# Large scale fine-tuning
DATA_ROOT="data"
N_GPU=8
N_EPOCH=15
BATCH_SIZE=4
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
HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.sft_training \
    datasets.train_names=[${TRAIN_DATA_NAME_LIST}] \
    datasets.cfgs.root=${DATA_ROOT} \
    trainer.args.num_epoch=${N_EPOCH} \
    trainer.args.train_batch_size=${BATCH_SIZE}

# Retrieval evaluation
N_GPU=4
DATA_ROOT="data"
CHECKPOINT="rmanluo/GFM-RAG-8M" # Or the path to your checkpoints
HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.sft_training \
    load_model_from_pretrained=${CHECKPOINT} \
    datasets.cfgs.root=${DATA_ROOT} \
    datasets.train_names=[] \
    +trainer.args.eval_batch_size=1 \
    trainer.args.do_train=false \
    trainer.args.do_eval=true \
    trainer.args.do_predict=true
