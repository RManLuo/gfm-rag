N_GPU=4
N_EPOCH=10
CHECKPOINT=null
PRETRAINED=null
BATCH_SIZE=4
SPLIT_GRAPH_TRAINING=false # Whether to split the large graph into smaller subgraphs for training or inference
SPLIT_GRAPH_INFERENCE=false
SPLIT_GRAPH_METHOD="metis" # "metis" or "contiguous"
N_DIM=1024
N_LAYERS="[${N_DIM},${N_DIM},${N_DIM},${N_DIM},${N_DIM},${N_DIM}]"
SAVE_BEST_ONLY=true
SAVE_PRETRAINED=true
USE_WANDB=false
RUN_NAME="stage2-finetune-g-reasoner"
DATA_ROOT="data"
INIT_DATASETS=false
MAX_DATA_IN_MEMORY=10
DATA_LOADING_WORKER=4

START_N=0
END_N=19
DATA_NAME_LIST="musique_train hotpotqa_train 2wikimultihopqa_train"
TRAIN_DATA_NAME_LIST=""
for DATA_NAME in ${DATA_NAME_LIST}; do
    for i in $(seq ${START_N} ${END_N}); do
        TRAIN_DATA_NAME_LIST="${TRAIN_DATA_NAME_LIST},${DATA_NAME}${i}"
    done
done
TRAIN_DATA_NAME_LIST=${TRAIN_DATA_NAME_LIST:1}
echo "TRAIN_DATA_NAME_LIST: [${TRAIN_DATA_NAME_LIST}]"

HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.sft_training \
    --config-path config/gfm_reasoner \
    save_pretrained=${SAVE_PRETRAINED} \
    wandb.enabled=${USE_WANDB} \
    wandb.name=${RUN_NAME} \
    model.entity_model.input_dim=${N_DIM} \
    model.entity_model.hidden_dims=${N_LAYERS} \
    trainer.args.resume_from_checkpoint=${CHECKPOINT} \
    load_model_from_pretrained=${PRETRAINED} \
    datasets.cfgs.root=${DATA_ROOT} \
    datasets.train_names=[${TRAIN_DATA_NAME_LIST}] \
    datasets.valid_names=[hotpotqa_test_v2,musique_test,2wikimultihopqa_test] \
    datasets.init_datasets=${INIT_DATASETS} \
    datasets.max_datasets_in_memory=${MAX_DATA_IN_MEMORY} \
    datasets.data_loading_workers=${DATA_LOADING_WORKER} \
    trainer.args.num_epoch=${N_EPOCH} \
    trainer.args.save_best_only=${SAVE_BEST_ONLY} \
    +trainer.training_mode=${TRAIN_MODE} \
    +trainer.args.eval_batch_size=${BATCH_SIZE} \
    trainer.args.split_graph_training=${SPLIT_GRAPH_TRAINING} \
    trainer.args.split_graph_inference=${SPLIT_GRAPH_INFERENCE} \
    trainer.args.split_graph_partition=${SPLIT_GRAPH_METHOD} \
    trainer.args.train_batch_size=${BATCH_SIZE}
