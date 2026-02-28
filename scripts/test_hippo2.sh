N_GPU=4
DATA_ROOT="data"
checkpoints=/nfsdata/data/lluo/gfm_rag/outputs/qa_finetune/2025-09-10/10-18-12/pretrained # I have changed the pre-train config into new format
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.sft_training --config-path config/gfm_reasoner \
    load_model_from_pretrained=${checkpoints} \
    datasets.cfgs.root=${DATA_ROOT} \
    datasets.train_names=[] \
    trainer.args.do_train=false \
    +trainer.args.eval_batch_size=1