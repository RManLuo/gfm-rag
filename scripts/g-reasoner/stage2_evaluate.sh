# Retrieval evaluation
N_GPU=2
DATA_ROOT="data"
CHECKPOINT="rmanluo/G-reasoner-34M"

HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.sft_training \
    --config-path config/gfm_reasoner \
    --config-name sft_training \
    load_model_from_pretrained=${CHECKPOINT} \
    +datasets.cfgs.skip_empty_target=true \
    datasets.cfgs.root=${DATA_ROOT} \
    datasets.train_names=[] \
    datasets.valid_names=[hotpotqa_test_v2,musique_test,2wikimultihopqa_test] \
    +trainer.args.eval_batch_size=1 \
    trainer.metrics=[hits@2,hits@5,recall@2,recall@5,mrr] \
    trainer.args.do_train=false \
    trainer.args.do_eval=true \
    trainer.args.do_predict=true \
    trainer.args.split_graph_inference=false \
    trainer.args.split_graph_partition=metis
