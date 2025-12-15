# G-reasoner Retrieval Evaluation on GraphRAG Benchmarks

N_GPU=1
DATA_ROOT="data" #
checkpoint=${PATH_TO_YOUR_CHECKPOINT} # Replace with your checkpoint path
python -m gfmrag.workflow.sft_training \
    --config-path=../config/gfm_reasoner \
    load_model_from_pretrained=${checkpoint} \
    +datasets.cfgs.skip_empty_target=false \
    datasets.cfgs.root=${DATA_ROOT} \
    datasets.train_names=[] \
    datasets.valid_names=[graphrag_bench_cs,graphrag_benchmark_medical,graphrag_benchmark_novel] \
    +trainer.args.eval_batch_size=1 \
    trainer.target_types=[document] \
    trainer.args.do_train=false \
    +trainer.args.do_predict=true \
    trainer.args.do_eval=false

# GFM-RAG Retrieval Evaluation on GraphRAG Benchmarks
N_GPU=1
DATA_ROOT="data" #
checkpoint=${PATH_TO_YOUR_CHECKPOINT} # Replace with your checkpoint path
python -m gfmrag.workflow.sft_training \
    --config-path=../config/gfm_rag \
    load_model_from_pretrained=${checkpoint} \
    +datasets.cfgs.skip_empty_target=false \
    datasets.cfgs.root=${DATA_ROOT} \
    datasets.train_names=[] \
    datasets.valid_names=[graphrag_bench_cs,graphrag_benchmark_medical,graphrag_benchmark_novel] \
    +trainer.args.eval_batch_size=1 \
    trainer.target_types=[document] \
    trainer.args.do_train=false \
    +trainer.args.do_predict=true \
    trainer.args.do_eval=false
