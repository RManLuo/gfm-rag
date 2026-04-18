DATA_ROOT="data"
DATA_NAME_LIST="graphrag_bench_cs graphrag_benchmark_medical graphrag_benchmark_novel"
for DATA_NAME in ${DATA_NAME_LIST}; do
    DATA_NAME=${DATA_NAME}
    CUDA_VISIBLE_DEVICES=0 python -m gfmrag.workflow.index_dataset \
    dataset.root=${DATA_ROOT} \
    dataset.data_name=${DATA_NAME} \
    openie_model.max_ner_tokens=2048 \
    ner_model.max_tokens=2048 \
    graph_constructor.num_processes=15 \
    force=false
done
