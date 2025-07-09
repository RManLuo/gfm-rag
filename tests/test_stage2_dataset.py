def test_graph_dataset() -> None:
    from omegaconf import OmegaConf

    from gfmrag.graph_index_datasets import GraphIndexDataset

    text_emb_cfgs = OmegaConf.create(
        {
            "_target_": "gfmrag.text_emb_models.Qwen3TextEmbModel",
            "text_emb_model_name": "Qwen/Qwen3-Embedding-0.6B",
            "normalize": True,
            "batch_size": 32,
            "query_instruct": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            "passage_instruct": None,
            "truncate_dim": 1024,
        }
    )
    dataset = GraphIndexDataset(
        root="data_full/new_graph_interface",
        data_name="hotpotqa_test",
        text_emb_model_cfgs=text_emb_cfgs,
        use_node_feat=True,
        use_relation_feat=True,
        use_edge_feat=False,
    )
    graph = dataset.graph
    assert graph.feat_dim == 1024


def test_graph_dataset_v1() -> None:
    from omegaconf import OmegaConf

    from gfmrag.graph_index_datasets import GraphIndexDatasetV1

    text_emb_cfgs = OmegaConf.create(
        {
            "_target_": "gfmrag.text_emb_models.Qwen3TextEmbModel",
            "text_emb_model_name": "Qwen/Qwen3-Embedding-0.6B",
            "normalize": True,
            "batch_size": 32,
            "query_instruct": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            "passage_instruct": None,
            "truncate_dim": 1024,
        }
    )
    dataset = GraphIndexDatasetV1(
        target_type="entity",
        root="data_full/new_graph_interface",
        data_name="hotpotqa_test",
        text_emb_model_cfgs=text_emb_cfgs,
        use_node_feat=False,
        use_relation_feat=True,
        use_edge_feat=False,
        inverse_relation_feat="text",
    )
    graph = dataset.graph
    assert graph.feat_dim == 1024


if __name__ == "__main__":
    # test_graph_dataset()
    test_graph_dataset_v1()
