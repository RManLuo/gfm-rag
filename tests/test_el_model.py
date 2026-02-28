def test_colbert_el_model() -> None:
    import json

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.kg_construction.entity_linking_model.ColbertELModel",
            "checkpoint_path": "tmp/colbertv2.0",
            "root": "tmp",
        }
    )

    el_model = instantiate(cfg)
    ner_entity_list = ["south chicago community hospital", "july 13 14  1966"]
    with open(
        "data_full/GPT-4o-mini/hotpotqa/processed/stage2/2929736454cf0fb4808976f0986c6230/ent2id.json"
    ) as fin:
        ent2id = json.load(fin)
    entity_list = list(ent2id.keys())
    el_model.index(entity_list)
    linked_entity_dict = el_model(ner_entity_list, topk=2)
    print(linked_entity_dict)
    assert isinstance(linked_entity_dict, dict)


def test_dpr_el_model() -> None:
    import json

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.kg_construction.entity_linking_model.DPRELModel",
            "model_name": "BAAI/bge-large-en-v1.5",
            "root": "tmp",
            "use_cache": True,
            "normalize": True,
        }
    )

    el_model = instantiate(cfg)
    ner_entity_list = ["south chicago community hospital", "july 13 14  1966"]
    with open(
        "data_full/GPT-4o-mini/hotpotqa/processed/stage2/2929736454cf0fb4808976f0986c6230/ent2id.json"
    ) as fin:
        ent2id = json.load(fin)
    entity_list = list(ent2id.keys())
    el_model.index(entity_list)
    linked_entity_list = el_model(ner_entity_list, topk=2)
    print(linked_entity_list)
    assert isinstance(linked_entity_list, dict)

def test_nv_el_model() -> None:
    import json

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.kg_construction.entity_linking_model.NVEmbedV2ELModel",
            "model_name": "nvidia/NV-Embed-v2",
            "query_instruct": "Instruct: Given a entity, retrieve entities that are semantically equivalent to the given entity\nQuery: ",
            "passage_instruct": None,
            "root": "tmp",
            "use_cache": True,
            "normalize": True,
            "topk": 5,
            "batch_size": 256,
        }
    )

    el_model = instantiate(cfg)
    ner_entity_list = ["what is one of the stars of  The Newcomers known for"]
    with open(
        "data/hotpotqa/raw/documents.json"
    ) as fin:
        documents = json.load(fin)
    docs = [f"{title}\n{text}" for title, text in documents.items()]
    el_model.index(docs)
    linked_entity_list = el_model(ner_entity_list, topk=5)
    print(linked_entity_list)
    assert isinstance(linked_entity_list, dict)

def test_qwen_el_model() -> None:
    import json

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.graph_index_construction.entity_linking_model.QWENELModel",
            "model_name": "Qwen/Qwen3-Embedding-8B",
            "query_instruct": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            "passage_instruct": None,
            "root": "tmp",
            "use_cache": True,
            "normalize": True,
            "topk": 5,
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.5,
            "batch_size": 256,
        }
    )

    el_model = instantiate(cfg)
    import pandas as pd
    edges = pd.read_csv("data/hotpotqa/processed/stage1/edges.csv", keep_default_na=False)
    docs = edges[edges["relation"] != "is_mentioned_in"][["source", "relation", "target"]].values.tolist()
    facts = [str(fact) for fact in docs]
    el_model.batch_index(facts)

if __name__ == "__main__":
    # test_colbert_el_model()
    # test_dpr_el_model()
    # test_nv_el_model()
    test_qwen_el_model()
