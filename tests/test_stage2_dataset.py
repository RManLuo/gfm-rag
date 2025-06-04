from gfmrag.datasets.kg_dataset import KGDataset
from gfmrag.datasets.qa_dataset import QADataset


def test_kg_dataset() -> None:
    from omegaconf import OmegaConf

    text_emb_cfgs = OmegaConf.create(
        {
            "_target_": "gfmrag.text_emb_models.BaseTextEmbModel",
            "text_emb_model_name": "sentence-transformers/all-mpnet-base-v2",
            "normalize": False,
            "batch_size": 32,
            "query_instruct": "",
            "model_kwargs": None,
        }
    )
    dataset = KGDataset(
        root="data",
        data_name="hotpotqa_test",
        text_emb_model_cfgs=text_emb_cfgs,
    )
    kg = dataset[0]
    assert kg.num_nodes == 82157
    assert kg.num_relations == 35768


def test_kg_datasets_with_documents() -> None:
    from omegaconf import OmegaConf

    from gfmrag.datasets.kg_dataset import KGDatasetWithDocuments

    text_emb_cfgs = OmegaConf.create(
        {
            "_target_": "gfmrag.text_emb_models.BaseTextEmbModel",
            "text_emb_model_name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "normalize": False,
            "batch_size": 4,
            "query_instruct": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            "model_kwargs": None,
        }
    )
    dataset = KGDatasetWithDocuments(
        root="data",
        data_name="hotpotqa_test",
        text_emb_model_cfgs=text_emb_cfgs,
    )
    kg = dataset[0]
    assert kg.num_nodes == 96983
    assert kg.num_relations == 45114


def test_qa_dataset() -> None:
    from omegaconf import OmegaConf

    text_emb_cfgs = OmegaConf.create(
        {
            "text_emb_model_name": "sentence-transformers/all-mpnet-base-v2",
            "normalize": False,
            "batch_size": 32,
            "query_instruct": "",
            "model_kwargs": None,
        }
    )
    dataset = QADataset(
        root="data",
        data_name="hotpotqa_train_example",
        text_emb_model_cfgs=text_emb_cfgs,
    )
    train_data, test_data = dataset._data
    assert len(train_data) == 800
    assert len(test_data) == 200


def test_qa_dataset_with_documents() -> None:
    from omegaconf import OmegaConf

    from gfmrag.datasets.qa_dataset import QADatasetWithDocuments

    text_emb_cfgs = OmegaConf.create(
        {
            "_target_": "gfmrag.text_emb_models.BaseTextEmbModel",
            "text_emb_model_name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "normalize": False,
            "batch_size": 16,
            "query_instruct": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            "model_kwargs": None,
        }
    )
    dataset = QADatasetWithDocuments(
        root="data",
        data_name="hotpotqa_train0",
        text_emb_model_cfgs=text_emb_cfgs,
    )
    train_data, test_data = dataset._data
    assert len(test_data) == 1000


if __name__ == "__main__":
    # test_kg_dataset()
    # test_qa_dataset()
    # test_kg_datasets_with_documents()
    test_qa_dataset_with_documents()
