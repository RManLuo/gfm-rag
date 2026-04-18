def test_qwen3_emb() -> None:
    from gfmrag.text_emb_models.qwen3_model import Qwen3TextEmbModel

    text_emb_model_name = "Qwen/Qwen3-Embedding-0.6B"
    truncate_dim = 512
    text_emb_model = Qwen3TextEmbModel(
        text_emb_model_name=text_emb_model_name,
        query_instruct="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        passage_instruct="Instruct: Given a passage of text, retrieve the most relevant information for the query\nPassage: ",
        truncate_dim=truncate_dim,
    )

    text = [
        "Hello, world!",
        "This is a test.",
        "Qwen3 is a powerful language model.",
    ]
    embeddings = text_emb_model.encode(text, is_query=True, show_progress_bar=False)
    assert len(embeddings) == len(text)
    assert embeddings.shape[1] == truncate_dim


if __name__ == "__main__":
    test_qwen3_emb()
