def test_llm_ner_model() -> None:
    import dotenv
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.kg_construction.ner_model.LLMNERModel",
            "llm_api": "openai",
            "model_name": "gpt-4o-mini",
        }
    )

    dotenv.load_dotenv()

    ner_model = instantiate(cfg)
    text = "When was the judge born who made notable contributions to the trial of the man who tortured, raped, and murdered eight student nurses from South Chicago Community Hospital on the night of July 13-14, 1966?"
    named_entities = ner_model(text)
    print(named_entities)
    assert isinstance(named_entities, list)


if __name__ == "__main__":
    test_llm_ner_model()
