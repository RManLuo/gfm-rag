def test_llm_openie_model() -> None:
    import dotenv
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.kg_construction.openie_model.LLMOPENIEModel",
            "llm_api": "openai",
            "model_name": "gpt-4o-mini",
        }
    )

    dotenv.load_dotenv()

    openie_model = instantiate(cfg)
    text = "Fred Gehrke\nClarence Fred Gehrke (April 24, 1918 \u2013 February 9, 2002) was an American football player and executive.\n He played in the National Football League (NFL) for the Cleveland / Los Angeles Rams, San Francisco 49ers and Chicago Cardinals from 1940 through 1950.\n To boost team morale, Gehrke designed and painted the Los Angeles Rams logo in 1948, which was the first painted on the helmets of an NFL team.\n He later served as the general manager of the Denver Broncos from 1977 through 1981.\n He is the great-grandfather of Miami Marlin Christian Yelich"
    res = openie_model(text)
    print(res)
    assert isinstance(res, dict)


if __name__ == "__main__":
    test_llm_openie_model()
