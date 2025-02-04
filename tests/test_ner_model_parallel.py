from concurrent.futures import ThreadPoolExecutor


def test_llm_ner_model_parallel() -> None:
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
    num_processes = 5

    dotenv.load_dotenv()

    ner_model = instantiate(cfg)

    texts = [
        "What relationship does Fred Gehrke have to the 23rd overall pick in the 2010 Major League Baseball Draft?",
        "What was the other name for the war between the Cherokee people and white settlers in 1793?",
        "Which university has more campuses, Dalhousie University or California State Polytechnic University, Pomona?",
        "Who was the football manager that played in the Football League Cup in 1985 and managed to lead the Birmingham City Football Club's 103rd season to finish in the 18th position?",
        "From 2003 to 2008 Tom Holland was responsible for what  ?",
    ]

    def process_text(text: str) -> tuple:
        ents = ner_model(text)
        return (text, ents)

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(process_text, texts))

    named_entities = {text: ents for text, ents in results}
    print(named_entities)


if __name__ == "__main__":
    test_llm_ner_model_parallel()
