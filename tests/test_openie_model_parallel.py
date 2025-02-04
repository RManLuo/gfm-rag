from concurrent.futures import ThreadPoolExecutor


def test_llm_ner_model_parallel() -> None:
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
    num_processes = 5

    dotenv.load_dotenv()
    openie_model = instantiate(cfg)
    texts = [
        "Fred Gehrke\nClarence Fred Gehrke (April 24, 1918 \u2013 February 9, 2002) was an American football player and executive.\n He played in the National Football League (NFL) for the Cleveland / Los Angeles Rams, San Francisco 49ers and Chicago Cardinals from 1940 through 1950.\n To boost team morale, Gehrke designed and painted the Los Angeles Rams logo in 1948, which was the first painted on the helmets of an NFL team.\n He later served as the general manager of the Denver Broncos from 1977 through 1981.\n He is the great-grandfather of Miami Marlin Christian Yelich"
        "Manny Machado\nManuel Arturo Machado (] ; born July 6, 1992) is an American professional baseball third baseman and shortstop for the Baltimore Orioles of Major League Baseball (MLB).\n He attended Brito High School in Miami and was drafted by the Orioles with the third overall pick in the 2010 Major League Baseball draft.\n He bats and throws right-handed.",
        "Christian Yelich\nChristian Stephen Yelich (born December 5, 1991) is an American professional baseball left fielder for the Miami Marlins of Major League Baseball (MLB).\n Yelich was drafted out of high school by the Marlins in the 1st round (23rd overall) of the 2010 Major League Baseball Draft.\n He stands 6 feet 3 inches and weighs 195 pounds.",
        "Taylor Duncan\nTaylor McDowell \"Dunc\" Duncan (May 12, 1953 in Memphis, Tennessee \u2013 January 3, 2004 in Asheville, North Carolina) was an American baseball infielder.\n Duncan, who was a college teammate of Leon Lee in Sacramento, was selected by the Atlanta Braves as the 10th overall pick of the 1971 Major League Baseball Draft.\n A year later he was traded to the Baltimore Orioles and spent five seasons playing for Orioles-affiliated minor league clubs.\n In September 1977 Duncan was claimed off waivers by the St. Louis Cardinals and made his major league debut, playing a handful of the remaining games.\n In the off-season Duncan changed teams again as the Oakland Athletics selected him in the Rule 5 draft.\n The 1978 season was Duncan's last in Major League Baseball: he appeared in 104 games of the 1978 season playing mostly third base.\n Duncan continued to play in the minor leagues until 1980.\n The obituary of The Sacramento Bee quoted a major league scout who believed that Duncan's career had been hampered by a broken ankle he suffered early in his minor league career.",
        "Kyle Parker\nKyle James Parker (born September 30, 1989) is an American professional baseball left fielder who is currently a free agent.\n Parker was highly regarded during his prep career as both a baseball and football player and chose to attend Clemson University to play both sports.\n After redshirting during his freshman season, Parker spent the 2009 and 2010 seasons as the starting quarterback for the Clemson Tigers football team.\n He was also an integral part of the school's baseball team.\n Parker was drafted by the Colorado Rockies as the 26th overall pick in the 2010 Major League Baseball Draft and made his Major League Baseball (MLB) debut with them in 2014.",
    ]

    def process_text(text: str) -> tuple:
        res = openie_model(text)
        return res

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(process_text, texts))

    print(results)


if __name__ == "__main__":
    test_llm_ner_model_parallel()
