import ast
import difflib
import json
import re
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter

from gfmrag.llms import ChatGPT

best_dspy_prompt = {
    "prog": {
        "lm": None,
        "traces": [],
        "train": [],
        "demos": [
            {
                "augmented": True,
                "question": "Are Imperial River (Florida) and Amaradia (Dolj) both located in the same country?",
                "fact_before_filter": '{"fact": [["imperial river", "is located in", "florida"], ["imperial river", "is a river in", "united states"], ["imperial river", "may refer to", "south america"], ["amaradia", "flows through", "ro ia de amaradia"], ["imperial river", "may refer to", "united states"]]}',
                "fact_after_filter": '{"fact":[["imperial river","is located in","florida"],["imperial river","is a river in","united states"],["amaradia","flows through","ro ia de amaradia"]]}',
            },
            {
                "augmented": True,
                "question": "When is the director of film The Ancestor 's birthday?",
                "fact_before_filter": '{"fact": [["jean jacques annaud", "born on", "1 october 1943"], ["tsui hark", "born on", "15 february 1950"], ["pablo trapero", "born on", "4 october 1971"], ["the ancestor", "directed by", "guido brignone"], ["benh zeitlin", "born on", "october 14  1982"]]}',
                "fact_after_filter": '{"fact":[["the ancestor","directed by","guido brignone"]]}',
            },
            {
                "augmented": True,
                "question": "In what geographic region is the country where Teafuone is located?",
                "fact_before_filter": '{"fact": [["teafuaniua", "is on the", "east"], ["motuloa", "lies between", "teafuaniua"], ["motuloa", "lies between", "teafuanonu"], ["teafuone", "is", "islet"], ["teafuone", "located in", "nukufetau"]]}',
                "fact_after_filter": '{"fact":[["teafuone","is","islet"],["teafuone","located in","nukufetau"]]}',
            },
            {
                "augmented": True,
                "question": "When did the director of film S.O.B. (Film) die?",
                "fact_before_filter": '{"fact": [["allan dwan", "died on", "28 december 1981"], ["s o b", "written and directed by", "blake edwards"], ["robert aldrich", "died on", "december 5  1983"], ["robert siodmak", "died on", "10 march 1973"], ["bernardo bertolucci", "died on", "26 november 2018"]]}',
                "fact_after_filter": '{"fact":[["s o b","written and directed by","blake edwards"]]}',
            },
            {
                "augmented": True,
                "question": "Do both films: Gloria (1980 Film) and A New Life (Film) have the directors from the same country?",
                "fact_before_filter": '{"fact": [["sebasti n lelio watt", "received acclaim for directing", "gloria"], ["gloria", "is", "1980 american thriller crime drama film"], ["a brand new life", "is directed by", "ounie lecomte"], ["gloria", "written and directed by", "john cassavetes"], ["a new life", "directed by", "alan alda"]]}',
                "fact_after_filter": '{"fact":[["gloria","is","1980 american thriller crime drama film"],["gloria","written and directed by","john cassavetes"],["a new life","directed by","alan alda"]]}',
            },
            {
                "augmented": True,
                "question": "What is the date of death of the director of film The Old Guard (1960 Film)?",
                "fact_before_filter": '{"fact": [["the old guard", "is", "1960 french comedy film"], ["gilles grangier", "directed", "the old guard"], ["the old guard", "directed by", "gilles grangier"], ["the old fritz", "directed by", "gerhard lamprecht"], ["oswald albert mitchell", "directed", "old mother riley series of films"]]}',
                "fact_after_filter": '{"fact":[["the old guard","is","1960 french comedy film"],["gilles grangier","directed","the old guard"],["the old guard","directed by","gilles grangier"]]}',
            },
            {
                "augmented": True,
                "question": "When is the composer of film Aulad (1968 Film) 's birthday?",
                "fact_before_filter": '{"fact": [["aulad", "has music composed by", "chitragupta shrivastava"], ["aadmi sadak ka", "has music by", "ravi"], ["ravi shankar sharma", "composed music for", "hindi films"], ["gulzar", "was born on", "18 august 1934"], ["aulad", "is a", "1968 hindi language drama film"]]}',
                "fact_after_filter": '{"fact":[["aulad","has music composed by","chitragupta shrivastava"],["aulad","is a","1968 hindi language drama film"]]}',
            },
            {
                "question": "How many households were in the city where Angelical Tears located?",
                "fact_before_filter": '{"fact": [["dow city", "had", "219 households"], ["tucson", "had", "229 762 households"], ["atlantic city", "has", "15 504 households"], ["angelical tears", "located in", "oklahoma city"], ["atlantic city", "had", "15 848 households"]]}',
                "fact_after_filter": '{"fact": [["angelical tears", "located in", "oklahoma city"]]}',
            },
            {
                "question": "Did the movies In The Pope'S Eye and Virgin Mountain, originate from the same country?",
                "fact_before_filter": '{"fact": [["virgin mountain", "released in", "icelandic cinemas"], ["virgin mountain", "directed by", "dagur k ri"], ["virgin mountain", "icelandic title is", "f si"], ["virgin mountain", "won", "2015 nordic council film prize"], ["virgin mountain", "is a", "2015 icelandic drama film"]]}',
                "fact_after_filter": '{"fact": [["virgin mountain", "released in", "icelandic cinemas"], ["virgin mountain", "directed by", "dagur k ri"], ["virgin mountain", "icelandic title is", "f si"], ["virgin mountain", "won", "2015 nordic council film prize"], ["virgin mountain", "is a", "2015 icelandic drama film"]]}',
            },
            {
                "question": "Which film has the director who died earlier, The Virtuous Model or Bulldog Drummond'S Peril?",
                "fact_before_filter": '{"fact": [["the virtuous model", "is", "1919 american silent drama film"], ["bulldog drummond s peril", "directed by", "james p  hogan"], ["the virtuous model", "directed by", "albert capellani"], ["bulldog drummond s revenge", "directed by", "louis king"], ["bulldog drummond s peril", "is", "american film"]]}',
                "fact_after_filter": '{"fact": [["the virtuous model", "is", "1919 american silent drama film"], ["bulldog drummond s peril", "directed by", "james p  hogan"], ["the virtuous model", "directed by", "albert capellani"], ["bulldog drummond s peril", "is", "american film"]]}',
            },
        ],
        "signature": {
            "instructions": 'You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. You must select up to 4 relevant facts from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer. The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts are relevant, return an empty list, {"fact": []}. The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. The future of critical decision-making relies on your ability to accurately filter and present relevant information.',
            "fields": [
                {"prefix": "Question:", "description": "Query for retrieval"},
                {
                    "prefix": "Fact Before Filter:",
                    "description": "Candidate facts to be filtered",
                },
                {
                    "prefix": "Fact After Filter:",
                    "description": "Filtered facts in JSON format",
                },
            ],
        },
        "system": 'Your input fields are:\n1. `question` (str): Query for retrieval\n2. `fact_before_filter` (str): Candidate facts to be filtered\n\nYour output fields are:\n1. `fact_after_filter` (Fact): Filtered facts in JSON format\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n[[ ## fact_after_filter ## ]]\n{fact_after_filter}        # note: the value you produce must be pareseable according to the following JSON schema: {"type": "object", "properties": {"fact": {"type": "array", "description": "A list of facts, each fact is a list of 3 strings: [subject, predicate, object]", "items": {"type": "array", "items": {"type": "string"}}, "title": "Fact"}}, "required": ["fact"], "title": "Fact"}\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is: \n        You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. You must select up to 4 relevant facts from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer. The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts are relevant, return an empty list, {"fact": []}. The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. The future of critical decision-making relies on your ability to accurately filter and present relevant information.',
    }
}


class Fact(BaseModel):
    fact: list[list[str]] = Field(
        description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]"
    )


class DSPyFilter:
    def __init__(self, hipporag: Any) -> None:
        """
        Initializes the object with the necessary configurations and templates for processing input and output messages.

        Parameters:
            hipporag : An object that provides the global configuration and the LLM model required for inference.

        Attributes:
            dspy_file_path : The file path for reranking as specified in the global configuration.
            one_input_template : A string template for formatting the input message with placeholders for specific fields.
            one_output_template : A string template for formatting the output message with specific fields.
            message_template : A template generated using the specified dspy file path.
            llm_infer_fn : A function reference for making inferences using the provided LLM model.
            model_name : The name of the language model as specified in the global configuration.
            default_gen_kwargs : A dictionary for storing the default generation keyword arguments.
        """
        dspy_file_path = "gfmrag/graph_index_construction/sft_constructors/hipporag2/dspy_prompts.json"
        self.one_input_template = """[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`."""
        self.one_output_template = """[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"""
        self.message_template = self.make_template(dspy_file_path)
        self.llm_infer_fn = ChatGPT(model_name_or_path="gpt-4o-mini", retry=5)
        self.model_name = "gpt-4o-mini"
        self.default_gen_kwargs: dict[str, Any] = {}

    def make_template(self, dspy_file_path: str) -> list[dict[str, str]]:
        if dspy_file_path is not None:
            dspy_saved = json.load(open(dspy_file_path))
        else:
            dspy_saved = best_dspy_prompt

        system_prompt = dspy_saved["prog"]["system"]
        message_template = [
            {"role": "system", "content": system_prompt},
        ]
        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            message_template.append(
                {
                    "role": "user",
                    "content": self.one_input_template.format(
                        question=demo["question"],
                        fact_before_filter=demo["fact_before_filter"],
                    ),
                }
            )
            message_template.append(
                {
                    "role": "assistant",
                    "content": self.one_output_template.format(
                        fact_after_filter=demo["fact_after_filter"]
                    ),
                }
            )
        return message_template

    def parse_filter(self, response: str) -> list:
        sections: list[tuple[str | None, list[str]]] = [(None, [])]  # [(None, [])]
        field_header_pattern = re.compile("\\[\\[ ## (\\w+) ## \\]\\]")
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        joined_sections: list[tuple[str | None, str]] = [
            (k, "\n".join(v).strip()) for k, v in sections
        ]
        parsed = []
        for k, value in joined_sections:
            if k == "fact_after_filter":
                try:
                    # fields[k] = parse_value(v, signature.output_fields[k].annotation) if _parse_values else v
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        try:
                            parsed_value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            parsed_value = value
                    parsed = TypeAdapter(Fact).validate_python(parsed_value).fact
                except Exception as e:
                    print(
                        f"Error parsing field {k}: {e}.\n\n\t\tOn attempting to parse the value\n```\n{value}\n```"
                    )

        return parsed

    def llm_call(self, question: str, fact_before_filter: str) -> str:
        # make prompt
        messages = deepcopy(self.message_template)
        messages.append(
            {
                "role": "user",
                "content": self.one_input_template.format(
                    question=question, fact_before_filter=fact_before_filter
                ),
            }
        )
        # call openai

        self.default_gen_kwargs["max_completion_tokens"] = 512

        response = self.llm_infer_fn.client.chat.completions.create(
            messages=messages, model=self.model_name, **self.default_gen_kwargs
        )
        response = response.choices[0].message.content.strip()
        return response

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.rerank(*args, **kwargs)

    def rerank(
        self,
        query: str,
        candidate_items: list[tuple],
        candidate_indices: list[int],
        len_after_rerank: int,
    ) -> tuple[list[int], list[tuple], dict]:
        fact_before_filter = {
            "fact": [list(candidate_item) for candidate_item in candidate_items]
        }
        try:
            # prediction = self.program(question=query, fact_before_filter=json.dumps(fact_before_filter))
            response = self.llm_call(query, json.dumps(fact_before_filter))
            generated_facts = self.parse_filter(response)
        except Exception as e:
            print("exception", e)
            generated_facts = []
        result_indices = []
        for generated_fact in generated_facts:
            closest_matched_fact = difflib.get_close_matches(
                str(generated_fact), [str(i) for i in candidate_items], n=1, cutoff=0.0
            )[0]
            try:
                result_indices.append(candidate_items.index(eval(closest_matched_fact)))
            except Exception as e:
                print("result_indices exception", e)

        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]
        return (
            sorted_candidate_indices[:len_after_rerank],
            sorted_candidate_items[:len_after_rerank],
            {"confidence": None},
        )
