# Adapt from: https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/named_entity_extraction_parallel.py
import logging
from typing import Literal

from langchain_community.chat_models import ChatLlamaCpp, ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from gfmrag.kg_construction.langchain_util import init_langchain_model
from gfmrag.kg_construction.utils import extract_json_dict, processing_phrases

from .base_model import BaseNERModel

logger = logging.getLogger(__name__)
# Disable OpenAI and httpx logging
# Configure logging level for specific loggers by name
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

query_prompt_one_shot_input = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?

"""
query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""

query_prompt_template = """
Question: {}

"""


class LLMNERModel(BaseNERModel):
    """A Named Entity Recognition (NER) model that uses Language Models (LLMs) for entity extraction.

    This class implements entity extraction using various LLM backends (OpenAI, Together, Ollama, llama.cpp)
    through the Langchain interface. It processes text input and returns a list of extracted named entities.

    Args:
        llm_api (Literal["openai", "together", "ollama", "llama.cpp"]): The LLM backend to use. Defaults to "openai".
        model_name (str): Name of the specific model to use. Defaults to "gpt-4o-mini".
        max_tokens (int): Maximum number of tokens in the response. Defaults to 1024.

    Methods:
        __call__: Extracts named entities from the input text.

    Raises:
        Exception: If there's an error in extracting or processing named entities.
    """

    def __init__(
        self,
        llm_api: Literal["openai", "together", "ollama", "llama.cpp"] = "openai",
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 1024,
    ):
        """Initialize the LLM-based NER model.

        Args:
            llm_api (Literal["openai", "together", "ollama", "llama.cpp"]): The LLM API provider to use.
                Defaults to "openai".
            model_name (str): Name of the language model to use.
                Defaults to "gpt-4o-mini".
            max_tokens (int): Maximum number of tokens for model output.
                Defaults to 1024.
        """

        self.llm_api = llm_api
        self.model_name = model_name
        self.max_tokens = max_tokens

        self.client = init_langchain_model(llm_api, model_name)

    def __call__(self, text: str) -> list:
        """Process text input to extract named entities using different chat models.

        This method handles entity extraction using various chat models (OpenAI, Ollama, LlamaCpp),
        with special handling for JSON mode responses.

        Args:
            text (str): The input text to extract named entities from.

        Returns:
            list: A list of processed named entities extracted from the text.
                 Returns empty list if extraction fails.

        Raises:
            None: Exceptions are caught and handled internally, logging errors when they occur.

        Examples:
            >>> ner_model = NERModel()
            >>> entities = ner_model("Sample text with named entities")
            >>> print(entities)
            ['Entity1', 'Entity2']
        """
        query_ner_prompts = ChatPromptTemplate.from_messages(
            [
                SystemMessage("You're a very effective entity extraction system."),
                HumanMessage(query_prompt_one_shot_input),
                AIMessage(query_prompt_one_shot_output),
                HumanMessage(query_prompt_template.format(text)),
            ]
        )
        query_ner_messages = query_ner_prompts.format_prompt()

        json_mode = False
        if isinstance(self.client, ChatOpenAI):  # JSON mode
            chat_completion = self.client.invoke(
                query_ner_messages.to_messages(),
                temperature=0,
                max_tokens=self.max_tokens,
                stop=["\n\n"],
                response_format={"type": "json_object"},
            )
            response_content = chat_completion.content
            chat_completion.response_metadata["token_usage"]["total_tokens"]
            json_mode = True
        elif isinstance(self.client, ChatOllama) or isinstance(
            self.client, ChatLlamaCpp
        ):
            response_content = self.client.invoke(query_ner_messages.to_messages())
            response_content = extract_json_dict(response_content)
            len(response_content.split())
        else:  # no JSON mode
            chat_completion = self.client.invoke(
                query_ner_messages.to_messages(),
                temperature=0,
                max_tokens=self.max_tokens,
                stop=["\n\n"],
            )
            response_content = chat_completion.content
            response_content = extract_json_dict(response_content)
            chat_completion.response_metadata["token_usage"]["total_tokens"]

        if not json_mode:
            try:
                assert "named_entities" in response_content
                response_content = str(response_content)
            except Exception as e:
                print("Query NER exception", e)
                response_content = {"named_entities": []}

        try:
            ner_list = eval(response_content)["named_entities"]
            query_ner_list = [processing_phrases(ner) for ner in ner_list]
            return query_ner_list
        except Exception as e:
            logger.error(f"Error in extracting named entities: {e}")
            return []
