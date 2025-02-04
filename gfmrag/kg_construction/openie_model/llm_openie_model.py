# Adapt from: https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/openie_with_retrieval_option_parallel.py
import json
import logging
from itertools import chain
from typing import Literal

import numpy as np
from langchain_community.chat_models import ChatLlamaCpp, ChatOllama
from langchain_openai import ChatOpenAI

from gfmrag.kg_construction.langchain_util import init_langchain_model
from gfmrag.kg_construction.openie_extraction_instructions import (
    ner_prompts,
    openie_post_ner_prompts,
)
from gfmrag.kg_construction.utils import extract_json_dict

from .base_model import BaseOPENIEModel

logger = logging.getLogger(__name__)
# Disable OpenAI and httpx logging
# Configure logging level for specific loggers by name
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class LLMOPENIEModel(BaseOPENIEModel):
    """
    A class for performing Open Information Extraction (OpenIE) using Large Language Models.

    This class implements OpenIE functionality by performing Named Entity Recognition (NER)
    and relation extraction using various LLM backends like OpenAI, Together, Ollama, or llama.cpp.

    Args:
        llm_api (Literal["openai", "together", "ollama", "llama.cpp"]): The LLM backend to use.
            Defaults to "openai".
        model_name (str): Name of the specific model to use. Defaults to "gpt-4o-mini".
        max_ner_tokens (int): Maximum number of tokens for NER output. Defaults to 1024.
        max_triples_tokens (int): Maximum number of tokens for relation triples output.
            Defaults to 4096.

    Attributes:
        llm_api: The LLM backend being used
        model_name: Name of the model being used
        max_ner_tokens: Token limit for NER
        max_triples_tokens: Token limit for relation triples
        client: Initialized language model client

    Methods:
        ner: Performs Named Entity Recognition on input text
        openie_post_ner_extract: Extracts relation triples after NER
        __call__: Main method to perform complete OpenIE pipeline

    Examples:
        >>> openie_model = LLMOPENIEModel()
        >>> result = openie_model("Emmanuel Macron is the president of France")
        >>> print(result)
        {'passage': 'Emmanuel Macron is the president of France', 'extracted_entities': ['Emmanuel Macron', 'France'], 'extracted_triples': [['Emmanuel Macron', 'president of', 'France']]}
    """

    def __init__(
        self,
        llm_api: Literal["openai", "together", "ollama", "llama.cpp"] = "openai",
        model_name: str = "gpt-4o-mini",
        max_ner_tokens: int = 1024,
        max_triples_tokens: int = 4096,
    ):
        """Initialize LLM-based OpenIE model.

        Args:
            llm_api (Literal["openai", "together", "ollama", "llama.cpp"]): The LLM API provider to use.
                Defaults to "openai".
            model_name (str): Name of the language model to use. Defaults to "gpt-4o-mini".
            max_ner_tokens (int): Maximum number of tokens for NER processing. Defaults to 1024.
            max_triples_tokens (int): Maximum number of tokens for triple extraction. Defaults to 4096.

        Attributes:
            llm_api: The selected LLM API provider
            model_name: Name of the language model
            max_ner_tokens: Token limit for NER
            max_triples_tokens: Token limit for triples
            client: Initialized language model client
        """
        self.llm_api = llm_api
        self.model_name = model_name
        self.max_ner_tokens = max_ner_tokens
        self.max_triples_tokens = max_triples_tokens

        self.client = init_langchain_model(llm_api, model_name)

    def ner(self, text: str) -> list:
        """
        Performs Named Entity Recognition (NER) on the input text using different LLM clients.

        Args:
            text (str): Input text to extract named entities from.

        Returns:
            list: A list of named entities extracted from the text. Returns empty list if extraction fails.

        Note:
            - For OpenAI client, uses JSON mode with specific parameters
            - For Ollama and LlamaCpp clients, extracts JSON from regular response
            - For other clients, extracts JSON from regular response without JSON mode
            - Handles exceptions by returning empty list and logging error
        """
        ner_messages = ner_prompts.format_prompt(user_input=text)

        try:
            if isinstance(self.client, ChatOpenAI):  # JSON mode
                chat_completion = self.client.invoke(
                    ner_messages.to_messages(),
                    temperature=0,
                    max_tokens=self.max_ner_tokens,
                    stop=["\n\n"],
                    response_format={"type": "json_object"},
                )
                response_content = chat_completion.content
                response_content = eval(response_content)

            elif isinstance(self.client, ChatOllama) or isinstance(
                self.client, ChatLlamaCpp
            ):
                response_content = self.client.invoke(ner_messages.to_messages())
                response_content = extract_json_dict(response_content)

            else:  # no JSON mode
                chat_completion = self.client.invoke(
                    ner_messages.to_messages(), temperature=0
                )
                response_content = chat_completion.content
                response_content = extract_json_dict(response_content)

            if "named_entities" not in response_content:
                response_content = []
            else:
                response_content = response_content["named_entities"]

        except Exception as e:
            logger.error(f"Error in extracting named entities: {e}")
            response_content = []

        return response_content

    def openie_post_ner_extract(self, text: str, entities: list) -> str:
        """
        Extracts open information (triples) from text using LLM, considering pre-identified named entities.

        Args:
            text (str): The input text to extract information from.
            entities (list): List of pre-identified named entities in the text.

        Returns:
            str: JSON string containing the extracted triples. Returns empty JSON object "{}" if extraction fails.

        Raises:
            Exception: Logs any errors that occur during the extraction process.

        Notes:
            - For ChatOpenAI client, uses JSON mode for structured output
            - For ChatOllama and ChatLlamaCpp clients, extracts JSON from unstructured response
            - For other clients, extracts JSON from response content
            - Uses temperature=0 and configured max_tokens for consistent outputs
        """
        named_entity_json = {"named_entities": entities}
        openie_messages = openie_post_ner_prompts.format_prompt(
            passage=text, named_entity_json=json.dumps(named_entity_json)
        )
        try:
            if isinstance(self.client, ChatOpenAI):  # JSON mode
                chat_completion = self.client.invoke(
                    openie_messages.to_messages(),
                    temperature=0,
                    max_tokens=self.max_triples_tokens,
                    response_format={"type": "json_object"},
                )
                response_content = chat_completion.content

            elif isinstance(self.client, ChatOllama) or isinstance(
                self.client, ChatLlamaCpp
            ):
                response_content = self.client.invoke(openie_messages.to_messages())
                response_content = extract_json_dict(response_content)
                response_content = str(response_content)
            else:  # no JSON mode
                chat_completion = self.client.invoke(
                    openie_messages.to_messages(),
                    temperature=0,
                    max_tokens=self.max_triples_tokens,
                )
                response_content = chat_completion.content
                response_content = extract_json_dict(response_content)
                response_content = str(response_content)

        except Exception as e:
            logger.error(f"Error in OpenIE: {e}")
            response_content = "{}"

        return response_content

    def __call__(self, text: str) -> dict:
        """
        Perform OpenIE on the given text.

        Args:
            text (str): input text

        Returns:
            dict: dict of passage, extracted entities, extracted_triples

                - passage (str): input text
                - extracted_entities (list): list of extracted entities
                - extracted_triples (list): list of extracted triples
        """
        res = {"passage": text, "extracted_entities": [], "extracted_triples": []}

        # ner_messages = ner_prompts.format_prompt(user_input=text)
        doc_entities = self.ner(text)
        try:
            doc_entities = list(np.unique(doc_entities))
        except Exception as e:
            logger.error(f"Results has nested lists: {e}")
            doc_entities = list(np.unique(list(chain.from_iterable(doc_entities))))

        triples = self.openie_post_ner_extract(text, doc_entities)
        res["extracted_entities"] = doc_entities
        try:
            res["extracted_triples"] = eval(triples)["triples"]
        except Exception:
            logger.error(f"Error in parsing triples: {triples}")

        return res
