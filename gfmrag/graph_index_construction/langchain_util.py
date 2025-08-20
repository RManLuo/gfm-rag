import os
from typing import Any

from langchain_community.chat_models import ChatLlamaCpp, ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether


def init_langchain_model(
    llm: str,
    model_name: str,
    temperature: float = 0.0,
    max_retries: int = 5,
    timeout: int = 60,
    **kwargs: Any,
) -> ChatOpenAI | ChatTogether | ChatOllama | ChatLlamaCpp:
    """
    Initialize a language model from the langchain library.
    :param llm: The LLM to use, e.g., 'openai', 'together'
    :param model_name: The model name to use, e.g., 'gpt-3.5-turbo'
    """
    if llm == "openai":
        # https://python.langchain.com/v0.1/docs/integrations/chat/openai/

        assert model_name.startswith("gpt-")
        return ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=model_name,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
            **kwargs,
        )
    elif llm == "nvidia":
        # https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints/

        return ChatNVIDIA(
            nvidia_api_key=os.environ.get("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1",
            model=model_name,
            temperature=temperature,
            **kwargs,
        )
    elif llm == "together":
        # https://python.langchain.com/v0.1/docs/integrations/chat/together/

        return ChatTogether(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            model=model_name,
            temperature=temperature,
            **kwargs,
        )
    elif llm == "ollama":
        # https://python.langchain.com/v0.1/docs/integrations/chat/ollama/

        return ChatOllama(model=model_name)  # e.g., 'llama3'
    elif llm == "llama.cpp":
        # https://python.langchain.com/v0.2/docs/integrations/chat/llamacpp/

        return ChatLlamaCpp(
            model_path=model_name, verbose=True
        )  # model_name is the model path (gguf file)
    else:
        # add any LLMs you want to use here using LangChain
        raise NotImplementedError(f"LLM '{llm}' not implemented yet.")
