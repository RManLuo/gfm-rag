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
    n_ctx: int | None = None,
    low_vram: bool = False,
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
        options = {}
        if n_ctx is not None:
            options["num_ctx"] = n_ctx

        return ChatOllama(
            model=model_name,  # e.g., 'llama3'
            temperature=temperature,
            options=options if options else None,
            **kwargs,
        )

    elif llm == "llama.cpp":
        # https://python.langchain.com/v0.2/docs/integrations/chat/llamacpp/
        llama_kwargs = {
            "model_path": model_name,  # model_name is the model path (gguf file)
            "temperature": temperature,
            "verbose": True,
        }

        if n_ctx is not None:
            llama_kwargs["n_ctx"] = n_ctx

        if low_vram:
            llama_kwargs["low_vram"] = True

        llama_kwargs.update(kwargs)

        return ChatLlamaCpp(**llama_kwargs)

    else:
        # add any LLMs you want to use here using LangChain
        raise NotImplementedError(f"LLM '{llm}' not implemented yet.")
