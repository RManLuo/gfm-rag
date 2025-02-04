import logging
import os

import dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .base_language_model import BaseLanguageModel

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


class HfCausalModel(BaseLanguageModel):
    """A class for handling Hugging Face causal language models with various configurations.

    This class provides functionality to load and use Hugging Face's causal language models
    with different precision types, quantization options, and attention implementations.

    Args:
        model_name_or_path : str
            The name or path of the pre-trained model to load
        maximun_token : int, optional
            Maximum number of tokens for the model input, by default 4096
        max_new_tokens : int, optional
            Maximum number of new tokens to generate, by default 1024
        dtype : str, optional
            Data type for model computation ('fp32', 'fp16', or 'bf16'), by default 'bf16'
        quant : str or None, optional
            Quantization option (None, '4bit', or '8bit'), by default None
        attn_implementation : str, optional
            Attention implementation method ('eager', 'sdpa', or 'flash_attention_2'),
            by default 'flash_attention_2'

    Methods:
        token_len(text: str) -> int
            Returns the number of tokens in the input text
        generate_sentence(llm_input: Union[str, list], system_input: str = "") -> Union[str, Exception]
            Generates text based on the input prompt or message list
    """

    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    QUANT = [None, "4bit", "8bit"]
    ATTEN_IMPLEMENTATION = ["eager", "sdpa", "flash_attention_2"]

    def __init__(
        self,
        model_name_or_path: str,
        maximun_token: int = 4096,
        max_new_tokens: int = 1024,
        dtype: str = "bf16",
        quant: None | str = None,
        attn_implementation: str = "flash_attention_2",
    ):
        assert quant in self.QUANT, f"quant should be one of {self.QUANT}"
        assert (
            attn_implementation in self.ATTEN_IMPLEMENTATION
        ), f"attn_implementation should be one of {self.ATTEN_IMPLEMENTATION}"
        assert dtype in self.DTYPE, f"dtype should be one of {self.DTYPE}"
        self.maximun_token = maximun_token
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, token=HF_TOKEN, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            token=HF_TOKEN,
            torch_dtype=self.DTYPE.get(dtype, None),
            load_in_8bit=quant == "8bit",
            load_in_4bit=quant == "4bit",
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        self.maximun_token = self.tokenizer.model_max_length
        self.generator = pipeline(
            "text-generation", model=model, tokenizer=self.tokenizer
        )

    def token_len(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))

    @torch.inference_mode()
    def generate_sentence(
        self, llm_input: str | list, system_input: str = ""
    ) -> str | Exception:
        """
        Generate sentence by using a Language Model.

        This method processes input (either a string or a list of messages) and generates text using the configured language model.
        If a system prompt is provided along with a string input, it will be included in the message structure.

        Args:
            llm_input (Union[str, list]): Input for the language model. Can be either a string containing the prompt,
                or a list of message dictionaries with 'role' and 'content' keys.
            system_input (str, optional): System prompt to be prepended to the input. Only used when llm_input is a string.
                Defaults to empty string.

            Union[str, Exception]: Generated text output from the language model if successful,
                or the Exception object if generation fails.

        Examples:
            >>> # Using string input with system prompt
            >>> model.generate_sentence("Tell me a joke", system_input="Be funny")

            >>> # Using message list input
            >>> messages = [
            ...     {"role": "system", "content": "Be helpful"},
            ...     {"role": "user", "content": "Tell me a joke"}
            ... ]
            >>> model.generate_sentence(messages)
        """
        # If the input is a list, it is assumed that the input is a list of messages
        if isinstance(llm_input, list):
            message = llm_input
        else:
            message = []
            if system_input:
                message.append({"role": "system", "content": system_input})
            message.append({"role": "user", "content": llm_input})
        try:
            outputs = self.generator(
                message,
                return_full_text=False,
                max_new_tokens=self.max_new_tokens,
                handle_long_generation="hole",
            )
            return outputs[0]["generated_text"].strip()  # type: ignore
        except Exception as e:
            return e
