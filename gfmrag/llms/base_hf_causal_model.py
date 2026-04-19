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
    """Hugging Face causal language-model wrapper.

    Args:
        model_name_or_path: Pretrained model name or local path.
        maximun_token: Maximum number of input tokens.
        max_new_tokens: Maximum number of generated tokens.
        dtype: Runtime dtype name.
        quant: Optional quantization mode.
        attn_implementation: Attention backend name.
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
        assert attn_implementation in self.ATTEN_IMPLEMENTATION, (
            f"attn_implementation should be one of {self.ATTEN_IMPLEMENTATION}"
        )
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
        """Generate text from a prompt string or a chat message list.

        Args:
            llm_input: Prompt string or chat message list.
            system_input: Optional system prompt used with string input.

        Returns:
            Generated text, or the raised exception instance when generation fails.
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
