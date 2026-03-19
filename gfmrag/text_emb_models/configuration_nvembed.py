from typing import Any, Literal

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.mistral import MistralConfig

NVEMBED_TYPE = "nvembed"
LATENT_ATTENTION_TYPE = "latent_attention"
BIDIR_MISTRAL_TYPE = "bidir_mistral"


class NVEmbedConfig(PretrainedConfig):
    model_type = NVEMBED_TYPE
    is_composition = False

    def __init__(
        self,
        latent_attention_config: dict[str, Any] | PretrainedConfig | None = None,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        padding_side: Literal["right", "left"] = "right",
        add_pad_token: bool = True,
        is_mask_instruction: bool = True,
        add_eos: bool = True,
        mask_type: str = "b",
        **kwargs: Any,
    ) -> None:
        if isinstance(latent_attention_config, dict):
            latent_attention_config["model_type"] = latent_attention_config.get(
                "model_type", LATENT_ATTENTION_TYPE
            )
            latent_attention_config = CONFIG_MAPPING[
                latent_attention_config["model_type"]
            ](**latent_attention_config)
        elif latent_attention_config is None:
            latent_attention_config = CONFIG_MAPPING[LATENT_ATTENTION_TYPE]()

        self.latent_attention_config = latent_attention_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = None

        self.text_config = text_config
        self.padding_side = padding_side
        self.is_mask_instruction = is_mask_instruction
        self.add_pad_token = add_pad_token
        self.add_eos = add_eos
        self.mask_type = mask_type
        self.hidden_size = kwargs.get("hidden_size", 4096)

        super().__init__(**kwargs)


class LatentAttentionConfig(PretrainedConfig):
    model_type = LATENT_ATTENTION_TYPE
    is_composition = False
    _name_or_path = "latent_attention"

    def __init__(
        self,
        num_latents_value: int = 512,
        num_cross_heads: int = 8,
        output_normalize: bool = True,
        hidden_dim: int = 4096,
        latent_dim: int = 4096,
        cross_dim_head: int = 4096,
        **kwargs: Any,
    ) -> None:
        self.num_latents_value = num_latents_value
        self.num_cross_heads = num_cross_heads
        self.output_normalize = output_normalize
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cross_dim_head = cross_dim_head
        super().__init__(**kwargs)


class BidirectionalMistralConfig(MistralConfig):
    model_type = BIDIR_MISTRAL_TYPE
    keys_to_ignore_at_inference = ["past_key_values"]


AutoConfig.register(NVEMBED_TYPE, NVEmbedConfig)
AutoConfig.register(LATENT_ATTENTION_TYPE, LatentAttentionConfig)
AutoConfig.register(BIDIR_MISTRAL_TYPE, BidirectionalMistralConfig)
