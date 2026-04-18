from contextlib import nullcontext
from functools import partial
from typing import Any, TypedDict, TypeVar, cast

import numpy as np
import torch
from datasets import Dataset
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    BatchEncoding,
    DataCollatorWithPadding,
    MistralConfig,
    MistralModel,
    PreTrainedTokenizerFast,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoTokenizer
from transformers.utils import logging

from .configuration_nvembed import (
    BidirectionalMistralConfig,
    LatentAttentionConfig,
    NVEmbedConfig,
)

logger = logging.get_logger(__name__)
T = TypeVar("T")


class NVEmbedFeatures(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pool_mask: torch.Tensor | None


class BidirectionalMistralModel(MistralModel):
    config_class = BidirectionalMistralConfig

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False
        self._attn_implementation = "eager"

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Cache | None] | BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            elif not isinstance(past_key_values, Cache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
            )

        if self._attn_implementation == "eager":
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, inputs_embeds.dtype
            )
        else:
            attention_mask = attention_mask if 0 in attention_mask else None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
                hidden_states = layer_outputs
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return (hidden_states, past_key_values if use_cache else None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


def _move_to_device(maybe_tensor: Any, device: torch.device) -> Any:
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device, non_blocking=device.type == "cuda")
    if isinstance(maybe_tensor, dict):
        return {
            key: _move_to_device(value, device) for key, value in maybe_tensor.items()
        }
    if isinstance(maybe_tensor, list):
        return [_move_to_device(x, device) for x in maybe_tensor]
    if isinstance(maybe_tensor, tuple):
        return tuple(_move_to_device(x, device) for x in maybe_tensor)
    return maybe_tensor


def move_to_device(sample: Any, device: torch.device) -> Any:
    if device.type == "cpu":
        return sample

    if len(sample) == 0:
        return {}
    return _move_to_device(sample, device)


def input_transform_func(
    tokenizer: PreTrainedTokenizerFast,
    examples: dict[str, list[str]],
    always_add_eos: bool,
    max_length: int,
    instruction: str,
) -> BatchEncoding:
    eos_token = tokenizer.eos_token or ""
    if always_add_eos:
        examples["input_texts"] = [
            instruction + input_example + eos_token
            for input_example in examples["input_texts"]
        ]
    batch_dict = tokenizer(
        examples["input_texts"],
        max_length=max_length,
        padding=True,
        return_token_type_ids=False,
        return_tensors="pt",
        truncation=True,
    )
    return batch_dict


class PreNorm(torch.nn.Module):
    def __init__(
        self, dim: int, fn: torch.nn.Module, context_dim: int | None = None
    ) -> None:
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)
        self.norm_context = (
            torch.nn.LayerNorm(context_dim) if exists(context_dim) else None
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = self.norm(x)
        if exists(self.norm_context):
            context = cast(torch.Tensor, kwargs["context"])
            normed_context = cast(torch.nn.LayerNorm, self.norm_context)(context)
            kwargs.update(context=normed_context)
        return cast(torch.Tensor, self.fn(x, **kwargs))


class GEGLU(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gates)


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            torch.nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


def exists(val: object | None) -> bool:
    return val is not None


def default(val: T | None, d: T) -> T:
    return val if val is not None else d


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = torch.nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = torch.nn.Linear(inner_dim, query_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True
        ):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class LatentAttentionModel(PreTrainedModel):
    config_class = LatentAttentionConfig

    def __init__(self, config: LatentAttentionConfig):
        super().__init__(config)
        ## cross-attention block
        num_latents, latent_dim, cross_heads, cross_dim_head = (
            config.num_latents_value,
            config.latent_dim,
            config.num_cross_heads,
            config.cross_dim_head,
        )
        dim = config.hidden_dim
        # init latent_attention and latents
        self.cross_attend_blocks = torch.nn.ModuleList(
            [
                PreNorm(
                    latent_dim,
                    Attention(
                        latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head
                    ),
                    context_dim=dim,
                ),
                PreNorm(latent_dim, FeedForward(latent_dim)),
            ]
        )
        self.output_normalize = config.output_normalize
        self.register_parameter(
            "latents", torch.nn.Parameter(torch.randn(num_latents, latent_dim))
        )

    def forward(
        self, hiddens: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        ## cross-attention block
        cross_attn = cast(PreNorm, self.cross_attend_blocks[0])
        cross_ff = cast(PreNorm, self.cross_attend_blocks[1])
        batch_size = hiddens.shape[0]
        x = repeat(self.latents, "n d -> b n d", b=batch_size)
        hiddens = cross_attn(hiddens, context=x, mask=attention_mask) + hiddens
        hiddens = cross_ff(hiddens) + hiddens
        if attention_mask is not None:
            s = torch.sum(hiddens * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            hiddens = s / d
            if self.output_normalize:
                hiddens = torch.nn.functional.normalize(hiddens, p=2, dim=-1)
        return hiddens


class NVEmbedModel(PreTrainedModel):
    config_class = NVEmbedConfig
    _no_split_modules = ["MistralDecoderLayer", "LatentAttentionModel"]

    def __init__(self, config: NVEmbedConfig):
        super().__init__(config)
        text_config = cast(PretrainedConfig | None, config.text_config)
        self.latent_attention_model = cast(
            LatentAttentionModel, AutoModel.from_config(config.latent_attention_config)
        )
        self.embedding_model = (
            cast(PreTrainedModel, AutoModel.from_config(text_config))
            if text_config is not None
            else None
        )
        self.tokenizer = (
            cast(
                PreTrainedTokenizerFast,
                AutoTokenizer.from_pretrained(text_config._name_or_path),
            )
            if text_config is not None
            else None
        )
        self.padding_side = config.padding_side
        self.is_mask_instruction = config.is_mask_instruction
        self.add_eos = config.add_eos
        self.mask_type = config.mask_type
        if config.add_pad_token and self.tokenizer is not None:
            self.add_pad_token()

    def _require_tokenizer(self) -> PreTrainedTokenizerFast:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        return self.tokenizer

    def _require_embedding_model(self) -> PreTrainedModel:
        if self.embedding_model is None:
            raise ValueError("Embedding model is not initialized.")
        return self.embedding_model

    def add_pad_token(self) -> None:
        tokenizer = self._require_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = self.padding_side

    def prepare_kwargs_from_batch(
        self,
        batch_dict: dict[str, torch.Tensor],
        instruction_lens: int,
        device: torch.device,
    ) -> NVEmbedFeatures:
        batch_dict = cast(dict[str, torch.Tensor], move_to_device(batch_dict, device))
        attention_mask = (
            batch_dict["attention_mask"].clone()
            if "attention_mask" in batch_dict
            else None
        )
        if (
            attention_mask is not None
            and self.padding_side == "right"
            and self.is_mask_instruction
            and instruction_lens > 0
        ):
            # Mask out the instruction tokens for mean-pooling
            attention_mask[:, :instruction_lens] = 0
        return {
            "input_ids": batch_dict["input_ids"].long(),
            "attention_mask": batch_dict["attention_mask"],
            "pool_mask": attention_mask,
        }

    @torch.no_grad()
    def _do_encode(
        self,
        prompts: list[str],
        batch_size: int = 1,
        instruction: str = "",
        max_length: int = 4096,
        num_workers: int = 32,
        **kwargs: Any,
    ) -> np.ndarray | torch.FloatTensor:
        tokenizer = self._require_tokenizer()
        embedding_model = self._require_embedding_model()
        dataset: Dataset = Dataset.from_dict({"input_texts": prompts})
        dataset.set_transform(
            partial(
                input_transform_func,
                tokenizer,
                always_add_eos=True,
                max_length=max_length,
                instruction=instruction,
            )
        )

        data_collator = DataCollatorWithPadding(tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=data_collator,
            pin_memory=True,
        )

        if (
            self.padding_side == "right"
            and self.is_mask_instruction
            and len(instruction) > 0
        ):
            instruction_lens = len(tokenizer.tokenize(instruction))
        else:
            instruction_lens = 0

        encoded_embeds: list[torch.Tensor] = []
        device = next(embedding_model.parameters()).device
        for batch_dict in tqdm(data_loader, desc="encoding", mininterval=10):
            features = self.prepare_kwargs_from_batch(
                batch_dict, instruction_lens, device=device
            )
            model_output = cast(dict[str, torch.Tensor], self(**features))
            embeds = model_output["sentence_embeddings"].squeeze(1)
            encoded_embeds.append(embeds)
        encoded_embeds_tensor = cast(torch.Tensor, torch.cat(encoded_embeds, axis=0))
        if "return_numpy" in kwargs and kwargs.get("return_numpy"):
            return encoded_embeds_tensor.cpu().detach().numpy()
        return encoded_embeds_tensor

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pool_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | dict[str, torch.Tensor]:
        autocast_ctx = torch.autocast if torch.cuda.is_available() else nullcontext
        embedding_model = self._require_embedding_model()
        with autocast_ctx("cuda"):
            ## decoder only layer
            outputs = embedding_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            ## latent attention layer
            embeds = self.latent_attention_model(
                outputs.last_hidden_state,
                pool_mask,
            )
        if not return_dict:
            return (embeds,)
        return {"sentence_embeddings": embeds}

    @torch.no_grad()
    def encode(
        self,
        prompts: list[str],
        instruction: str = "",
        max_length: int = 4096,
        **kwargs: Any,
    ) -> torch.Tensor:
        tokenizer = self._require_tokenizer()
        embedding_model = self._require_embedding_model()
        if (
            self.padding_side == "right"
            and self.is_mask_instruction
            and len(instruction) > 0
        ):
            instruction_lens = len(tokenizer.tokenize(instruction))
        else:
            instruction_lens = 0

        device = next(embedding_model.parameters()).device
        batch_dict = input_transform_func(
            tokenizer,
            {"input_texts": [prompt for prompt in prompts]},
            always_add_eos=True,
            max_length=max_length,
            instruction=instruction,
        )

        features: NVEmbedFeatures = self.prepare_kwargs_from_batch(
            batch_dict, instruction_lens, device=device
        )
        model_output = cast(dict[str, torch.Tensor], self(**features))
        return model_output["sentence_embeddings"].squeeze(1)


## AutoModel Register
AutoModel.register(NVEmbedConfig, NVEmbedModel)
AutoModel.register(LatentAttentionConfig, LatentAttentionModel)
AutoModel.register(BidirectionalMistralConfig, BidirectionalMistralModel)

## Register for auto class
NVEmbedModel.register_for_auto_class("AutoModel")
LatentAttentionModel.register_for_auto_class("AutoModel")
BidirectionalMistralModel.register_for_auto_class("AutoModel")
