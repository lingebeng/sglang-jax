# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0

"""Inference-only Mistral model compatible with HF and native Mistral weights."""

import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.dtype_config import DtypeConfig
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.llama import LlamaForCausalLM, LlamaMLP, LlamaModel
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.utils.weight_utils import WeightLoader

logger = logging.getLogger(__name__)


def _get_sliding_window(config: PretrainedConfig) -> int | None:
    if hasattr(config, "sliding_window"):
        return config.sliding_window
    return getattr(config, "sliding_window_size", None)


class MistralAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        partial_rotary_factor: int | None = None,
        rope_is_neox_style: bool = True,
        max_position_embeddings: int = 8192,
        sliding_window_size: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        attention_bias: bool = False,
        dtype_config: DtypeConfig | None = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads
        self.head_dim = head_dim or self.hidden_size // self.q_head_num

        if dtype_config is None:
            dtype_config = DtypeConfig(default_dtype=dtype)

        if partial_rotary_factor is None:
            partial_rotary_factor = 1

        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.scaling = self.head_dim**-0.5

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype_config.get_dtype("q_proj"),
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype_config.get_dtype("k_proj"),
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype_config.get_dtype("v_proj"),
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype_config.get_dtype("o_proj"),
            mesh=mesh,
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=rope_is_neox_style,
            rope_scaling=rope_scaling,
            dtype=dtype,
        )

        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            softmax_dtype=dtype_config.get_optional_dtype("softmax"),
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> jax.Array:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(-1, self.q_head_num, self.head_dim)
        k = k.reshape(-1, self.kv_head_num, self.head_dim)
        v = v.reshape(-1, self.kv_head_num, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(
            q, k, v, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool
        )

        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class MistralDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        dtype_config: DtypeConfig | None = None,
    ) -> None:
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        if dtype_config is None:
            dtype_config = DtypeConfig(default_dtype=dtype)

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        sliding_window_size = _get_sliding_window(config)
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)

        self.self_attn = MistralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None),
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            sliding_window_size=sliding_window_size,
            attention_bias=attention_bias,
            dtype=dtype,
            mesh=mesh,
            dtype_config=dtype_config.get_config("self_attn"),
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            dtype=dtype,
            mesh=mesh,
            dtype_config=dtype_config.get_config("mlp"),
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype_config.get_dtype("input_layernorm"),
            dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype_config.get_dtype("post_attention_layernorm"),
            dtype=dtype,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None,
    ):
        layer_callback_flag = []
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        layer_norm_callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "input_layernorm_output", "INPUT_LAYERNORM", self.layer_id
        )
        layer_callback_flag.append(layer_norm_callback_flag)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        attn_callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "self_attn_output", "SELF_ATTN", self.layer_id
        )
        layer_callback_flag.append(attn_callback_flag)
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        mlp_callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "mlp_output", "MLP", self.layer_id
        )
        layer_callback_flag.append(mlp_callback_flag)

        return hidden_states, residual, kv_fused, layer_callback_flag


class MistralModel(LlamaModel):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        is_draft_model: bool = False,
        dtype_config: DtypeConfig | None = None,
    ) -> None:
        nnx.Module.__init__(self)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        if dtype_config is None:
            dtype_config = DtypeConfig(default_dtype=dtype)

        if not is_draft_model:
            self.embed_tokens = Embed(
                config.vocab_size,
                config.hidden_size,
                dtype=dtype,
                kernel_axes=("tensor", None),
                param_dtype=dtype_config.get_dtype("embed_tokens"),
                mesh=mesh,
            )
            self.layers = nnx.data(
                [
                    MistralDecoderLayer(
                        config=config,
                        layer_id=i,
                        dtype=dtype,
                        dtype_config=dtype_config.get_config("layers"),
                        mesh=mesh,
                    )
                    for i in range(config.num_hidden_layers)
                ]
            )
            self.norm = RMSNorm(
                config.hidden_size,
                epsilon=config.rms_norm_eps,
                param_dtype=dtype_config.get_dtype("norm"),
            )
        self.layers_to_capture = []


class MistralForCausalLM(LlamaForCausalLM):
    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        cfg = mc.hf_text_config
        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None:
            head_dim = cfg.hidden_size // cfg.num_attention_heads
            cfg.head_dim = head_dim
            mc.hf_config.head_dim = head_dim

        sliding_window = _get_sliding_window(cfg)
        cfg.sliding_window = sliding_window
        mc.hf_config.sliding_window = sliding_window
        mc.sliding_window = sliding_window

        mc.head_dim = head_dim
        if getattr(mc, "v_head_dim", None) is None:
            mc.v_head_dim = head_dim

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        dtype_config: DtypeConfig | None = None,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        if dtype_config is None:
            dtype_config = DtypeConfig(default_dtype=dtype)
            logger.info("MistralForCausalLM config dtype: %s", dtype)
        else:
            if dtype != dtype_config.default_dtype:
                raise ValueError(
                    f"Global dtype ({dtype}) is not the same as the default dtype provided in dtype_config ({dtype_config.default_dtype})."
                )
            logger.info("MistralForCausalLM using dtype_config: %s", dtype_config)

        self.model = MistralModel(
            config, dtype=self.dtype, dtype_config=dtype_config.get_config("model"), mesh=mesh
        )

        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=dtype_config.get_dtype("lm_head"),
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)
        self.capture_aux_hidden_states = False

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        hf_mappings = self._create_llama_weight_mappings()
        if self._checkpoint_uses_mistral_native_format(loader):
            weight_mappings = self._create_mistral_native_weight_mappings(hf_mappings)
        else:
            weight_mappings = hf_mappings

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("mistral weights loaded successfully!")

    @staticmethod
    def _checkpoint_uses_mistral_native_format(loader: WeightLoader) -> bool:
        if loader.dummy_mode:
            return False

        weight_info = loader._scan_weight_info()
        hf_sentinels = (
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.0.self_attn.q_proj.weight",
        )
        native_sentinels = (
            "tok_embeddings.weight",
            "norm.weight",
            "layers.0.attention.wq.weight",
        )
        has_hf_layout = any(key in weight_info for key in hf_sentinels)
        has_native_layout = any(key in weight_info for key in native_sentinels)
        if has_hf_layout and has_native_layout:
            logger.info(
                "Mistral checkpoint contains both HF and native weight names; "
                "using HF/Llama layout."
            )
        return has_native_layout and not has_hf_layout

    def _create_mistral_native_weight_mappings(self, hf_mappings: dict) -> dict:
        mappings = {
            "tok_embeddings.weight": hf_mappings["model.embed_tokens.weight"],
            "norm.weight": hf_mappings["model.norm.weight"],
        }

        if "lm_head.weight" in hf_mappings:
            mappings["output.weight"] = hf_mappings["lm_head.weight"]

        for layer_idx in range(self.config.num_hidden_layers):
            hf_prefix = f"model.layers.{layer_idx}"
            native_prefix = f"layers.{layer_idx}"

            mappings[f"{native_prefix}.attention_norm.weight"] = hf_mappings[
                f"{hf_prefix}.input_layernorm.weight"
            ]
            mappings[f"{native_prefix}.ffn_norm.weight"] = hf_mappings[
                f"{hf_prefix}.post_attention_layernorm.weight"
            ]
            mappings[f"{native_prefix}.attention.wq.weight"] = hf_mappings[
                f"{hf_prefix}.self_attn.q_proj.weight"
            ]
            mappings[f"{native_prefix}.attention.wk.weight"] = hf_mappings[
                f"{hf_prefix}.self_attn.k_proj.weight"
            ]
            mappings[f"{native_prefix}.attention.wv.weight"] = hf_mappings[
                f"{hf_prefix}.self_attn.v_proj.weight"
            ]
            mappings[f"{native_prefix}.attention.wo.weight"] = hf_mappings[
                f"{hf_prefix}.self_attn.o_proj.weight"
            ]
            mappings[f"{native_prefix}.feed_forward.w1.weight"] = hf_mappings[
                f"{hf_prefix}.mlp.gate_proj.weight"
            ]
            mappings[f"{native_prefix}.feed_forward.w2.weight"] = hf_mappings[
                f"{hf_prefix}.mlp.down_proj.weight"
            ]
            mappings[f"{native_prefix}.feed_forward.w3.weight"] = hf_mappings[
                f"{hf_prefix}.mlp.up_proj.weight"
            ]

        return mappings


class MistralForCausalLMMistralFormat(MistralForCausalLM):
    """Mistral loaded from native Mistral checkpoint key names."""


EntryClass = [MistralForCausalLM, MistralForCausalLMMistralFormat]
