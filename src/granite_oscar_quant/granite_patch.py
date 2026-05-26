from __future__ import annotations

import math
from collections.abc import Callable
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict

from .config import OscarKVConfig

_ORIGINAL_FORWARD_ATTR = "_granite_oscar_original_forward"
_OSCAR_CONFIG_ATTR = "_granite_oscar_config"
_GRANITE_SYMBOLS_ATTR = "_granite_oscar_symbols"


class _AttentionSymbols(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    family: str
    attention_cls: Any
    apply_rotary_pos_emb: Callable[..., Any]
    repeat_kv: Callable[..., Any]


def apply_oscar_to_granite(model: torch.nn.Module, config: OscarKVConfig | None = None) -> int:
    """Patch supported Granite attention modules in a loaded model."""

    config = OscarKVConfig() if config is None else OscarKVConfig.model_validate(config)

    supported_attention = _load_granite_attention_symbols()
    patched = 0

    for module in model.modules():
        symbols = _symbols_for_module(module, supported_attention)
        if symbols is None:
            continue

        if not hasattr(module, _ORIGINAL_FORWARD_ATTR):
            setattr(module, _ORIGINAL_FORWARD_ATTR, module.forward)
        setattr(module, _OSCAR_CONFIG_ATTR, config)
        setattr(module, _GRANITE_SYMBOLS_ATTR, symbols)
        module.forward = MethodType(_granite_attention_forward_with_oscar, module)
        patched += 1

    if patched == 0:
        supported = ", ".join(symbol.family for symbol in supported_attention)
        raise ValueError(
            "No supported Granite attention modules were found. Expected one "
            f"of: {supported}. Granite 4.0 1B Base uses GraniteMoeHybridAttention, "
            "which requires a recent transformers release."
        )

    return patched


def restore_granite_attention(model: torch.nn.Module) -> int:
    """Restore original Granite attention forward methods on a patched model."""

    restored = 0
    for module in model.modules():
        original = getattr(module, _ORIGINAL_FORWARD_ATTR, None)
        if original is not None:
            module.forward = original
            delattr(module, _ORIGINAL_FORWARD_ATTR)
            for attr in (_OSCAR_CONFIG_ATTR, _GRANITE_SYMBOLS_ATTR):
                if hasattr(module, attr):
                    delattr(module, attr)
            restored += 1
    return restored


def _load_granite_attention_symbols() -> tuple[_AttentionSymbols, ...]:
    symbols: list[_AttentionSymbols] = []
    import_errors: list[ImportError] = []

    try:
        from transformers.models.granite.modeling_granite import (  # type: ignore
            GraniteAttention,
            apply_rotary_pos_emb,
            repeat_kv,
        )
    except ImportError as exc:
        import_errors.append(exc)
    else:
        symbols.append(
            _AttentionSymbols(
                family="GraniteAttention",
                attention_cls=GraniteAttention,
                apply_rotary_pos_emb=apply_rotary_pos_emb,
                repeat_kv=repeat_kv,
            )
        )

    try:
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (  # type: ignore
            GraniteMoeHybridAttention,
            apply_rotary_pos_emb,
            repeat_kv,
        )
    except ImportError as exc:
        import_errors.append(exc)
    else:
        symbols.append(
            _AttentionSymbols(
                family="GraniteMoeHybridAttention",
                attention_cls=GraniteMoeHybridAttention,
                apply_rotary_pos_emb=apply_rotary_pos_emb,
                repeat_kv=repeat_kv,
            )
        )

    if symbols:
        return tuple(symbols)

    message = (
        "Could not import supported Hugging Face Granite attention classes. "
        "Install transformers>=4.56 for Granite 4.0 1B Base support."
    )
    if import_errors:
        raise ImportError(message) from import_errors[-1]
    raise ImportError(message)


def _symbols_for_module(
    module: torch.nn.Module,
    supported_attention: tuple[_AttentionSymbols, ...],
) -> _AttentionSymbols | None:
    for symbols in supported_attention:
        if isinstance(module, symbols.attention_cls):
            return symbols
    return None


def _ensure_oscar_quantizer(module: torch.nn.Module) -> None:
    config: OscarKVConfig = getattr(module, _OSCAR_CONFIG_ATTR)
    args = config.as_namespace()

    if not hasattr(module, "quarot_quantizer"):
        try:
            from kv_cache_compression.quarot_utils import init_quarot  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OScaR-KV-Quant is not installed in this environment. Run "
                "`bash scripts/install_oscar_dependency.sh` from this repo, or "
                "install https://github.com/ZunhaiSu/OScaR-KV-Quant manually."
            ) from exc

        init_quarot(
            module,
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            k_groupsize=args.k_groupsize,
            v_groupsize=args.v_groupsize,
            k_sym=args.k_sym,
            v_sym=args.v_sym,
            k_clip_ratio=args.k_clip_ratio,
            v_clip_ratio=args.v_clip_ratio,
            residual_length=args.residual_length,
            k_token_rotation=args.k_token_rotation,
            k_norm_factoring=args.k_norm_factoring,
            use_hadamard=args.use_hadamard,
            offline_v_hadamard=args.offline_v_hadamard,
        )
    else:
        quantizer = module.quarot_quantizer
        for attr in ("committed_k_len", "committed_v_len"):
            if hasattr(quantizer, attr):
                setattr(quantizer, attr, 0)


def _granite_attention_forward_with_oscar(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    symbols: _AttentionSymbols = getattr(self, _GRANITE_SYMBOLS_ATTR)

    input_shape = hidden_states.shape[:-1]
    q_len = input_shape[-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        query_states, key_states = symbols.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if q_len > 1:
        _ensure_oscar_quantizer(self)

    if hasattr(self, "quarot_quantizer"):
        query_states, key_states, value_states = self.quarot_quantizer.process_kv(
            query_states,
            key_states,
            value_states,
        )

    if past_key_values is not None:
        cache_kwargs = {"cache_position": kwargs.get("cache_position")}
        if position_embeddings is not None:
            cache_kwargs.update({"sin": sin, "cos": cos})
        key_states, value_states = _update_cache(
            past_key_values,
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs,
        )

    attn_output, attn_weights = _eager_granite_attention(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        symbols.repeat_kv,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=getattr(self, "scaling", 1.0 / math.sqrt(self.head_dim)),
        output_attentions=kwargs.get("output_attentions", False),
    )

    if hasattr(self, "quarot_quantizer") and past_key_values is not None:
        cache_key_states, cache_value_states = _get_layer_cache(past_key_values, self.layer_idx)
        if q_len > 1:
            cache_key_states, cache_value_states = self.quarot_quantizer.quantize_prefill(
                cache_key_states,
                cache_value_states,
            )
        else:
            cache_key_states, cache_value_states = self.quarot_quantizer.quantize_kv_cache(
                cache_key_states,
                cache_value_states,
            )
        _set_layer_cache(past_key_values, self.layer_idx, cache_key_states, cache_value_states)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _update_cache(
    cache: Any,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return cache.update(key_states, value_states, layer_idx, cache_kwargs)
    except TypeError as first_error:
        try:
            return cache.update(key_states, value_states, layer_idx)
        except TypeError:
            raise first_error


def _eager_granite_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    repeat_kv: Callable[..., torch.Tensor],
    *,
    dropout: float,
    scaling: float,
    output_attentions: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        if attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights


def _get_layer_cache(cache: Any, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    key_cache = _read_cache_tensor(cache, layer_idx, "key")
    value_cache = _read_cache_tensor(cache, layer_idx, "value")
    return key_cache, value_cache


def _set_layer_cache(
    cache: Any,
    layer_idx: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cache.key_cache[layer_idx] = key_states
        cache.value_cache[layer_idx] = value_states
        return

    layer = _cache_layer(cache, layer_idx)
    if hasattr(layer, "keys") and hasattr(layer, "values"):
        layer.keys = key_states
        layer.values = value_states
        return
    if hasattr(layer, "key_cache") and hasattr(layer, "value_cache"):
        layer.key_cache = key_states
        layer.value_cache = value_states
        return

    raise TypeError(f"Unsupported cache layer type: {type(layer)!r}")


def _read_cache_tensor(cache: Any, layer_idx: int, kind: str) -> torch.Tensor:
    legacy_attr = f"{kind}_cache"
    if hasattr(cache, legacy_attr):
        return getattr(cache, legacy_attr)[layer_idx]

    layer = _cache_layer(cache, layer_idx)
    for attr in (f"{kind}s", legacy_attr):
        if hasattr(layer, attr):
            tensor = getattr(layer, attr)
            if tensor is not None:
                return tensor

    raise TypeError(f"Could not read {kind} cache from {type(cache)!r}")


def _cache_layer(cache: Any, layer_idx: int) -> Any:
    if hasattr(cache, "layers"):
        return cache.layers[layer_idx]
    raise TypeError(f"Unsupported cache type: {type(cache)!r}")
