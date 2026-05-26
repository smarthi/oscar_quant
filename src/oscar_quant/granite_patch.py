"""Runtime attention patching for Granite models.

OScaR-KV-Quant works by changing how key/value tensors are stored in the
generation cache. Hugging Face Transformers does not expose that as a simple
configuration switch, so this module replaces supported Granite attention
`forward` methods with a compatible implementation that calls OScaR at the
right points in prefill and decode.

The code intentionally keeps the patch local to a loaded model instance. It does
not modify global Transformers classes, which makes experimentation safer in
notebooks and scripts that may load both patched and unpatched models.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict

from .config import OscarKVConfig
from .kv_cache_utils import (
    OSCAR_CONFIG_ATTR,
    ensure_oscar_quantizer,
    quantize_layer_cache_after_attention,
    update_cache,
)

_ORIGINAL_FORWARD_ATTR = "_granite_oscar_original_forward"
_GRANITE_SYMBOLS_ATTR = "_granite_oscar_symbols"


class _AttentionSymbols(BaseModel):
    """Imported symbols needed to patch one Granite attention family.

    What it does:
        Stores the attention class plus its model-family-specific rotary
        embedding and KV-head repetition helpers.

    Why it exists:
        Granite 4.0 1B Base uses `GraniteMoeHybridAttention`, while earlier
        transformer Granite models use `GraniteAttention`. The two classes have
        similar attention math, but their symbols live in different Transformers
        modules.

    How it helps:
        The patcher can support multiple Granite families without hard-coding
        imports inside the hot attention path or duplicating forward logic.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    family: str
    attention_cls: Any
    apply_rotary_pos_emb: Callable[..., Any]
    repeat_kv: Callable[..., Any]


def apply_oscar_to_granite(model: torch.nn.Module, config: OscarKVConfig | None = None) -> int:
    """Patch supported Granite attention modules in a loaded model.

    What it does:
        Walks every module in the model, finds supported Granite attention
        classes, stores their original `forward` method, attaches the validated
        OScaR config and symbol bundle, and replaces `forward` with the OScaR
        aware implementation.

    Why it exists:
        OScaR needs to process key/value tensors during the attention call and
        then quantize the cached tensors after the cache is updated. Hugging
        Face's public generation API does not provide a hook at exactly that
        location.

    How it helps:
        Callers can keep using normal `model.generate` while the KV cache behind
        each supported attention layer is compressed by OScaR.
    """

    config = OscarKVConfig() if config is None else OscarKVConfig.model_validate(config)

    supported_attention = _load_granite_attention_symbols()
    patched = 0

    for module in model.modules():
        symbols = _symbols_for_module(module, supported_attention)
        if symbols is None:
            continue

        if not hasattr(module, _ORIGINAL_FORWARD_ATTR):
            setattr(module, _ORIGINAL_FORWARD_ATTR, module.forward)
        setattr(module, OSCAR_CONFIG_ATTR, config)
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
    """Restore original Granite attention forward methods on a patched model.

    What it does:
        Finds modules that were patched by `apply_oscar_to_granite`, restores
        their saved `forward` method, and removes adapter-specific attributes.

    Why it exists:
        Interactive sessions may need to compare patched and unpatched behavior
        without reloading model weights from disk or the network.

    How it helps:
        The patch becomes reversible at the model-instance level, which makes
        notebooks, tests, and benchmarks less stateful.
    """

    restored = 0
    for module in model.modules():
        original = getattr(module, _ORIGINAL_FORWARD_ATTR, None)
        if original is not None:
            module.forward = original
            delattr(module, _ORIGINAL_FORWARD_ATTR)
            for attr in (OSCAR_CONFIG_ATTR, _GRANITE_SYMBOLS_ATTR):
                if hasattr(module, attr):
                    delattr(module, attr)
            restored += 1
    return restored


def _load_granite_attention_symbols() -> tuple[_AttentionSymbols, ...]:
    """Import supported Granite attention families from Transformers.

    What it does:
        Attempts to import classic Granite attention and Granite 4
        GraniteMoeHybrid attention, collecting the helper functions that each
        implementation uses for RoPE and grouped-query attention.

    Why it exists:
        Different Transformers releases may contain one family, both families,
        or neither. Importing opportunistically lets the adapter work with the
        widest useful range while still producing a clear error when Granite 4
        support is missing.

    How it helps:
        The main patching function can operate over a tuple of supported symbol
        bundles instead of carrying release-specific branching logic.
    """
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
    """Return the symbol bundle matching a module, if it is supported.

    What it does:
        Checks whether a module is an instance of any imported Granite attention
        class and returns the corresponding `_AttentionSymbols`.

    Why it exists:
        `model.modules()` yields every layer in the network, including norms,
        MLPs, embeddings, and container modules. Only attention layers should be
        patched.

    How it helps:
        Keeps the patch loop small and makes it explicit that class matching is
        the gate before replacing a module's `forward` method.
    """
    for symbols in supported_attention:
        if isinstance(module, symbols.attention_cls):
            return symbols
    return None


def _granite_attention_forward_with_oscar(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Replacement `forward` for supported Granite attention layers.

    What it does:
        Reimplements the standard Granite attention flow: project Q/K/V, apply
        RoPE when available, let OScaR transform the tensors, update the
        generation cache, compute eager attention, then quantize the cached
        K/V tensors for future decode steps.

    Why it exists:
        The key integration point is between cache update and future cache
        reuse. OScaR needs unquantized tensors for the current attention
        computation but quantized tensors stored for later tokens.

    How it helps:
        Granite can keep using Hugging Face's `generate` loop while the memory
        footprint of accumulated KV cache is reduced by OScaR.
    """
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
        ensure_oscar_quantizer(self)

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
        key_states, value_states = update_cache(
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

    if past_key_values is not None:
        quantize_layer_cache_after_attention(self, past_key_values, self.layer_idx, q_len)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


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
    """Compute Granite attention with explicit eager PyTorch operations.

    What it does:
        Repeats grouped-query KV heads, applies the causal/padding mask,
        softmaxes attention weights in float32, applies dropout, and multiplies
        by values to produce the attention output.

    Why it exists:
        Fused attention kernels hide intermediate tensors that OScaR needs to
        process and later quantize. Eager attention keeps the integration point
        visible and easier to audit.

    How it helps:
        The patched path behaves like normal Granite attention while remaining
        compatible with OScaR's tensor transformations.
    """
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
