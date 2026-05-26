"""Runtime attention patching for Gemma4 text attention.

Gemma4's OScaR integration is deliberately separate from the Granite patch.
Both families use the same upstream OScaR quantizer and cache helpers, but
Gemma4 has model-specific attention details that matter:

* `Gemma4TextAttention` applies Q/K RMSNorm before RoPE.
* It can alternate full and sliding attention masks through `layer_type`.
* Later layers may share K/V states from an earlier source layer.

The shared-KV behavior is the important wrinkle. Source layers own the K/V cache
and are the only layers whose cache tensors should be quantized. Shared layers
reuse source-layer keys, so they need only the Q-side OScaR rotation to keep
attention scores in the same rotated space.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MethodType
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict

from .config import OscarKVConfig
from .kv_cache_utils import (
    OSCAR_CONFIG_ATTR,
    ensure_oscar_quantizer,
    quantize_layer_cache_after_attention,
    rotate_query_like_oscar,
    update_cache,
)

_ORIGINAL_FORWARD_ATTR = "_gemma4_oscar_original_forward"
_GEMMA4_SYMBOLS_ATTR = "_gemma4_oscar_symbols"


class _Gemma4Symbols(BaseModel):
    """Imported Transformers symbols needed by the Gemma4 patch.

    What it does:
        Stores the Gemma4 text attention class plus the upstream RoPE and eager
        attention helpers that the replacement forward function mirrors.

    Why it exists:
        Gemma4 support is new enough that importing these symbols at package
        import time would make unrelated APIs fragile. Loading them only when a
        caller asks for Gemma4 OScaR support produces clearer version errors.

    How it helps:
        The hot forward path can use concrete callables without repeating lazy
        import logic or depending on global monkeypatches.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    attention_cls: Any
    apply_rotary_pos_emb: Callable[..., torch.Tensor]
    eager_attention_forward: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]


def apply_oscar_to_gemma4(model: torch.nn.Module, config: OscarKVConfig | None = None) -> int:
    """Patch Gemma4 text attention modules in a loaded model.

    What it does:
        Walks the loaded model, finds every `Gemma4TextAttention` instance,
        saves its original forward method, attaches the validated OScaR config
        and Gemma4 symbol bundle, and replaces `forward` with the OScaR-aware
        Gemma4 implementation.

    Why it exists:
        Gemma4's KV cache is created inside text attention during generation.
        OScaR needs to rotate/process K/V before cache update and quantize the
        stored cache after the current attention computation.

    How it helps:
        Callers can use normal Hugging Face `generate(...)` on Gemma4 while
        source attention layers maintain an OScaR-compressed KV cache and
        shared-KV layers keep their queries in the correct rotated space.
    """
    config = OscarKVConfig() if config is None else OscarKVConfig.model_validate(config)
    symbols = _load_gemma4_attention_symbols()
    patched = 0

    for module in model.modules():
        if not isinstance(module, symbols.attention_cls):
            continue

        if not hasattr(module, _ORIGINAL_FORWARD_ATTR):
            setattr(module, _ORIGINAL_FORWARD_ATTR, module.forward)
        setattr(module, OSCAR_CONFIG_ATTR, config)
        setattr(module, _GEMMA4_SYMBOLS_ATTR, symbols)
        module.forward = MethodType(_gemma4_text_attention_forward_with_oscar, module)
        patched += 1

    if patched == 0:
        raise ValueError(
            "No Gemma4TextAttention modules were found. Gemma4-E2B requires a "
            "Transformers release with `transformers.models.gemma4` support and "
            "a loaded Gemma4 model such as `google/gemma-4-E2B`."
        )

    return patched


def restore_gemma4_attention(model: torch.nn.Module) -> int:
    """Restore original Gemma4 attention forward methods on a patched model.

    What it does:
        Finds Gemma4 attention modules previously patched by
        `apply_oscar_to_gemma4`, restores the saved forward method, and removes
        adapter-specific attributes.

    Why it exists:
        Benchmarks and notebooks often compare patched and unpatched generation
        paths in the same Python process.

    How it helps:
        Gemma4 OScaR patching remains reversible at the model-instance level
        without requiring another model download or weight reload.
    """
    restored = 0
    for module in model.modules():
        original = getattr(module, _ORIGINAL_FORWARD_ATTR, None)
        if original is not None:
            module.forward = original
            delattr(module, _ORIGINAL_FORWARD_ATTR)
            for attr in (OSCAR_CONFIG_ATTR, _GEMMA4_SYMBOLS_ATTR):
                if hasattr(module, attr):
                    delattr(module, attr)
            restored += 1
    return restored


def _load_gemma4_attention_symbols() -> _Gemma4Symbols:
    """Import Gemma4 text attention symbols from Transformers.

    What it does:
        Imports `Gemma4TextAttention`, `apply_rotary_pos_emb`, and
        `eager_attention_forward` from the official Transformers Gemma4 module.

    Why it exists:
        The patch must follow Gemma4's current forward math closely, and the
        model family may not exist in older Transformers installations.

    How it helps:
        Users get a direct installation/version error instead of a vague
        attribute failure after model loading has already started.
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import (  # type: ignore
            Gemma4TextAttention,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
    except ImportError as exc:
        raise ImportError(
            "Could not import Hugging Face Gemma4 text attention. Install a "
            "recent Transformers release with Gemma4 support."
        ) from exc

    return _Gemma4Symbols(
        attention_cls=Gemma4TextAttention,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        eager_attention_forward=eager_attention_forward,
    )


def _gemma4_text_attention_forward_with_oscar(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    past_key_values: Any | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Replacement `forward` for Gemma4 text attention.

    What it does:
        Mirrors the official Gemma4 text attention flow: project and normalize
        query/key/value tensors, apply RoPE, run the OScaR tensor transform,
        update the source-layer cache, preserve Gemma4 shared-KV semantics,
        compute eager attention, and quantize source-layer cached K/V tensors
        after attention.

    Why it exists:
        OScaR's cache compression point is inside attention, after the cache is
        updated but before future decode calls read it. Gemma4's public
        generation API does not expose a hook there, especially for shared-KV
        layers that do not own their own K/V projections.

    How it helps:
        Gemma4-E2B can use runtime OScaR KV-cache quantization while preserving
        the model's full/sliding mask routing and shared-KV layer behavior.
    """
    symbols: _Gemma4Symbols = getattr(self, _GEMMA4_SYMBOLS_ATTR)

    input_shape = hidden_states.shape[:-1]
    q_len = input_shape[-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = symbols.apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
    query_states = query_states.transpose(1, 2)

    if getattr(self, "is_kv_shared_layer", False):
        key_states, value_states = _shared_kv_for_gemma4(self, shared_kv_states, query_states.device)
        query_states = rotate_query_like_oscar(self, query_states)
    else:
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = symbols.apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        if q_len > 1:
            ensure_oscar_quantizer(self)

        if hasattr(self, "quarot_quantizer"):
            query_states, key_states, value_states = self.quarot_quantizer.process_kv(
                query_states,
                key_states,
                value_states,
            )

        if past_key_values is not None:
            key_states, value_states = update_cache(
                past_key_values,
                key_states,
                value_states,
                self.layer_idx,
            )

        if getattr(self, "store_full_length_kv", False):
            _store_shared_kv_for_gemma4(self, shared_kv_states, key_states, value_states)

    attn_output, attn_weights = symbols.eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    if past_key_values is not None and not getattr(self, "is_kv_shared_layer", False):
        quantize_layer_cache_after_attention(self, past_key_values, self.layer_idx, q_len)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _shared_kv_for_gemma4(
    module: torch.nn.Module,
    shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return shared K/V states for a Gemma4 shared-KV attention layer.

    What it does:
        Looks up the current layer's `layer_type` in Gemma4's `shared_kv_states`
        dictionary and moves the tensors to the query device.

    Why it exists:
        Gemma4 shared-KV layers intentionally do not own K/V projection weights
        or cache entries. They consume the most recent full-length source-layer
        K/V states for their attention type.

    How it helps:
        The replacement forward can keep shared layers small and explicit while
        producing a clear error if the expected source layer did not populate
        the shared-state dictionary.
    """
    if shared_kv_states is None or module.layer_type not in shared_kv_states:
        raise KeyError(
            "Gemma4 shared-KV state was missing for layer type "
            f"{module.layer_type!r}. This usually means a source attention layer "
            "did not run before a shared-KV layer."
        )

    key_states, value_states = shared_kv_states[module.layer_type]
    return key_states.to(device), value_states.to(device)


def _store_shared_kv_for_gemma4(
    module: torch.nn.Module,
    shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    """Store full-length K/V states for later Gemma4 shared-KV layers.

    What it does:
        Writes the source layer's current full-length key/value tensors into the
        shared-state dictionary under `module.layer_type`.

    Why it exists:
        Gemma4 reuses source-layer K/V tensors in later shared layers, including
        cases where sliding-window cache entries no longer contain the full
        sequence. The shared dictionary is the official per-forward handoff.

    How it helps:
        Shared layers receive the same source K/V tensors they would see in the
        unpatched model, with OScaR's Q/K rotation already applied to keys.
        Cache quantization is performed afterward so same-forward shared layers
        are not forced to consume newly quantized tensors.
    """
    if shared_kv_states is None:
        raise ValueError("Gemma4 shared_kv_states must be provided when storing shared K/V tensors.")
    shared_kv_states[module.layer_type] = key_states, value_states
