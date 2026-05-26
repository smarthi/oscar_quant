"""High-level loaders for OScaR KV-patched models.

This module is the most direct answer to "give me the OScaR KV patched model."
It loads a supported Hugging Face model, applies the matching runtime attention
patch, and returns a Pydantic wrapper whose `model` field is the patched model
object. The wrapper is intentionally in-memory because KV-cache quantization is
a runtime behavior, not a new set of saved model weights.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import OscarKVConfig
from .models import DEFAULT_GEMMA4_E2B_MODEL_ID, DEFAULT_GRANITE_MODEL_ID


class OscarPatchedGraniteModel(BaseModel):
    """Container returned by `load_oscar_patched_granite`.

    What it does:
        Holds the loaded Hugging Face model, tokenizer, selected model id,
        validated OScaR KV config, and the number of attention layers patched.

    Why it exists:
        The old low-level API patched a model in place and returned only an
        integer. That was useful for diagnostics but not expressive enough when
        the desired output is "the patched Granite model." This wrapper makes
        the patched model the primary returned object.

    How it helps:
        Callers can keep the result, inspect `patched_attention_layers`, and use
        `result.model.generate(...)` directly. The `generate_text` convenience
        method provides the same patched path with less boilerplate.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    model_id: str
    model: Any
    tokenizer: Any
    kv_config: OscarKVConfig
    patched_attention_layers: int = Field(ge=1)

    def generate_text(self, prompt: str, **generate_kwargs: Any) -> str:
        """Generate text with the already patched Granite model.

        What it does:
            Tokenizes a prompt, moves tensors to the model's device, runs
            `model.generate` with `use_cache=True`, and decodes only the newly
            generated continuation.

        Why it exists:
            The patched model is useful only when generation uses the KV cache.
            This helper keeps that requirement visible and avoids repeating
            tokenizer/device boilerplate in quick experiments.

        How it helps:
            A caller can load once, then call `generate_text` repeatedly while
            every generation run uses the OScaR-patched attention path.
        """
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        generation_defaults: dict[str, Any] = {
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        generation_defaults.update(generate_kwargs)
        generation_defaults = {
            key: value for key, value in generation_defaults.items() if value is not None
        }

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_defaults)

        new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)


class OscarPatchedGemma4Model(BaseModel):
    """Container returned by `load_oscar_patched_gemma4`.

    What it does:
        Holds the loaded Gemma4 Hugging Face model, processor, selected model
        id, validated OScaR KV config, and number of Gemma4 text attention
        layers patched.

    Why it exists:
        Gemma4-E2B is multimodal and normally loads through
        `AutoModelForImageTextToText` plus `AutoProcessor`, so it should not be
        squeezed into the Granite tokenizer wrapper.

    How it helps:
        Callers receive an explicit patched Gemma4 object with a text-generation
        convenience method while retaining direct access to the underlying
        processor and Hugging Face model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    model_id: str
    model: Any
    processor: Any
    kv_config: OscarKVConfig
    patched_attention_layers: int = Field(ge=1)

    def generate_text(self, prompt: str, **generate_kwargs: Any) -> str:
        """Generate text with the already patched Gemma4 model.

        What it does:
            Uses the Gemma4 processor to create text-only model inputs, moves
            tensor inputs to the model device, runs `model.generate` with
            `use_cache=True`, and decodes only the newly generated continuation.

        Why it exists:
            The patched Gemma4 path is useful only when generation uses the KV
            cache. Gemma4's processor/tokenizer split also adds enough
            boilerplate that a small convenience method keeps examples clearer.

        How it helps:
            A caller can load once, then call `generate_text` repeatedly while
            source Gemma4 attention layers use the OScaR-compressed cache and
            shared-KV layers use the query-only rotation path.
        """
        import torch

        inputs = _processor_text_inputs(self.processor, prompt)
        device = next(self.model.parameters()).device
        inputs = {
            name: tensor.to(device) if hasattr(tensor, "to") else tensor
            for name, tensor in inputs.items()
        }

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        generation_defaults: dict[str, Any] = {
            "use_cache": True,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        generation_defaults.update(generate_kwargs)
        generation_defaults = {
            key: value for key, value in generation_defaults.items() if value is not None
        }

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_defaults)

        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            new_tokens = generated[:, input_ids.shape[-1] :]
        else:
            new_tokens = generated
        return tokenizer.decode(new_tokens[0], skip_special_tokens=True)


def load_oscar_patched_granite(
    model_id: str = DEFAULT_GRANITE_MODEL_ID,
    kv_config: OscarKVConfig | dict[str, Any] | None = None,
    *,
    torch_dtype: Any = "auto",
    device_map: str | dict[str, Any] | None = "auto",
    attn_implementation: str = "eager",
    trust_remote_code: bool = False,
    **model_kwargs: Any,
) -> OscarPatchedGraniteModel:
    """Load Granite and return an OScaR KV-patched model wrapper.

    What it does:
        Loads the tokenizer and model with Hugging Face Transformers, applies
        `apply_oscar_to_granite` to the loaded model, and returns an
        `OscarPatchedGraniteModel` containing the patched model object.

    Why it exists:
        OScaR KV quantization is applied by changing attention behavior at
        runtime. There is no separate `.safetensors` artifact that represents
        "the KV-patched model"; the patched object in memory is the artifact
        callers need.

    How it helps:
        User code can treat the return value as the baseline deliverable:
        `patched = load_oscar_patched_granite()`, then use `patched.model` or
        `patched.generate_text(...)`. This makes the intended output explicit.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .granite_patch import apply_oscar_to_granite

    resolved_config = OscarKVConfig() if kv_config is None else OscarKVConfig.model_validate(kv_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )

    patched_layers = apply_oscar_to_granite(model, resolved_config)
    return OscarPatchedGraniteModel(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        kv_config=resolved_config,
        patched_attention_layers=patched_layers,
    )


def load_oscar_patched_gemma4(
    model_id: str = DEFAULT_GEMMA4_E2B_MODEL_ID,
    kv_config: OscarKVConfig | dict[str, Any] | None = None,
    *,
    torch_dtype: Any = "auto",
    device_map: str | dict[str, Any] | None = "auto",
    attn_implementation: str = "eager",
    trust_remote_code: bool = False,
    **model_kwargs: Any,
) -> OscarPatchedGemma4Model:
    """Load Gemma4-E2B and return an OScaR KV-patched model wrapper.

    What it does:
        Loads Gemma4 through `AutoModelForImageTextToText` and
        `AutoProcessor`, applies `apply_oscar_to_gemma4` to the text attention
        stack, and returns an `OscarPatchedGemma4Model` containing the patched
        model object.

    Why it exists:
        Gemma4's runtime patch is model-family-specific. It needs to handle
        shared-KV layers and processor-based multimodal loading, so a dedicated
        loader is clearer and safer than overloading the Granite path.

    How it helps:
        User code can call `load_oscar_patched_gemma4()` and immediately use
        `patched.model.generate(...)` or `patched.generate_text(...)` with the
        OScaR attention patch already installed.
    """
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Gemma4-E2B loading requires a recent Transformers release that "
            "provides AutoModelForImageTextToText and AutoProcessor."
        ) from exc

    from .gemma4_patch import apply_oscar_to_gemma4

    resolved_config = OscarKVConfig() if kv_config is None else OscarKVConfig.model_validate(kv_config)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )

    patched_layers = apply_oscar_to_gemma4(model, resolved_config)
    return OscarPatchedGemma4Model(
        model_id=model_id,
        model=model,
        processor=processor,
        kv_config=resolved_config,
        patched_attention_layers=patched_layers,
    )


def _processor_text_inputs(processor: Any, prompt: str) -> dict[str, Any]:
    """Build text-only Gemma4 inputs from an AutoProcessor-like object.

    What it does:
        Calls the processor with `text=prompt` first, then falls back to the
        tokenizer-style positional call if the installed processor uses that
        shape.

    Why it exists:
        Gemma4's processor API is newer than the classic tokenizer-only path.
        This helper absorbs small call-shape differences without making the
        public wrapper method noisy.

    How it helps:
        `OscarPatchedGemma4Model.generate_text(...)` stays focused on generation
        while still working with processor implementations that resemble
        tokenizers for text-only prompts.
    """
    try:
        return dict(processor(text=prompt, return_tensors="pt"))
    except TypeError:
        return dict(processor(prompt, return_tensors="pt"))
