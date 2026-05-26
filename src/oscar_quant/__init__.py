"""Public package API for OScaR quantization tools.

The package root exposes the lightweight Pydantic models immediately and lazily
loads the torch-heavy patching/export functions only when callers ask for them.
That keeps configuration and schema imports cheap in tools, tests, and
notebooks that do not need to instantiate a model.
"""

from .config import OscarKVConfig
from .schemas import BenchmarkReport, BenchmarkRun

__all__ = [
    "ArtifactQuantizationConfig",
    "BenchmarkReport",
    "BenchmarkRun",
    "OscarKVConfig",
    "OscarPatchedGemma4Model",
    "OscarPatchedGraniteModel",
    "QuantizedArtifactReport",
    "SafetensorFile",
    "apply_oscar_to_gemma4",
    "apply_oscar_to_granite",
    "load_oscar_patched_gemma4",
    "load_oscar_patched_granite",
    "quantize_granite_to_safetensors",
    "quantize_model_to_safetensors",
    "restore_gemma4_attention",
    "restore_granite_attention",
]


def __getattr__(name: str):
    """Lazily import torch-backed patch helpers.

    What it does:
        Resolves torch-backed patch helpers and the high-level patched-model
        loader only when those names are requested.

    Why it exists:
        Importing the patch module imports torch and later touches Hugging Face
        model symbols. Delaying that work keeps simple config/schema imports
        fast and avoids surprising import failures in environments that are only
        inspecting metadata.

    How it helps:
        Downstream code can use `from oscar_quant import OscarKVConfig`
        without paying the model-runtime import cost, while still getting
        convenient package-root imports for the patched-model entry points.
    """
    if name in {"apply_oscar_to_granite", "restore_granite_attention"}:
        from .granite_patch import apply_oscar_to_granite, restore_granite_attention

        return {
            "apply_oscar_to_granite": apply_oscar_to_granite,
            "restore_granite_attention": restore_granite_attention,
        }[name]
    if name in {"apply_oscar_to_gemma4", "restore_gemma4_attention"}:
        from .gemma4_patch import apply_oscar_to_gemma4, restore_gemma4_attention

        return {
            "apply_oscar_to_gemma4": apply_oscar_to_gemma4,
            "restore_gemma4_attention": restore_gemma4_attention,
        }[name]
    if name in {
        "OscarPatchedGemma4Model",
        "OscarPatchedGraniteModel",
        "load_oscar_patched_gemma4",
        "load_oscar_patched_granite",
    }:
        from .loader import (
            OscarPatchedGemma4Model,
            OscarPatchedGraniteModel,
            load_oscar_patched_gemma4,
            load_oscar_patched_granite,
        )

        return {
            "OscarPatchedGemma4Model": OscarPatchedGemma4Model,
            "OscarPatchedGraniteModel": OscarPatchedGraniteModel,
            "load_oscar_patched_gemma4": load_oscar_patched_gemma4,
            "load_oscar_patched_granite": load_oscar_patched_granite,
        }[name]
    if name in {
        "ArtifactQuantizationConfig",
        "QuantizedArtifactReport",
        "SafetensorFile",
        "quantize_granite_to_safetensors",
        "quantize_model_to_safetensors",
    }:
        from .artifact import (
            ArtifactQuantizationConfig,
            QuantizedArtifactReport,
            SafetensorFile,
            quantize_granite_to_safetensors,
            quantize_model_to_safetensors,
        )

        return {
            "ArtifactQuantizationConfig": ArtifactQuantizationConfig,
            "QuantizedArtifactReport": QuantizedArtifactReport,
            "SafetensorFile": SafetensorFile,
            "quantize_granite_to_safetensors": quantize_granite_to_safetensors,
            "quantize_model_to_safetensors": quantize_model_to_safetensors,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
