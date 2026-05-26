"""Model profiles shared by quantization and runtime patching tools.

The codebase supports more than one model family now: Granite remains the
baseline, Gemma4-E2B has its own runtime attention patch, and both families can
use the `.safetensors` weight artifact exporter. Keeping model-specific
defaults in one module prevents CLI, README, tests, and implementation code
from drifting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ModelProfileName = Literal["granite-4.0-1b-base", "gemma4-e2b"]
ModelAutoClass = Literal["causal-lm", "image-text-to-text"]

DEFAULT_GRANITE_MODEL_ID = "ibm-granite/granite-4.0-1b-base"
DEFAULT_GEMMA4_E2B_MODEL_ID = "google/gemma-4-E2B"
DEFAULT_ARTIFACT_PROFILE_NAME: ModelProfileName = "granite-4.0-1b-base"


class ModelProfile(BaseModel):
    """Model-family defaults used by shared artifact code.

    What it does:
        Stores the profile name, Hugging Face model id, Transformers auto-model
        class choice, default artifact directory stem, and whether the profile
        is supported by a runtime OScaR KV-cache patch.

    Why it exists:
        Granite and Gemma do not load through exactly the same Transformers auto
        class or runtime patch module. A profile lets shared code use one
        family-aware source of truth while isolating those details.

    How it helps:
        Adding a future model becomes a small profile addition instead of a new
        copy of the exporter.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    name: ModelProfileName
    model_id: str
    auto_model_class: ModelAutoClass
    artifact_dir_stem: str
    description: str
    supports_oscar_kv_cache: bool = False

    def default_output_dir(self, quantization: str) -> Path:
        """Return a default artifact directory for a quantization method.

        What it does:
            Combines the model profile's directory stem with the quantization
            name and places it under `artifacts/`.

        Why it exists:
            Beginners should not need to invent output paths just to try the
            exporter, but paths should still be predictable and model-specific.

        How it helps:
            `--profile gemma4-e2b` naturally saves under a Gemma directory while
            Granite keeps its existing default layout.
        """
        suffix = quantization.replace("_weight_only", "").replace("_", "-")
        return Path("artifacts") / f"{self.artifact_dir_stem}-{suffix}"


MODEL_PROFILES: dict[ModelProfileName, ModelProfile] = {
    "granite-4.0-1b-base": ModelProfile(
        name="granite-4.0-1b-base",
        model_id=DEFAULT_GRANITE_MODEL_ID,
        auto_model_class="causal-lm",
        artifact_dir_stem="granite-4.0-1b-base",
        description="IBM Granite 4.0 1B Base text model",
        supports_oscar_kv_cache=True,
    ),
    "gemma4-e2b": ModelProfile(
        name="gemma4-e2b",
        model_id=DEFAULT_GEMMA4_E2B_MODEL_ID,
        auto_model_class="image-text-to-text",
        artifact_dir_stem="gemma-4-e2b",
        description="Google Gemma 4 E2B multimodal model",
        supports_oscar_kv_cache=True,
    ),
}


def resolve_model_profile(
    profile: ModelProfileName | None = None,
    *,
    model_id: Optional[str] = None,
    auto_model_class: Optional[ModelAutoClass] = None,
) -> ModelProfile:
    """Resolve a model profile plus optional CLI/API overrides.

    What it does:
        Starts from a named profile, then optionally replaces the model id or
        auto-model class.

    Why it exists:
        The built-in profiles cover the expected Granite and Gemma paths, while
        advanced users may still need to point at a local repo or custom model
        id with the same loading semantics.

    How it helps:
        Artifact code can consume one `ModelProfile` object instead of juggling
        raw profile names, model ids, and auto-class overrides.
    """
    base = MODEL_PROFILES[profile or DEFAULT_ARTIFACT_PROFILE_NAME]
    if model_id is None and auto_model_class is None:
        return base
    return base.model_copy(
        update={
            "model_id": model_id or base.model_id,
            "auto_model_class": auto_model_class or base.auto_model_class,
        }
    )
