from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict, Field


class OscarKVConfig(BaseModel):
    """Configuration passed to upstream OScaR's init_quarot helper."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    k_bits: int = Field(default=2, ge=2)
    v_bits: int = Field(default=2, ge=2)
    k_groupsize: int = Field(default=32, gt=0)
    v_groupsize: int = Field(default=32, gt=0)
    k_sym: bool = False
    v_sym: bool = False
    k_clip_ratio: float = Field(default=1.0, gt=0)
    v_clip_ratio: float = Field(default=1.0, gt=0)
    residual_length: int = Field(default=0, ge=0)
    k_token_rotation: bool = False
    k_norm_factoring: bool = True
    use_hadamard: bool = True
    offline_v_hadamard: bool = True

    def as_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(**self.model_dump())
