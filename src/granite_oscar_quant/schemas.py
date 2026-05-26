from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    elapsed_seconds: float = Field(ge=0)
    new_tokens: int = Field(ge=0)
    tokens_per_second: Optional[float] = Field(default=None, ge=0)
    text: str
    cuda_peak_allocated_gib: Optional[float] = Field(default=None, ge=0)
    patched_attention_layers: Optional[int] = Field(default=None, ge=0)


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    prompt_tokens: int = Field(ge=0)
    k_bits: int = Field(ge=2)
    v_bits: int = Field(ge=2)
    runs: List[BenchmarkRun]
