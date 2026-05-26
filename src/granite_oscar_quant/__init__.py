"""IBM Granite adapter for OScaR-KV-Quant."""

from .config import OscarKVConfig
from .schemas import BenchmarkReport, BenchmarkRun

__all__ = [
    "BenchmarkReport",
    "BenchmarkRun",
    "OscarKVConfig",
    "apply_oscar_to_granite",
    "restore_granite_attention",
]


def __getattr__(name: str):
    if name in {"apply_oscar_to_granite", "restore_granite_attention"}:
        from .granite_patch import apply_oscar_to_granite, restore_granite_attention

        return {
            "apply_oscar_to_granite": apply_oscar_to_granite,
            "restore_granite_attention": restore_granite_attention,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
