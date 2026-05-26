from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cli import _dtype, _oscar_config
from .granite_patch import apply_oscar_to_granite
from .models import DEFAULT_GRANITE_MODEL_ID
from .schemas import BenchmarkReport, BenchmarkRun


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=_dtype(args.dtype),
        device_map=args.device_map,
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = _move_inputs_to_model(inputs, model)
    generation_kwargs = _generation_kwargs(args, tokenizer)

    runs: list[BenchmarkRun] = []

    _warmup(model, inputs, generation_kwargs, args.warmup_runs)
    runs.append(_timed_generate("baseline", model, tokenizer, inputs, generation_kwargs))

    if not args.baseline_only:
        patched_layers = apply_oscar_to_granite(model, _oscar_config(args))
        print(f"patched_granite_attention_layers={patched_layers}", file=sys.stderr)
        _warmup(model, inputs, generation_kwargs, args.warmup_runs)
        oscar_result = _timed_generate("oscar_kv_quant", model, tokenizer, inputs, generation_kwargs)
        oscar_result = oscar_result.model_copy(update={"patched_attention_layers": patched_layers})
        runs.append(oscar_result)

    report = BenchmarkReport(
        model_id=args.model_id,
        prompt_tokens=int(inputs["input_ids"].shape[-1]),
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        runs=runs,
    )
    print(report.model_dump_json(indent=2))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline Granite generation with and without OScaR KV-cache quantization."
    )
    parser.add_argument("--model-id", default=DEFAULT_GRANITE_MODEL_ID)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--k-bits", type=int, default=2)
    parser.add_argument("--v-bits", type=int, default=2)
    parser.add_argument("--k-groupsize", type=int, default=32)
    parser.add_argument("--v-groupsize", type=int, default=32)
    parser.add_argument("--k-sym", action="store_true")
    parser.add_argument("--v-sym", action="store_true")
    parser.add_argument("--k-clip-ratio", type=float, default=1.0)
    parser.add_argument("--v-clip-ratio", type=float, default=1.0)
    parser.add_argument("--residual-length", type=int, default=0)
    parser.add_argument("--k-token-rotation", action="store_true")
    parser.add_argument("--disable-k-norm-factoring", action="store_true")
    parser.add_argument("--disable-hadamard", action="store_true")
    parser.add_argument("--disable-offline-v-hadamard", action="store_true")
    return parser.parse_args(argv)


def _generation_kwargs(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def _move_inputs_to_model(inputs: Any, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def _warmup(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    generation_kwargs: dict[str, Any],
    runs: int,
) -> None:
    for _ in range(runs):
        with torch.inference_mode():
            model.generate(**inputs, **generation_kwargs)


def _timed_generate(
    label: str,
    model: torch.nn.Module,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    generation_kwargs: dict[str, Any],
) -> BenchmarkRun:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start

    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    new_token_count = int(new_tokens.shape[-1])
    return BenchmarkRun(
        label=label,
        elapsed_seconds=round(elapsed_seconds, 6),
        new_tokens=new_token_count,
        tokens_per_second=round(new_token_count / elapsed_seconds, 3) if elapsed_seconds > 0 else None,
        text=tokenizer.decode(new_tokens[0], skip_special_tokens=True),
        cuda_peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        if torch.cuda.is_available()
        else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
