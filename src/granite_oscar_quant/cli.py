from __future__ import annotations

import argparse
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import OscarKVConfig
from .granite_patch import apply_oscar_to_granite
from .models import DEFAULT_GRANITE_MODEL_ID


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dtype = _dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=args.device_map,
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
    )

    patched_layers = apply_oscar_to_granite(model, _oscar_config(args))
    print(f"patched_granite_attention_layers={patched_layers}", file=sys.stderr)

    prompt = args.prompt
    if args.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        generated = model.generate(**inputs, **generate_kwargs)

    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    print(tokenizer.decode(new_tokens[0], skip_special_tokens=True))

    if torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        print(f"cuda_peak_allocated_gib={peak_gib:.2f}", file=sys.stderr)

    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IBM Granite with OScaR KV-cache quantization.")
    parser.add_argument("--model-id", default=DEFAULT_GRANITE_MODEL_ID)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
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


def _dtype(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _oscar_config(args: argparse.Namespace) -> OscarKVConfig:
    return OscarKVConfig(
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
        k_norm_factoring=not args.disable_k_norm_factoring,
        use_hadamard=not args.disable_hadamard,
        offline_v_hadamard=not args.disable_offline_v_hadamard,
    )


if __name__ == "__main__":
    raise SystemExit(main())
