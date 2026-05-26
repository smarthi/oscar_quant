# OScaR KV Quant for IBM Granite

This project adds a small adapter for running IBM Granite causal language models
with the OScaR-KV-Quant technique.

OScaR is a KV-cache quantization method. It does not quantize model weights. The
adapter patches Granite attention layers so prefill and decode cache tensors flow
through OScaR's Canalized Rotation and Omni-Token Scaling quantizer.

## What this supports

- Baseline model: `ibm-granite/granite-4.0-1b-base`
- Granite 4.0 attention exposed through
  `transformers.models.granitemoehybrid.modeling_granitemoehybrid.GraniteMoeHybridAttention`
- Earlier transformer Granite attention exposed through
  `transformers.models.granite.modeling_granite.GraniteAttention`
- Dynamic Hugging Face generation cache
- INT2 KV cache by default, matching the common OScaR configuration
- Pydantic v2 classes for quantization config and benchmark result JSON

The project is intentionally baselined on Granite 4.0 1B Base. Hugging Face
exposes that model with `model_type="granitemoehybrid"` and architecture
`GraniteMoeHybridForCausalLM`; the 1B base config uses attention layers with
RoPE, so this adapter patches its attention modules directly.

## Setup

OScaR's upstream package builds CUDA extensions, so install it from source in the
same Python environment as this repo.

```bash
cd /Users/suneel.marti/opensourceprojects/oscar-granite-kv-quant
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install -e .
bash scripts/install_oscar_dependency.sh
```

If you do not use `uv`, create and activate a normal virtualenv, then run:

```bash
python -m pip install -e .
bash scripts/install_oscar_dependency.sh
```

The script clones upstream OScaR into `third_party/OScaR-KV-Quant` and installs it
editable into the active environment.

## Baseline Granite 4.0 1B Base

Run a baseline comparison between vanilla generation and OScaR KV-cache
quantized generation:

```bash
granite-oscar-baseline \
  --prompt "The capital of France is" \
  --max-new-tokens 64 \
  --k-bits 2 \
  --v-bits 2
```

The command prints JSON with latency, generated tokens, tokens/sec, generated
text, and CUDA peak memory when CUDA is available.

To measure only the unpatched model:

```bash
granite-oscar-baseline \
  --baseline-only \
  --prompt "The capital of France is" \
  --max-new-tokens 64
```

## Generate With Granite

```bash
granite-oscar-generate \
  --prompt "Explain KV-cache quantization in one paragraph." \
  --max-new-tokens 128 \
  --k-bits 2 \
  --v-bits 2
```

You can also run the module directly:

```bash
python -m granite_oscar_quant.cli \
  --prompt "Write a concise Granite deployment checklist."
```

## Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from granite_oscar_quant import OscarKVConfig, apply_oscar_to_granite

model_id = "ibm-granite/granite-4.0-1b-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager",
)

patched_layers = apply_oscar_to_granite(
    model,
    OscarKVConfig(k_bits=2, v_bits=2, k_groupsize=32, v_groupsize=32),
)
print(f"Patched {patched_layers} Granite attention layers")
```

## Notes

- Use CUDA for realistic speed and memory behavior. The upstream OScaR project is
  CUDA-extension based.
- This adapter keeps attention in an eager implementation while patched. That
  makes the cache path explicit and keeps the integration easy to audit.
- The first prompt prefill is stored in full precision for the current attention
  call and then quantized in cache for subsequent decode steps, following the
  upstream OScaR pattern.

## Source References

- IBM Granite 4.0 1B Base model card:
  https://huggingface.co/ibm-granite/granite-4.0-1b-base
- Granite 4.0 1B Base config:
  https://huggingface.co/ibm-granite/granite-4.0-1b-base/blob/main/config.json
- Hugging Face GraniteMoeHybrid docs:
  https://huggingface.co/docs/transformers/en/model_doc/granitemoehybrid
