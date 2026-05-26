# OScaR KV Quant for IBM Granite

This project loads IBM Granite 4.0 1B Base and patches its attention layers so
generation uses the OScaR-KV-Quant KV-cache quantization path.

The main Python output is an in-memory OScaR KV-patched Granite model:

```python
from granite_oscar_quant import load_oscar_patched_granite

patched_granite = load_oscar_patched_granite()
model = patched_granite.model
tokenizer = patched_granite.tokenizer
```

`model` is a normal Hugging Face causal language model object, but its supported
Granite attention layers have been patched so the KV cache is processed by
OScaR during generation.

## What This Is

OScaR-KV-Quant is a **KV-cache quantization** technique. During autoregressive
generation, transformer models store key/value tensors so future tokens can
reuse previous attention results. That stored cache can become large for long
contexts. This adapter inserts OScaR into the Granite attention path so those
cached key/value tensors are quantized.

This repo is baselined on:

- Model: `ibm-granite/granite-4.0-1b-base`
- Hugging Face architecture: `GraniteMoeHybridForCausalLM`
- Attention class: `GraniteMoeHybridAttention`
- Default KV precision: INT2 keys and INT2 values
- Python: 3.12+

The project also keeps compatibility hooks for older transformer-style Granite
attention exposed as `GraniteAttention`.

## What This Is Not

This project does **not** produce a new quantized `.safetensors` file.

It does **not** quantize model weights.

It does **not** fine-tune Granite.

It does **not** export a standalone model artifact that stays patched after
process exit.

The patch is runtime behavior. You load Granite, patch the in-memory attention
modules, and then call `generate()`.

## Requirements

For the real OScaR path, use a Linux machine with an NVIDIA CUDA GPU. OScaR's
upstream package builds CUDA extensions, so CPU-only machines and macOS are not
good targets for the full integration.

You need:

- Python 3.12+
- `git`
- A CUDA-capable NVIDIA GPU
- NVIDIA driver compatible with the PyTorch CUDA wheel you install
- Enough disk space for the Granite model, Python packages, and OScaR build
- Optional: a Hugging Face token if model downloads require authentication in
  your environment

The helper script installs `torch==2.6.0+cu124` from the PyTorch CUDA 12.4 wheel
index. If your machine needs a different CUDA/PyTorch combination, edit
`scripts/install_oscar_dependency.sh` before running it.

## Quickstart

From a fresh shell:

```bash
cd /Users/suneel.marti/opensourceprojects/oscar-granite-kv-quant
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
bash scripts/install_oscar_dependency.sh
```

If you use `uv`, this is equivalent:

```bash
cd /Users/suneel.marti/opensourceprojects/oscar-granite-kv-quant
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install -e .
bash scripts/install_oscar_dependency.sh
```

The install script clones upstream OScaR into
`third_party/OScaR-KV-Quant`, initializes its submodules, installs the CUDA
PyTorch wheel, and installs OScaR editable into the active environment.

Check that this package imports:

```bash
python -c "from granite_oscar_quant import OscarKVConfig; print(OscarKVConfig())"
```

That command does not download Granite. It only confirms that your Python
environment can see this package.

## Optional Hugging Face Login

If model download fails with an authentication or gated-model error, log in:

```bash
huggingface-cli login
```

Then rerun the command that failed. Hugging Face will cache downloaded model
files on your machine, so the first run is usually the slowest.

## First Smoke Test

Run one generation request through the patched model:

```bash
granite-oscar-generate \
  --prompt "Explain KV-cache quantization in one paragraph." \
  --max-new-tokens 128 \
  --k-bits 2 \
  --v-bits 2
```

Expected stderr:

```text
patched_granite_attention_layers=<positive integer>
```

Expected stdout:

```text
<generated text from Granite>
```

The exact text and speed depend on your hardware, installed packages, and model
generation settings.

You can also run the module directly:

```bash
python -m granite_oscar_quant.cli \
  --prompt "Write a concise Granite deployment checklist."
```

## Main Python API

Use this when you want the patched Granite model object as the output.

```python
from granite_oscar_quant import OscarKVConfig, load_oscar_patched_granite

patched_granite = load_oscar_patched_granite(
    kv_config=OscarKVConfig(
        k_bits=2,
        v_bits=2,
        k_groupsize=32,
        v_groupsize=32,
    ),
    torch_dtype="auto",
    device_map="auto",
)

print(patched_granite.model_id)
print(patched_granite.patched_attention_layers)

# This is the OScaR KV-patched Granite Hugging Face model object.
model = patched_granite.model
tokenizer = patched_granite.tokenizer

text = patched_granite.generate_text(
    "Explain KV-cache quantization in one sentence.",
    max_new_tokens=64,
)
print(text)
```

`load_oscar_patched_granite(...)` returns an `OscarPatchedGraniteModel` Pydantic
wrapper with these important fields:

- `model`: the loaded Granite model with OScaR-patched attention layers
- `tokenizer`: the matching Hugging Face tokenizer
- `kv_config`: the validated OScaR KV quantization config
- `patched_attention_layers`: how many attention layers were patched
- `model_id`: the Hugging Face model id that was loaded

If `patched_attention_layers` is zero, the loader raises an error instead of
returning a silently unpatched model.

## Baseline Benchmark

Run a before/after comparison between vanilla generation and OScaR KV-cache
quantized generation:

```bash
granite-oscar-baseline \
  --prompt "The capital of France is" \
  --max-new-tokens 64 \
  --k-bits 2 \
  --v-bits 2
```

The command prints JSON:

```json
{
  "model_id": "ibm-granite/granite-4.0-1b-base",
  "prompt_tokens": 5,
  "k_bits": 2,
  "v_bits": 2,
  "runs": [
    {
      "label": "baseline",
      "elapsed_seconds": 1.234,
      "new_tokens": 64,
      "tokens_per_second": 51.864,
      "text": "...",
      "cuda_peak_allocated_gib": 3.21,
      "patched_attention_layers": null
    },
    {
      "label": "oscar_kv_quant",
      "elapsed_seconds": 1.456,
      "new_tokens": 64,
      "tokens_per_second": 43.956,
      "text": "...",
      "cuda_peak_allocated_gib": 2.74,
      "patched_attention_layers": 24
    }
  ]
}
```

The numbers above are examples. Real values depend on GPU, prompt length,
generation length, CUDA/PyTorch versions, and OScaR settings.

To measure only the unpatched model:

```bash
granite-oscar-baseline \
  --baseline-only \
  --prompt "The capital of France is" \
  --max-new-tokens 64
```

## Important CLI Options

- `--model-id`: Hugging Face model id. Defaults to
  `ibm-granite/granite-4.0-1b-base`.
- `--max-new-tokens`: number of tokens to generate.
- `--dtype`: one of `auto`, `bfloat16`, `float16`, or `float32`.
- `--device-map`: passed to Hugging Face model loading. Defaults to `auto`.
- `--k-bits` and `--v-bits`: key/value cache quantization bit widths.
- `--k-groupsize` and `--v-groupsize`: quantization group sizes.
- `--temperature`: `0.0` means greedy decoding; values above zero enable
  sampling.
- `--trust-remote-code`: passed to Hugging Face loading when needed.

## How It Works

At a high level:

1. Load the Granite model and tokenizer with Hugging Face Transformers.
2. Find supported Granite attention modules.
3. Save each module's original `forward` method.
4. Replace `forward` with an OScaR-aware eager attention implementation.
5. During prompt prefill, initialize OScaR for each attention layer.
6. During generation, let OScaR process key/value tensors and quantize cached
   K/V tensors.
7. Keep using normal Hugging Face `model.generate(...)`.

The adapter uses eager attention because OScaR needs access to intermediate
key/value tensors and the cache update path. Fused attention kernels tend to hide
those details.

The first prompt prefill is used for current attention computation and then the
cache is quantized for later decode steps. Later tokens reuse the quantized
cache.

## Project Layout

```text
src/granite_oscar_quant/
  __init__.py       Public package exports
  benchmark.py      Baseline vs OScaR benchmark CLI
  cli.py            Single generation CLI
  config.py         Pydantic OScaR KV config
  granite_patch.py  Runtime Granite attention patch
  loader.py         High-level patched Granite model loader
  models.py         Default model id
  schemas.py        Pydantic benchmark result schemas

scripts/
  install_oscar_dependency.sh

tests/
  test_config.py
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'kv_cache_compression'`

OScaR is not installed in the active environment. Activate your virtualenv and
run:

```bash
bash scripts/install_oscar_dependency.sh
```

### CUDA extension build fails

Check that:

- You are on a CUDA-capable NVIDIA system.
- Your NVIDIA driver supports the CUDA version used by the PyTorch wheel.
- Your Python environment is active.
- You have build tools installed.
- The PyTorch version in `scripts/install_oscar_dependency.sh` matches your
  system.

If you need a different PyTorch wheel, edit the script before running it.

### `No supported Granite attention modules were found`

The loaded model did not contain `GraniteMoeHybridAttention` or
`GraniteAttention` modules. Make sure:

- You are using `ibm-granite/granite-4.0-1b-base`, or another supported Granite
  model.
- You installed a recent enough `transformers` version.
- You did not load a model family with a different attention implementation.

### Out of memory

Try:

- Lower `--max-new-tokens`.
- Use a shorter prompt.
- Use `--dtype bfloat16` or `--dtype float16`.
- Close other GPU workloads.
- Use a smaller model if you changed `--model-id`.

### The CLI prints text, but I wanted a patched model

Use the Python API:

```python
from granite_oscar_quant import load_oscar_patched_granite

patched_granite = load_oscar_patched_granite()
model = patched_granite.model
```

The CLI is for running generation. The Python API is for getting the patched
model object.

### Can I save the patched model?

Not as a new quantized weight file. The patch changes runtime attention and
KV-cache behavior. Save your Python code/config, then call
`load_oscar_patched_granite(...)` again when you start a new process.

## Source References

- IBM Granite 4.0 1B Base model card:
  https://huggingface.co/ibm-granite/granite-4.0-1b-base
- Granite 4.0 1B Base config:
  https://huggingface.co/ibm-granite/granite-4.0-1b-base/blob/main/config.json
- Hugging Face GraniteMoeHybrid docs:
  https://huggingface.co/docs/transformers/en/model_doc/granitemoehybrid
- OScaR-KV-Quant:
  https://github.com/ZunhaiSu/OScaR-KV-Quant
