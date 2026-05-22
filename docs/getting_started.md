# Getting Started

This guide shows the manual Cerebellum workflow from model weights to a final
mixed-precision GGUF.

## Prerequisites

- Python 3.10 or newer.
- A local install of this package:

  ```bash
  pip install -e ".[dev]"
  ```

- llama.cpp with `llama-quantize` and, for measured ablation, a perplexity
  runner such as `llama-perplexity`.
- A source GGUF, normally F16/BF16 or another high precision source.
- Enough disk space for temporary GGUFs. Measured ablation creates one temporary
  quant per tested tensor unless your runner deletes it after each test.

## Step 1: Generate an Imatrix

The streaming generator reads safetensors one tensor at a time and writes the
legacy imatrix binary format used by llama.cpp:

```bash
python -m osmosis.imatrix_stream \
  --model <hf-model-id-or-local-safetensors-dir> \
  --output cerebellum_imatrix.dat \
  -v
```

Use `osmosis.imatrix_gen` instead when you want optional activation calibration:

```bash
python -m osmosis.imatrix_gen \
  --model <local-model-dir> \
  --output cerebellum_imatrix.dat \
  --num-samples 8 \
  -v
```

Use `--no-calibrate` with `imatrix_gen` if you want weight-only behavior.

## Step 2: Build a Baseline GGUF

Use stock llama.cpp. Cerebellum does not replace the GGUF quantizer:

```bash
llama-quantize \
  --imatrix cerebellum_imatrix.dat \
  source-f16.gguf \
  baseline-Q4_K_M.gguf \
  Q4_K_M
```

Run perplexity on the baseline and record the value in `ablation_results.json`.

## Step 3: Run Measured Ablations

For each tensor or tensor group you want to test, write a one-line
`tensor_types.txt` file that forces that tensor down to a low precision:

```text
blk.10.ffn_down.weight=q2_K
```

Build a temporary ablation GGUF:

```bash
llama-quantize \
  --imatrix cerebellum_imatrix.dat \
  --tensor-type-file tensor_types.txt \
  source-f16.gguf \
  ablation-temp.gguf \
  Q4_K_M
```

Run perplexity on `ablation-temp.gguf`, then record the result:

```json
{
  "baseline_ppl": 7.034,
  "tests": {
    "layer_10.mlp.down_proj": {
      "gguf_tensor": "blk.10.ffn_down.weight",
      "ppl": 7.091
    }
  }
}
```

The `tests` keys can use your own human-readable names. The allocator uses
`gguf_tensor` and `ppl`.

## Step 4: Allocate Tensor Types

Ask Cerebellum to classify the measured tensors and spend bits under a target
file size:

```bash
python -m osmosis.cerebellum \
  --ablation ablation_results.json \
  --source-gguf source-f16.gguf \
  --imatrix cerebellum_imatrix.dat \
  --budget-gb 12.0 \
  --output final_tensor_types.txt \
  -v
```

The command prints tensor classifications, a layer heatmap, placement notes,
and the exact `llama-quantize` command shape for the final build.

## Step 5: Build the Final GGUF

Use the emitted tensor type file with stock llama.cpp:

```bash
llama-quantize \
  --imatrix cerebellum_imatrix.dat \
  --tensor-type-file final_tensor_types.txt \
  source-f16.gguf \
  model-cerebellum.gguf \
  Q4_K_M
```

The output is a normal GGUF. Users do not need Cerebellum installed to run it.

## Proxy Allocation Path

Measured ablation is slower but more direct. For early exploration, you can use
weight-only sensitivity:

```bash
python -m osmosis.sensitivity_stream \
  --model <hf-model-id-or-local-safetensors-dir> \
  --output sensitivity_multi.json \
  -v

python -m osmosis.budget \
  --sensitivity sensitivity_multi.json \
  --source-gguf source-f16.gguf \
  --imatrix cerebellum_imatrix.dat \
  --budget-gb 12.0 \
  --output final_tensor_types.txt \
  -v
```

Label this as proxy-based if you publish the result.

## What To Keep

For a reproducible release, keep:

- source model name and source GGUF hash,
- imatrix file hash,
- ablation or sensitivity JSON,
- final `tensor_types.txt`,
- allocator command and budget settings,
- final GGUF hash,
- benchmark summary and detailed outputs,
- runtime flags used for all benchmark runs.

These files are release artifacts. They belong in public model repositories or
public benchmark directories once they are sanitized. The local scripts that
produced them do not need to be public.

For ablation-driven releases, upload the Cerebellum ablation output itself, not
only the final score. At minimum, keep the measured `ablation_results.json`, the
final `tensor_types.txt`, and the allocator command used to turn one into the
other.

In this repository, small release recipe files belong under `model-data/`.
Benchmark summaries and detailed answer files belong under `benchmarks/`.
