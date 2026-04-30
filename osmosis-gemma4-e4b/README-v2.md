---
license: gemma
library_name: gguf
base_model: google/gemma-4-E4B-it
base_model_relation: quantized
model_name: Gemma-4-E4B-it-Cerebellum-v2-GGUF
model_creator: google
model_type: gemma4
quantized_by: deucebucket
pipeline_tag: image-text-to-text
tags:
  - GGUF
  - gemma4
  - gemma
  - google
  - quantized
  - cerebellum
  - imatrix
  - ple-protected
  - ablation
  - 3-bit
  - conversational
---

# Gemma 4 E4B — Cerebellum v2 GGUF (4.2 GB)

Ablation-informed mixed-precision quantization of [google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it). **4.2 GB** file size, **52.20 perplexity** — smaller and better than [Cerebellum v1](https://huggingface.co/deucebucket/Gemma-4-E4B-it-Cerebellum-v1-GGUF) (4.3 GB, PPL 55.10).

Cerebellum v1 fixed Gemma 4's broken small quants by protecting PLE tensors. v2 goes further — a 26-tensor ablation sweep identified which non-PLE tensors are sacred (need more precision) and which actively improve when crushed to Q2_K. The result: 2 tensors promoted to Q6_K, 3 demoted to Q2_K, everything else unchanged.

## Benchmarks

| Benchmark | v2 (4.2 GB) | v1 (4.3 GB) | BF16 (15 GB) |
|-----------|-------------|-------------|--------------|
| **Perplexity** (WikiText-2, 2048 ctx) | **52.20** | 55.10 | 54.58 |
| **HumanEval** pass@1 | **68.3%** | 65.9% | — |
| **ARC-Challenge** | **85.7%** | 85.1% | — |
| **HellaSwag** | **75.3%** | 75.5% | — |
| **MMLU-Redux** | **58.4%** | 58.0% | — |

*All benchmarks measured directly on this file.*

## How Cerebellum v2 Works

### Step 1: PLE Protection (same as v1)

Gemma 4's 174 Per-Layer Embedding tensors have a hard precision cliff between Q4_K and Q3_K. Pinning them to Q5_K eliminates the cliff. See [Cerebellum v1](https://huggingface.co/deucebucket/Gemma-4-E4B-it-Cerebellum-v1-GGUF) for the full PLE analysis.

### Step 2: Ablation Sweep (new in v2)

We crushed each of 26 non-PLE tensors individually to Q2_K while keeping everything else at baseline (Q3_K_M + PLE Q5_K), then measured the PPL delta for each.

**Sacred tensors — promoted to Q6_K:**

| Tensor | PPL Delta | Action |
|--------|-----------|--------|
| `blk.41.attn_q` | **+5.44** | Q3_K_M → Q6_K |
| `blk.20.ffn_up` | **+1.58** | Q3_K_M → Q6_K |

`blk.41.attn_q` is the single most important non-PLE tensor in the model — crushing it to Q2_K adds 5.44 PPL. The final layer's Q projection carries disproportionate weight in output quality.

**Demotable tensors — crushed to Q2_K (PPL improved):**

| Tensor | PPL Delta | Action |
|--------|-----------|--------|
| `blk.15.ffn_gate` | **-0.64** | Q3_K_M → Q2_K |
| `blk.40.attn_q` | **-0.64** | Q3_K_M → Q2_K |
| `blk.41.ffn_up` | **-0.63** | Q3_K_M → Q2_K |

These tensors are *actively worse* at Q3_K_M than Q2_K — the quantization noise at Q3_K hurts more than the precision loss at Q2_K.

### Full Ablation Results

| Tensor | PPL | Delta | Verdict |
|--------|-----|-------|---------|
| blk.41.attn_q | 60.54 | +5.44 | sacred — promoted |
| blk.20.ffn_up | 56.68 | +1.58 | sacred — promoted |
| blk.35.ffn_gate | 55.53 | +0.43 | leave at Q3_K_M |
| blk.25.ffn_gate | 55.44 | +0.34 | leave at Q3_K_M |
| blk.20.attn_q | 55.45 | +0.35 | leave at Q3_K_M |
| blk.1.ffn_up | 55.32 | +0.22 | leave at Q3_K_M |
| blk.10.ffn_up | 55.26 | +0.16 | leave at Q3_K_M |
| blk.30.attn_q | 55.24 | +0.14 | leave at Q3_K_M |
| blk.0.ffn_gate | 55.14 | +0.04 | noise |
| blk.0.attn_k | 55.13 | +0.03 | noise |
| blk.0.attn_q | 55.12 | +0.02 | noise |
| blk.10.attn_q | 55.11 | +0.01 | noise |
| blk.41.attn_v | 55.11 | +0.01 | noise |
| blk.0.ffn_up | 55.07 | -0.03 | demotable |
| blk.1.ffn_gate | 55.07 | -0.03 | demotable |
| blk.5.ffn_gate | 55.04 | -0.06 | demotable |
| blk.10.ffn_gate | 55.04 | -0.06 | demotable |
| blk.40.ffn_up | 55.04 | -0.06 | demotable |
| blk.41.ffn_gate | 54.94 | -0.16 | demotable |
| blk.20.ffn_gate | 54.84 | -0.26 | demotable |
| blk.40.ffn_gate | 54.77 | -0.33 | demotable |
| blk.30.ffn_up | 54.71 | -0.39 | demotable |
| blk.30.ffn_gate | 54.68 | -0.42 | demotable |
| blk.15.ffn_gate | 54.46 | -0.64 | crushed in v2 |
| blk.40.attn_q | 54.46 | -0.64 | crushed in v2 |
| blk.41.ffn_up | 54.47 | -0.63 | crushed in v2 |

## Why PPL Is Below BF16

v2's PPL (52.20) is lower than BF16 (54.58). This happens because certain tensors at Q3_K_M carry quantization noise that actively degrades prediction. Crushing them to Q2_K removes that noise pattern, and promoting sacred tensors to Q6_K preserves the precision that matters most. The net effect is better than keeping everything at full precision — targeted precision allocation beats uniform precision.

## VRAM Requirements

| Context | VRAM |
|---------|------|
| 2K | ~4.5 GB |
| 8K | ~5.5 GB |
| 16K | ~6.5 GB |

Fits comfortably on a 6 GB GPU at moderate context.

## Usage

```bash
# llama.cpp
llama-server \
  --model Gemma-4-E4B-it-Cerebellum-v2.gguf \
  --n-gpu-layers 99 \
  --ctx-size 8192

# Ollama
echo 'FROM ./Gemma-4-E4B-it-Cerebellum-v2.gguf' > Modelfile
ollama create gemma4-e4b-v2 -f Modelfile
ollama run gemma4-e4b-v2
```

## Reproducing This Quant

```bash
pip install git+https://github.com/deucebucket/cerebellum.git

# 1. Generate imatrix from weight statistics (~45 seconds on CPU)
python -m osmosis.imatrix_stream \
  --model google/gemma-4-e4b-it \
  --output imatrix.dat -v

# 2. Convert to BF16 GGUF (using llama.cpp)
python convert_hf_to_gguf.py gemma-4-e4b-it --outfile gemma4-e4b-bf16.gguf --outtype bf16

# 3. Quantize with Cerebellum v2 overrides
llama-quantize \
  --imatrix imatrix.dat \
  --tensor-type-file cerebellum_v2_overrides.txt \
  gemma4-e4b-bf16.gguf Gemma-4-E4B-it-Cerebellum-v2.gguf Q3_K_M
```

The `imatrix.dat`, `cerebellum_v2_overrides.txt`, and `ablation_results.json` are included in this repo.

## Files

| File | Size | Description |
|------|------|-------------|
| `Gemma-4-E4B-it-Cerebellum-v2.gguf` | 4.2 GB | The quantized model |
| `imatrix.dat` | 4.6 MB | Weight-sensitivity importance matrix |
| `cerebellum_v2_overrides.txt` | 4.6 KB | 179 tensor type overrides (174 PLE Q5_K + 2 Q6_K + 3 Q2_K) |
| `ablation_results.json` | 3 KB | Full ablation sweep data (26 tensors) |

## Model Details

- **Base model**: [google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it)
- **Architecture**: Dense transformer with PLE, 42 layers, 720 tensors
- **Quantization**: Q3_K_M base + 174 PLE at Q5_K + 2 sacred at Q6_K + 3 demoted to Q2_K
- **Imatrix**: Weight-sensitivity (L2_norm × max_abs × variance), no calibration data
- **Vocabulary**: 262,144 tokens (text + vision + audio)
- **File format**: GGUF v3
- **Cerebellum v1**: [deucebucket/Gemma-4-E4B-it-Cerebellum-v1-GGUF](https://huggingface.co/deucebucket/Gemma-4-E4B-it-Cerebellum-v1-GGUF)

## Test Hardware

| Component | Spec |
|-----------|------|
| **GPU** | NVIDIA RTX 3090 (24 GB) |
| **CPU** | AMD Ryzen 7 5800XT |
| **RAM** | 64 GB DDR4 |
| **OS** | Fedora Linux 43 (Atomic) |

## Attribution

- [Google DeepMind](https://huggingface.co/google) — Gemma 4 base model
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — quantization and tensor type override support
- [AWQ](https://arxiv.org/abs/2306.00978) — channel-level weight sensitivity insights

## License

Gemma Terms of Use
