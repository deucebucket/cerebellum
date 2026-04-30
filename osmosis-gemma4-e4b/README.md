---
license: gemma
library_name: gguf
base_model: google/gemma-4-E4B-it
base_model_relation: quantized
model_name: Gemma-4-E4B-it-Cerebellum-v1-GGUF
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
  - 3-bit
  - conversational
---

# Gemma 4 E4B — Cerebellum v1 GGUF (4.3 GB)

PLE-aware mixed-precision quantization of [google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it). **4.3 GB** file size, **55.10 perplexity** — smaller than Q4_K_M (5.0 GB) with better quality.

Every other Gemma 4 E4B quant below Q4_K_M on HuggingFace is broken. Q3_K_M without PLE protection gives PPL 104.74 — nearly 2x the BF16 baseline. This file fixes it by pinning Gemma 4's Per-Layer Embedding tensors to Q5_K, recovering full quality at 14% less size than Q4_K_M.

## Benchmarks

| Benchmark | Score | Questions |
|-----------|-------|-----------|
| **Perplexity** (WikiText-2, 2048 ctx) | **55.10** | — |
| **HumanEval** pass@1 | **65.9%** | 164 |
| **ARC-Challenge** | **85.1%** | 1,172 |
| **HellaSwag** | **75.5%** | 10,042 |
| **MMLU-Redux** | **58.0%** | 2,400 |

*All benchmarks measured directly on this file (Q3_K_M + PLE Q5_K).*

## The PLE Problem

Gemma 4 introduced Per-Layer Embeddings (PLE) — auxiliary residual signals injected at every decoder layer. Standard quantization treats them like regular tensors. They aren't. These 174 tensors have a hard precision cliff between Q4_K and Q3_K:

| Quant | Size | PPL (WikiText-2) | vs BF16 | Status |
|-------|------|-------------------|---------|--------|
| BF16 | 15 GB | 54.58 | baseline | |
| Q4_K_M | 5.0 GB | 55.74 | +2.1% | Fine |
| Q3_K_M + PLE Q6_K | 4.6 GB | 54.63 | +0.1% | Excellent |
| **Q3_K_M + PLE Q5_K** | **4.3 GB** | **55.10** | **+1.0%** | **This file** |
| Q3_K_M + PLE Q4_K | 3.9 GB | 56.52 | +3.6% | Good |
| Q3_K_M | 4.6 GB | 104.74 | +91.9% | Broken — PLE destroyed |
| Q2_K + PLE Q8_0 | 4.8 GB | 62.72 | +14.9% | Usable — non-PLE damage |
| Q2_K | 4.1 GB | 7296.76 | +13,268% | Destroyed |

The cliff is entirely caused by PLE tensors falling below their precision threshold. Pinning them to Q5_K eliminates the cliff at minimal size cost.

### PLE Protection Level Sweep

We tested four protection levels to find the minimum viable precision:

| PLE Level | Size | PPL | vs BF16 |
|-----------|------|-----|---------|
| Q4_K | 3.9 GB | 56.52 | +3.6% |
| **Q5_K** | **4.3 GB** | **55.10** | **+1.0%** |
| Q6_K | 4.6 GB | 54.63 | +0.1% |
| Q8_0 | 5.2 GB | 55.82 | +2.3% |

Q5_K is the sweet spot — half the overhead of Q8_0, better PPL than Q8_0 (55.10 vs 55.82), and 14% smaller than standard Q4_K_M.

## Why BF16 PPL Is 54.58

Gemma 4 is a multimodal model with a 262K vocabulary that includes vision and audio tokens. This inflates perplexity on text-only benchmarks compared to text-only models. Community reports of "broken quantization" at Q4_K_M were comparing against Gemma 3's lower baseline (~7-10 PPL), not Gemma 4's actual BF16 performance. Q4_K_M degradation is only 2.1% — quantization works fine when PLE tensors are handled correctly.

**Note:** The [Gemma 4 E4B base model](https://github.com/ggml-org/llama.cpp/issues/22407) shows BF16 PPL = 7.11 and Q4_K_M PPL = 23.06 without imatrix — the cliff hits one level higher. Our imatrix pushes the cliff from Q5→Q4 down to Q4→Q3, and explicit PLE pinning eliminates it entirely.

## What PLE Tensors Are

Per-Layer Embeddings are Gemma 4's mechanism for injecting a learned residual signal at each transformer layer:

- `per_layer_token_embd` — per-layer token embeddings (global, large)
- `per_layer_model_proj` — projection from PLE space to model space (global)
- `blk.{0-41}.inp_gate` — gating weight per layer
- `blk.{0-41}.proj` — projection weight per layer
- Plus associated norms and scales

174 tensors total. At Q5_K, they add ~0.3 GB overhead compared to leaving them at Q3_K_M.

## VRAM Requirements

| Context | VRAM |
|---------|------|
| 2K | ~5 GB |
| 8K | ~6 GB |
| 16K | ~7 GB |

Fits comfortably on a 6 GB GPU at moderate context.

## Usage

```bash
# llama.cpp
llama-server \
  --model Gemma-4-E4B-it-Cerebellum-v1.gguf \
  --n-gpu-layers 99 \
  --ctx-size 8192

# Ollama
echo 'FROM ./Gemma-4-E4B-it-Cerebellum-v1.gguf' > Modelfile
ollama create gemma4-e4b -f Modelfile
ollama run gemma4-e4b
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

# 3. Quantize with PLE overrides (Q5_K protection)
llama-quantize \
  --imatrix imatrix.dat \
  --tensor-type-file ple_overrides.txt \
  gemma4-e4b-bf16.gguf Gemma-4-E4B-it-Cerebellum-v1.gguf Q3_K_M
```

The `imatrix.dat` and `ple_overrides.txt` files are included in this repo.

## Files

| File | Size | Description |
|------|------|-------------|
| `Gemma-4-E4B-it-Cerebellum-v1.gguf` | 4.3 GB | The quantized model |
| `imatrix.dat` | 4.6 MB | Weight-sensitivity importance matrix |
| `ple_overrides.txt` | 4.5 KB | 174 tensor type overrides (PLE → Q5_K) |

## Imatrix Method

The importance matrix is computed from weight statistics rather than calibration data:

1. For each weight tensor, compute channel sensitivity: `L2_norm × max_abs × variance`
2. Write importance scores in llama.cpp's imatrix binary format
3. Feed to `llama-quantize --imatrix` for informed bit allocation

No calibration data. No GPU required. ~45 seconds on CPU.

## What's Next: Cerebellum v2

This release protects PLE tensors based on architectural knowledge — we knew they'd be fragile. Cerebellum v2 will run a full ablation sweep on the remaining ~546 non-PLE tensors to find additional sensitivity and further optimize precision allocation within the same size budget. The [Cerebellum tooling](https://github.com/deucebucket/cerebellum) automates this process.

## Model Details

- **Base model**: [google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it)
- **Architecture**: Dense transformer with PLE, 42 layers, 720 tensors
- **Quantization**: Q3_K_M base with 174 PLE tensors at Q5_K
- **Imatrix**: Weight-sensitivity (L2_norm × max_abs × variance), no calibration data
- **Vocabulary**: 262,144 tokens (text + vision + audio)
- **File format**: GGUF v3

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
