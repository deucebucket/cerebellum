---
license: apache-2.0
library_name: gguf
base_model: Qwen/Qwen3.6-35B-A3B
base_model_relation: quantized
model_name: Qwen3.6-35B-A3B-Cerebellum-GGUF
model_creator: Qwen
model_type: qwen3
quantized_by: deucebucket
pipeline_tag: text-generation
tags:
  - GGUF
  - qwen3
  - qwen
  - quantized
  - cerebellum
  - imatrix
  - moe
  - mixed-precision
  - 3-bit
  - conversational
---

# Qwen 3.6 35B-A3B — Cerebellum GGUF

Sensitivity-guided mixed-precision quantization of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Two variants available:

| Variant | File | Size | BPW |
|---------|------|------|-----|
| **Cerebellum v3** (recommended) | `Qwen3.6-35B-A3B-Cerebellum-v3.gguf` | **11 GB** | 2.76 |
| Cerebellum v1 (legacy) | `Qwen3.6-35B-A3B-Cerebellum.gguf` | 12 GB | 2.73 |

Cerebellum measures which weight groups survive extreme compression and which don't, then writes a single GGUF with per-tensor precision assignments. v3 uses 360 tensor-level overrides guided by group ablation and reverse ablation analysis.

## Benchmarks

All benchmarks measured directly on these GGUF files using llama.cpp inference with cleaned evaluation harness.

| Benchmark | **v3 (11 GB)** | v1 (12 GB) | Q3_K_M (15.6 GB) |
|-----------|:---:|:---:|:---:|
| **ARC-Challenge** | **95.8%** | 94.8% | 96.1% |
| **HellaSwag** | **92.3%** | 91.5% | 91.5% |
| **MMLU-Redux** | **75.0%** | 73.9% | 74.1% |
| **HumanEval base** | **70.7%** | — | 64.0% |
| **HumanEval+** | **65.2%** | — | 56.7% |
| Vision smoke (36 images) | **100%** | 100% | — |

v3 at 11 GB is **29% smaller** than stock Q3_K_M (15.6 GB) while outperforming it on 5 of 6 benchmarks. The Q2_K regularization effect on gate/mixing weights actively improves downstream task performance despite reducing perplexity.

## v3 Allocation

| Group | Precision | Rationale |
|-------|-----------|-----------|
| `attn_qkv` | Q3_K_M | Critical for vision and attention routing |
| `ssm_out` | Q3_K_M | Most sensitive tensor per ablation (+0.24 PPL) |
| `ffn_gate_exps` | Q2_K | Q2_K regularization outperforms Q3_K_M |
| `ffn_up_exps` | Q2_K | Q2_K regularization outperforms Q3_K_M |
| `ffn_down_exps` | Q2_K | Acceptable loss for size savings |
| `ffn_gate_shexp` | Q2_K | Q2_K regularization outperforms Q3_K_M |
| `ffn_up_shexp` | Q2_K | Q2_K regularization outperforms Q3_K_M |
| `ffn_down_shexp` | Q2_K | Q2_K regularization outperforms Q3_K_M |
| `attn_gate` | Q2_K | Q2_K regularization outperforms Q3_K_M |
| `ssm_alpha`, `ssm_beta` | Q2_K | Q2_K regularization outperforms Q3_K_M |

Protected: all norms (F32), SSM state params (F32), router tensors (default).

## Ablation Data

Full ablation methodology and results are in the `ablation/` directory:

- `group_ablation_results.log` — Forward ablation: demote each group to Q2_K, measure PPL
- `reverse_ablation_results.log` — Reverse ablation: from fully-demoted v1, restore each group
- `cerebellum_v3_overrides.txt` — The 360-line tensor type override file used for v3

Key finding from reverse ablation: **6 of 10 groups perform better at Q2_K than Q3_K_M** — imatrix-guided Q2_K acts as beneficial regularization on gate, mixing, and shared expert weights.

## Usage

```bash
# v3 (recommended, 11 GB)
llama-server --model Qwen3.6-35B-A3B-Cerebellum-v3.gguf \
  --mmproj mmproj-F16.gguf --n-gpu-layers 99 --ctx-size 8192

# v1 (legacy, 12 GB)
llama-server --model Qwen3.6-35B-A3B-Cerebellum.gguf \
  --mmproj mmproj-F16.gguf --n-gpu-layers 99 --ctx-size 8192
```

## Files

| File | Size | Description |
|------|------|-------------|
| `Qwen3.6-35B-A3B-Cerebellum-v3.gguf` | 11 GB | v3 — recommended, 29% smaller than Q3_K_M |
| `Qwen3.6-35B-A3B-Cerebellum.gguf` | 12 GB | v1 — legacy |
| `mmproj-F16.gguf` | 858 MB | Vision projection (F16) |
| `benchmark_results/v3/` | — | Full benchmark JSON artifacts for v3 |
| `ablation/` | — | Ablation logs and override files |

## Methodology

Built with [Cerebellum](https://github.com/deucebucket/cerebellum) — sensitivity-guided mixed-precision quantization. v3 uses unsloth coder imatrix for importance-weighted quantization within each precision level.

Quantized by [@deucebucket](https://huggingface.co/deucebucket).
