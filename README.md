# Cerebellum — Ablation-Informed Quantization for LLMs

Tensor-level precision allocation for GGUF quantization. Measures tensor sensitivity and assigns quantization types under a target size budget. Models for people who are poor.

## How It Works

1. **Ablate** — crush each tensor to Q2_K individually, measure the perplexity impact
2. **Classify** — label tensors by measured sensitivity and PPL delta
3. **Allocate** — high-sensitivity tensors get higher precision, low-sensitivity tensors stay at Q2_K, everything else fills in to meet the size budget
4. **Multi-pass promotion** — tensors climb quant levels (Q2_K → Q3_K → ... → Q8_0) across passes until the budget is exhausted

The result: smaller files with less quality loss than uniform quants of the same size. Sometimes demoting tensors actually *improves* the model.

## Released Models

All models available at [huggingface.co/deucebucket](https://huggingface.co/deucebucket). Full benchmark data (including per-question answers) in [`benchmarks/`](benchmarks/).

### Summary Table

| Model | Architecture | Size | HumanEval | ARC | HellaSwag | MMLU | PPL |
|-------|-------------|------|-----------|-----|-----------|------|-----|
| **Granite 4.1 30B** v2 | Dense | 13 GB | **82.3%** | 91.6% | 88.9% | 73.5% | 8.49 |
| **Qwen 3.6 27B** v4 | Dense | 12 GB | 81.1% | 96.8% | 92.2% | 82.5% | 7.03 |
| **Qwen 3.6 35B-A3B** v1 | SSM+MoE | 12 GB | 75.0% | 94.8% | 91.5% | 73.9% | — |
| **Gemma 4 26B-A4B** v6 | MoE (128 exp) | 11.7 GB | 72.0% | 95.6% | 84.7% | 71.2% | 12,054 |
| **Qwen 3 30B-A3B** v2 | MoE | — | 72.0% | — | — | — | — |
| **Gemma 4 E4B** v2 | Dense (PLE) | 4.2 GB | 68.3% | 85.7% | 75.3% | 58.4% | 52.20 |
| **Qwen 3 14B** v2 | Dense | — | 65.9% | — | — | — | — |
| **Granite 4.0-H-Small** v1 | Mamba2+MoE | 14.2 GB | — | 90.7% | 87.1% | 68.6% | 6.46 |
| **Qwen 3 32B** v2 | Dense | — | 45.1% | — | — | — | — |

All benchmarks measured locally on RTX 3090 with llama.cpp. Temperature=0, no thinking mode. HumanEval uses completions API with pre-filled think tokens.

**2026-05-03 Score Corrections:** Found and fixed bugs in the benchmark scripts. HumanEval had a fence-stripping bug that destroyed indentation (all models affected, scores were ~6-8 points too low). ARC had 19 questions misjudged due to numeric label handling. HellaSwag had 108 empty responses incorrectly counted as wrong answers. Only Qwen 3.6 27B v4 has been re-benchmarked so far. Other models will be updated as they get re-run. Full audit trail in [BENCHMARK_CORRECTIONS.md](osmosis-qwen36-27b/benchmark_results/BENCHMARK_CORRECTIONS.md).

### Granite 4.1 30B — Cerebellum v2 (13 GB)

Best code performance. Dense 30B model, 64 layers, GQA. Demoted 3 attention groups (attn_k, attn_q, attn_output) that were *hurting* the model at Q3_K_M. Saves 1 GB over uniform Q3_K_M with +1.4% PPL.

HF: [deucebucket/Granite-4.1-30B-Cerebellum-GGUF](https://huggingface.co/deucebucket/Granite-4.1-30B-Cerebellum-GGUF)

| Benchmark | Score |
|-----------|-------|
| HumanEval pass@1 | **82.3%** |
| ARC-Challenge | 91.6% |
| HellaSwag | 88.9% |
| MMLU | 73.5% |
| WikiText PPL | 8.4912 |

### Qwen 3.6 27B — Cerebellum v4 (12 GB)

Best overall knowledge scores. Dense transformer, 64 layers. 181 tensor overrides across 5 precision levels. Multi-pass promotion from full ablation sweep.

HF: [deucebucket/Qwen3.6-27B-Cerebellum-v4-GGUF](https://huggingface.co/deucebucket/Qwen3.6-27B-Cerebellum-v4-GGUF)

| Benchmark | Score | Notes |
|-----------|-------|-------|
| HumanEval pass@1 | **81.1%** | corrected (was 75.0%, script bug) |
| ARC-Challenge | **96.8%** | corrected (was 95.1%, label mismatch) |
| HellaSwag | **92.2%** | corrected (was 91.2%, empty response bug) |
| MMLU | **82.5%** | |
| MMLU-Redux | 76.6% | confirmed (was 77.1%, run variance) |
| WikiText PPL | **7.034** | |

Recommended sampling: temperature=0. Tested across the full benchmark suite, temp=0 outperforms temp=0.3 on all benchmarks (HumanEval 81.1% vs 78.7%, MC benchmarks within noise). The aggressive quantization leaves no room for randomness to find a better path.

Speed: 71 tok/s prompt, 36.5 tok/s generation (RTX 3090, full offload)

### Qwen 3.6 35B-A3B — Cerebellum v1 (12 GB)

Hybrid SSM (Mamba-2) + MoE with 256 experts. Only 3B active parameters per token. Same performance as the 27B dense model at fraction of the compute. 2.73 BPW average.

HF: [deucebucket/Qwen3.6-35B-A3B-Cerebellum-GGUF](https://huggingface.co/deucebucket/Qwen3.6-35B-A3B-Cerebellum-GGUF)

| Benchmark | Score |
|-----------|-------|
| HumanEval pass@1 | 75.0% |
| ARC-Challenge | 94.8% |
| HellaSwag | 91.5% |
| MMLU-Redux | 73.9% |

### Gemma 4 26B-A4B — Cerebellum v6 (11.7 GB)

MoE with 128 experts per layer, 4B active. Six internal iterations of ablation: group-level, per-layer, PLE protection, reverse ablation, and MoE router surgery. Router layer 8 demotion to Q8_0 was the final gain.

HF: [deucebucket/Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF](https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF)

| Benchmark | Score |
|-----------|-------|
| HumanEval pass@1 | 72.0% |
| ARC-Challenge | **95.6%** |
| HellaSwag | 84.7% |
| MMLU-Redux | 71.2% |
| WikiText PPL | 12,054 |

Version history: v1 (65.2%) → v2 (65.9%) → v3 (67.1%) → v4 (69.5%) → v5 (71.3%) → **v6 (72.0%)**

### Gemma 4 E4B — Cerebellum v2 (4.2 GB)

Smallest model. PLE-protected + 26-tensor ablation. v2 beats BF16 perplexity (52.20 vs 54.58) at 4.2 GB because the ablation found tensors that benefit from demotion.

HF: [deucebucket/Gemma-4-E4B-it-Cerebellum-v2-GGUF](https://huggingface.co/deucebucket/Gemma-4-E4B-it-Cerebellum-v2-GGUF)

| Benchmark | v2 (4.2 GB) | v1 (4.3 GB) | BF16 (15 GB) |
|-----------|-------------|-------------|--------------|
| WikiText PPL | **52.20** | 55.10 | 54.58 |
| HumanEval | **68.3%** | 65.9% | — |
| ARC-Challenge | 85.7% | 85.1% | — |
| HellaSwag | 75.3% | 75.5% | — |
| MMLU-Redux | 58.4% | 58.0% | — |

### Granite 4.0-H-Small — Cerebellum v1 (14.2 GB)

Hybrid Mamba-2 + Transformer MoE. First Cerebellum build for this architecture. Found that routed expert weights are sensitive while shared experts tolerate demotion (opposite of dense MoE patterns).

HF: [deucebucket/Granite-4.0-H-Small-Cerebellum-GGUF](https://huggingface.co/deucebucket/Granite-4.0-H-Small-Cerebellum-GGUF)

| Benchmark | Score |
|-----------|-------|
| ARC-Challenge | 90.7% |
| HellaSwag | 87.1% |
| MMLU-Redux | 68.6% |
| WikiText PPL | 6.4580 |

## Key Findings

- **Demoting tensors can improve the model.** Granite 4.1 attention K/Q/output showed lower PPL at Q2_K than Q3_K. Gemma E4B v2 beats BF16 PPL.
- **MoE fragility is in expert weights, not auxiliary signals.** Opposite of dense models where attention projections are most sensitive.
- **Hybrid SSM models have hard precision cliffs.** SSM parameters break below 4-bit. No gradual degradation, just NaN.
- **PLE tensors in Gemma 4 have a cliff between Q4_K and Q3_K.** Q3_K_M without PLE protection: PPL 104.74. With PLE Q5_K: PPL 55.10.
- **Cross-layer effects are ~86% additive.** Single-tensor ablation deltas predict multi-tensor outcomes with ~14% attenuation.
- **Same-layer interaction effects are strong.** Crushing two FFN tensors in the same layer: 87% regression (interaction ratio 0.13).

## Usage

### Ablation Sweep

```bash
python -m osmosis.cerebellum ablate \
    --base-gguf model-Q2_K.gguf \
    --tensors ablation_plan.json \
    --output ablation_results.json
```

### Budget Allocation

```bash
python -m osmosis.cerebellum allocate \
    --ablation ablation_results.json \
    --budget 12.0 \
    --output tensor_types.txt
```

### Build the GGUF

```bash
llama-quantize --imatrix imatrix.dat \
    --tensor-type @tensor_types.txt \
    model-f16.gguf model-cerebellum.gguf Q2_K
```

### Streaming Quantizer (any model size, constant RAM)

Processes one tensor at a time. Peak RAM is ~300 MB regardless of model size. 122B model on 4 GB RAM, no problem.

```bash
# Basic requantization
python tools/streaming_quantize.py model-Q4_K.gguf output-Q2_K.gguf --type q2_K

# Mixed precision with Cerebellum tensor map
python tools/streaming_quantize.py model.gguf output.gguf \
    --type q2_K --override-file tensor_types.txt

# Preserve original types, only override specific tensors
python tools/streaming_quantize.py model.gguf output.gguf \
    --type keep --override-file promote_attn_qkv.txt

# Inspect tensors
python tools/streaming_quantize.py model.gguf --info

# Dry run (calculate output size)
python tools/streaming_quantize.py model.gguf --dry-run --type q3_K
```

~50 MB/s with native libggml (auto-detected), falls back to pure Python without it.

### Imatrix Generation

Fast imatrix from weight statistics, ~60 seconds on CPU:

```bash
python -m osmosis.imatrix_stream \
    --model Qwen/Qwen3.6-27B \
    --output osmosis_imatrix.dat -v
```

## Architecture Support

- **Dense transformers** (Qwen 3.6, Granite 4.1, Qwen 3) — full ablation + allocation
- **MoE** (Gemma 4 26B-A4B, Qwen 3.6 35B-A3B, Qwen 3 30B-A3B) — expert-level ablation + router surgery
- **Hybrid SSM+MoE** (Granite 4.0-H-Small, Qwen 3.5 9B) — SSM-aware with hard precision floors
- **PLE models** (Gemma 4 E4B) — per-layer embedding protection

## Project Structure

```
tools/
└── streaming_quantize.py      # Streaming GGUF quantizer (constant RAM)

scripts/
├── benchmark_humaneval.py     # HumanEval eval with pre-fill trick
└── patch_thinking_default.py  # Patch GGUF chat template (thinking off)

benchmarks/                    # Full benchmark data organized by model
├── qwen36-27b/               # Qwen 3.6 27B results + detailed answers
├── qwen36-35b-a3b/           # Qwen 3.6 35B-A3B results
├── gemma4-e4b/               # Gemma 4 E4B results
├── gemma4-26b-a4b/           # Gemma 4 26B-A4B results
├── granite41-30b/            # Granite 4.1 30B results + detailed answers
├── granite4-h-small/         # Granite 4.0-H-Small results
├── qwen3-30b-a3b/           # Qwen 3 30B-A3B results
├── qwen3-32b/               # Qwen 3 32B results
└── qwen3-14b/               # Qwen 3 14B results

osmosis/
├── cerebellum.py             # Ablation-informed precision allocator
├── imatrix_stream.py         # Streaming imatrix generation
├── imatrix_gen.py            # Standard imatrix generation
└── imatrix_format.py         # llama.cpp imatrix binary format writer
```

## Test Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 3090 (24 GB) |
| CPU | AMD Ryzen 7 5800XT |
| RAM | 64 GB DDR4 |
| OS | Fedora Linux 43 (Atomic) |

## License

Apache 2.0
