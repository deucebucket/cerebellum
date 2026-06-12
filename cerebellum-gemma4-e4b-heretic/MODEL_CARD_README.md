---
license: apache-2.0
license_link: https://huggingface.co/google/gemma-4-E4B-it/blob/main/LICENSE
library_name: gguf
base_model: coder3101/gemma-4-E4B-it-heretic
base_model_relation: quantized
model_name: Gemma-4-E4B-it-Heretic-Cerebellum-GGUF
model_creator: Google
model_type: gemma4
quantized_by: deucebucket
pipeline_tag: text-generation
tags:
  - GGUF
  - gemma4
  - gemma
  - quantized
  - cerebellum
  - imatrix
  - mixed-precision
  - 3-bit
  - ple
  - heretic
  - uncensored
  - abliterated
model-index:
- name: Gemma-4-E4B-it-Heretic-Cerebellum-GGUF
  results:
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: AI2 Reasoning Challenge (25-Shot)
      type: ai2_arc
      config: ARC-Challenge
      split: test
      args:
        num_few_shot: 25
    metrics:
    - name: normalized accuracy
      type: acc_norm
      value: 0.8737
    source:
      name: Local audited benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Gemma-4-E4B-it-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: HellaSwag (10-Shot)
      type: hellaswag
      split: validation
      args:
        num_few_shot: 10
    metrics:
    - name: accuracy
      type: acc
      value: 0.7498
    source:
      name: Local audited benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Gemma-4-E4B-it-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: MMLU-Redux (5-Shot)
      type: cais/mmlu
      config: all
      split: test
      args:
        num_few_shot: 5
    metrics:
    - name: accuracy
      type: acc
      value: 0.5863
    source:
      name: Local audited benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Gemma-4-E4B-it-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: HumanEval+ (pass@1)
      type: openai_humaneval
      split: test
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.6524
    source:
      name: Local audited benchmark run (RTX 3090, llama.cpp) — chat no-think harness, patched evalplus
      url: https://huggingface.co/deucebucket/Gemma-4-E4B-it-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: WikiText-2 Perplexity
      type: wikitext
      config: wikitext-2-raw-v1
      split: test
    metrics:
    - name: perplexity
      type: perplexity
      value: 50.61
    source:
      name: Local audited benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Gemma-4-E4B-it-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
---

# Gemma 4 E4B Heretic — Cerebellum GGUF

Sensitivity-guided mixed-precision quantization of
[coder3101/gemma-4-E4B-it-heretic](https://huggingface.co/coder3101/gemma-4-E4B-it-heretic),
which is itself a decensored variant of
[google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
produced by coder3101 using [Heretic](https://github.com/p-e-w/heretic) v1.2.0.

All future Heretic versions of this build will live in this repository.
Version identifiers appear only in filenames, not in the repo name.

## Files

| File | Size | Description |
|------|------|-------------|
| `Gemma-4-E4B-it-Heretic-Cerebellum-v1-Q3_K_M.gguf` | **4.2 GiB** (~4.51 GB, 4,498,725,440 bytes) | Cerebellum v2 recipe — recommended |

No vision mmproj is included. Stock Cerebellum v2 ships none either — Gemma 4 E4B's vision projector
is not publicly available in a distributable form.

## Provenance

1. **Base architecture**: [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) — Google DeepMind (Apache-2.0)
2. **Heretic variant**: [coder3101/gemma-4-E4B-it-heretic](https://huggingface.co/coder3101/gemma-4-E4B-it-heretic) — coder3101.
   The safetensors from that repository were converted to F16 GGUF and used
   as the direct quantization source (single quantization step from F16).
   coder3101 applied Heretic v1.2.0 with the Arbitrary-Rank Ablation (ARA)
   method (with row-norm preservation), targeting layers 21–42.
   Their reported result: **0.0058 KL divergence** from base,
   **3/100 refusals** vs 99/100 on the original model.
3. **Quantization**: Cerebellum v2 recipe transferred verbatim from the stock
   [deucebucket/Gemma-4-E4B-it-Cerebellum-v2-GGUF](https://huggingface.co/deucebucket/Gemma-4-E4B-it-Cerebellum-v2-GGUF)
   build — same 179-entry tensor-type override file, same imatrix.
   The heretic build lands within 384 bytes of the stock v2 file
   (4,498,725,440 vs 4,498,725,056 bytes) — byte-class identical.

## Benchmarks

Benchmarks run on these GGUF files directly using llama.cpp on RTX 3090.
All numbers are audited; every failed answer was manually verified as a genuine
model error — audit reports are in `benchmark_results/AUDIT_e4b.md`.
Full per-question detail (summary JSON, samples JSONL, EvalPlus eval JSON,
adversarial audit reports) is in `benchmark_results/` in this repository.

### Heretic Cerebellum v1 (4.2 GiB) vs baselines

| Benchmark | Heretic Cerebellum v1 (4.2 GiB) | Stock Cerebellum v2 (4.2 GiB) | Notes |
|-----------|:---:|:---:|---|
| Wiki PPL (ctx 2048, 32 chunks) | **50.61 ± 1.23** | 51.35 ± 1.26 | RTX 3090, identical invocation — see note |
| ARC-Challenge | **87.37%** (1172 q) | 85.7% | 25-shot |
| HellaSwag | 74.98% (10042 q) | 75.3% | 10-shot |
| MMLU-Redux | **58.63%** (2400 q) | 58.4% | 5-shot — see footnote |
| HumanEval base (chat, no-think) | **70.12%** (164 q) | 68.3% | pass@1, patched evalplus harness |
| HumanEval+ (chat, no-think) | **65.24%** | not published | pass@1, patched evalplus harness |

Stock Cerebellum v2 is the same tensor allocation applied to the non-heretic base.

#### Perplexity measurement note

Stock v2's published card number of 52.20 was measured under a different protocol.
The 51.35 figure above is a same-night, same-invocation pair run alongside the
heretic build (ctx 2048, 32 chunks, RTX 3090) and is the directly comparable
baseline. The heretic build's 50.61 ± 1.23 is 0.74 points lower than that
paired measurement; error bars overlap at the edges, making this a marginal
improvement consistent with the abliteration applying a slight distributional
shift rather than introducing noise.

#### MMLU-Redux footnote

The audit (BE-16) identified 12 truly identical duplicate entries in the MMLU-Redux
dataset resulting from a subject-boundary edge case in the benchmark runner.
These entries inflate the reported score by ≤0.04pp (58.6654% → 58.6250% without
duplicates), which is below the 0.1pp non-blocking threshold. The reported 58.63%
is rounded from 58.625% and is trustworthy within standard rounding tolerance.

#### HumanEval methodology note

Zero fabricated completions. One model-authored stub (`HumanEval/79`,
`decimal_to_binary`) — a literal `pass` placeholder — was correctly scored as
fail and does not inflate pass@1. Stock v2's published 68.3% base was measured
with the upstream evalplus harness; HumanEval+ was not published for stock v2.
The heretic numbers above use the patched no-think chat harness (same harness
used for all Cerebellum builds since the harness audit, May 2026).

## Heretic Abliteration Details (from coder3101)

The following parameters are as reported in coder3101's model card and are
reproduced here for downstream reference.

| Parameter | Value |
|-----------|-------|
| Method | Arbitrary-Rank Ablation (ARA) with row-norm preservation |
| Targeted layers | 21–42 |
| preserve_good_behavior_weight | 0.3873 |
| steer_bad_behavior_weight | 0.0003 |
| overcorrect_relative_weight | 0.8555 |
| neighbor_count | 14 |

Tool: [Heretic](https://github.com/p-e-w/heretic) v1.2.0,
method: Arbitrary-Rank Ablation (ARA) with row-norm preservation.

## Cerebellum v2 Tensor Allocation

Same allocation as the stock build. Listed here for reference.
179 overrides over a Q3_K_M base: 174 PLE tensors pinned to Q5_K,
2 sacred promotions to Q6_K, 3 demotions to Q2_K.

| Group | Precision | Rationale |
|-------|-----------|-----------|
| Per-layer embedding (PLE) tensors × 174 | Q5_K | Without PLE protection: PPL ~104 at Q3_K_M; with PLE@Q5_K: PPL ~55. Q4_K→Q3_K cliff is catastrophic for this architecture |
| 2 sacred attention tensors | Q6_K | Highest per-tensor PPL sensitivity per ablation |
| 3 ffn tensors | Q2_K | Reverse ablation confirmed Q2_K neutral or beneficial at these positions |
| Norm tensors | F32 | Protected; standard practice |
| Bulk ffn / remaining attention | Q3_K_M | Base precision, imatrix-guided |

The PLE protection is the load-bearing element of this recipe. Gemma 4's
per-layer embedding tensors degrade catastrophically at Q3_K_M — the
Q4_K→Q3_K transition produces a PPL cliff rather than a gradual slope, going
from ~55 to ~104 without PLE pins. This finding is documented in the stock v2
ablation logs.

## Runtime — Casual Deployment

```bash
llama-server \
  --model Gemma-4-E4B-it-Heretic-Cerebellum-v1-Q3_K_M.gguf \
  --n-gpu-layers 99 \
  --ctx-size 8192
```

Fits comfortably in 6 GB VRAM at 8K context on an RTX 3060/3070/3080 class card.

## Reproduction

Standard Cerebellum recipe. The tensor-type override file and ablation logs
from the stock v2 build apply directly.

```bash
# 1. imatrix (constant ~300 MB RAM)
python -m osmosis.imatrix_stream \
    --model gemma-4-E4B-it-heretic-f16.gguf \
    --output imatrix.dat

# 2. quantize with stock llama-quantize
llama-quantize \
    --imatrix imatrix.dat \
    --tensor-type-file cerebellum_v2_overrides.txt \
    gemma-4-E4B-it-heretic-f16.gguf \
    Gemma-4-E4B-it-Heretic-Cerebellum-v1-Q3_K_M.gguf \
    Q3_K_M
```

The 179-entry tensor override file (`cerebellum_v2_overrides.txt`) is included
in this repository alongside the ablation logs.

## Benchmark Artifacts

Summary JSONs, per-question JSONL samples, EvalPlus eval JSON files, and
adversarial audit reports (`AUDIT_e4b.md`) are in `benchmark_results/` in this
repository per project policy.

## Credits

- Base model: [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) — Google DeepMind
- Heretic variant: [coder3101/gemma-4-E4B-it-heretic](https://huggingface.co/coder3101/gemma-4-E4B-it-heretic) — coder3101
- Abliteration tool: [Heretic](https://github.com/p-e-w/heretic) v1.2.0 by p-e-w
- GGUF runtime: [llama.cpp](https://github.com/ggml-org/llama.cpp)
- Quantization method and workflow: [Cerebellum](https://github.com/deucebucket/cerebellum) — deucebucket
