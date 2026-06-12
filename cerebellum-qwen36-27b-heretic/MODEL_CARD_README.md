---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/LICENSE
library_name: gguf
base_model: llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF
base_model_relation: quantized
model_name: Qwen3.6-27B-Heretic-Cerebellum-GGUF
model_creator: Qwen
model_type: qwen3
quantized_by: deucebucket
pipeline_tag: image-text-to-text
tags:
  - GGUF
  - qwen3
  - qwen
  - quantized
  - cerebellum
  - imatrix
  - hybrid-ssm
  - mixed-precision
  - 2-bit
  - heretic
  - uncensored
  - abliterated
model-index:
- name: Qwen3.6-27B-Heretic-Cerebellum-GGUF
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
      value: 0.9693
    source:
      name: Local audited benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Qwen3.6-27B-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
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
      value: 0.9014
    source:
      name: Local audited benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Qwen3.6-27B-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
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
      value: 0.7621
    source:
      name: Local audited benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Qwen3.6-27B-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
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
      value: 0.8476
    source:
      name: Local audited benchmark run (RTX 3090, llama.cpp) — chat no-think harness
      url: https://huggingface.co/deucebucket/Qwen3.6-27B-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
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
      value: 6.925
    source:
      name: Local audited benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Qwen3.6-27B-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
---

# Qwen 3.6 27B Heretic — Cerebellum GGUF

Sensitivity-guided mixed-precision quantization of
[llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF),
which is itself a decensored variant of
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
produced by llmfan46 using [Heretic](https://github.com/p-e-w/heretic) v1.2.0.

All future Heretic versions of this build will live in this repository.
Version identifiers appear only in filenames, not in the repo name.

## Files

| File | Size | Description |
|------|------|-------------|
| `Qwen3.6-27B-Heretic-Cerebellum-v1-Q2_K_Mixed.gguf` | **12.87 GB** (12,866,587,680 bytes) | Cerebellum v4 recipe — recommended |
| `Qwen3.6-27B-mmproj-BF16.gguf` | ~888 MB (931,146,048 bytes) | Vision projector, passed through unmodified from llmfan46's repo |

The vision projector is required for multimodal (image/video) use.
It is identical to the file distributed by llmfan46 and is included here
for single-repo convenience only.

## Provenance

1. **Base architecture**: [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) — Qwen Team (Apache-2.0)
2. **Heretic variant**: [llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF) — llmfan46.
   The BF16 GGUF from that repository was used as the direct quantization source.
   llmfan46 applied Heretic v1.2.0 with the Magnitude-Preserving Orthogonal
   Ablation (MPOA) method, targeting `attn.o_proj`, `attn.out_proj`, and
   `mlp.down_proj`. Their reported result: **0.0021 KL divergence** from base,
   **6/100 refusals** vs 92/100 on the original model, MMLU accuracy 85.61%
   vs 86.65% on the original.
3. **Quantization**: Cerebellum v4 recipe transferred verbatim from the stock
   [deucebucket/Qwen3.6-27B-Cerebellum-v4-GGUF](https://huggingface.co/deucebucket/Qwen3.6-27B-Cerebellum-v4-GGUF)
   build — same 181-entry tensor-type override file, same coder imatrix
   (ncall=8). The heretic build is verified to land within 4,400 bytes of the
   stock v4 file (12,866,587,680 vs 12,866,583,200 bytes) — byte-class
   identical.

## Benchmarks

Benchmarks run on these GGUF files directly using llama.cpp on RTX 3090.
All numbers are audited; every failed answer was manually verified as a genuine
model error — audit reports are in `benchmark_results/AUDIT_*.md`.
Full per-question detail (summary JSON, samples JSONL, EvalPlus eval JSON,
adversarial audit reports) is in `benchmark_results/` in this repository.

### Heretic Cerebellum v1 (12.87 GB) vs baselines

| Benchmark | Heretic Cerebellum v1 (12.87 GB) | Stock Cerebellum v4 (12.87 GB) | Notes |
|-----------|:---:|:---:|---|
| Wiki PPL (ctx 2048, 32 chunks) | 6.925 ± 0.095 | 6.681 ± 0.091 | RTX 3090, identical invocation |
| ARC-Challenge | **96.93%** (1172 q) | 96.76% | 25-shot |
| HellaSwag | **90.14%** (10042 q) | 92.29% | 10-shot, stock re-measured same-night same-harness |
| MMLU-Redux | **76.21%** (2400 q) | 76.58% | 5-shot |
| HumanEval base (chat, no-think) | **89.63%** (164 problems) | 90.85% | pass@1, evalplus patched harness |
| HumanEval+ (chat, no-think) | **84.76%** | 87.20% | pass@1, evalplus patched harness |
| Vision smoke | **100%** (24/24) | — | basic image description |
| RealWorldQA | **78.0%** (n=50, ±2%) | — | single-question granularity ±2% per question |

Stock Cerebellum v4 is the same tensor allocation applied to the non-heretic base.

#### HumanEval methodology note

The May-published "81.10%" for stock v4 was measured with thinking mode enabled
and is **not comparable** to the numbers above. The publishable comparison is
the same-night, same-harness, no-thinking chat pair on the patched harness
(zero fabricated completions, verified): heretic **89.63 base / 84.76 plus**
vs stock v4 **90.85 base / 87.20 plus**.

For transparency, the raw-completions endpoint pair is also reported:
heretic 65.24% base / 60.98% plus vs stock v4 62.20% base / 56.71% plus.
Both raw-completions numbers were affected by a now-fixed harness timeout bug
(BE-15); the raw-completions endpoint under-serves this model family. Full
details are in `benchmark_results/`.

## Heretic Abliteration Details (from llmfan46)

The following parameters are as reported in llmfan46's model card and are
reproduced here for downstream reference.

| Parameter | Value |
|-----------|-------|
| direction_index | 30.38 |
| attn.out_proj.max_weight | 1.58 |
| attn.out_proj.max_weight_position | 38.93 |
| attn.out_proj.min_weight | 1.51 |
| attn.out_proj.min_weight_distance | 32.78 |
| mlp.down_proj.max_weight | 1.80 |
| mlp.down_proj.max_weight_position | 41.28 |
| mlp.down_proj.min_weight | 0.54 |
| mlp.down_proj.min_weight_distance | 43.66 |
| attn.o_proj.max_weight | 1.99 |
| attn.o_proj.max_weight_position | 48.06 |
| attn.o_proj.min_weight | 1.75 |
| attn.o_proj.min_weight_distance | 39.00 |

Targeted components: `attn.o_proj`, `attn.out_proj`, `mlp.down_proj`.

Tool: [Heretic](https://github.com/p-e-w/heretic) v1.2.0,
method: Magnitude-Preserving Orthogonal Ablation (MPOA)
([reference](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)).

## Cerebellum v4 Tensor Allocation

Same allocation as the stock build. Listed here for reference.
181 sacred promotions over a Q2_K base: 70 Q5_K / 41 Q6_K / 7 Q8_0 /
22 Q4_K / 19 Q3_K / 22 Q2_K (explicit).

| Group | Precision | Rationale |
|-------|-----------|-----------|
| SSM state parameters | F32 | Hard-fail below 4-bit — NaN with no gradual degradation |
| SSM `in_proj_a/b`, `A_log`, `dt_bias`, `conv1d`, `in_proj_z` | ≥ Q4 minimum | 4-bit floor enforced per hybrid SSM ablation results |
| Most-sensitive attention tensors | Q5_K / Q6_K / Q8_0 | Sacred-pinned per per-tensor PPL ablation |
| Norm tensors | F32 | Protected; standard practice |
| Bulk ffn / remaining attention | Q2_K | Base precision, imatrix-guided |

Protected: all norms (F32), SSM recurrent state (F32).

## Perplexity Note

Wiki PPL for the Heretic build (6.925 ± 0.095) is 0.244 higher than the stock
Cerebellum v4 (6.681 ± 0.091). The error bars do not overlap, indicating a
measurable distributional shift from the abliteration step. The delta is
consistent with what llmfan46 reported (MMLU drop of ~1 point) and is not
attributable to quantization differences — the two builds use identical
tensor allocations on a byte-class-identical file.

## Runtime — Casual Deployment

```bash
llama-server \
  --model Qwen3.6-27B-Heretic-Cerebellum-v1-Q2_K_Mixed.gguf \
  --mmproj Qwen3.6-27B-mmproj-BF16.gguf \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --jinja
```

`--jinja` is required for Qwen3.6. The `enable_thinking` chat-template flag
only takes effect when the Jinja template path is active; without it, the
model defaults to thinking mode on every request.

Non-thinking requests require an explicit flag at the API level:
```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

Qwen3.6 does not support the `/think` and `/nothink` soft-switch tokens
used by Qwen3.5. Thinking mode is on by default.

## Recommended Sampling Parameters

From the official Qwen3.6-27B documentation.

| Mode | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|------|-------------|-------|-------|-------|------------------|--------------------|
| Thinking — general | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 1.0 |
| Thinking — precise coding (WebDev) | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Non-thinking (instruct) | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

`presence_penalty` can be adjusted between 0 and 2 to reduce repetition loops;
higher values may occasionally cause language mixing.

## Reproduction

Standard Cerebellum recipe. The tensor-type override file and ablation logs
from the stock v4 build apply directly.

```bash
# 1. imatrix (constant ~300 MB RAM)
python -m osmosis.imatrix_stream \
    --model Qwen3.6-27B-uncensored-heretic-v2-BF16.gguf \
    --output imatrix.dat

# 2. quantize with stock llama-quantize
llama-quantize \
    --imatrix imatrix.dat \
    --tensor-type-file cerebellum_v4_overrides.txt \
    Qwen3.6-27B-uncensored-heretic-v2-BF16.gguf \
    Qwen3.6-27B-Heretic-Cerebellum-v1-Q2_K_Mixed.gguf \
    Q2_K
```

The imatrix used for this build was generated from the coder corpus
(ncall=8; same corpus as the stock Cerebellum v4 build).

The 181-entry tensor override file (`cerebellum_v4_overrides.txt`) is included
in this repository alongside the ablation logs.

## Benchmark Artifacts

Summary JSONs, per-question JSONL samples, EvalPlus eval JSON files, and
adversarial audit reports (`AUDIT_*.md`) are in `benchmark_results/` in this
repository per project policy.

## Credits

- Base model: [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) — Qwen Team
- Heretic variant and BF16 source: [llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF) — llmfan46
- Abliteration tool: [Heretic](https://github.com/p-e-w/heretic) v1.2.0 by p-e-w
- GGUF runtime: [llama.cpp](https://github.com/ggml-org/llama.cpp)
- Quantization method and workflow: [Cerebellum](https://github.com/deucebucket/cerebellum) — deucebucket
