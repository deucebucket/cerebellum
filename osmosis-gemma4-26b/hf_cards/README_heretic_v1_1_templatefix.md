---
license: gemma
library_name: gguf
base_model: coder3101/gemma-4-26B-A4B-it-heretic
base_model_relation: quantized
model_name: Gemma-4-26B-A4B-it-Heretic-Cerebellum-v1.1-templatefix-GGUF
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
  - moe
  - 3-bit
  - templatefix
---

# Gemma 4 26B-A4B-it Heretic Cerebellum GGUF

This repository contains GGUF builds derived from
`coder3101/gemma-4-26B-A4B-it-heretic`.

## 2026-05-22 Update

Added:

```text
gemma-4-26B-A4B-it-heretic-cerebellum-v1.1-templatefix.gguf
sha256: 103f973317a0daa2d59f94559c64ae7925257606b8c105c9dbdc8996a86310b1

gemma-4-26B-A4B-it-heretic.mmproj-f16.gguf
sha256: ed0e132c0ec1c97437e7eef555f654fd17ee0c090fbd1ffeea54e30402680476
```

The v1.1 file keeps the existing Heretic Cerebellum tensor allocation and
updates GGUF/runtime-facing metadata for Gemma 4 chat-template use. Older v1
files remain in the repository for reproducibility.

## Tested Runtime

Runtime used for the 2026-05-22 templatefix checks:

```text
llama.cpp fork: https://github.com/deucebucket/llama.cpp
branch: cerebellum/gemma4-runtime-fixes
fork commit: ded491334 fix: harden Gemma 4 server budgets
base build: b8930-59fa0b455
```

Server shape used locally:

```bash
llama-server \
  --model gemma-4-26B-A4B-it-heretic-cerebellum-v1.1-templatefix.gguf \
  --mmproj gemma-4-26B-A4B-it-heretic.mmproj-f16.gguf \
  --n-gpu-layers 99 \
  --ctx-size 65536 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja \
  --reasoning auto \
  --media-path /tmp/
```

Normal no-thinking requests used:

```json
{
  "chat_template_kwargs": {"enable_thinking": false},
  "thinking_budget_tokens": 0
}
```

Bounded-thinking smoke requests used `thinking_budget_tokens: 128`.

## 2026-05-22 Templatefix Test Artifacts

Creative-writing smoke files:

```text
creative_eval_20260522/heretic_v1_1_templatefix_creative_summary.json
```

Non-coding tool-use files:

```text
agentic_eval_20260522/README.md
agentic_eval_20260522/heretic_v1_1_noncoding_agentic_tools_adjudicated_summary.json
agentic_eval_20260522/heretic_v1_1_noncoding_agentic_tools_strict_retry_summary.json
```

Observed 2026-05-22 results from those artifacts:

| Area | Harness | Observed result |
|---|---|---|
| No-thinking output channel | six creative prompts | `reasoning_len=0` in recorded outputs |
| Template leakage markers | six creative prompts | no `<think>` marker or template marker recorded by checker |
| Creative prompt finishes | six creative prompts | six stop finishes in summary |
| First non-coding tool run | three OpenAI-style tool tasks | required tools called; warnings recorded for date preservation and repeated `search_notes` calls |
| Strict non-coding retry | three stricter OpenAI-style tool tasks | `schedule_strict`, `release_notes_strict`, `creative_brief_strict` listed in `pass_cases` |

The non-coding tool harness used mock tools named `list_calendar`,
`create_calendar_hold`, `search_notes`, `save_note`, and `add_task`. It did not
test code editing.

## Historical Same-Repo Benchmark Artifacts

The following artifacts are from the earlier Heretic v1 run on 2026-05-18.
They are included as same-repository measurements for the previous v1 file, not
as new v1.1 benchmark measurements.

| Artifact | Observed result |
|---|---:|
| `benchmark_results/gemma4_26b_heretic_cerebellum_v1_arc_results.json` | ARC-Challenge 95.48 |
| `benchmark_results/gemma4_26b_heretic_cerebellum_v1_hellaswag_results.json` | HellaSwag 83.49 |
| `benchmark_results/gemma4_26b_heretic_cerebellum_v1_mmlu_redux_results.json` | MMLU-Redux 71.42 |
| `benchmark_results/gemma4_26b_heretic_cerebellum_v1_evalplus_chat_results.json` | HumanEval base 92.07, HumanEval+ 89.63 |
| `benchmark_results/gemma4_26b_heretic_cerebellum_v1_vision_results.json` | local vision smoke 6/6 |

For Gemma 4 HumanEval/EvalPlus, the recorded fresh run used chat completions:

```text
llama-server --jinja --reasoning auto
chat_template_kwargs: {"enable_thinking": false}
thinking_budget_tokens: 0
BENCH_WORKERS=1
BENCH_MAX_TOKENS=768
```

The local audit notes for that run recorded:

```text
0 prompt echoes
0 repeated target function definitions
0 pass-only outputs
2 syntax failures
```

## Files and Provenance

Main v1.1 GGUF:

```text
source base: coder3101/gemma-4-26B-A4B-it-heretic
quantization family: mixed-precision GGUF
recipe lineage: Cerebellum v6 tensor allocation transferred to matching Heretic tensor layout
```

Matching mmproj:

```text
gemma-4-26B-A4B-it-heretic.mmproj-f16.gguf
```

## Notes

- The 2026-05-22 tests were run on local `llama-server`.
- The opencode coding-agent test is not used as a model-card result. In one
  internal White and Black project run, the model connected through the harness
  and ran a Godot test, then produced malformed edit-tool calls.
- The creative-writing checks are smoke tests plus mechanical checks, not a
  human preference benchmark.
- The non-coding tool checks use mocked tools and fixed task definitions.
- The first non-coding tool run is preserved because it records useful warning
  cases; the strict retry summary is preserved separately.

## Credits

- Source model: `coder3101/gemma-4-26B-A4B-it-heretic`
- Original Gemma family: Google Gemma Team
- GGUF/runtime: llama.cpp
- Method and quantization workflow: deucebucket/osmosis Cerebellum pipeline
- Local test artifacts: deucebucket Cerebellum workflow
