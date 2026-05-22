---
license: gemma
library_name: gguf
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
model_name: Gemma-4-26B-A4B-it-Cerebellum-v6.1-templatefix-GGUF
model_creator: google
model_type: gemma4
quantized_by: deucebucket
pipeline_tag: text-generation
tags:
  - GGUF
  - gemma4
  - gemma
  - google
  - quantized
  - cerebellum
  - imatrix
  - moe
  - 3-bit
  - templatefix
---

# Gemma 4 26B-A4B-it Cerebellum GGUF

This repository contains GGUF builds derived from
`google/gemma-4-26B-A4B-it`.

## 2026-05-22 Update

Added:

```text
gemma-4-26B-A4B-it-cerebellum-v6.1-templatefix.gguf
sha256: d24229facdef8360a7ffa8b37a50e1de636b9139a5eba0efe899828e45ae7989

gemma-4-26b-a4b-it.mmproj.gguf
sha256: b762c43119ebdc3e3c36d929d958e827fac35b03278dda9203f87131aee1f185
```

The v6.1 file keeps the v6 tensor allocation and updates GGUF/runtime-facing
metadata for Gemma 4 chat-template use. The update was tested with
`llama-server --jinja --reasoning auto` and request-level no-thinking controls.

Older files in this repository are retained for reproducibility.

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
  --model gemma-4-26B-A4B-it-cerebellum-v6.1-templatefix.gguf \
  --mmproj gemma-4-26b-a4b-it.mmproj.gguf \
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
creative_eval_20260522/regular_v6_1_templatefix_creative_summary.json
creative_eval_20260522/regular_v6_1_templatefix_creative_rerun_longcaps_summary.json
```

Non-coding tool-use files:

```text
agentic_eval_20260522/README.md
agentic_eval_20260522/regular_v6_1_noncoding_agentic_tools_strict_summary.json
```

Observed 2026-05-22 results from those artifacts:

| Area | Harness | Observed result |
|---|---|---|
| No-thinking output channel | six creative prompts | `reasoning_len=0` in recorded outputs |
| Template leakage markers | six creative prompts | no `<think>` marker or template marker recorded by checker |
| Creative long-cap rerun | four prompts rerun after initial length caps | four stop finishes in rerun summary |
| Non-coding tool workflow | three strict OpenAI-style tool tasks | `schedule_strict`, `release_notes_strict`, `creative_brief_strict` listed in `pass_cases` |

The non-coding tool harness used mock tools named `list_calendar`,
`create_calendar_hold`, `search_notes`, `save_note`, and `add_task`. It did not
test code editing.

## Historical Same-Repo Benchmark Artifacts

The following benchmark artifacts are from the earlier v6 line and the local
Q3_K_M baseline. They are included as historical same-project measurements, not
as new v6.1 measurements.

| Artifact set | ARC-Challenge | HellaSwag | MMLU-Redux | HumanEval note |
|---|---:|---:|---:|---|
| `q3km_baseline_*` | 95.2218 | 86.5664 | 73.6667 | `q3km_baseline_humaneval_results.json`: 62.2 pass@1 |
| `cerebellum_v6_*` | 95.5631 | 84.55 | 71.3333 | v6 HumanEval artifacts are retained but marked for audit in local notes |

For Gemma 4 HumanEval/EvalPlus, the local protocol now uses chat completions,
not raw completions:

```text
llama-server --jinja --reasoning auto
chat_template_kwargs: {"enable_thinking": false}
thinking_budget_tokens: 0
BENCH_WORKERS=1
```

## Files and Provenance

Main v6.1 GGUF:

```text
source base: google/gemma-4-26B-A4B-it
quantization family: mixed-precision GGUF
recipe lineage: Cerebellum v6 tensor allocation
base quant lineage: Q3_K_M with bartowski imatrix
```

Matching mmproj:

```text
gemma-4-26b-a4b-it.mmproj.gguf
```

## Notes

- The 2026-05-22 tests were run on local `llama-server`.
- The opencode coding-agent test is not used as a model-card result. In one
  internal White and Black project run, the model connected through the harness
  and ran a Godot test, then produced malformed edit-tool calls.
- The creative-writing checks are smoke tests plus mechanical checks, not a
  human preference benchmark.
- The non-coding tool checks use mocked tools and fixed task definitions.

## Credits

- Base model: Google Gemma Team, `google/gemma-4-26B-A4B-it`
- Imatrix source used in the v6 lineage: bartowski,
  `bartowski/google_gemma-4-26B-A4B-it-GGUF`
- GGUF/runtime: llama.cpp
- Method and quantization workflow: deucebucket/osmosis Cerebellum pipeline
- Local test artifacts: deucebucket Cerebellum workflow
