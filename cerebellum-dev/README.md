# Cerebellum Dev

Private factory notes, experiments, and automation live here. Do not publish
this tree to the public `origin` repo.

This directory intentionally keeps legacy Cerebellum and old Osmosis-era
artifacts. Do not delete historical logs, ablation traces, benchmark outputs,
or older scripts just because the current CLI has moved on. Some old data is
still useful for comparing whether the current hill-climber actually improves
quality or only improves PPL on one corpus.

Artifact cleanup starts with `ARTIFACT_INVENTORY.md`. That inventory categorizes
legacy model trees, raw ablation data, benchmark outputs, private scripts,
large binaries, public-risk files, and cleanup candidates. Nothing listed there
is approved for deletion without a verified backup and a separate cleanup plan.

## Current Focus (updated 2026-06-11)

- **Active build:** Heretic Qwen 3.6 35B from `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF`
  (plain BF16, 69.4 GB, non-MTP). Recipe = v3 transfer (360-entry override file reused verbatim
  from stock, no re-ablation). Full pipeline in `forensics_2026-06-11/RECIPE_heretic_qwen36_35b.md`.
- **Hill-climber mode (osmosis/hillstep.py exhaustive wiki-PPL-only pass) is DEPRECATED.**
  The Gemma 4 12B block-10 checkpoint achieved -35% wiki PPL but -14 pts HumanEval+. Wiki PPL
  alone is not a safe objective. See DEAD_PATHS.md §DP-1 for full evidence.
  Targeted hillstep as an optional post-scan add-on after group-first ablation remains valid.

## Key Reference Documents (added 2026-06-11)

**WINNING_METHOD.md** — canonical formula: group ablation (~23 probes) -> reverse/ladder
checks -> override file -> stock llama-quantize -> benchmark gates. Contains exact commands
and final numbers for all three proven instantiations:
- Gemma 4 26B v6/v6.1 (91 overrides, bartowski imatrix, router surgery blk.8 Q8_0)
- Qwen 3.6 27B v4 (23 sparse PPL probes, budget allocator, 181 overrides)
- Qwen 3.6 35B v3 (360-entry all-Q2_K expert demotion on Q3_K_M base)
Plus the heretic-transfer protocol (verbatim override map, same imatrix, no re-ablation;
verify non-MTP source first).

**DEAD_PATHS.md** — archived dead pathways. Every failed approach with what it was,
when it died, why, and evidence pointer. Includes hillstep exhaustive mode, MTP-preserved
heretic source, raw /v1/completions for chat models, harmful 205-entry Gemma imatrix,
wiki-only 9B imatrix, critical-layer over-demotion, unauditable Brainloop runs, and six
logging failures (LF-1 through LF-6) still ambiguous on disk.

**forensics_2026-06-11/BENCHMARK_ERRORS_CATALOG.md** — merged catalog of all 14
benchmark/test/logging setup errors found in session forensics. Includes all Codex-found
bugs plus Claude-session-found bugs: fence stripping, ARC numeric labels, HellaSwag
empty responses, cache contamination, 36% and 0% false lows, MMLU slot-context abort,
enable_thinking empties, EvalPlus null parser, wrong-filename quantize log, and unauditable
Brainloop runs.

**forensics_2026-06-11/** — raw forensic mining reports from 2026-06-11 session archaeology:
- `claude_sessions.md` — authoritative: full build evidence, benchmark incidents, hillstep data
- `other_clis_sessions.md` — Codex/OpenCode/Gemini session catalog (384 Codex sessions)
- `RECIPE_heretic_qwen36_35b.md` — step-by-step plan for current active build

## What Goes Here

- raw ablation traces and candidate logs
- devlogs and investigation notes
- private dashboard/control-plane specs
- benchmark debugging notes and detailed artifacts
- pipeline automation and scripts not safe for public release
- cross-model pattern summaries
- failed approaches that may be useful later

## What Does Not Go Public

- tensor-selection heuristics
- streaming quant internals
- raw candidate/event logs
- local paths, account names, tokens, or private machine details
- devlogs explaining why a method works
- unfinished automation or dashboard code

The public repo should carry model cards, benchmark summaries, release metadata,
safe recipes, and sanitized artifacts only.

## Current CLI Work

Recent private-dev branches added:

- resumable `cerebellum run/resume/watch/recover/cleanup/rollback`
- screenshot-safe `watch --public`
- locked layer map in private watch
- ETA with expected completion day/time
- public/private package manifests and public audit/export commands
- benchmark report, benchmark plan, benchmark manifest, benchmark audit
- dashboard model/artifact/benchmark/audit ingest APIs
- Dynamic GGUF comparison with tensor map reference support
- frontier benchmark adapter plan layer for MMLU-Pro, GPQA-Diamond, MMMLU,
  HLE no-tools, and LiveCodeBench v6
- CPU-offload pipeline planning for huge models such as GLM-5.1

## Benchmark Reality

We have not fully benched a finished model from the current hill-climber path
yet. Treat old benchmark rows as historical evidence, not final proof for this
new method. Publishable claims need:

- final GGUF hash and size
- benchmark summary JSON
- detailed JSONL/sample artifacts where applicable
- audit output for wrong answers, parser failures, empty responses, and EvalPlus
  completion quality
- runtime flags and server settings

## Legacy Notes

Old `osmosis-*` directories and older Cerebellum experiment folders are archived
evidence. Keep them unless there is an explicit cleanup plan and a verified
backup. The package directory is still named `osmosis/` during the rename, but
new user-facing docs and artifacts should use `Cerebellum`.
