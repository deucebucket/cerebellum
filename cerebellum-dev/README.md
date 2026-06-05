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

## Current Focus

- Gemma 4 12B visible hill-climber run is the active live experiment.
- The run starts from an explicit Q4_K baseline, then tests each tensor across
  demotion and promotion levels.
- Unlike the earlier group-first experiments, this pass walks every quantizable
  tensor. That should produce deeper data, but final quality still needs full
  benchmark validation before claims are published.
- The expected proof chain is PPL drop, coherent chat smoke checks, final GGUF,
  then audited benchmarks.

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
