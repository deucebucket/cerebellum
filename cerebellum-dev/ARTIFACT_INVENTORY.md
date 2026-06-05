# Cerebellum Artifact Inventory

Generated from the local Cerebellum workspace on 2026-06-05.

This is a preservation-first inventory. Nothing in this file is approval to
delete data. Cleanup means categorize, move to safer storage, deduplicate, or
exclude from public release after a verified backup exists.

## Policy

- Keep legacy Cerebellum and Osmosis-era evidence unless a separate cleanup plan
  identifies a backed-up copy and a clear reason to remove the local file.
- Treat raw ablation data, tensor maps, devlogs, dashboard state, scripts, local
  paths, benchmark details, and selection notes as private-dev material.
- Public `origin` should only receive sanitized model cards, benchmark
  summaries, safe docs, release metadata, and reviewed public assets.
- Large binaries should be tracked by location, hash, model/source, and purpose
  before moving or deleting.

## Storage Categories

| Category | Contents | Default Action |
| --- | --- | --- |
| `archive/legacy-models` | Old model experiment trees, model cards, logs, tensor maps, benchmark outputs. | Keep, summarize, and back up. |
| `archive/benchmarks` | Detailed JSONL, EvalPlus outputs, MCQ detailed rows, benchmark logs. | Keep; needed for audit and score correction. |
| `archive/ablation` | Raw PPL logs, candidate JSON, tensor overrides, final type maps. | Keep private; useful for method comparison. |
| `archive/devnotes` | Devlogs, issue drafts, specs, research notes. | Keep private; sanitize only selected high-level summaries for public. |
| `archive/binaries` | GGUFs, imatrix GGUF/DAT, large generated files. | Keep until hashed and provenance is recorded. |
| `scratch/cache` | `__pycache__`, local caches, temporary generated outputs. | Cleanup candidate after confirming no experiment state depends on it. |
| `public-candidates` | Model-card-ready summaries, release notes, public benchmark summaries. | Audit with `cerebellum public-audit` before public use. |

## Coarse Workspace Summary

Top-level legacy/model directories found:

- `cerebellum-dev`
- `cerebellum-gemma4-codex`
- `cerebellum-granite41-30b`
- `cerebellum-qwen35-9b`
- `osmosis-gemma4-26b`
- `osmosis-gemma4-e2b`
- `osmosis-gemma4-e4b`
- `osmosis-granite4-h-small`
- `osmosis-granite41-30b`
- `osmosis-qwen3-14b`
- `osmosis-qwen3-30b`
- `osmosis-qwen3-32b`
- `osmosis-qwen35-122b`
- `osmosis-qwen35-9b`
- `osmosis-qwen36-27b`
- `osmosis-qwen36-35b`

Approximate sizes from `du -sh`:

| Path | Size | Initial Category |
| --- | ---: | --- |
| `osmosis-gemma4-26b` | 24G | `archive/legacy-models`, `archive/binaries`, `archive/benchmarks` |
| `osmosis-gemma4-e2b` | 4.4G | `archive/legacy-models`, `archive/binaries`, `archive/benchmarks` |
| `cerebellum-dev` | 1.4G | `archive/devnotes`, `archive/ablation`, private factory |
| `osmosis-qwen3-30b` | 1.2G | `archive/legacy-models`, `archive/benchmarks` |
| `cerebellum-qwen35-9b` | 574M | `archive/legacy-models`, `archive/ablation` |
| `osmosis-granite41-30b` | 497M | `archive/legacy-models` |
| `osmosis-qwen3-32b` | 494M | `archive/legacy-models` |
| `osmosis-qwen35-122b` | 484M | `archive/legacy-models`, `archive/binaries` |
| `cerebellum-granite41-30b` | 399M | `archive/legacy-models`, `archive/benchmarks` |
| `cerebellum-gemma4-codex` | 306M | `archive/legacy-models`, `archive/benchmarks` |
| `osmosis-qwen36-27b` | 186M | `archive/legacy-models`, `archive/benchmarks`, `archive/ablation` |
| `osmosis-gemma4-e4b` | 43M | `archive/legacy-models`, `archive/benchmarks` |
| `osmosis-qwen36-35b` | 35M | `archive/legacy-models`, `archive/benchmarks` |
| `osmosis-granite4-h-small` | 11M | `archive/legacy-models`, `archive/benchmarks` |
| `osmosis-qwen35-9b` | 9.6M | `archive/legacy-models`, `archive/benchmarks` |
| `checkpoints` | 3.4M | active/local checkpoint material |
| `db` | 200K | local dashboard/control-plane state |
| `benchmark_results` | 80K | root benchmark artifact bucket |
| `research` | 20K | `archive/devnotes` |
| `spaces` | 12K | public/release candidate after audit |
| `taildrop` | 536K | local transfer/drop material, inspect before cleanup |

## Miscellaneous Root Inventory

| Category | Count | Size | Retention | Risk | Suggested Storage |
| --- | ---: | ---: | --- | --- | --- |
| Root `benchmark_results/` | 2 files | 80K | High if audited/published | Public: medium | Public artifact only after audit; otherwise private dev. |
| `checkpoints/` | 81 files | 3.4M | Medium-high | Public: medium; includes local paths | Private dev or local archive. |
| `db/` | 5 files | 200K | High for local analysis | Public: high | Private dev only. |
| `research/` | 4 files | 20K | Medium-high | Public: medium if unpublished | Private until polished. |
| `spaces/` | 3 files, 1 tracked | 12K | Medium | Public: medium, deployment-specific | Separate Space repo or private staging. |
| `taildrop/` | 1 image | 536K | Low/unknown | Public/private: high until identified | Personal archive or later delete candidate. |
| Root logs/watch files | 4 files | 28K | Low | Public/private: medium | Disposable local logs after attribution. |
| Root images | 3 PNG | 3.8M | High if branding | Public: low | Public docs/assets if intended. |
| Caches | `.cache`, `.pytest_cache`, `.ruff_cache`, `unsloth_compiled_cache` | about 2.7M visible | Low | Public: noisy/high | Ignore/local disposable. |
| `.opencode/` | 3 visible untracked, 3439 total | 57M | Medium for local agent setup | Public: high | Private config/cache, never `origin`. |
| Per-model/result trees | 16 root dirs | 34G total | Mixed, often high | Public: high unless curated | Private dev, game-drive archive, or curated HF artifacts. |

`git ls-files --others --exclude-standard` reported 2,266 untracked files,
about 7.4G by file size. Largest untracked buckets:

| Bucket | Untracked Count | Untracked Size |
| --- | ---: | ---: |
| `osmosis-gemma4-e2b` | 71 | 1.34G |
| `cerebellum-dev` | 252 | 1.34G |
| `osmosis-gemma4-26b` | 565 | 1.28G |
| `osmosis-qwen3-30b` | 53 | 1.15G |
| `osmosis-granite41-30b` | 17 | 496M |
| `osmosis-qwen3-32b` | 18 | 494M |
| `cerebellum-granite41-30b` | 25 | 398M |

Notable root-level untracked files: `AGENTS.md`, `cerebellum_logo.png`,
`cerebellum_logo_500kb.png`, `cerebellum_banner_500kb.png`, `llama.log`,
`osmosis-granite41-30b-download.log`, `speculative.log`,
`watch_20260604-141238`, `wikitext-test.txt`, `start.md`, and `placebo.md`.

## Coarse File-Type Counts

These counts are approximate and based on file names under top-level legacy
directories.

| Path | GGUF | JSON/JSONL | Logs/Out | Tensor Maps/Overrides | Benchmark Files | Docs/Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cerebellum-dev` | 0 | 63 | 61 | 16 | 7 | 20 |
| `cerebellum-gemma4-codex` | 0 | 28 | 18 | 0 | 47 | 0 |
| `cerebellum-granite41-30b` | 0 | 15 | 10 | 0 | 25 | 0 |
| `cerebellum-qwen35-9b` | 0 | 325 | 6 | 46 | 75 | 12 |
| `osmosis-gemma4-26b` | 3 | 156 | 235 | 147 | 128 | 7 |
| `osmosis-gemma4-e2b` | 1 | 27 | 25 | 15 | 27 | 1 |
| `osmosis-gemma4-e4b` | 0 | 28 | 36 | 9 | 27 | 2 |
| `osmosis-granite4-h-small` | 0 | 6 | 16 | 0 | 13 | 1 |
| `osmosis-granite41-30b` | 0 | 0 | 6 | 0 | 0 | 1 |
| `osmosis-qwen3-14b` | 0 | 1 | 10 | 7 | 2 | 1 |
| `osmosis-qwen3-30b` | 0 | 18 | 11 | 0 | 18 | 1 |
| `osmosis-qwen3-32b` | 0 | 0 | 8 | 7 | 0 | 1 |
| `osmosis-qwen35-122b` | 1 | 23 | 57 | 15 | 25 | 1 |
| `osmosis-qwen35-9b` | 0 | 9 | 9 | 2 | 15 | 0 |
| `osmosis-qwen36-27b` | 0 | 247 | 79 | 50 | 143 | 0 |
| `osmosis-qwen36-35b` | 0 | 43 | 4 | 2 | 6 | 1 |

## Large Binaries

| Size | Path | Initial Action |
| ---: | --- | --- |
| 10.94 GiB | `osmosis-gemma4-26b/gemma-4-26B-A4B-it-cerebellum-v6.gguf` | Hash and provenance-record before moving. |
| 3.01 GiB | `osmosis-gemma4-e2b/google_gemma-4-E2B-it-Q3_K_M.gguf` | Keep as baseline/reference for E2B comparisons. |
| 1.11 GiB | `osmosis-gemma4-26b/mmproj-google_gemma-4-26B-A4B-it-f16.gguf` | Keep with Gemma 4 26B release evidence. |
| 0.33 GiB | `osmosis-qwen35-122b/imatrix_unsloth.gguf` | Inspect purpose; likely private calibration/build artifact. |
| 0.05 GiB | `osmosis-gemma4-26b/google_gemma-4-26B-A4B-it-imatrix.gguf` | Keep until imatrix provenance is recorded. |

## High-Value Evidence

- `osmosis-gemma4-e2b/README.md` records the old E2B result, benchmark deltas,
  and three-layer override recipe. Keep as a reference for comparing the new
  full hill-climber against the older group-first/layer-drill method.
- `osmosis-gemma4-e2b/benchmark_results/` contains baseline, v1, and v2
  benchmark artifacts. Keep for score regression checks.
- `osmosis-gemma4-26b/` contains large GGUFs, release notes, benchmark logs,
  overrides, and model-card templates. Keep private until public-safe material
  is explicitly re-exported.
- `cerebellum-dev/DEVLOG_2026-06-03_gemma4-12b.md` records Gemma 4 12B
  conversion and run context. Keep private; sanitize only high-level claims.
- `cerebellum-qwen35-9b/` and `osmosis-qwen36-27b/` contain dense ablation and
  rowblock artifacts useful for cross-model pattern mining.
- `cerebellum-dev/SPEC_benchmark_gates.md` is the benchmark integrity
  source-of-truth.
- `cerebellum-dev/SPEC_api_control_plane.md` and
  `cerebellum-dev/ISSUE_cerebellum_api.md` are current API/control-plane
  direction.
- `cerebellum-dev/ablation_cross_model_summary.md`,
  `cerebellum-dev/ablation_cross_model_summary.json`, and
  `cerebellum-dev/ablation_cross_model_records.csv` are reusable cross-model
  research records.
- `cerebellum-dev/sparse-upcycling/docs/` contains current MoE/upcycling
  research and best-practice notes.
- `docs/benchmark_protocol.md`, `docs/public_release_scope.md`, and
  `docs/cerebellum_success_patterns.md` are public-release discipline docs.

## Private Dev Detail

Agent scan of `cerebellum-dev/` found:

- 5 devlogs in `cerebellum-dev/`, plus `docs/devlog/2026-06-04-cerebellum-cli.md`.
- 12 spec/issue docs, including API, dashboard, benchmark gates, agent runner,
  and agent routing.
- about 10 private research summaries, plus a sampled set of root docs that
  mention Cerebellum.
- about 50 benchmark/eval/result artifacts in private dev, including sparse
  upcycling PPL logs and comparison summaries.
- 63 private script/tool/test files: 56 Python files and 7 shell scripts after
  excluding pycache.
- 10 `.pt` checkpoints in `cerebellum-dev/conch-poc/checkpoints-*`, with the
  largest around 276 MB.

Private-dev scan found public-risk patterns such as absolute local paths,
private repo references, localhost/server commands, provider/backend routing
docs, diagnostic dumps, and run logs. No obvious literal API keys surfaced in
the sampled scan, but config-adjacent files and logs still require private-only
treatment unless audited.

## Public-Risk Material

- Raw ablation JSON, candidate logs, tensor maps, and override files can reveal
  method details.
- Devlogs/specs/issues in `cerebellum-dev` can reveal pipeline internals and
  local strategy.
- Benchmark detailed JSONL is valuable but should not be published until
  reviewed for prompts, local paths, model IDs, and scoring bugs.
- Local scripts, run shell files, dashboard DBs, caches, and transfer folders
  should stay out of `origin`.
- Also keep private: `agent_bench`, `harm_check`, `steering`,
  `user_bench_results`, refusal vectors/tests, internal automation logs,
  `.cache`, and huge server logs.

## Per-Family Inventory Notes

Counts below are path/name matches and may overlap. They are useful for triage,
not exact semantic counts.

### Gemma

- `osmosis-gemma4-26b` (24G): high-value `benchmark_results*`, `hf_cards`,
  release README/docs, `cerebellum_v*_overrides.txt`, PLE overrides,
  `imatrix.dat`, and final
  `gemma-4-26B-A4B-it-cerebellum-v6.gguf`. Private/risk: devlogs,
  `FULL_EXPERIMENT_LOG.md`, `research_log.md`, `.cache` with incomplete HF
  download material, and huge server logs.
- `osmosis-gemma4-e2b` (4.4G): high-value `benchmark_results`, override
  variants, README, and GGUF. Private/risk: scripts, logs, and server log.
- `osmosis-gemma4-e4b` (43M): high-value PLE override set,
  `ablation_results.json`, HumanEval benchmark artifacts, README/PPL docs.
- `cerebellum-gemma4-codex` (306M): high-value EvalPlus chat samples/eval
  JSON, ARC/HellaSwag/MMLU logs. Private/risk: developer log, transfer plan,
  and huge server log.

### Granite

- `osmosis-granite4-h-small` (11M): high-value PPL/ablation logs and benchmark
  results.
- `osmosis-granite41-30b` (497M): mostly logs and ablation evidence; huge
  `server_v2.log` is private/internal noise until proven otherwise.
- `cerebellum-granite41-30b` (399M): high-value benchmark results comparing
  baseline and Cerebellum, including EvalPlus samples/results. Private/risk:
  huge server logs.

### Qwen

- `osmosis-qwen3-14b` (528K): high-value ablation/tensor maps and small
  benchmark record. Private/risk: scripts/logs.
- `osmosis-qwen3-30b` (1.2G): high-value benchmark results. Private/risk:
  large ablation server log.
- `osmosis-qwen3-32b` (494M): high-value tensor maps and ablation evidence.
  Private/risk: large server log.
- `osmosis-qwen35-122b` (484M): high-value ablation results,
  `cerebellum_imatrix.dat`, benchmark results, tensor/build logs.
  Private/risk: pipeline scripts, `.cache`, internal automation logs.
- `osmosis-qwen35-9b` (9.6M): high-value `ablation_results.json`,
  `tensor_types_*`, and benchmark results.
- `osmosis-qwen36-27b` (186M): high-value `ablation_results.json`,
  `brain_scan*.json`, tensor maps, coding ablation/promotion dirs, and
  benchmark results. Private/risk: refusal vectors/tests and interaction logs.
- `osmosis-qwen36-35b` (35M): high-value verified benchmark dirs, overrides,
  and README. Private/risk: investigation devlog.
- `cerebellum-qwen35-9b` (574M): high-value `ablation_results_v2.json`,
  `variants/tensor_types_*`, benchmark results, and rowblock JSONs.
  Private/risk: `agent_bench`, Godot workspaces, `harm_check`, `steering`,
  `user_bench_results`, scripts, dev guide, and findings.

## Cleanup Candidates Requiring Verification

These are not approved for deletion. They are candidates for separate review:

- `__pycache__/` under private dev folders.
- duplicate benchmark logs once the canonical summary/detail artifacts are
  identified and backed up.
- local caches such as `unsloth_compiled_cache/`.
- transfer/drop folders such as `taildrop/`.
- stale root logs after their corresponding experiment directory is identified.

## Next Inventory Tasks

1. Hash all GGUF/imatrix binaries and attach source/model/purpose metadata.
2. Produce per-model manifests for Gemma 4, Qwen, and Granite trees.
3. Identify canonical benchmark result bundles and mark duplicate logs.
4. Mark public-safe candidates that can be exported through `public-export`.
5. Decide an archive layout on the game drive or another backup target before
   moving any heavy artifacts.
