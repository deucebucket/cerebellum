# ISSUE: Build the Cerebellum API as the Source of Truth

## Problem

The dashboard is currently a queue UI, not a Cerebellum control plane.
It only knows about jobs created through `/api/jobs`, while the real project
state lives in shell-run artifacts:

- `cerebellum-*/benchmark_results/`
- `osmosis-*/benchmark_results/`
- GGUFs on `/var/home/deucebucket/games/`
- ablation JSON, tensor override files, imatrix files, findings docs
- live benchmark logs such as `*_evalplus_run.log`

That is why the dashboard showed nothing useful: the data existed, but no API
ingested it, normalized it, or rendered it as model cards.

## Immediate Fix Landed

Added `/api/model-cards` to `osmosis/dashboard/server.py`.

It scans local `cerebellum-*` and legacy `osmosis-*` experiment directories,
then returns model-card-shaped records:

- GGUF artifacts and sizes
- benchmark summary JSONs
- live benchmark progress parsed from run logs
- docs/findings files
- ablation/variant presence

Added a `Models` dashboard page that consumes this endpoint.

This is a stopgap. It proves the UI can show real data, but discovery is still
read-only and file-system based.

## Required API

The proper API should make Cerebellum a local model-lab service. The dashboard,
CLI, automation, and future agents should all talk to this API instead of
scraping folders independently.

### Core Resources

`Model`

- stable id, display name, source repo/path
- architecture, parameter count, tokenizer, modalities
- source files, F16 GGUF, quant artifacts
- model family links: baseline, Cerebellum variants, heretic variants

`Artifact`

- type: `source`, `f16_gguf`, `quant_gguf`, `imatrix`, `ablation`, `tensor_types`,
  `benchmark_result`, `model_card`, `upload_manifest`
- path, size, checksum, created_at, source job id
- public/private visibility flag for origin/dev safety

`Experiment`

- model id
- method: uniform quant, Cerebellum, heretic, PLE, MTP, rowblock, etc.
- calibration corpus, imatrix metadata, allocator settings
- output artifacts
- findings notes

`BenchmarkRun`

- model artifact id
- benchmark name, harness version, command, server settings
- result summary, detailed artifact paths
- audit status: syntax checks, fence checks, prompt echo checks, wrong-answer inspection

`Job`

- queued/running/completed/failed
- phase, progress, logs, subprocess metadata
- associated model/experiment/artifacts

`ModelCard`

- generated markdown
- benchmark tables
- audit statement
- artifact links
- HF upload readiness checklist

### Endpoints

Read:

- `GET /api/models`
- `GET /api/models/{id}`
- `GET /api/models/{id}/card`
- `GET /api/models/{id}/benchmarks`
- `GET /api/models/{id}/artifacts`
- `GET /api/experiments`
- `GET /api/experiments/{id}`
- `GET /api/jobs`
- `GET /api/jobs/{id}/logs`
- `GET /api/atlas/tensors`
- `GET /api/health`

Ingest:

- `POST /api/ingest/scan`
- `POST /api/ingest/path`
- `POST /api/ingest/benchmark-result`
- `POST /api/ingest/ablation`
- `POST /api/ingest/gguf`

Run:

- `POST /api/jobs/convert`
- `POST /api/jobs/imatrix`
- `POST /api/jobs/ablate`
- `POST /api/jobs/allocate`
- `POST /api/jobs/build`
- `POST /api/jobs/benchmark`
- `POST /api/jobs/publish`

Generate:

- `POST /api/models/{id}/card/generate`
- `POST /api/models/{id}/card/render`
- `POST /api/models/{id}/hf/prepare`
- `POST /api/models/{id}/hf/upload`

Agent:

- `POST /api/agents/opencode/run`
- `GET /api/agents/runs/{id}`
- `POST /api/agents/runs/{id}/review`

### Database

SQLite is fine, but the schema needs to move beyond `jobs`.

Minimum tables:

- `models`
- `artifacts`
- `experiments`
- `benchmark_runs`
- `benchmark_audits`
- `jobs`
- `event_logs`
- `model_cards`
- `tensor_sensitivity`

The file scanner should populate these tables idempotently. Folder names are
not identity; checksums and canonical model ids are.

## Benchmark Integrity Rules

The API must refuse to mark a benchmark as publishable until audits exist.

EvalPlus required audit fields:

- `ast_syntax_fail_count`
- `fence_count`
- `prompt_echo_count`
- `repeated_target_def_count`
- `pass_only_count`
- `cop_out_count`
- inspected wrong-answer sample ids
- harness name and git revision

MCQ required audit fields:

- unknown answer count
- empty response fallback count
- parse method
- first wrong-answer sample path

## Opencode as an Agent

Yes, opencode can be turned into an agent runner.

It already supports:

- `opencode run`
- `opencode serve`
- `opencode attach`
- `opencode agent`
- `opencode acp`
- `opencode export`

The reliable pattern is not "let opencode go wild"; it is:

1. Create an isolated workspace.
2. Start an opencode run with a fixed agent config and model endpoint.
3. Capture session export, logs, git diff, and test output.
4. Run an independent reviewer pass.
5. Only mark success if tests pass and the reviewer signs off.

### Reviewer Loop

`opencode-runner`:

- runs the task
- enforces timeout and workspace boundaries
- records files touched, commands run, and final diff

`reviewer`:

- reads task, diff, and logs
- runs deterministic checks
- flags missing deliverables, broken tests, hallucinated files, unsafe commands
- can request one bounded repair pass

`supervisor`:

- owns stop conditions
- prevents infinite repair loops
- writes final verdict into `benchmark_runs` or `agent_runs`

For model evaluation, this is better than a raw opencode score because it
separates:

- model tool-call ability
- opencode integration behavior
- task completion
- test correctness
- reviewer-confirmed quality

## Implementation Plan

1. Add schema + migration helper for models/artifacts/benchmarks.
2. Promote the current `/api/model-cards` scanner into `POST /api/ingest/scan`.
3. Store scan results in SQLite instead of recomputing every page load.
4. Build `GET /api/models/{id}/card` from normalized DB rows.
5. Fix dashboard worker command contracts, especially benchmark invocation.
6. Add benchmark audit records and publishability gates.
7. Add opencode agent-run resource with reviewer loop.
8. Replace the current dashboard queue/history pages with model-centric pages.

## Acceptance Criteria

- Dashboard shows every local Cerebellum model and variant without manual entry.
- The current running benchmark appears within 30 seconds.
- Each model card shows artifacts, benchmark summaries, audit status, and docs.
- A fresh shell-run benchmark can be ingested and audited without editing DB rows.
- HF upload cannot proceed unless required benchmark artifacts and audit records exist.
- Opencode runs produce a structured verdict: pass/fail, files changed, tests run, reviewer notes.
