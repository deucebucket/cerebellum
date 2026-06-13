# SPEC: Cerebellum API & Control Plane

## Goal
Replace the current file-scraping `/api/model-cards` stopgap with a proper REST API backed by a normalized SQLite schema. All pipeline operations, artifact discovery, benchmark ingestion, and agent orchestration go through this API.

---

## Resource Model

### `Model`
- `id: str` — stable canonical id (`cerebellum-gemma-4-9b`)
- `display_name: str`, `source_repo: str`, `architecture: str`
- `params_b: float`, `tokenizer: str`, `modalities: list[str]`
- `variant_of: str?` — link to parent model for Cerebellum/heretic variants
- `created_at`, `updated_at`

### `Artifact`
- `id: int`, `model_id: str` (FK), `experiment_id: int?` (FK)
- `type: enum` — `source`, `f16_gguf`, `quant_gguf`, `imatrix`, `ablation`, `tensor_types`, `benchmark_result`, `model_card`, `upload_manifest`
- `path: str`, `size_bytes: int`, `sha256: str?`, `created_at`
- `visibility: enum` — `public`, `private` (controls origin/dev filtering)

### `Experiment`
- `id: int`, `model_id: str` (FK)
- `method: enum` — `uniform_quant`, `cerebellum`, `heretic`, `ple`, `mtp`, `rowblock`
- `config: json` — calibration corpus, imatrix metadata, allocator settings
- `findings: text?` — human notes
- `output_artifact_ids: list[int]`

### `BenchmarkRun`
- `id: int`, `artifact_id: int` (FK → artifact of quant GGUF)
- `benchmark: str` — `humaneval_plus`, `arc`, `hellaswag`, `mmlu`, `mmlu_redux`
- `harness: str`, `harness_revision: str`, `command: str`, `server_settings: json`
- `results: json` — parsed pass@1, accuracy, etc.
- `detailed_path: str?` — full results JSON / samples file
- `status: enum` — `pending`, `running`, `completed`, `failed`
- `audit_id: int?` (FK → benchmark_audits)

### `BenchmarkAudit`
- `id: int`, `benchmark_run_id: int` (FK)
- `auditor: str` — `human`, `opencode_reviewer`, `automated_script`
- `passed: bool`
- For EvalPlus: `ast_syntax_fail_count`, `fence_count`, `prompt_echo_count`, `repeated_target_def_count`, `pass_only_count`, `cop_out_count`, `inspected_sample_ids: json`
- For MCQ: `unknown_answer_count`, `empty_response_fallback_count`, `parse_method`, `first_wrong_sample_path`
- `notes: text?`, `created_at`

### `Job`
- Current `jobs` table extended with `model_id` FK and `experiment_id` FK
- `phase: str`, `progress: float`, `error_message`, `worker_id`, `config`

### `ModelCard`
- `id: int`, `model_id: str` (FK)
- `generated_markdown: text`, `audit_statement: text?`
- `hf_ready: bool`, `hf_uploaded_at: datetime?`
- `created_at`, `version: int`

### `EventLog`
- Keep existing table; add `resource_type` and `resource_id` columns for generic eventing.

---

## SQLite Schema Migration

Add tables, preserve existing `jobs`, `schedules`, `model_watches`, `event_logs`:

```python
# New tables added to models.py
class Model(Base): __tablename__ = "models"
class Artifact(Base): __tablename__ = "artifacts"
class Experiment(Base): __tablename__ = "experiments"
class BenchmarkRun(Base): __tablename__ = "benchmark_runs"
class BenchmarkAudit(Base): __tablename__ = "benchmark_audits"
class ModelCard(Base): __tablename__ = "model_cards"
class AgentRun(Base): __tablename__ = "agent_runs"  # see SPEC_agent_runner.md
```

Migration strategy: add tables with `CREATE TABLE IF NOT EXISTS`, add FK columns to `jobs` via ALTER TABLE (SQLite supports ADD COLUMN). No destructive migrations.

---

## Endpoints

### Read
| Method | Path | Status |
|--------|------|--------|
| GET | `/api/models` | New |
| GET | `/api/models/{id}` | New |
| GET | `/api/models/{id}/card` | New — rendered model card |
| GET | `/api/models/{id}/benchmarks` | New |
| GET | `/api/models/{id}/artifacts` | New |
| GET | `/api/experiments` | New |
| GET | `/api/experiments/{id}` | New |
| GET | `/api/experiments/{id}/artifacts` | New |
| GET | `/api/benchmarks` | New — list all benchmark runs |
| GET | `/api/benchmarks/{id}` | New — single run with audit |
| GET | `/api/benchmarks/{id}/audit` | New |
| GET | `/api/jobs` | Existing |
| GET | `/api/jobs/{id}` | Existing |
| GET | `/api/jobs/{id}/logs` | Existing |
| GET | `/api/health` | Existing |

### Ingest (File System → DB)
| Method | Path | Status |
|--------|------|--------|
| POST | `/api/ingest/scan` | New — full filesystem scan, upserts models/artifacts |
| POST | `/api/ingest/path` | New — ingest a single path (GGUF, imatrix, etc.) |
| POST | `/api/ingest/benchmark-result` | New — parse and store a single benchmark JSON |
| POST | `/api/ingest/ablation` | New — ingest ablation_results.json |
| POST | `/api/ingest/gguf` | New — register a single GGUF as artifact |

### Run (Pipeline)
| Method | Path | Status |
|--------|------|--------|
| POST | `/api/jobs/convert` | New — create convert job |
| POST | `/api/jobs/imatrix` | New — create imatrix job |
| POST | `/api/jobs/ablate` | New — create ablation job |
| POST | `/api/jobs/allocate` | New — create budget allocation job |
| POST | `/api/jobs/build` | New — create quant build job |
| POST | `/api/jobs/benchmark` | New — create benchmark job |
| POST | `/api/jobs/publish` | New — create HF upload job |

### Generate (Model Card / HF)
| Method | Path | Status |
|--------|------|--------|
| POST | `/api/models/{id}/card/generate` | New — generate card markdown from DB |
| POST | `/api/models/{id}/card/render` | New — render preview HTML |
| POST | `/api/models/{id}/hf/prepare` | New — stage files for HF upload |
| POST | `/api/models/{id}/hf/upload` | New — upload to HF (blocked without audit) |

### Agent
| Method | Path | Status |
|--------|------|--------|
| POST | `/api/agents/opencode/run` | New — launch opencode task (see SPEC_agent_runner.md) |
| GET | `/api/agents/runs/{id}` | New — get run status + verdict |
| POST | `/api/agents/runs/{id}/review` | New — trigger independent review pass |

### Atlas / Sensitivity
| Method | Path | Status |
|--------|------|--------|
| GET | `/api/atlas/tensors` | New — cross-model tensor sensitivity summary |
| GET | `/api/atlas/models/{id}/tensors` | New — per-model tensor sensitivity |

---

## Acceptance Criteria

- `POST /api/ingest/scan` discovers every `cerebellum-*` dir and populates DB within 30s.
- `GET /api/models` returns all discovered models; `GET /api/models/{id}/benchmarks` returns their benchmark runs.
- `POST /api/jobs/benchmark` creates a job row and the worker picks it up.
- HF upload (`POST /api/models/{id}/hf/upload`) returns 412 Precondition Failed if audit records are missing for any claimed benchmark.
- Every endpoint returns consistent JSON envelopes: `{"data": ..., "error": null}` or `{"data": null, "error": "..."}`.

---

## First Milestone

1. Add schema + migration in `models.py` (7 new tables, FK cols on `jobs`).
2. Implement `POST /api/ingest/scan` — port the current `discover_model_cards()` logic to upsert into `models` and `artifacts`.
3. Implement `GET /api/models`, `GET /api/models/{id}`, `GET /api/models/{id}/artifacts`.
4. Add `GET /api/atlas/tensors` returning a summary of all ablation data.
5. Deploy and verify the dashboard loads from DB instead of scanning on every page load.
