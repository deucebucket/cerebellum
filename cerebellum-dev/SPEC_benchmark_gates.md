# SPEC: Benchmark Ingestion, Audit & Publishability Gates

## Goal
Every benchmark result ingested into the DB must pass automated and manual audit gates before it can be published. The API enforces these gates at the HF upload boundary, not as a suggestion.

---

## Ingestion Pipeline

```
Shell script / llama-server output
  → benchmark_*.json (raw results)
  → POST /api/ingest/benchmark-result
    → Parse + normalize → BenchmarkRun row
    → Link to model artifact (quant GGUF) by name/checksum
    → Return { benchmark_run_id, warnings }
```

### Supported Benchmark Formats

| Benchmark | Key Fields Parsed | File Pattern |
|-----------|------------------|-------------|
| HumanEval+ (EvalPlus) | `pass_at_1_plus`, `pass_at_1_base`, `total_problems`, `results` | `*_evalplus_results.json` |
| HumanEval+ (chat template) | `pass_at_1_plus`, `pass_at_1_base` | `*_evalplus_chat_results.json` |
| ARC | `accuracy`, `total` | `*_arc_results.json` |
| HellaSwag | `accuracy`, `total` | `*_hellaswag_results.json` |
| MMLU | `accuracy`, `total` | `*_mmlu_results.json` |
| MMLU-Redux | `accuracy`, `total` | `*_mmlu_redux_results.json` |
| PPL | `perplexity` | `*_ppl_results.json` |

### Ingestion Idempotency

- Dedup by `sha256` of the raw result file. If the same file is ingested twice, return the existing `benchmark_run_id`.
- If the file has changed (new sha256), create a new `BenchmarkRun` and mark the old one as `superseded`.

---

## Audit Gates

### Gate 1: Automated Pre-Audit (always runs on ingestion)

For EvalPlus results, the `POST /api/ingest/benchmark-result` handler runs these checks synchronously:

1. **AST Syntax Check** — Run `ast.parse()` on every generated solution. Count failures. `ast_syntax_fail_count` must be 0.
2. **Fence Check** — Count code fences (```) in solutions. `fence_count` must be 0 (fences already stripped in proper runs).
3. **Prompt Echo Check** — Scan for prompt text leaked into output. `prompt_echo_count` must be 0.
4. **Repeated Target Definition** — Count solutions that re-define the target function. `repeated_target_def_count` must be 0.
5. **Pass-Only Output** — Count solutions that only contain `pass`. `pass_only_count` must be 0.
6. **Cop-Out Check** — Count `...`, `# TODO`, `raise NotImplementedError`. `cop_out_count` must be 0.

For MCQ benchmarks (ARC/HellaSwag/MMLU):

1. **Unknown Answer Count** — responses that don't match expected answer format.
2. **Empty Response Fallback** — responses that were empty and received a fallback.
3. **Parse Method** — record which parse strategy was used (regex, logprobs, etc.).

Results stored in `BenchmarkAudit` with `auditor = "automated_script"`.

### Gate 2: Wrong-Answer Inspection (human or reviewer agent)

EvalPlus:
- Random sample 15 wrong answers (or all if <15 wrong).
- For each, read the full solution, verify it's a real failure (not a parse issue).
- Store `inspected_sample_ids: list[int]` in the audit record.

MCQ:
- Dump first 30 wrong answers via `jq 'select(.correct == false)' detailed.jsonl | head -30`.
- Identify systematic issues (answer format mismatch, model refusal, truncation).
- Store `first_wrong_sample_path` and summary note.

Results stored in `BenchmarkAudit` with `auditor = "human"` or `auditor = "opencode_reviewer"`.

### Gate 3: Syntax/Indentation Cluster Check

Before marking publishable:
- Group AST failures by error type. If 3+ failures share the same SyntaxError message, flag as a systematic issue.
- Systematic issues require human review and notes before publishability.

---

## Publishability Function

```
def is_publishable(benchmark_run_id: int) -> Publishability:
    run = get_benchmark_run(benchmark_run_id)
    audits = get_audits(benchmark_run_id)

    if run.status != "completed":
        return Publishability(ready=False, reason="Benchmark not completed")

    # Require at least automated audit
    auto_audit = [a for a in audits if a.auditor == "automated_script"]
    if not auto_audit:
        return Publishability(ready=False, reason="Automated audit required")

    latest_auto = max(auto_audit, key=lambda a: a.created_at)
    if not latest_auto.passed:
        return Publishability(ready=False, reason=f"Automated audit failed: {latest_auto.notes}")

    # EvalPlus requires human/reviewer inspection
    if run.benchmark in ("humaneval_plus",):
        inspection_audits = [a for a in audits if a.auditor in ("human", "opencode_reviewer")]
        if not inspection_audits:
            return Publishability(ready=False, reason="Wrong-answer inspection required")
        latest_inspect = max(inspection_audits, key=lambda a: a.created_at)
        if not latest_inspect.passed:
            return Publishability(ready=False, reason=f"Inspection failed: {latest_inspect.notes}")

    return Publishability(ready=True)
```

### Enforcement Points

- `POST /api/models/{id}/hf/upload` — calls `is_publishable()` for every benchmark claimed in the model card. Returns 412 with a list of failures if any gate fails.
- `POST /api/models/{id}/card/generate` — includes an `hf_ready` boolean and a `gates` block listing each benchmark's publishability status.
- Dashboard Model Detail page — shows red/green gate indicators per benchmark.

---

## Audit API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/benchmarks/{id}/audit` | Create or update audit record |
| GET | `/api/benchmarks/{id}/audit` | Get all audit records for a run |
| POST | `/api/benchmarks/{id}/inspect` | Trigger automated re-audit (re-runs AST checks) |
| GET | `/api/benchmarks/{id}/publishability` | Returns `{ready, reasons[]}` |

---

## Acceptance Criteria

- Ingesting a benchmark JSON with AST failures creates a `BenchmarkRun` + automated `BenchmarkAudit` with `passed=false`.
- The dashboard shows a red "Audit Failed" badge for that benchmark.
- Submitting a human audit record with `passed=true` updates the gate.
- `POST /api/models/{id}/hf/upload` returns 412 when any claimed benchmark lacks required audits.
- A brand new EvalPlus result ingested via `POST /api/ingest/benchmark-result` goes from not-ready → automated-audit-done → inspection-done → publishable in the UI.

---

## First Milestone

1. Add `BenchmarkAudit` model and `audit_id` FK to `BenchmarkRun`.
2. Implement automated pre-audit checks in the `POST /api/ingest/benchmark-result` handler for EvalPlus.
3. Implement `GET /api/benchmarks/{id}/publishability` and `POST /api/benchmarks/{id}/audit`.
4. Wire publishability gate into `POST /api/models/{id}/hf/upload` (return 412).
5. Add gate indicators to the dashboard Model Detail page (red/green per benchmark).
