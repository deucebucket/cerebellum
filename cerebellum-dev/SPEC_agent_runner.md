# SPEC: Opencode-as-Agent Runner & Independent Reviewer Loop

## Goal
Run opencode as a controlled agent for software tasks (benchmarking, analysis, code review) with an independent reviewer pass that prevents hallucinations, unsafe commands, and false passes.

---

## Architecture

```
API POST /api/agents/opencode/run
  → Supervisor creates isolated workspace
  → Spawns opencode run with fixed config + model endpoint
  → Captures: session export, logs, git diff, test output, exit code
  → Supervisor stores raw run as AgentRun (status=running)
  → (Optional) POST /api/agents/runs/{id}/review triggers reviewer
  → Reviewer produces verdict → AgentRun.verdict updated
  → Supervisor enforces max 1 repair pass if requested
```

---

## Database Model

```python
class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: int (PK)
    task: str                  # The task prompt given to opencode
    model_endpoint: str        # e.g. "http://localhost:8000/v1"
    agent_config: str          # JSON blob — opencode config overrides
    workspace_path: str        # Isolated work directory
    status: str                # running, completed, failed, reviewing, done

    # Captured outputs
    session_export_path: str?  # Path to opencode export artifact
    log_path: str?             # Full run log
    git_diff_path: str?        # Diff of workspace changes
    test_output_path: str?     # Test run output
    exit_code: int?

    # Verdict
    verdict: str?              # pass, fail, needs_repair
    verdict_detail: str?       # JSON — reviewer notes, flags, scores
    reviewer_model: str?       # Which model did the review (e.g. "big-pickle")

    created_at, completed_at
```

---

## Runner (Supervisor)

The supervisor is a Python module (`osmosis/agents/supervisor.py`) that:

1. **Creates workspace**: `tempfile.mkdtemp(prefix="opencode-run-")` inside `cerebellum-dev/agent_workspace/`.
2. **Prepares context**: copies in task description, any reference files, a minimal README.
3. **Spawns opencode**:
   ```
   opencode run --model <endpoint> --agent-config <config> --task <task file>
   ```
   Wrapped in `subprocess` with timeout (default 30min, configurable per task type).
4. **Captures output**:
   - `stdout/stderr → log file`
   - Watches for `opencode export` artifact location via log parsing
   - Runs `git init && git add -A && git diff --cached --stat` in workspace after completion
5. **Stores result** in `agent_runs` table.
6. **Cleans up** workspace after configurable retention period.

### Timeout & Retry

- Default timeout: 30min for code tasks, 60min for analysis tasks.
- On timeout: set status to `failed`, verdict to `fail`, detail includes "Timeout exceeded".
- No automatic retry — supervisor only retries if reviewer requests a single repair pass.

---

## Reviewer Loop

The reviewer is a separate opencode run (or direct API call) that evaluates the runner's output deterministically.

### Trigger

```
POST /api/agents/runs/{id}/review
  Body: { "model": "big-pickle" | "nim" | "deepseek" }
```

Returns immediately with `{ "status": "reviewing" }`. Reviewer runs async.

### Reviewer Prompt Template

```
You are a code review agent. Evaluate the following task result.

TASK:
{task}

FILES CHANGED:
{git_diff}

TEST OUTPUT:
{test_output}

RUN LOG (last 200 lines):
{log_tail}

Check:
1. Does the solution actually solve the task? (pass/fail)
2. Are all required deliverables present? (list missing)
3. Do all provided tests pass? (note any failures)
4. Are there hallucinated files or commands? (list)
5. Are there unsafe operations? (rm -rf, eval, curl to unknown hosts)
6. Is the code idiomatic and consistent with the codebase?

Respond with JSON:
{
  "verdict": "pass" | "fail" | "needs_repair",
  "score": 0-100,
  "missing_deliverables": [],
  "hallucinated_files": [],
  "unsafe_commands": [],
  "test_failures": [],
  "repair_request": "..."  // only if needs_repair
}
```

### Repair Pass

If verdict is `needs_repair`:
- Supervisor creates a follow-up task: "Fix these issues: {repair_request}"
- Same workspace, same opencode config.
- Runs with 15min timeout (shorter — repairs should be targeted).
- After repair, reviewer runs again with `repair_attempt: true` in context.
- If verdict is still `needs_repair` or `fail`, final verdict is `fail`. No third attempt.

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/agents/opencode/run` | Launch a new agent run |
| GET | `/api/agents/runs` | List all runs, filterable by status |
| GET | `/api/agents/runs/{id}` | Get run details + verdict |
| POST | `/api/agents/runs/{id}/review` | Trigger (or re-trigger) reviewer pass |
| GET | `/api/agents/runs/{id}/logs` | Stream full log |
| POST | `/api/agents/runs/{id}/cancel` | Kill running agent process |
| DELETE | `/api/agents/runs/{id}` | Clean up workspace + DB row |

---

## Use Cases

### 1. Automated Benchmark Audit
- Task: "Audit EvalPlus results in {path}. Run ast.parse on every solution, count fence/prompt-echo/repeated-def issues, sample 15 wrong answers and check if they are real failures."
- Review: Verifies counts match audit outputs, flags any missed issues.
- Outcome: `BenchmarkAudit` record with `auditor = "opencode_reviewer"`.

### 2. Benchmark Run Validation
- Task: "Check that benchmark results in {path} have no prompt echoes, trailing fences, or pass-only outputs. Report counts for each."
- Review: Verifies the agent ran the correct checks.
- Outcome: Gate passes or fails.

### 3. Code Change Agent (future)
- Task: "Fix the HumanEval fence-stripping bug in scripts/benchmark_evalplus_chat.py"
- Review: Verifies the fix, runs existing tests, checks for regressions.
- Outcome: Committed fix or failed attempt.

---

## Acceptance Criteria

- `POST /api/agents/opencode/run` with a valid task creates an `AgentRun`, starts opencode, and captures logs + diff.
- `POST /api/agents/runs/{id}/review` triggers the reviewer, which returns a structured verdict within 5min.
- A task that intentionally introduces an AST syntax error gets caught by the reviewer (verdict = `fail`).
- Repair pass: a `needs_repair` verdict followed by a fix attempt gets re-reviewed, and the cycle terminates.
- Supervisor prevents infinite loops (max 1 repair pass).
- Agent runs are isolated — no cross-workspace file access.

---

## First Milestone

1. Implement `supervisor.py` — workspace creation, opencode spawning, output capture, timeout enforcement.
2. Add `agent_runs` table to `models.py`.
3. Implement `POST /api/agents/opencode/run` and `GET /api/agents/runs/{id}` endpoints.
4. Implement reviewer as a lightweight HTTP call (OpenAI-compatible chat completions endpoint).
5. Wire repair pass logic (max 1 loop).
6. Test with a known task: audit an EvalPlus result file and compare to manual audit.
