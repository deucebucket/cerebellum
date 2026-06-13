# SPEC: Agent Worker Routing — NIM / DeepSeek / Big Pickle

## Goal
The agent runner (`SPEC_agent_runner.md`) must support routing tasks to different inference backends. Each backend has distinct cost, speed, capability, and availability profiles. The routing decision is configurable per task type and can be overridden per run.

---

## Available Backends

### 1. Big Pickle Free (Opencode Provider, Default)
- **Opencode model id**: `opencode/big-pickle`
- **Cost**: Free tier via opencode provider.
- **GPU impact**: None on the local 3090. This is the point of using it while
  local Cerebellum benchmarks are running.
- **Capability**: Good default for background planning, issue decomposition,
  lightweight code/docs edits, and audit scaffolding.
- **Availability**: Depends on opencode provider access, not local llama-server.
- **Limits**: Expect turn/session limits. The supervisor must poll and bump
  long sessions forward instead of assuming one run finishes the whole task.

### 2. NIM / NVIDIA Provider
- **Opencode model ids**: examples include
  `nvidia/deepseek-ai/deepseek-v4-flash`,
  `nvidia/deepseek-ai/deepseek-v4-pro`,
  `nvidia/qwen/qwen3-coder-480b-a35b-instruct`,
  `nvidia/nvidia/llama-3.3-nemotron-super-49b-v1.5`.
- **Cost/GPU impact**: Remote/provider path as configured in opencode; no local
  VRAM pressure unless a separate local NIM container is intentionally started.
- **Capability**: Strong reviewer, coding, and structured analysis options.
- **Availability**: Depends on provider keys/quotas.
- **Limits**: Tool-call behavior and context vary by model. Store exact model id
  on every agent run.

### 3. DeepSeek via Opencode-Go / Opencode
- **Opencode model ids**: `opencode-go/deepseek-v4-flash`,
  `opencode-go/deepseek-v4-pro`, `opencode/deepseek-v4-flash-free`.
- **Cost/GPU impact**: Remote/provider path; no local VRAM pressure.
- **Capability**: Strong hard-analysis and code-review backend. Use when Big
  Pickle stalls, produces shallow specs, or a reviewer pass needs more bite.
- **Availability**: Requires provider access and network.
- **Limits**: Data leaves the machine. Do not send credentials, private tokens,
  or unpublished private model weights/logs unless explicitly approved.

---

## Routing Matrix

| Task Type | Default Backend | Alternative | Notes |
|-----------|----------------|-------------|-------|
| Benchmark audit (AST checks) | Big Pickle Free | NIM/DeepSeek | Simple checks; preserve local VRAM |
| Wrong-answer inspection | Big Pickle Free | DeepSeek V4 | Needs reasoning over samples |
| Reviewer pass | DeepSeek V4 | NIM/Qwen Coder | Review benefits from strongest model |
| Code/docs generation | Big Pickle Free | DeepSeek V4 | Use docs-only scope for background work |
| Analysis / data query | NIM/Nemotron | DeepSeek V4 | Strong structured-output models |
| Emergency fallback | opencode/deepseek-v4-flash-free | Big Pickle Free | Use whichever provider is currently healthy |

---

## Configuration

### Environment Variables

```
AGENT_DEFAULT_OPENCODE_MODEL=opencode/big-pickle
AGENT_REVIEWER_OPENCODE_MODEL=opencode-go/deepseek-v4-pro
AGENT_FAST_FALLBACK_OPENCODE_MODEL=opencode/deepseek-v4-flash-free
AGENT_NIM_REVIEWER_MODEL=nvidia/qwen/qwen3-coder-480b-a35b-instruct
```

### Per-Run Override

The `POST /api/agents/opencode/run` endpoint accepts:

```json
{
  "task": "Audit EvalPlus results in /path/to/results.json",
  "model": "opencode/big-pickle",
  "reviewer_model": "opencode-go/deepseek-v4-pro",
  "agent_config": { ... }
}
```

- `backend` — overrides the runner backend.
- `reviewer_backend` — overrides the reviewer backend.
- `model_endpoint` — fully custom endpoint (e.g. a cloud provider or another local server).
- If both are omitted, the routing matrix defaults apply.

---

## Backend Health Check

Before launching a run, the supervisor checks backend availability:

```
GET {endpoint}/health  (or model list endpoint)
```

If the backend is unreachable:
- Log warning: `Backend {name} unavailable, falling back to {fallback}`
- Use the fallback from the routing matrix.
- If all backends are unavailable, return `503 Service Unavailable` from the API.

### Health Status Endpoint

```
GET /api/agents/backends
{
  "backends": {
    "big-pickle": {
      "available": true,
      "model": "big-pickle-q4_K_M",
      "latency_ms": 45,
      "gpu_used_mb": 8124,
      "gpu_total_mb": 12288
    },
    "nim": {
      "available": false,
      "error": "Connection refused on localhost:8000"
    },
    "deepseek": {
      "available": true,
      "model": "deepseek-r1",
      "latency_ms": 320
    }
  }
}
```

---

## OpenCode Config Per Backend

Each backend needs slightly different opencode config:

### Big Pickle Config
```json
{
  "model": "http://localhost:8080/v1",
  "temperature": 0.3,
  "max_tokens": 8192,
  "tool_use": "auto"
}
```

### NIM Config
```json
{
  "model": "http://localhost:8000/v1",
  "temperature": 0.3,
  "max_tokens": 4096,
  "tool_use": "auto",
  "stop": ["<|im_end|>"]
}
```

### DeepSeek Config
```json
{
  "model": "deepseek-chat",
  "api_base": "https://api.deepseek.com/v1",
  "temperature": 0.3,
  "max_tokens": 8192,
  "tool_use": "auto"
}
```

These configs are stored in `osmosis/agents/backends.json` and loaded by the supervisor based on the routing decision.

---

## Acceptance Criteria

- `GET /api/agents/backends` reports availability and latency for all three backends.
- `POST /api/agents/opencode/run` with `backend: "nim"` routes to the NIM endpoint.
- If Big Pickle is down, the supervisor falls back to NIM for default tasks and logs the fallback.
- `POST /api/agents/runs/{id}/review` uses the reviewer backend from the run config (defaults to DeepSeek).
- The dashboard shows backend status on the Agents page.
- A run with an unreachable backend returns 503 with the list of backends tried.

---

## First Milestone

1. Create `osmosis/agents/backends.json` with the three backend profiles.
2. Implement backend health check in `supervisor.py` — try each endpoint's `/health` or `/v1/models`.
3. Add `GET /api/agents/backends` endpoint.
4. Wire backend selection into `POST /api/agents/opencode/run` — read `backend` field, select config, validate availability, fallback if needed.
5. Test: start Big Pickle llama-server, run an agent task with `backend: "big-pickle"`, verify it completes. Then stop llama-server, run again, verify fallback to NIM or error.
