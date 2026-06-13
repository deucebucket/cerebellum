# SPEC: Dashboard Data Model & UX

## Goal
Replace the current queue-focused jQuery dashboard with a model-centric React app. Every local Cerebellum model and variant appears as a card with artifacts, benchmarks, audit status, and docs — zero manual entry.

---

## Data Model (Drives Every View)

### Model Card Object (returned by `GET /api/models/{id}/card`)
```
{
  "model": {
    "id": "cerebellum-granite-3.2-8b",
    "display_name": "Granite 3.2 8B",
    "architecture": "granite",
    "params_b": 8.0,
    "source_repo": "ibm-granite/granite-3.2-8b-base",
    "variant_of": null
  },
  "artifacts": {
    "f16_gguf": { "name": "granite-3.2-8b.f16.gguf", "size_gb": 15.8, "sha256": "abc..." },
    "quants": [
      { "name": "granite-3.2-8b-Q4_K_M.gguf", "size_gb": 4.7, "method": "cerebellum", "experiment_id": 1 },
      { "name": "granite-3.2-8b-Q4_K_M.gguf", "size_gb": 4.7, "method": "uniform", "experiment_id": 2 }
    ],
    "imatrix": { "name": "imatrix.dat", "size_gb": 0.3 },
    "ablation": { "exists": true, "path": "..." },
    "tensor_types": { "exists": true, "path": "..." }
  },
  "benchmarks": {
    "humaneval_plus": { "pass_at_1": 0.72, "audit": { "passed": true, "inspected_samples": 15 }, "status": "completed" },
    "arc": { "accuracy": 0.68, "audit": null, "status": "completed" },
    "hellaswag": { "accuracy": 0.71, "audit": null, "status": "running", "progress": 42 }
  },
  "docs": [
    { "name": "FINDINGS.md", "size_kb": 12.3 },
    { "name": "DEVLOG.md", "size_kb": 45.1 }
  ],
  "hf_status": {
    "ready": false,
    "reason": "Benchmark audit missing for hellaswag"
  },
  "updated_at": "2026-05-20T10:30:00Z"
}
```

---

## Page Structure

### Page 1: Dashboard (Home)
**Purpose**: At-a-glance status of the lab.

Views:
- **Active Jobs Panel** — list of running/queued jobs from `GET /api/jobs`. Polled via WebSocket or 10s `setInterval`. Shows model name, phase, progress bar, elapsed time.
- **Recent Models** — last 6 updated model cards from `GET /api/models?sort=updated_at&limit=6`. Each shows name, latest score, status dot.
- **System Health** — disk gauge, GPU gauges (from `GET /api/health`), free space alert.
- **Quick Actions** — "Scan for Models", "New Benchmark Run", "Upload to HF".

### Page 2: Models (Model Browser)
**Purpose**: Browse all local models and variants.

Views:
- **Card Grid** — each card shows: model name, architecture, GGUF count, best benchmark score, audit status (green check / red X / amber running), "HF Ready" badge.
- **Search/Filter** — by architecture, by method (Cerebellum / uniform / heretic), by audit status.
- **Click → Model Detail** page.

### Page 3: Model Detail
**Purpose**: Every artifact, benchmark, and doc for a single model.

Views (tabs or sections):
1. **Overview** — model metadata, source repo link, variant chain (parent + children shown as a tree).
2. **Artifacts** — table of GGUFs (name, size, method, experiment link), imatrix, ablation files. Download links.
3. **Benchmarks** — table: benchmark name, score, run date, audit status, "Re-run" button. Each row expands to raw results + audit details. Side-by-side compare checkbox.
4. **Audit Panel** — for each benchmark: audit record fields, auditor name, pass/fail, inspected sample IDs. If not audited, shows "Audit Required" with link to audit tool.
5. **Model Card** — rendered markdown from `GET /api/models/{id}/card`, with "Regenerate" and "Upload to HF" buttons.
6. **Docs** — file list with viewer/editor links.

### Page 4: Jobs (Pipeline Queue)
**Purpose**: Full job queue with create controls.

Views:
- **Queue Table** — existing jobs list with status badges, progress bars, phase labels. Cancel/retry/delete actions.
- **New Job Form** — select model, pipeline steps, config overrides. Posts to `/api/jobs`.
- **History** — completed/failed jobs with duration and result summary.

### Page 5: Sensitivity Atlas
**Purpose**: Cross-model tensor sensitivity visualization.

Views:
- **Model Multi-Select** — pick 1-5 models to overlay.
- **Heatmap Grid** — layers × tensor groups, color intensity = PPL delta (green=robust, red=fragile). SVG-based, drag-to-zoom, hover tooltip with exact delta.
- **Tensor Inspector** — click a cell to see model-specific ablation value, quant level applied, recommendation.
- **Pattern Summary** — auto-generated text: "attn_v is universally fragile across Granite models; ffn_gate is crushable in MoE routers."

### Page 6: Quant Profile Builder
**Purpose**: Visual override editor + budget calculator.

Views:
- **Tensor Group List** — drag-and-drop rows organized by tensor type. Each row has current quant level and a dropdown.
- **Budget Calculator** — real-time file size estimate as you adjust. Shows delta from baseline Q4_K_M.
- **Preset Selector** — Code, Reasoning, Balanced, Minimal. Loads preset overrides.
- **Export** — generates tensor_types.txt and optionally queues a build job.

### Page 7: Model Card Generator
**Purpose**: Generate and publish model cards.

Views:
- **Data Pull** — reads from DB (benchmarks, artifacts, ablation).
- **Editor** — text area with generated markdown, editable.
- **Preview** — rendered HTML preview.
- **Upload** — "Upload to HF" button (blocked unless all gates pass; shows checklist if blocked).

---

## UX Rules

1. **No spinners on navigation** — preload data via React Router loaders. Skeleton screens for slow API calls (HF search).
2. **Real-time updates** — WebSocket for job progress, ablation streaming, benchmark running. Fallback to polling.
3. **Everything clickable** — every score, artifact, and log line is a link or expandable.
4. **Dark theme** — base `#0d1117`, accent `#58a6ff` (GitHub-dark inspired, matching current template).
5. **Data density** — compact tables, minimal whitespace, monospace for all data. Grafana × Linear aesthetic.
6. **Keyboard shortcuts** — `g h` home, `g m` models, `g j` jobs, `?` help overlay.

---

## Tech Stack (Frontend)

- **Framework**: React 18+ with Vite
- **Routing**: React Router v6 (loader pattern for data preload)
- **UI**: Tailwind CSS (matching current CSS variables)
- **Components**: shadcn/ui (Button, Card, Table, Tabs, Dialog, Select)
- **Charts**: Recharts (line, bar) + custom SVG (heatmap)
- **State**: React Query (server-state caching, refetch intervals)
- **WebSocket**: `useWebSocket` hook with auto-reconnect
- **Location**: `cerebellum-dev/dashboard/` (standalone, not inside `osmosis/` package)

---

## Acceptance Criteria

- Dashboard loads in <2s. Every local model appears without manual entry.
- Active benchmark appears in the job panel within 30s.
- Each model card shows artifacts, benchmark summaries, audit badges, and docs.
- A shell-run benchmark can be ingested (via scan or manual POST) and appears in the model detail page within 30s.
- HF Upload button is disabled with tooltip explaining which gate failed.
- Sensitivity atlas heatmap renders for any model with ablation data.

---

## First Milestone

1. Scaffold React app in `cerebellum-dev/dashboard/` with Vite + React Router.
2. Implement Pages 1-3 (Home, Model Browser, Model Detail) consuming the API from `SPEC_api_control_plane.md`.
3. Add WebSocket connection to `/ws` for live job updates.
4. Delete the old `index.html` SPA and point `GET /` at the new app.
5. Ship: dashboard shows real model data, job queue, and live updates.
