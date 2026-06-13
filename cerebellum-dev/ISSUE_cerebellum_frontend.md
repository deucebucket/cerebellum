# Cerebellum Command Center — Full Frontend Spec

## Vision
The backbone of Cerebellum development. A professional, polished web UI that is the complete pipeline — from model discovery to quantization to benchmarking to publishing. Everything visual, everything at a glance, everything saveable. Built for someone who thinks in results, not config files. Integrates with PMS for local/remote LLM inference, model management, and AI-assisted analysis.

---

## Module 1: Model Discovery & Download
- Browse HuggingFace trending/new models (HF API)
- Filter by architecture, size, license, task, popularity
- Show model cards, parameter counts, architecture details
- One-click download: bf16 GGUF + auto-generate imatrix
- Track download progress with ETA
- Auto-detect tensor groups and layer count for any architecture
- **Compare against existing Cerebellum quants** (yours + community)
- Suggest models to quantize based on popularity + architecture compatibility

## Module 2: Autopilot Control Panel
- **Launch** ablation runs on any downloaded model
- Configure: sample layers, skip groups, thresholds, base quant, disk budget
- **Real-time streaming** of ablation progress (WebSocket)
- **Live sensitivity heatmap**: layers × tensor groups, color intensity = PPL delta
- Pause/resume/cancel runs
- Disk space monitor with auto-cleanup gauge
- GPU utilization overlay (nvidia-smi integration)
- Queue multiple models for sequential autopilot runs
- Full log viewer with search/filter

## Module 3: Benchmark Suite
- **Run any benchmark** on any model: PPL, HumanEval, ARC, HellaSwag, MMLU-Redux, GSM8K, TruthfulQA
- Runnable and **re-runnable** — click to re-run, results versioned and saved
- **Benchmark OTHER models** — pull any GGUF from HF, run full suite, get baselines
- Side-by-side comparison tables (v4 vs v5 vs competitor vs bf16 baseline)
- Historical benchmark tracking with trend charts
- Export results as markdown tables or JSON
- **PMS integration**: deploy model to PMS model folder, run inference benchmarks through PMS API
- Move/copy models to PMS model folder for serving

## Module 4: Sensitivity Atlas
- **Cross-model visualization**: which tensor types are universally sensitive vs crushable
- Per-model heatmaps with drill-down to individual tensor results
- Overlay multiple models to find patterns
- Predict sensitivity curves for untested models based on atlas data
- **Task-specific view**: show which tensors affect code vs reasoning vs knowledge
- Export override files for any configuration

## Module 5: Task-Specific Quantization Builder
- Visual override editor: drag tensor groups between quant levels
- Budget calculator: shows file size impact of each change in real-time
- **Preset profiles**: Code, Reasoning, Balanced, Minimal
- Build custom profiles by selecting which benchmarks to optimize for
- A/B test profiles: quantize both, benchmark both, show winner
- One-click quantize from profile

## Module 6: Model Card Generator
- **PMS LLM integration**: use local/remote LLM to auto-generate model cards
- Pull all data from Cerebellum: benchmarks, override count, ablation insights, file size
- Generate professional markdown model cards
- Preview before publishing
- One-click upload to HuggingFace (model + model card)
- Template system for consistent branding

## Module 7: Data Explorer & Chat
- **Ask questions about your data**: "Which tensor is most sensitive in Gemma 4?" "How does attn_v compare across models?"
- PMS chat integration — query local or remote LLM about your ablation data
- All experiment data in a queryable format
- Visualize any slice of data on demand
- Export charts and tables

## Module 8: Dashboard (Home)
- **At-a-glance status**: running jobs, recent results, disk usage, GPU status
- Latest HF model releases in target architectures
- Quick-launch buttons for common tasks
- Recent benchmark results summary
- Notification feed: ablation milestones, benchmark completions, model downloads

---

## Tech Stack
- **Backend**: FastAPI (Python — same ecosystem as all our scripts)
- **Frontend**: React + Tailwind CSS + shadcn/ui (professional, dark mode, responsive)
- **Charts**: Recharts or Plotly for heatmaps, bar charts, trend lines
- **Real-time**: WebSocket for live ablation streaming
- **Database**: SQLite (local, portable, no setup) — stores all results, benchmarks, atlas data
- **PMS Integration**: HTTP API to localhost:7700 for LLM chat, model management
- **HF Integration**: `huggingface_hub` Python API + HF REST API
- **GPU Monitoring**: nvidia-smi polling

## PMS Integration Points
- `mcp__pms__pms_chat` — Ask LLM about data, generate model cards
- `mcp__pms__pms_list_models` — Show local models
- `mcp__pms__pms_scan_models` — Detect new models in folder
- `mcp__pms__pms_start_service` / `pms_stop_service` — Manage model serving
- `mcp__pms__pms_system_status` — GPU/system health
- Model folder: copy/move quantized GGUFs to PMS model directory for serving

## Design Requirements
- **Professional as fuck** — dark theme, clean typography, subtle animations
- **Data-dense but readable** — no wasted space, information hierarchy
- **Name-worthy** — the UI should look like it belongs to a real product
- **Accessible** — no ML PhD required to understand what's happening
- **Fast** — no spinners, instant navigation, lazy-load heavy charts

## MVP (Build First)
1. Dashboard with running job status + GPU monitor
2. Autopilot launcher with live log streaming + sensitivity heatmap
3. Benchmark runner with results table + comparison view
4. Model browser + one-click download

## Phase 2
5. Sensitivity atlas with cross-model overlay
6. Task-specific quant builder with budget calculator
7. PMS chat integration for data questions
8. Model card generator + HF upload

## Phase 3
9. Full data explorer with natural language queries
10. Predictive sensitivity curves
11. Community sharing (override files, atlas data)
12. Multi-GPU job scheduling
