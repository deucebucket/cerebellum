# Cerebellum CLI Toolbox Scope

Cerebellum should be a full quantization toolbox, not only one search strategy.

Core goal: make high-quality mixed-precision GGUF quantization practical on
ordinary systems without large datacenter storage or multiple high-VRAM GPUs.

## Principles

- Portable by default. Do not assume `/var/home/deucebucket`, a specific game
  drive, a specific distrobox, or a specific llama.cpp build path.
- Teachable without being noisy. A new user should be able to install
  Cerebellum, point it at a model, and understand the flow without reading
  source code.
- Resource-aware. The CLI should inspect CPU, RAM, VRAM, CUDA visibility, disk
  space, scratch directories, and available llama.cpp binaries before planning.
- Low-space capable. Large-model workflows must avoid requiring source,
  baseline, all candidates, and final output to coexist on one filesystem.
- Durable. Every long-running action must be resumable after lockup, power loss,
  OOM, or process death.
- Queryable. Append-only JSONL remains source of truth; SQLite is a rebuildable
  index for cross-run comparison.
- Automation-first. Every human view should have a JSON/API equivalent for AI
  agents, web UIs, scripts, HF uploads, and GitHub archival.

## Low-space quant modes

- `single-candidate`: keep source, current baseline, and one candidate only.
- `two-slot-pipeline`: keep source, current baseline, one PPL input, and one
  quantizing `.tmp.gguf` candidate.
- `external-scratch`: put temporary candidates on the largest writable drive
  while compact metadata stays under the run directory.
- `regenerate-winner`: keep only tensor-type state and regenerate accepted GGUF
  when needed if disk cannot retain baseline plus winner.
- `remote-source`: support source GGUF on a different mount or network path.
- `resume-clean`: classify stale `.tmp.gguf` files by manifest ownership before
  deletion.

## System discovery commands

- `cerebellum hill system`
- `cerebellum hill system --json`
- `cerebellum hill plan-space --source-gguf model.gguf --levels q3_K,q5_K`
- `cerebellum hill tutorial`
- `cerebellum hill tutorial --topic low-space`
- `cerebellum hill tips on|off`

System discovery should report:

- OS, hostname, Python version.
- CPU model/count and available RAM.
- GPUs, VRAM total/free/used, CUDA visibility.
- Filesystems, free space, mount points, and recommended scratch roots.
- llama.cpp binaries discovered in `PATH`, common build dirs, and env overrides.
- Container/distrobox availability.
- HF/GitHub auth status without printing secrets.

## Future web/API surface

- HTTP API over the same run store.
- Endpoints for system info, run list, run report, events, DB families/models,
  upload dry-runs, and scheduler state.
- Web UI should be a client of the public API, not a separate source of truth.

## User education

The CLI should explain:

- What each tool does.
- The recommended flow for a new model.
- Why scratch space matters.
- What imatrix, tensor-type files, candidate tests, PPL, and reports are.
- Which outputs are source-of-truth and which are regenerable.
- How to resume after a crash.

Tutorials should be concise and contextual:

- Default human commands may show one short tip and a next command.
- `--json`, `--plain`, `--quiet`, and `CEREBELLUM_TIPS=0` should suppress tips.
- Tips should never obscure errors, paths, decisions, or command outputs.
- A persistent setting can live at `~/.config/cerebellum/config.json`.

## Upload/report scope

- HF upload: model artifacts, benchmark artifacts, summaries, event logs, and
  model-card-ready report sections.
- GitHub upload: compact research metadata, issue-ready summaries, schedule
  manifests, and reproducibility records.
- Infographic export: compact JSON with cards, deltas, component summaries, and
  decision trail.

## Non-goals

- Do not hide failure details behind pretty UI.
- Do not make SQLite the only source of truth.
- Do not assume a full model can be duplicated several times.
- Do not require signing in for local quantization, reports, or API use.

## Portability and Live Dashboard Notes

- Cerebellum must run on normal host installs without distrobox. `--distrobox NAME`
  is only an optional adapter when a user keeps llama.cpp/CUDA/ROCm inside a
  container or toolbox.
- The CLI should auto-explain binary setup: use `llama-quantize` and
  `llama-perplexity` on `PATH`, or pass `--quantize-bin` and `--perplexity-bin`.
- The live dashboard should show current baseline GGUF size, active candidate or
  temp GGUF size, and most recent measured candidate size.
- Progress should always include a bar, raw tensor counts, and percent.
- Live panes should stay bounded for readability. Default to short windows and
  expose limits such as `--events-limit N` and `--measurements-limit N`.
- True scrollable panes require a curses/Textual-style TUI mode. The bounded
  dashboard remains the portable default; a future full TUI can add independent
  scroll focus for events, measurements, and files.
- Failure detection should combine state, event age, process existence, child
  process age, and disk/free-space waits so a running state never looks dead.
