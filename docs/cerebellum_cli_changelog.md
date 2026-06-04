# Cerebellum CLI changelog

## 2026-06-04

### Added

- Public `cerebellum` console entrypoint.
- `cerebellum run` as the user-facing quant search command.
- `cerebellum watch` compact live dashboard.
- `cerebellum watch --tui` scrollable curses UI with panes for events,
  measurements, processes/GPU, and files.
- `cerebellum stop` for clean stop/repair of run state.
- `cerebellum doctor` for portable setup checks.
- `cerebellum runs` filesystem run-library browser with filters.
- `cerebellum schedule --template` and `--dry-run`.
- `cerebellum provenance` for transparent GGUF metadata inspection/generation.
- `cerebellum finalize` for final reports, model-card text, provenance
  sidecars, and optional GGUF metadata injection.
- `cerebellum package` for a portable upload manifest.
- Expanded local API endpoints for runs, run detail, events, measurements,
  reports, and provenance.
- PPL profiles: `wiki`, `agentic`, `code`, `math`, `dialogue`,
  `all-around`, and `custom`.
- Live dashboard panels for:
  - progress bar and percent
  - active tensor/job
  - current/active/recent GGUF sizes
  - activity health and process heartbeat
  - ETA/resource panel
  - recent measurements
  - event stream
- Semantic dashboard coloring for labels, PPL, quant levels, tensors, deltas,
  timers, warnings, and active processes.
- Future-run subprocess heartbeat events for long quant/PPL jobs.
- Transparent `cerebellum.*` metadata block generation for attribution and
  auditability.
- ANSI-aware compact grid padding so semantic colors work without corrupting
  table alignment.
- README screenshots and CLI workflow documentation.
- Compact watch grid renderer with heavier framing and side-by-side operations,
  resources, GGUF sizes, and ETA cells.

### Changed

- New generated artifacts use `cerebellum_*` names instead of internal legacy
  names.
- User-facing docs and help now use Cerebellum terminology.
- Distrobox is documented as an optional execution adapter, not a requirement.
- Run library browsing no longer requires manual SQLite import for basic status.
- PPL profile lookup now prefers portable locations:
  `CEREBELLUM_CORPUS_ROOT`, `./corpora`, and
  `~/.cache/cerebellum/corpora`.
- Upload dry-runs now include finalize/provenance sidecars when present.

### Validated

- Real Qwen3-0.6B smoke run with llama.cpp quant/PPL, watched live, then stopped.
- Real Gemma 4 12B run launched through `cerebellum run` and watched live.
- Gemma run remained resumable/active while CLI changes were developed.
- Dashboard screenshots regenerated and committed.

### Known gaps

- Compact dashboard grid exists, but still needs ANSI-aware per-cell coloring so
  semantic colors can be applied without breaking alignment.
- ETA is rough early in a run and should improve with tensor/layer history.
- Full API depth, upload UX, and eventual web UI remain roadmap work.

## 2026-06-04 low-space and recovery controls

- Added targeted tensor selection flags for `cerebellum run`: `--layers` and `--tensor-regex`.
- Added low-space execution flags: `--low-space`, `--serial-candidates`, and `--prune-measured-candidates`.
- Added `cerebellum cleanup` dry-run/execute command for safe temp cleanup while preserving state, events, candidates, checkpoints, reports, and manifests.
- Added `cerebellum rollback` for clean resume boundaries: first N locked tensors, before a layer, or newest partial layer removal.
- Fixed compact dashboard overflow by wrapping box content and splitting resource/job rows.
- Colored `better` and `worse` measurement markers separately from tensor names.

## 2026-06-04 candidate pruning policy

- Kept overlapped CPU/GPU candidate flow as the default for speed.
- Changed measured-candidate cleanup so non-winning GGUFs are deleted as soon as their PPL result is recorded, while preserving the best-so-far candidate for possible locking.
- Added `--keep-measured-candidates` for diagnostic runs that intentionally retain all measured candidates until tensor end.
- Added `--hard-free-floor-gb` with a default of `10.0`; Cerebellum will wait before launching another quant job unless enough space remains for the estimated candidate plus the hard floor.

## 2026-06-04 backup and partial cleanup controls

- Added `cerebellum backup RUN_DIR --to BACKUP_ROOT` to copy critical metadata, event logs, candidate logs, tensor type files, timing data, and checkpoints to a separate location.
- Added `cerebellum run --backup-root BACKUP_ROOT` for automatic metadata mirroring during long runs.
- Added `cerebellum cleanup --partials` for deleting partial tensor temp directories after a stopped/interrupted run; it refuses to touch partial temp while the runner is active unless `--force` is used.
- Clarified rollback output: state rollback is durable, but if the baseline GGUF has already advanced beyond the rollback point, the model artifact must be rebuilt from the rolled-back tensor type map before resuming.

## 2026-06-04 rollback baseline rebuild

- Rollback now writes a rolled-back `cerebellum_current_tensor_types.txt` and marks the current baseline GGUF invalid.
- The next `cerebellum run` resume rebuilds `artifacts/current_baseline.gguf` from the rolled-back tensor map before continuing, preventing JSON state and GGUF artifact divergence.

## 2026-06-04 dashboard storage visibility

- Added live `tmp` and `artifacts` footprint display to compact and TUI watch views.
- Added hard free-space floor display to the compact operations grid so disk gating is visible during long runs.

## 2026-06-04 resume and recovery planner

- Added `cerebellum resume RUN_DIR` so an interrupted run can be restarted from durable manifest/state without reconstructing the original long command.
- Resume supports operational overrides for `--low-space`, `--backup-root`, `--distrobox`, `--min-free-gb`, and `--hard-free-floor-gb`.
- Added `cerebellum recover RUN_DIR` to print run status, active runner detection, last locked tensor, partial temp directories, storage footprint, and exact resume/cleanup/backup commands.

## 2026-06-04 tutorials and AI API expansion

- Expanded `cerebellum tutorial` with concrete topics for recovery, low-space operation, targeted testing, API automation, and provenance.
- Added read-only API endpoints for `/recover`, `/export`, `/package`, `/system`, `/space`, `/tutorial`, and `/commands`.
- API now exposes AI-safe recovery plans and command templates while keeping destructive actions as CLI-only operations.

## 2026-06-04 API schema endpoint

- Added `/schema` to expose endpoint parameters, return intent, safety policy, and tutorial topics for AI clients.
- Updated API startup endpoint list to include `/schema`.

## 2026-06-04 big-run smoke and wrapper fixes

- Fixed the top-level `cerebellum` wrapper so new commands are reachable: `resume`, `recover`, `cleanup`, `rollback`, and `backup`.
- Added new commands to the top-level help text.
- Smoke-checked syntax, tutorials, Gemma live status/watch/recover, API read-only endpoints, report/export/package, metadata backup, cleanup guard, rollback dry-run, system, doctor, and space planning.
- Improved llama.cpp binary discovery by checking portable local build locations in addition to environment variables and PATH.

## 2026-06-04 neutral delta and precise size display

- Recent measurements now show `=` for exactly neutral PPL deltas instead of leaving the marker blank.
- Neutral deltas are cyan, improvements remain green, and regressions remain red.
- GGUF size display in watch views now uses denser precision such as `8.001 GiB` so tiny tensor changes do not appear as identical rounded `8.0 GiB` values.

## 2026-06-04 size delta coloring

- Compact watch measurement sizes are now colored against the current baseline GGUF size: blue for smaller candidates, orange for larger candidates, cyan for equal size.

## 2026-06-04 read-only self-test command

- Added `cerebellum self-test` for non-mutating CLI/API smoke checks.
- `self-test --run-dir RUN_DIR` checks run state, manifest, recovery payload, report payload, and package payload without launching quantization or changing run data.
- Added `/self-test?run_dir=RUN_DIR` to the read-only API surface.
