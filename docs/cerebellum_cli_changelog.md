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
- README screenshots and CLI workflow documentation.
- Compact watch grid renderer with heavier framing and side-by-side operations,
  resources, GGUF sizes, and ETA cells.

### Changed

- New generated artifacts use `cerebellum_*` names instead of internal legacy
  names.
- User-facing docs and help now use Cerebellum terminology.
- Distrobox is documented as an optional execution adapter, not a requirement.
- Run library browsing no longer requires manual SQLite import for basic status.

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
