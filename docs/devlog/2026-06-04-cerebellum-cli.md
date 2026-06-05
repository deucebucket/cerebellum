# Devlog: Cerebellum CLI buildout

Date: 2026-06-04

## Context

The goal is a full Cerebellum CLI, not a background-only runner. The CLI needs
to be visible, durable, portable, research-friendly, and capable of running real
model jobs while preserving data integrity.

## Live run

- Model: `gemma-4/gemma-4-12b-it`
- Run ID: `gemma4-12b-cerebellum-q4km-wiki-visible-20260604`
- Run dir:
  `/var/home/deucebucket/games/cerebellum-runs/families/gemma-4/gemma-4-12b-it/sources/google-f16/runs/gemma4-12b-cerebellum-q4km-wiki-visible-20260604`
- Profile: `wiki`
- Current observed progress after CLI work: at least `2/616` tensors locked.
- Current observed PPL after first two locks: `2345.9479`.

## Work completed

- Added `cerebellum` console script.
- Added public package shim under `cerebellum/`.
- Added compact live dashboard.
- Added scrollable curses TUI via `watch --tui`.
- Added run stop/repair command.
- Added portable setup doctor.
- Added run library table and filters.
- Added schedule template and dry-run.
- Added provenance metadata inspection/generation.
- Added finalize command for sidecar/model-card generation and optional GGUF
  metadata injection.
- Added package manifest command and expanded upload sidecar list.
- Expanded local API server for future automation/web UI.
- Added PPL profile support.
- Added live resource/ETA view.
- Added GGUF size visibility.
- Added semantic coloring.
- Added subprocess heartbeat logging for future long-running jobs.
- Added README and screenshot artifacts.
- Implemented first compact grid renderer pass for `cerebellum watch`, merging
  progress/resources, active job/GGUF sizes, and ETA/confidence into a denser
  operations panel.
- Restored semantic colors in compact grid using ANSI-aware padding/clipping.

## Commits pushed to `dev`

- `c541535 feat: add Cerebellum CLI dashboard`
- `0b8587c fix: show Cerebellum live activity health`
- `281d927 feat: improve Cerebellum live dashboard detail`
- `270149c feat: add Cerebellum ETA and resource panel`
- `bd2e413 feat: add scrollable Cerebellum watch TUI`
- `636c521 feat: add Cerebellum portable doctor`
- `d3e999c feat: improve Cerebellum run library browsing`
- `520ee91 docs: refresh Cerebellum CLI dashboard assets`
- `aa72ea8 feat: record subprocess heartbeats`
- `8f01e23 feat: add Cerebellum schedule template dry-run`
- `2079fe6 docs: plan Cerebellum CLI layout redesign`

## Design notes

- Distrobox is optional. It is only used on this workstation because host
  llama.cpp binaries cannot see CUDA libraries directly.
- Cerebellum provenance should be transparent GGUF metadata, not a hidden
  watermark. The goal is attribution and stripped-metadata detection.
- Compact watch should remain bounded and screenshot-friendly.
- Scrollable research browsing belongs in `watch --tui`.
- The next visual pass should merge progress, timing, resources, activity, and
  health into a heavier framed grid.

## Next work

- Implement compact dashboard grid renderer.
- Improve layer-based ETA once more layers complete.
- Expand API endpoints for run queries, events, measurements, and reports.
- API now has first-pass run/event/measurement/report/provenance endpoints.
- Improve upload/auth UX for HF/GitHub.
- Continue screenshots after layout changes.

## Low-space recovery pass

Disk pressure on the Gemma 4 12B run exposed that per-tensor candidate fanout can hold five ~8.6 GiB GGUFs at once. Added low-space run controls to serialize candidate testing and prune measured candidate GGUFs immediately. Added cleanup and rollback commands so logs/progress remain durable while partial tensor temp can be discarded and resumed from the last locked tensor/layer boundary.

## Candidate pruning refinement

Adjusted low-space handling after live Gemma 4 observation: normal mode remains overlapped for speed, but measured losers are now pruned immediately by default. Disk gating now estimates one candidate GGUF from the current baseline size and requires that plus a 10 GiB hard floor before starting the next quant job.

## Backup and partial cleanup pass

Added metadata backup plumbing for catastrophic recovery. `--backup-root` mirrors small critical files outside the heavy temp/artifact directory, and `cerebellum backup` can be run manually. Partial cleanup is explicit and guarded so active PPL/quant files are not removed by mistake. Rollback now warns that artifact state and JSON state can diverge if rolling back after the baseline GGUF has already been promoted.

## Rollback artifact correction

Rollback is now operational rather than state-only. It rewrites the tensor type map and sets `baseline_invalid_after_rollback`, causing the next resume to rebuild and remeasure the baseline before continuing from the rolled-back boundary.

## Storage visibility in watch UI

Added run storage footprint to the live dashboards. The user can now see candidate temp growth, artifact size, active GGUF size, disk free space, and the hard floor from inside Cerebellum instead of checking `du`/`df` separately.

## Resume and recovery planner

Added first-class resume/recover commands. This removes the need to preserve or reconstruct the original run command after a lockup. `recover` reports whether partial temp cleanup is safe, shows disk pressure, and emits the exact commands for resume, metadata backup, and partial cleanup.

## Tutorial and AI API pass

Added tutorial coverage for the new operational controls and expanded the local API so an AI agent can inspect runs, plan recovery, read tutorials, get command templates, export reports, inspect system/space, and package metadata without shell scraping. Destructive actions remain CLI-only for now.

## API schema endpoint

Added `/schema` so AI clients can discover available read-only endpoints and state-changing CLI templates without prompt-guessing. This is the bridge toward future tool calling and web UI integration.

## Big-run smoke pass

Ran smoke checks against the live Gemma 4 12B run instead of launching a tiny model. Found and fixed the top-level command wrapper missing new commands. Confirmed `recover`, `watch --once`, `status`, API read-only endpoints, backup, cleanup guard, rollback dry-run, report/export/package, system, doctor, and plan-space paths. Doctor initially missed llama.cpp binaries because they were outside PATH; added portable common build-path discovery.

## Neutral deltas and size precision

Live Gemma 4 showed normalization tensor tests with identical PPL at current precision. Added an explicit `=` marker for neutral deltas and denser GGUF size formatting so small per-tensor file-size differences are visible in the dashboard.

## Size delta coloring

Added baseline-relative candidate size coloring to the compact dashboard so users can visually track whether candidate GGUFs are shrinking or growing versus the current baseline.

## Self-test command

Added a built-in read-only smoke path so future validation does not require manually chaining help, tutorial, recover, report, package, and API calls. This supports big-run smoke without starting a separate tiny model.

## Cerebellum imatrix surface

Folded imatrix into the public Cerebellum CLI. Users can now run `cerebellum imatrix` instead of legacy `python -m osmosis.imatrix_stream`. The old modules remain implementation compatibility while the package rename is in flight.

## Project-aware imatrix flow

Connected imatrix generation to the same Cerebellum project tree used by runs. Each model source can now own imatrix data, run directories, reports, and a `cerebellum_project.json` manifest so the workflow naturally continues from imatrix into Cerebellum quant search.

## Project browser

Added a project browser so the new imatrix project layout is usable from the CLI and API. Existing runs without `cerebellum_project.json` still show up via run manifest fallback.

## Live Gemma 4 12B resume pass

Resumed `gemma4-12b-cerebellum-q4km-wiki-visible-20260604` after a stale partial
PPL process. Backed up metadata, cleaned only the partial tensor temp, and
restarted the run in low-space mode. While monitoring, fixed runner detection
for `cerebellum resume`, added wall-clock ETA completion time, preserved event
ID continuity across resumed processes, and redacted secret-like environment
values from process command display. The live run continued through
`blk.0.post_attention_norm.weight` q3_K PPL and into q2_K quantization.

## Norm tensor rollback and quant override fix

The live Gemma 4 12B run exposed a real pipeline bug: norm tensors were being
tested even though llama.cpp leaves `*_norm.weight` tensors unquantized. This
created no-op candidate GGUFs with repeated PPL/size values. Backed up the bad
trace, rolled the run from 12 locked tensors back to the last valid 7 locked
tensors, and resumed from the rebuilt baseline.

Fixed Cerebellum tensor discovery and explicit tensor-file filtering to skip
quantizer-excluded tensors. Tensor-type files now use exact escaped regex
patterns for current llama.cpp matching. Watch/recover now anchor active state
to the latest run epoch after rollback. Post-fix validation on
`blk.1.ffn_down.weight` produced distinct q3/q2 sizes and PPL deltas.

## Gemma 4 source conversion gotcha

Recorded the separate Gemma 4 12B source-conversion requirement: the HF class is
`Gemma4UnifiedForConditionalGeneration`, with the text backbone under
`model.language_model.*`. llama.cpp's Gemma4Model already strips that prefix,
but the converter must register the architecture name on Gemma4Model before
building the F16 GGUF. A valid converted source should report
`general.architecture=gemma4` and llama.cpp-style `blk.*` tensor names.
