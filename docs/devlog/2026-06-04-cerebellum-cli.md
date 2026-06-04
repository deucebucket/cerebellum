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
- Added PPL profile support.
- Added live resource/ETA view.
- Added GGUF size visibility.
- Added semantic coloring.
- Added subprocess heartbeat logging for future long-running jobs.
- Added README and screenshot artifacts.
- Implemented first compact grid renderer pass for `cerebellum watch`, merging
  progress/resources, active job/GGUF sizes, and ETA/confidence into a denser
  operations panel.

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
- Compact watch should remain bounded and screenshot-friendly.
- Scrollable research browsing belongs in `watch --tui`.
- The next visual pass should merge progress, timing, resources, activity, and
  health into a heavier framed grid.

## Next work

- Implement compact dashboard grid renderer.
- Add ANSI-aware cell coloring to compact grid rows.
- Improve layer-based ETA once more layers complete.
- Expand API endpoints for run queries, events, measurements, and reports.
- Improve upload/auth UX for HF/GitHub.
- Continue screenshots after layout changes.
