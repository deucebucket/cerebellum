# Cerebellum Architecture Map & Reorganization Plan

Status: PROPOSAL — nothing has been moved or deleted. Evidence untouched.
Scope: `/var/home/deucebucket/ai-drive/cerebellum` (+ knowledge surfaces of sibling repos).

READ-ONLY ZONES at survey time (other agents actively writing — inventoried, untouched):
`cerebellum-gemma4-12b/`, `cerebellum-north-mini-code/` (exists only at `/var/home/deucebucket/games/cerebellum-north-mini-code`), `cerebellum-dev/modal_harness/`, `/var/home/deucebucket/games/cerebellum-*`.

---

## 1. Current state — top-level inventory (2026-06-12)

Repo total is dominated by two items: `cerebellum-dev/conch-poc` (32G, brainloop's
nested repo with GGUFs/checkpoints) and `scripts/.bench-venv` (8.2G regenerable venv).

### ENGINE CODE (the product)

| Item | Size | Files | Notes |
|---|---:|---:|---|
| `cerebellum/` | 2.3M | 52 | Public CLI package (entrypoint, imatrix wrapper, `_legacy/`). |
| `osmosis/` | 164K | 41 | Engine package, rename-in-flight. `hillstep.py` = deprecated method, kept as optional add-on. `dashboard/` gitignored. |
| `scripts/` | 8.2G* | 78,777* | *~316K / ~50 files of real scripts; the rest is `.bench-venv` (8.2G, 78,726 files) + `__pycache__`. Bench runners, ablation drivers, audit tools. |
| `tests/` | 1.2M | 17 | Local-only (gitignored), pytest. |
| `pyproject.toml`, `.gitignore`, `LICENSE` | — | 3 | Root config. |

### EVIDENCE (preservation-locked — never move without verified backup)

| Item | Size | Files | Notes |
|---|---:|---:|---|
| `osmosis-gemma4-e2b/` | 3.1G | 78 | Contains GGUF-scale artifacts. |
| `osmosis-gemma4-26b/` | 1.4G | 585 | v6 imatrix + overrides live here (shipped-model evidence). |
| `osmosis-qwen35-122b/` | 373M | 112 | |
| `cerebellum-qwen35-9b/` | 330M | 8,997 | High file count (per-question bench detail). |
| `osmosis-qwen36-27b/` | 186M | 382 | v4 overrides + imatrix (shipped-model evidence). |
| `benchmarks/` | 120M | 96 | Released-model benchmark data, public. |
| `cerebellum-gemma4-12b/` | 79M | 50 | **READ-ONLY ZONE (active agent).** |
| `osmosis-gemma4-e4b/` | 43M | 77 | |
| `osmosis-qwen36-35b/` | 35M | 216 | |
| `osmosis-qwen3-30b/` | 34M | 53 | |
| `cerebellum-gemma4-codex/` | 32M | 49 | |
| `cerebellum-granite41-30b/` | 23M | 25 | |
| `osmosis-granite4-h-small/` | 11M | 31 | |
| `cerebellum-qwen36-27b-heretic/` | 11M | 23 | |
| `osmosis-qwen35-9b/` | 9.6M | 22 | |
| `osmosis-qwen3-32b/` | 4.4M | 19 | |
| `osmosis-granite41-30b/` | 4.2M | 17 | |
| `cerebellum-gemma4-e2b-heretic/` | 2.9M | 10 | |
| `cerebellum-qwen36-35b-heretic/` | 844K | 19 | |
| `osmosis-qwen3-14b/` | 528K | 25 | |
| `cerebellum-gemma4-e4b-heretic/` | 324K | 8 | |
| `cerebellum-glm47-flash/` | 44K | 4 | |
| `spaces/` | 12K | 3 | HF space artifact (qwen36-cerebellum). |

### DOCS

| Item | Size | Files | Notes |
|---|---:|---:|---|
| `docs/` | 1.1M | 77 | Mixed public design docs (memory controller, tensor backend patch, benchmark protocol) + gitignored dated findings. Healthy; the gitignore split handles public/private. |
| `README.md`, `AGENTS.md`, `CLAUDE.md`, `llms.txt` | 56K | 4 | Root knowledge surface. |

### FACTORY — `cerebellum-dev/` (32G total; 35,877 files; git-tracked on `dev` remote only, 156 tracked files)

Subclassification of its ~48 top-level files + 12 dirs:

| Class | Items |
|---|---|
| **Canon truth** | `WINNING_METHOD.md`, `DEAD_PATHS.md`, `BACKLOG.md`, `README.md` (factory index), `ARTIFACT_INVENTORY.md` (+ `artifact_inventory.json`, 32M), `PIPELINE_VERIFICATION_2026-06-12.md`, **`knowledge/` (this dir, new)** |
| **Devlogs** | `DEVLOG_2026-05-01_qwen36_35b_start.md`, `DEVLOG_2026-05-01_router_road_mapping.md`, `DEVLOG_2026-05-02_granite4_h_small_start.md`, `DEVLOG_2026-06-03_gemma4-12b.md`, `FULL_EXPERIMENT_LOG.md`, `REBRAND_LOG.md`, `rtx3090_training_speed_notes.md` |
| **Specs** | `SPEC_agent_routing.md`, `SPEC_agent_runner.md`, `SPEC_api_control_plane.md`, `SPEC_benchmark_gates.md`, `SPEC_dashboard_ux.md`, `UI_DESIGN_SPEC.md` |
| **Issues (drafts)** | `ISSUE_cerebellum_api.md`, `ISSUE_cerebellum_frontend.md`, `ISSUE_conch_shell_architecture.md`, `ISSUE_reasoning_drift_benchmark.md`, `ISSUE_tensor_bridges.md` |
| **Playbooks / reconstruction** | `OG_CEREBELLUM_RECONSTRUCTION_GUIDES_2026-06-07.md`, `QWEN36_27B_V1_TO_V4_PLAYBOOK_2026-06-07.md`, `QWEN3_0_6B_OG_V4_METHOD_SMOKE_2026-06-07.md`, `HERETIC_FLEET_PLAN.md` |
| **Forensics** | `forensics_2026-06-11/` (204K), `geminichat-6-9-2026-1042pm.txt` |
| **Research / planning** | `RESEARCH_IDEAS.md`, `COMPARISON_122B.md`, `COMMUNITY_FEEDBACK_2026-06-12.md`, `community_feedback_data_2026-06-12/` (180K) |
| **Data summaries** | `ablation_cross_model_{records.csv,summary.json,summary.md}`, `benchmark_compare_qwen36_sample.{json,md}` |
| **Harnesses / pipeline code** | `cerebellum_autopilot.py`, `cerebellum_budget.py`, `streaming_quantize.py`, `build_calibration_corpus.py`, `imatrix_vs_ablation.py`, `test_reasoning_loop.py`, `run_gate2_{9b,granite30b,pipeline}.sh`, `modal_harness/` (**READ-ONLY ZONE**), `hf_jobs_eval/`, `tools/`, `tests/`, `tool_tests/`, `calibration/` |
| **Member project** | `conch-poc/` (32G) — **brainloop's home**: nested git repo, own remote (`cerebellum-brainloop.git`, PUBLIC), own CLAUDE.md, ops-log law. Contains 15G + 5.8G brainloop GGUFs, 5.4G rag-experiment, ~5G of `checkpoints-*` dirs. Inventory only — managed by brainloop's own process. |
| **Side experiments** | `sparse-upcycling/` (17M), `history/` (6.3M), `planka-backups/` (464K) |

### SCRATCH-OR-DEAD (root-level mess — the actual cleanup surface)

| Item | Size | Verdict (proposed) |
|---|---:|---|
| `scripts/.bench-venv/` | **8.2G** | DELETE-CANDIDATE (regenerable venv; recreate with a pinned requirements file first; confirm no bench run is live) |
| `.opencode/` | 57M | DELETE-CANDIDATE after salvaging `cerebellum-db.md` + `openpets.md` notes → archive (bulk is node_modules) |
| `checkpoints/` | 9.9M | ARCHIVE (hillstep-era checkpoints; deprecated path but it's evidence of DP-1) |
| `db/` | 368K | ARCHIVE (hillstep SQLite + import scripts; deprecated path evidence) |
| `cerebellum_logo*.png`, `cerebellum_banner_500kb.png` (6 files at root) | 7.9M | MOVE → `assets/` (needs README link check — approval) |
| `taildrop/` | 536K | ARCHIVE (one reddit screenshot) |
| `wikitext-test.txt` | 1.3M | KEEP at root (live PPL corpus used by scripts) |
| `Understanding External GPUs ... Gemini.pdf` | 164K | MOVE → `research/external/` |
| `placebo.md` | 12K | MOVE → `research/` (essay, not a project doc) |
| `start.md` | 12K | ARCHIVE (Apr 30 onboarding doc, superseded by CLAUDE.md + knowledge/INDEX.md; salvage anything not yet in canon first) |
| `benchmark_results/` (root) | 80K | MOVE contents → proper per-model evidence dirs (megazord, qwen2.5-coder-7b samples), then remove empty dir |
| `watch_20260604-141238` (file) | 16K | DELETE-CANDIDATE (stale hillstep watch dump) |
| `run_overnight.sh`, `run_phase2.sh` | 8K | ARCHIVE (April-era one-shots) |
| `llama.log`, `speculative.log`, `osmosis-granite41-30b-download.log` | 12K | DELETE-CANDIDATE (stray logs) |
| `osmosis.egg-info/`, `cerebellum.egg-info/` | 68K | DELETE-CANDIDATE (regenerated by pip; already gitignored) |
| `.ruff_cache/`, `.pytest_cache/`, `.playwright-mcp/`, `scripts/__pycache__/` | ~450K | DELETE-CANDIDATE (caches) |
| `configs/`, `evalplus/`, `.codex/`, `.agents/` (empty dirs) | 0 | DELETE-CANDIDATE (empty) |
| `cerebellum-dev/conch-poc/brainloop-ggml-weights.bak` | 393M | DELETE-CANDIDATE **but defer to brainloop** (inside its repo; flag, don't touch) |
| `cerebellum-dev/geminichat-6-9-2026-1042pm.txt` | 24K | ARCHIVE → `cerebellum-dev/forensics_2026-06-11/` (it's forensic source material) |

**Delete-candidate total: ~8.66 GB** (8.2G venv + 57M .opencode + 393M brainloop .bak + ~1M small items). All listed for user approval only — nothing deleted.

### UNKNOWN / verify with user

| Item | Size | Question |
|---|---:|---|
| `research/` | 24K | `memory.md` ("Brain in a Jar" philosophy) + `external/` — looks like a proto-knowledge dir. Proposal: fold `research/memory.md` → `knowledge/` or keep as essay shelf alongside `placebo.md`. |
| `tools/` (root) | 52K | Only `__pycache__` — orphaned? Whose import path? |
| `.cache/huggingface` | 4K | Stray HF cache root — harmless, candidate for deletion. |
| Root `public` git remote → cerebellum-brainloop.git | — | Why does the *parent* repo have brainloop's remote? Leak risk if someone pushes the wrong branch. Recommend removing the remote from the parent (approval). |

---

## 2. Sibling knowledge surfaces (inventory only — not reorganized)

| Repo | Location | CLAUDE.md | Other knowledge |
|---|---|---|---|
| brainloop | `cerebellum/cerebellum-dev/conch-poc` (nested repo, public remote) | yes, at its root | RESEARCH_LOG.md, RESULTS.md, EXPERIMENTAL_PATHS.md, DEADBLOCK_STATUS.md, README.md, append-only ops log ("Logging Is Law") |
| clanker | `/var/home/deucebucket/ai-drive/clanker` | yes (+ stale `CLAUDE.md.bak`) | README, SPEC.md, CHANGELOG, docs/, benchmarks/, llms.txt/llms-full.txt, its own `placebo.md` copy; project memory holds `user-working-style.md` |
| clanker-soul | `/var/home/deucebucket/ai-drive/clanker-soul` | yes + AGENTS.md | README, CHANGELOG, docs/, logs/, examples/ |
| clanker-drift | `/var/home/deucebucket/ai-drive/clanker-drift` | **no** (AGENTS.md only) | CLANKER_DRIFT_INTEGRATION.md |
| games partition | `/var/home/deucebucket/games/cerebellum-*` | — | **READ-ONLY ZONES** (active build areas: north-mini-code, runs, staging, quants, etc.) |

---

## 3. Proposed target architecture

Principles: evidence never moves; the git public/private boundary is untouched
(knowledge canon lives under `cerebellum-dev/` → dev remote only); running
campaigns (gemma4-12b, north-mini-code, modal_harness, games/*) see zero path changes.

```
cerebellum/                          # repo root — stays the head project
├── CLAUDE.md / AGENTS.md            # + new ALWAYS-READ-FIRST block (drafted in §5)
├── README.md, LICENSE, llms.txt, pyproject.toml, wikitext-test.txt
├── assets/                          # NEW — logos/banner moved off root (approval)
├── cerebellum/  osmosis/  scripts/  tests/        # engine, unchanged
├── docs/                            # unchanged (gitignore already splits public/private)
├── benchmarks/  osmosis-*/  cerebellum-*/         # EVIDENCE — frozen in place
├── research/                        # essay shelf: placebo.md, memory.md, external/ (+ eGPU pdf)
├── archive/                         # NEW, gitignored — dated bins for scratch
│   └── 2026-06-root-sweep/          # start.md, run_*.sh, taildrop/, checkpoints/, db/, watch file, salvaged .opencode notes
└── cerebellum-dev/                  # FACTORY (dev remote only)
    ├── knowledge/                   # ★ CANON — INDEX, CURRENT_METHOD, PROJECT_HIERARCHY, ARCHITECTURE_MAP, STORY (incoming)
    ├── WINNING_METHOD.md  DEAD_PATHS.md  BACKLOG.md  README.md  ARTIFACT_INVENTORY.md
    ├── devlogs/   specs/   issues/  playbooks/      # NEW subdirs — file-by-file moves of the loose .md sprawl (approval; low risk, git-tracked renames)
    ├── forensics_2026-06-11/  community_feedback_data_2026-06-12/
    ├── modal_harness/ (read-only)  hf_jobs_eval/  tools/  tests/  calibration/
    └── conch-poc/                   # brainloop — untouched, its own repo
```

Canon placement decision: `cerebellum-dev/knowledge/` (not a root `KNOWLEDGE/`)
because cerebellum-dev is already the private, dev-remote-tracked zone — a root
KNOWLEDGE/ would be a new public-leak surface and another root item. The Story doc
slots at `cerebellum-dev/knowledge/STORY.md`; `BACKLOG.md` stays at
`cerebellum-dev/BACKLOG.md` and is linked from INDEX.md.

### Proposed moves table

| From | To | Risk | Needs approval? |
|---|---|---|---|
| `cerebellum-dev/knowledge/` creation + 4 canon files | — | none (additive) | **DONE** (this change) |
| Root logo/banner PNGs (6) | `assets/` | README/HF links may reference root paths — grep first | YES |
| `placebo.md`, eGPU PDF | `research/` | none | no (cosmetic) |
| `start.md`, `run_overnight.sh`, `run_phase2.sh`, `taildrop/`, `watch_20260604-141238`, `checkpoints/`, `db/` | `archive/2026-06-root-sweep/` | low — verify no script references `checkpoints/` or `db/` (hillstep is deprecated but the dashboard may read `db/cerebellum.db`) | YES (db/, checkpoints/); no (rest) |
| Root `benchmark_results/*.jsonl` | matching evidence dirs | low — they're bench evidence; pick correct model dir | YES (evidence-adjacent) |
| `cerebellum-dev` loose .md sprawl | `devlogs/` `specs/` `issues/` `playbooks/` subdirs | low — git-tracked renames on dev; update README.md factory index links | YES (one batch PR on dev) |
| `scripts/.bench-venv` | delete after freezing `requirements-bench.txt` | medium — active bench runs use it | YES + confirm idle |
| Parent repo `public` remote removal | — | removes a wrong-push foot-gun | YES |
| Evidence dirs (`osmosis-*`, `cerebellum-*`, `benchmarks/`) | **no moves** | — | locked |

### Rename-in-flight note

The `osmosis/` → `cerebellum/` package rename stays a separate, code-level task
(imports, pyproject, per-model dir names). Out of scope here; tracked in BACKLOG.md.

---

## 4. Anti-forgetting wiring (drafted, NOT applied)

Mechanism: every repo's CLAUDE.md opens with a 6-line pointer block. Short enough
that it never gets skimmed past; everything deep lives one hop away in INDEX.md.

### Draft A — cerebellum `CLAUDE.md` (insert at top, right after the title line)

```markdown
## ALWAYS READ FIRST
Canonical truth lives in `cerebellum-dev/knowledge/INDEX.md` — read it before any
build, benchmark, publish, or "future work" suggestion. Non-negotiables:
the OG group-first bench-gated formula is the method (`knowledge/CURRENT_METHOD.md`);
hillstep exhaustive mode is deprecated; PPL never gates alone; evidence dirs are
preservation-locked; no AI attribution in public output. If a doc conflicts with
the knowledge canon, the canon wins.
```

### Draft B — brainloop (`cerebellum-dev/conch-poc/CLAUDE.md`, insert after "What This Is")

```markdown
## Umbrella context
Brainloop is a member project of Cerebellum (the head). Cross-project truth:
`../knowledge/INDEX.md` (hierarchy, current method, who-offers-what-data).
This repo is PUBLIC and nested inside the private cerebellum-dev tree — git here
hits the brainloop remote, never cerebellum's. No AI attribution in anything public.
```

### Draft C — clanker (`/var/home/deucebucket/ai-drive/clanker/CLAUDE.md`, insert after "What This Is")

```markdown
## Umbrella context
Clanker is a member project of the Cerebellum umbrella (head repo:
`/var/home/deucebucket/ai-drive/cerebellum`). Cross-project hierarchy and shared
truth: `cerebellum/cerebellum-dev/knowledge/INDEX.md` (read PROJECT_HIERARCHY.md
there before cross-project work). No AI attribution in public output.
```

### Draft D — clanker-soul (`/var/home/deucebucket/ai-drive/clanker-soul/CLAUDE.md`, same position; mirror into its AGENTS.md)

```markdown
## Umbrella context
Clanker-soul is a member project of the Cerebellum umbrella (head repo:
`/var/home/deucebucket/ai-drive/cerebellum`). Cross-project hierarchy and shared
truth: `cerebellum/cerebellum-dev/knowledge/INDEX.md`. Emotional-coordinate data
contracts with clanker are described in PROJECT_HIERARCHY.md there.
No AI attribution in public output.
```

(Optional Draft E — clanker-drift has no CLAUDE.md; if one is created, use Draft C
with the name swapped.)

Also recommended (approval): add one line to the cerebellum auto-memory MEMORY.md
pointing at `cerebellum-dev/knowledge/INDEX.md` as the canon entrypoint, so memory
and disk agree on where truth lives.

---

## 5. OPS LOG (append-only, dated)

- **2026-06-12 04:0x CDT — survey**: walked repo top-level (du + file counts), subclassified cerebellum-dev (48 loose files, 12 dirs), inventoried sibling knowledge surfaces (clanker, clanker-soul, clanker-drift, conch-poc/brainloop), confirmed read-only zones untouched (gemma4-12b, modal_harness, games/cerebellum-*; north-mini-code found at games/, not in repo root). Discovered: cerebellum-dev is git-TRACKED on dev remote (156 files), not gitignored; conch-poc is a nested public repo; parent repo carries a stray `public` remote to brainloop.
- **2026-06-12 04:0x CDT — create canon (additive only)**: created `cerebellum-dev/knowledge/` with INDEX.md, PROJECT_HIERARCHY.md, CURRENT_METHOD.md, ARCHITECTURE_MAP.md (this file). No moves, no deletions, no git commits, no CLAUDE.md edits — pointer blocks drafted in §4 await approval.
