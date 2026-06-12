# Rebrand Log: osmosis -> cerebellum

## 2026-06-12 — START

Executing the full code rebrand. Constraints honored:
- Two campaigns live: Gemma 12B ablation driver (PID 1889080, `scripts/ablate_multidomain.py`, verified stdlib-only — NOT touched) and an auto-continuation script (will call `python -m osmosis.budget`). Modal watchdog imports nothing local.
- Strategy: move module code into `cerebellum/`; convert `osmosis/` into a deprecation-shim package (sys.modules aliasing + runpy passthrough for `python -m`), so `import osmosis.X` and `python -m osmosis.X` keep working unchanged.
- Legacy gitignored experimental modules (ab_test, benchmark, cli, gguf_reader, gguf_writer, loader, micro_quant, pipeline, repair*, sensitivity*) move to `cerebellum/_legacy/`, with `osmosis.X` shims kept.
- Historical artifact dirs (`osmosis-*`/) and `/var/home/deucebucket/games/osmosis-quants` untouched.

Baseline test run (before any change): `python -m pytest tests/ -v` -> **225 passed, 3 warnings** (11.46s).

Planned move map:
- osmosis/{budget,cerebellum,hillstep,imatrix_gen,imatrix_stream,micro_quantizer,sensitivity_stream}.py -> cerebellum/
- osmosis/dashboard/{__init__,models,scheduler,server,worker}.py + templates/ -> cerebellum/dashboard/
- osmosis/{ab_test,benchmark,cli,gguf_reader,gguf_writer,loader,micro_quant,pipeline,repair,repair_nim,sensitivity,sensitivity_multi}.py -> cerebellum/_legacy/
- osmosis/__init__.py + per-module shims remain as deprecation layer.

HF: will move deucebucket/Qwen3.6-27B-Osmosis-Q2_K-GGUF -> deucebucket/Qwen3.6-27B-Cerebellum-Q2K-GGUF and rename the GGUF inside via server-side copy+delete.

## 2026-06-12 — END (rebrand executed)

**Move map (all completed):**
- `osmosis/{budget,cerebellum,hillstep,imatrix_gen,imatrix_stream,micro_quantizer,sensitivity_stream}.py` -> `cerebellum/`
- `osmosis/dashboard/{__init__,models,scheduler,server,worker}.py` + `templates/` -> `cerebellum/dashboard/`
- `osmosis/{ab_test,benchmark,cli,gguf_reader,gguf_writer,loader,micro_quant,pipeline,repair,repair_nim,sensitivity,sensitivity_multi}.py` -> `cerebellum/_legacy/` (gitignored from origin via `cerebellum/_legacy/`; `cli.py` stays tracked as before)
- `osmosis/` is now a deprecation shim package. Each module emits a `DeprecationWarning`, then aliases itself via `sys.modules[__name__] = cerebellum.<target>` (identity-preserving: `osmosis.X is cerebellum.X`), and `python -m osmosis.X` passes through with `runpy.run_module(..., run_name="__main__", alter_sys=True)`. `osmosis/dashboard/__init__.py` stays a real package init (no self-alias) to avoid duplicate module execution via `__path__`.
- Previously-tracked osmosis paths keep tracked shims; previously-untracked legacy/dashboard pieces have untracked local shims — tracked-path set preserved exactly.

**Not touched (by design):** `scripts/ablate_multidomain.py` (running, PID 1889080, verified stdlib-only), historical `osmosis-*` artifact dirs, `/var/home/deucebucket/games/osmosis-quants` (added `cerebellum-quants` symlink alongside), dated findings logs (`docs/qwen35_9b_findings_log.md` etc.), `run_overnight.sh`/`run_phase2.sh` (dead — reference the nonexistent `/ai-drive/osmosis` tree; the `osmosis` console script they call still works via `cerebellum._legacy.cli:main`).

**Verification:**
- `python -m osmosis.budget --help` -> prints cerebellum.budget usage. `python -c "import osmosis.cerebellum, osmosis.hillstep, osmosis.imatrix_stream, osmosis.imatrix_gen, osmosis.budget"` -> OK with DeprecationWarnings. Identity asserts pass (no duplicate dashboard modules).
- pytest before: **225 passed**; after: **225 passed** (absolute path `/var/home/deucebucket/ai-drive/cerebellum/tests/`).
- `ruff check osmosis` -> all checks passed (pre-existing lint debt in moved code left as-is to keep the move pure; confirmed present at HEAD before the move).
- Editable install reinstalled as dist `cerebellum` 0.2.0 (old `osmosis` dist uninstalled after, console scripts `cerebellum` + deprecated `osmosis` both regenerated and verified).

**Code references updated:** `cerebellum/dashboard/*` imports + worker subprocess invocations (`python -m cerebellum.*`), hillstep liveness check now matches both `cerebellum.hillstep` and legacy `osmosis.hillstep` cmdlines, public-audit path guard covers both `osmosis/dashboard/` and `cerebellum/dashboard/`, `pyproject.toml` (name=cerebellum, `osmosis` script -> `cerebellum._legacy.cli:main`), tests, `scripts/build_variant.sh`.

**Docs:** CLAUDE.md + AGENTS.md (rename-complete language, kept consistent), README.md, docs/cli_reference.md, docs/getting_started.md, docs/multi_domain_ablation.md (local-only), docs/mamba_hybrid_findings.md, cerebellum-dev/README.md.

**HF:** `deucebucket/Qwen3.6-27B-Osmosis-Q2_K-GGUF` -> `deucebucket/Qwen3.6-27B-Cerebellum-Q2K-GGUF` (old URL 307-redirects, verified). GGUF renamed server-side in one commit (CommitOperationCopy + CommitOperationDelete + README update): `qwen3.6-27b-osmosis-imatrix-Q2_K.gguf` -> `qwen3.6-27b-cerebellum-imatrix-Q2_K.gguf` (commit ae61350). README model-index name, benchmark URLs, GGUF refs and repro commands updated; `osmosis_imatrix.dat` file kept (README refs match the real file).

**Commits (local only, NOT pushed, no Co-Authored-By trailers):** 174e503 code move, 29e9afe shim package, 73c86d7 tests, f4297ae docs, + this log. Caveat: commits 174e503/73c86d7 include pre-existing in-flight working-tree edits to hillstep.py, dashboard/server.py, cli.py and the two tracked test files (noted in commit bodies) — they could not be split from the move.
