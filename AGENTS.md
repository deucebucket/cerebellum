# Repository Guidelines

## Two Remotes — This Matters

- `origin` → public release (`cerebellum`). Ships data, reproducible recipes, benchmark summaries, docs. **Never push pipeline code, credentials, or devlogs.**
- `dev` → private (`cerebellum-dev`). Exploratory pipeline work, devlogs, unfinished experiments, local automation. Default to `dev` when unsure.
- `.gitignore` excludes `scripts/`, `tests/`, `run_*.sh` from `origin` (only `!scripts/ablation_run.sh` survives).

## Project Rename — Osmosis → Cerebellum

The rename is **complete** (2026-06-12). All module code lives in `cerebellum/`; `osmosis/` is a deprecation-shim package whose modules re-export from `cerebellum` (with a `DeprecationWarning`) so `import osmosis.X` / `python -m osmosis.X` keep working for in-flight runs and old scripts. Don't add code to `osmosis/`; don't write new `osmosis.*` imports. **Don't create new artifacts with `osmosis_*` names** (use `cerebellum_*`). New per-model dirs should be `cerebellum-<model>/`; old `osmosis-{model}/` artifact dirs keep their names (evidence).

## Package Layout

| Dir | Role |
|-----|------|
| `cerebellum/` | The engine package: CLI (`cerebellum.cli:main`), ablation, budget, imatrix, hillstep, dashboard. `cerebellum/_legacy/` holds old experimental modules (local-only). |
| `osmosis/` | Deprecation shim package re-exporting from `cerebellum`. No real code. |
| `scripts/` | One-shot benchmark/driver scripts (gitignored from origin). `scripts/run_benchmarks.sh` is the benchmark entry. |
| `tests/` | 6 test files (gitignored from origin). `test_gguf.py` (2 tests), `test_packing.py` (6 tests), `test_ablate_rowblocks.py`, `test_benchmark_hle_no_tools.py`, `test_cerebellum_cli.py`, `test_dashboard_control_plane.py`. |
| `osmosis-{model}/` | Per-model artifact dirs (ablation results, override files, GGUFs). |
| `cerebellum-dev/` | Private experiments, devlogs, streaming quantizer. **Not in `origin`.** |

## Build, Test, Lint

```bash
pip install -e ".[dev]"                                    # pytest + ruff
python -m pytest /var/home/deucebucket/ai-drive/cerebellum/tests/test_packing.py::test_4bit_roundtrip -v  # single test
python -m pytest /var/home/deucebucket/ai-drive/cerebellum/tests/ -v    # all tests (~12 items)
ruff check cerebellum osmosis tests scripts
```

`pyproject.toml` declares `>=3.10`. System Python is 3.14 where `dill`/`datasets` are broken — prefer `pyarrow.parquet` + `hf_hub_download` over `load_dataset()`. Use a 3.10 distrobox for anything needing `datasets`.

**Test path quirk**: `tests/` is gitignored so `pytest tests/...` (relative) may collect 0 items. Use absolute paths or `python -m pytest` from within the `tests/` directory.

## Local System Constraints

- CPU/RAM/GPU: Ryzen 7 5800XT (8C/16T), 62 GiB RAM, RTX 3090 24 GiB VRAM.
- Drives: `/var/home/deucebucket/ai-drive` (nvme1n1p1), `/var/home/deucebucket/games` (nvme1n1p2), `/var/home` (nvme0n1p3).
- Drive space is an execution/cleanup problem, not a research blocker. Check `df -h` before large builds. Offload to game drive when needed.
- F16 source GGUFs (~2 bytes/param) often won't fit alongside output on the main drive.

## Core Engine + Workflow

| File | Role |
|------|------|
| `cerebellum/cerebellum.py` | Ablation runner + precision allocator |
| `cerebellum/budget.py` | Budget-constrained bit allocator, size estimator, promotion logic |
| `cerebellum/imatrix_stream.py` | Streaming imatrix generator (~300 MB RAM, any model size) |
| `cerebellum/imatrix_gen.py` | Calibration text generation for imatrix |

Manual workflow:
```bash
python -m cerebellum.imatrix_stream --model f16.gguf --output imatrix.dat
python -m cerebellum.cerebellum ablate --source-gguf f16.gguf --imatrix imatrix.dat --output ablation_results.json
python -m cerebellum.budget --sensitivity ablation_results.json --source-gguf f16.gguf --budget-gb 12.0 --output tensor_types.txt
llama-quantize --imatrix imatrix.dat --tensor-type-file tensor_types.txt f16.gguf out.gguf Q2_K
```

The `cerebellum` CLI (`cerebellum run`, `cerebellum plan-space`, `cerebellum watch`, etc.) wraps this into resumable hill-climber runs. See `cerebellum tutorial overview` for the flow.

CUDA commands run via distrobox: `distrobox enter ai -- <command>`.
llama.cpp binaries: `/var/home/deucebucket/ai-drive/llama.cpp/build/bin/`.

## Key Docs (Read Before Architecture Decisions)

- `docs/cerebellum_memory_controller.md` — three-tier shard architecture (VRAM/RAM/NVMe), row-block protocol, multi-model shard fusion. This is the endgame.
- `docs/llama_cpp_tensor_backend_patch.md` — llama.cpp patch for per-tensor backend override.
- `docs/mamba_hybrid_findings.md` — SSM hard-fail below 4-bit.
- `docs/benchmark_protocol.md` — bench invariants, audit gates, release artifact checklist.
- `cerebellum-dev/RESEARCH_IDEAS.md` — quantization-as-routing-bias, conch shell topology, sub-tensor row-level quant, reasoning drift bench.

## Benchmark Invariants

- **Server**: `llama-server -ngl 99 --parallel 4 -c 24576` covers the whole suite. 6144 per slot — HumanEval+ needs ≥4096 output tokens.
- **Worker counts**: HumanEval+ = 1 (`BENCH_WORKERS=1`, sequential). ARC/HellaSwag/MMLU = 4 workers. `scripts/run_benchmarks.sh` sets these explicitly per case.
- **Thinking must be disabled for published no-thinking benches**: serve with `--reasoning off --reasoning-budget 0` when the model/runtime supports it. `llama-cpp-python` has no equivalent — use `llama-server` for all inference.
- **Gemma 4 HumanEval+ exception**: raw `/v1/completions` bypasses Gemma 4's chat template and can produce garbage. Use `scripts/benchmark_evalplus_chat.py` against `llama-server --jinja --reasoning auto`, with request payload `chat_template_kwargs: {"enable_thinking": false}` and `thinking_budget_tokens: 0`. Keep `BENCH_WORKERS=1`.
- **Audit wrong answers before publishing any score**. For MCQ: `jq 'select(.correct == false)' detailed.jsonl | head -30`. For EvalPlus: count `base_status`/`plus_status`, run `ast.parse(solution)` over every sample, and inspect failures plus pass examples. Every prior benchmark bug in this project would have been caught by this step.
- **EvalPlus anti-misgrade checks**: verify `0` prompt echoes, `0` repeated target function definitions, `0` pass-only outputs, and no cluster of syntax/indentation failures. If a correction changes the score materially, do a fresh full rerun and publish the fresh number.
- **Bug history** (all baked into current scripts, don't regress): HumanEval fence-stripping (cost 6-8 pts), numeric label mismatch (ARC: 19 questions), empty response fallback (HellaSwag: 108 questions), trailing markdown fences (84% of HumanEval+ wrongs), Gemma 4 raw-completions/chat-template misrun (3.05%/17.07% false lows; fresh audited result 89.63% HumanEval+), cache contamination across models.
- **HF benchmark artifacts**: every uploaded model must include a clear `benchmark_results/` directory in the HF repo with summary JSONs and, for HumanEval+/EvalPlus, the samples JSONL plus EvalPlus eval JSON. The model card must point to the measured results and not rely on chat-only claims.

## Architecture-Specific Gotchas (Read Before Ablating)

- **Hybrid SSM**: `in_proj_a/b`, `A_log`, `dt_bias`, `conv1d`, `in_proj_z` hard-fail below 4-bit (NaN). Force 4-bit minimum.
- **MoE**: Expert weights are fragile, not router/aux signals — opposite of dense intuition. Routed experts > shared experts in sensitivity.
- **PLE** (Gemma 4 E4B): Q4_K → Q3_K cliff. Without PLE protection: PPL 104; with PLE@Q5_K: PPL 55.
- **Dense**: Sometimes demoting attention K/Q/output *improves* PPL (Granite 4.1). Trust the ablation data.

## Commit & PR Style

Prefixes: `feat:`, `fix:`, `docs:`, `data:`, `results:`. Imperative and scoped — `fix: normalize HumanEval indentation`. PRs touching benchmarks must include affected model dir and benchmark deltas.

## Reference

`CLAUDE.md` has deeper architecture detail (memory controller, tensor backend patch, multi-domain ablation methodology) — read it before making architecture decisions or proposing new research directions.
