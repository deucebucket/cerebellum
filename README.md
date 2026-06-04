# Cerebellum

Cerebellum is a small set of engine files for building ablation-informed GGUF
quants.

The core idea is simple: do not guess which tensors can survive lower
precision. Measure the damage, then spend bits where the measured damage says
they matter.

The Python package is still named `osmosis` for compatibility with existing
scripts and old experiment files. New documentation and release artifacts use
the Cerebellum name.

## What This Repo Contains

Public Cerebellum is intentionally narrow:

- `cerebellum imatrix` generates llama.cpp-compatible imatrix files from
  safetensors without loading the full model into RAM.
- `cerebellum imatrix --mode calibrated` generates an imatrix with optional activation
  calibration.
- `osmosis/cerebellum.py` reads measured ablation results and writes a
  `llama-quantize` tensor type override file.
- `osmosis/budget.py` reads weight-sensitivity curves and writes a
  budget-fitting tensor type override file.
- `osmosis/micro_quantizer.py` contains shared sensitivity helpers used by the
  imatrix generators.

The public repo is not meant to ship dashboards, local automation, private
devlogs, credentials, or one-off model-building scripts. Private exploration
lives in `cerebellum-dev`; public release artifacts live here.

Public release artifacts do include benchmark results, Cerebellum ablation
results, and the configuration needed to reproduce a release: source/model
hashes, imatrix provenance, ablation result JSON, tensor type files, allocator
settings, runtime flags, and benchmark artifacts.

## How Cerebellum Works

1. Generate an imatrix.

   The imatrix gives `llama-quantize` per-channel importance data in the binary
   format it already understands.

2. Measure tensor damage.

   For ablation-driven builds, start from a sane baseline such as `Q4_K_M`,
   force one tensor or tensor group down to `Q2_K`, run perplexity, and record
   the delta from baseline.

3. Classify tensors.

   Negative delta means the lower-precision tensor was not hurting that
   calibration run. Positive delta means the tensor needs protection. Small
   deltas are treated as noise.

4. Allocate precision under a size budget.

   Cerebellum uses `llama-quantize --dry-run` to get real tensor sizes, then
   writes `tensor_types.txt`. That file is passed back to stock
   `llama-quantize`.

5. Build and audit the final GGUF.

   The final artifact is still a normal GGUF. Cerebellum is the measurement and
   allocation process used to choose its mixed tensor types.

## Quickstart

For a fuller walkthrough, see
[docs/getting_started.md](docs/getting_started.md). For exact CLI options, see
[docs/cli_reference.md](docs/cli_reference.md).

Install the local package:

```bash
pip install -e ".[dev]"
```

## Cerebellum CLI

The current private/dev CLI is exposed as `cerebellum`. It is designed for
visible, resumable quantization runs rather than silent one-off scripts:

```bash
cerebellum system
cerebellum doctor
cerebellum plan-space --source-gguf source-f16.gguf
cerebellum run \
  --source-gguf source-f16.gguf \
  --profile wiki \
  --family gemma-4 \
  --model-name gemma-4-12b-it \
  --source-name google-f16 \
  --data-root /path/to/cerebellum-runs \
  --scratch-root /large/scratch/drive \
  --base-type Q4_K_M \
  --start-type q4_K \
  --levels q3_K,q2_K,q5_K,q6_K,f16
cerebellum watch /path/to/run
cerebellum watch /path/to/run --tui
cerebellum report /path/to/run
```

`--distrobox NAME` is optional. It is only needed on machines where the
llama.cpp binaries must run inside a container/toolbox to see CUDA or ROCm
libraries. On a normal install, put `llama-quantize` and `llama-perplexity` on
`PATH` or pass `--quantize-bin` / `--perplexity-bin`; Cerebellum itself is not
tied to distrobox or to this workstation.

Run `cerebellum doctor` after install. It checks llama.cpp binaries, GPU
visibility, writable data roots, local PPL profile corpora, and explains how to
fix missing pieces.

For unattended multi-model queues:

```bash
cerebellum schedule --template > cerebellum_schedule.json
cerebellum schedule --file cerebellum_schedule.json --dry-run
cerebellum schedule --file cerebellum_schedule.json
```

Local automation/API server:

```bash
cerebellum api --host 127.0.0.1 --port 8931 --data-root /path/to/cerebellum-runs
```

Initial endpoints include `/health`, `/runs`, `/run`, `/events`,
`/measurements`, `/report`, `/provenance`, and `/db/families`.

Transparent provenance metadata:

```bash
cerebellum provenance --run-dir /path/to/run
cerebellum provenance --gguf model.gguf
cerebellum provenance --run-dir /path/to/run --gguf model.gguf --hash-files
cerebellum finalize --run-dir /path/to/run --gguf model.gguf
cerebellum package /path/to/run
```

Cerebellum provenance uses visible `cerebellum.*` GGUF metadata keys and report
hashes. It is intended for attribution and auditability, not hidden watermarking.
`finalize` writes metadata sidecars and a model-card block everywhere; if a
compatible `gguf-set-metadata` tool is installed, `--inject` can tag the GGUF.
`package` writes a portable upload manifest with the sidecars Cerebellum expects
to travel with a published GGUF.

PPL/calibration target is explicit. Use `--profile wiki`, `--profile agentic`,
`--profile code`, `--profile math`, `--profile dialogue`, `--profile all-around`,
or `--profile custom --corpus FILE`. The chosen profile and resolved corpus are
written into manifest, state, reports, exports, and the live terminal dashboard
so results remain comparable later.

The live dashboard shows run identity, selected PPL profile, active quant/PPL
work, live process health, current/active GGUF sizes, timing totals, recent
measurements, and the event stream. The dashboard stays bounded by default;
increase visible rows with `--events-limit N` and `--measurements-limit N`. Use
`cerebellum watch RUN_DIR --tui` for an interactive terminal UI with independent
scrollable panes for events, measurements, processes/GPU, and files.
CLI layout redesign notes live in
[docs/cerebellum_cli_layout_notes.md](docs/cerebellum_cli_layout_notes.md).
Current CLI changelog:
[docs/cerebellum_cli_changelog.md](docs/cerebellum_cli_changelog.md).
Captured dev snapshots:

- [Qwen3 0.6B smoke dashboard](docs/cerebellum_cli_smoke_snapshot.png)
- [Gemma 4 12B live dashboard](docs/cerebellum_cli_gemma4_12b_live.png)

Install or build llama.cpp separately. The commands below assume
`llama-quantize` is on `PATH`. You can also pass `--quantize-bin` to the
allocator commands.

Generate a streaming imatrix from a Hugging Face model ID or local safetensors
directory:

```bash
cerebellum imatrix \
  --model <hf-model-id-or-local-path> \
  --output cerebellum_imatrix.dat \
  -v
```

Build a plain imatrix quant for a baseline:

```bash
llama-quantize \
  --imatrix cerebellum_imatrix.dat \
  source-f16.gguf \
  baseline-Q4_K_M.gguf \
  Q4_K_M
```

After you have ablation results, ask Cerebellum to allocate tensor types for a
target file size:

```bash
python -m osmosis.cerebellum \
  --ablation ablation_results.json \
  --source-gguf source-f16.gguf \
  --imatrix cerebellum_imatrix.dat \
  --budget-gb 12.0 \
  --output tensor_types.txt \
  -v
```

Build the final GGUF with stock llama.cpp:

```bash
llama-quantize \
  --imatrix cerebellum_imatrix.dat \
  --tensor-type-file tensor_types.txt \
  source-f16.gguf \
  model-cerebellum.gguf \
  Q4_K_M
```

## File Roles

| File | Role |
| --- | --- |
| `cerebellum_imatrix.dat` | Binary llama.cpp imatrix consumed by `llama-quantize --imatrix`. |
| `ablation_results.json` | Measured baseline PPL plus per-tensor PPL after forced low-precision tests. |
| `sensitivity_multi.json` | Weight-only multi-depth sensitivity report consumed by `osmosis.budget`. |
| `tensor_types.txt` | Final `tensor_name=qtype` overrides consumed by `llama-quantize --tensor-type-file`. |
| `source-f16.gguf` | Full precision or high precision GGUF used as quantization source. |
| `model-cerebellum.gguf` | Final normal GGUF artifact. |

## Ablation Result Schema

`osmosis.cerebellum` accepts the historical single-domain format:

```json
{
  "baseline_ppl": 7.034,
  "tests": {
    "layer_10.mlp.down_proj": {
      "gguf_tensor": "blk.10.ffn_down.weight",
      "ppl": 7.091
    }
  }
}
```

It also accepts a multi-domain format:

```json
{
  "baseline_ppl": {"wiki": 7.0, "code": 4.2, "math": 6.1, "dialogue": 8.3},
  "tests": {
    "layer_10.mlp.down_proj": {
      "gguf_tensor": "blk.10.ffn_down.weight",
      "ppl": {"wiki": 7.1, "code": 4.4, "math": 6.1, "dialogue": 8.2}
    }
  }
}
```

For multi-domain results, pass a named profile or explicit weights:

```bash
python -m osmosis.cerebellum \
  --ablation ablation_results.json \
  --source-gguf source-f16.gguf \
  --budget-gb 12.0 \
  --objective-weights code \
  --output tensor_types.txt
```

Built-in profiles are `general`, `code`, `chat`, and `balanced`. If no weights
are supplied for multi-domain input, the allocator defaults to wiki-only for
backward compatibility.

## Alternative Proxy Path

When full ablation is too expensive, `osmosis.budget` can allocate from
weight-only multi-depth sensitivity data:

```bash
python -m osmosis.sensitivity_stream \
  --model <hf-model-id-or-local-path> \
  --output sensitivity_multi.json \
  -v

python -m osmosis.budget \
  --sensitivity sensitivity_multi.json \
  --source-gguf source-f16.gguf \
  --imatrix cerebellum_imatrix.dat \
  --budget-gb 12.0 \
  --output tensor_types.txt \
  -v
```

This is faster than measured ablation, but it is a proxy. Published releases
should state which path was used.

## Architecture Notes

Cerebellum is empirical. The allocator should follow measurements, not a fixed
belief about which tensor classes matter.

- Dense transformer models often have local tensor groups that tolerate or even
  benefit from demotion.
- MoE models can put sensitivity in expert weights rather than the router or
  obvious auxiliary signals.
- Hybrid SSM models can have hard precision floors; some SSM tensors should not
  be pushed below 4-bit without explicit testing.
- Gemma-style per-layer embedding or projection tensors can show sharp cliffs
  between adjacent quant levels.

See [docs/mamba_hybrid_findings.md](docs/mamba_hybrid_findings.md) for one
example of architecture-specific guardrails.

## Benchmarking

Do not publish a score from a single summary number. Keep the detailed outputs,
inspect wrong answers, and rerun if a harness bug is found.

See [docs/benchmark_protocol.md](docs/benchmark_protocol.md) for the release
artifact checklist and audit gate.

Minimum release artifacts for a benchmarked model:

- summary JSON files,
- detailed JSONL answer files for MCQ tasks,
- EvalPlus samples and EvalPlus eval JSON for HumanEval/EvalPlus,
- exact runtime flags and model file hashes,
- notes describing whether thinking/reasoning was enabled or disabled.

For each released model, keep a reproducibility bundle next to the benchmark
artifacts when practical:

- `ablation_results.json` or `sensitivity_multi.json`,
- `tensor_types.txt`,
- allocator command and arguments,
- source GGUF hash,
- imatrix hash or source,
- final GGUF hash,
- llama.cpp commit or release,
- server and benchmark command flags.

## Development Notes

The public `origin` remote is for release-ready engine files, docs, benchmark
artifacts, and reproducible recipes. The private `dev` remote is for exploratory
automation, local scripts, unpublished logs, and unfinished experiments.

When unsure, keep it out of `origin`.

## License

Apache-2.0

### Recovery, low-space runs, and targeted testing

Cerebellum keeps durable run state in the run directory: `state.json`, `cerebellum_events.jsonl`, `cerebellum_candidates.jsonl`, `cerebellum_current_tensor_types.txt`, and `checkpoints/`. Heavy candidate GGUFs live under `tmp/` and can be discarded after a stopped run.

Useful controls:

```bash
# Never launch another quant job unless one estimated candidate plus 10 GiB remains.
cerebellum run ... --hard-free-floor-gb 10

# Mirror small critical metadata/checkpoints to another drive during a long run.
cerebellum run ... --backup-root /path/to/backup-root

# Manual metadata backup.
cerebellum backup RUN_DIR --to /path/to/backup-root

# Dry-run safe cleanup; add --yes to execute.
cerebellum cleanup RUN_DIR --partials

# Target only specific layers or tensor names.
cerebellum run ... --layers 0,1,8-12
cerebellum run ... --tensor-regex 'blk\\.12\\.(attn_q|attn_k)\\.weight'

# Roll state back to a clean boundary.
cerebellum rollback RUN_DIR --last-completed-layer --yes
```

Measured non-winning candidate GGUFs are pruned immediately by default, while the CPU/GPU overlap remains enabled for speed. Use `--keep-measured-candidates` only for diagnostic runs where retaining every candidate file is worth the disk cost.

After rollback, Cerebellum marks the current baseline GGUF invalid. The next resume rebuilds `artifacts/current_baseline.gguf` from the rolled-back `cerebellum_current_tensor_types.txt` before continuing.

Resume and recovery helpers:

```bash
# Resume from the existing manifest/state.
cerebellum resume RUN_DIR

# Resume in safer low-space mode after disk pressure.
cerebellum resume RUN_DIR --low-space --hard-free-floor-gb 10

# Ask Cerebellum what to do after a crash/lockup.
cerebellum recover RUN_DIR
```

### Tutorials and AI automation API

Cerebellum ships built-in tutorials so users and agents can discover the flow without reading source code:

```bash
cerebellum tutorial list
cerebellum tutorial overview
cerebellum tutorial recovery
cerebellum tutorial low-space
cerebellum tutorial targeting
cerebellum tutorial api
cerebellum tutorial provenance
```

The local API exposes read-only automation data for AI agents and future web UI work:

```bash
cerebellum api --host 127.0.0.1 --port 8931 --data-root DATA_ROOT
```

Read-only endpoints include:

```text
/health
/runs
/run?run_dir=RUN_DIR
/events?run_dir=RUN_DIR&limit=100
/measurements?run_dir=RUN_DIR&limit=100
/report?run_dir=RUN_DIR
/export?run_dir=RUN_DIR&kind=ai
/export?run_dir=RUN_DIR&kind=infographic
/recover?run_dir=RUN_DIR
/provenance?run_dir=RUN_DIR&gguf=MODEL.gguf
/package?run_dir=RUN_DIR
/system
/space?source_gguf=MODEL.gguf&margin_gb=20
/tutorial?topic=recovery
/commands
/db/families
```

State-changing operations remain CLI-only for now: `run`, `resume`, `stop`, `cleanup`, `rollback`, `backup`, `finalize`, and `upload`.

Measurement table colors:

```text
PPL delta: green = better, red = worse, cyan `=` = unchanged.
GGUF size: blue = smaller than current baseline, orange = larger than current baseline, cyan = equal/neutral.
```

Read-only smoke check:

```bash
cerebellum self-test
cerebellum self-test --run-dir RUN_DIR
```

The API also exposes `/self-test?run_dir=RUN_DIR` for automation agents.

### Cerebellum imatrix

Imatrix generation is part of Cerebellum. The legacy `osmosis.imatrix_*` modules remain only as compatibility internals while the package rename finishes.

```bash
# Streaming mode: safest default for large models.
cerebellum imatrix \
  --model HF_OR_LOCAL_SAFETENSORS_MODEL \
  --output cerebellum_imatrix.dat \
  -v

# Optional calibrated mode: loads the model and can blend activation stats.
cerebellum imatrix \
  --model HF_OR_LOCAL_MODEL \
  --output cerebellum_imatrix.dat \
  --mode calibrated \
  --num-samples 8
```

Use the output with `cerebellum run --imatrix cerebellum_imatrix.dat`.

### Cerebellum project layout

Cerebellum groups each model source as a project:

```text
DATA_ROOT/
  families/
    FAMILY/
      MODEL/
        sources/
          SOURCE/
            cerebellum_project.json
            imatrix/
              cerebellum_imatrix.dat
            runs/
              RUN_ID/
```

If `cerebellum imatrix` is run with `--family`, `--model-name`, and `--source-name`, the imatrix is written into that project and Cerebellum prints the next `cerebellum run --imatrix ...` command.

```bash
cerebellum imatrix \
  --model HF_OR_LOCAL_SAFETENSORS_MODEL \
  --family gemma-4 \
  --model-name gemma-4-12b-it \
  --source-name google-f16 \
  --source-gguf /models/gemma-4-12b-it-f16.gguf \
  --data-root /data/cerebellum-runs
```

Project browsing:

```bash
cerebellum project --data-root DATA_ROOT
cerebellum project --data-root DATA_ROOT --family gemma-4 --json
```

The local API also exposes `/projects?family=...&model=...&source=...`.
