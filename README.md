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

- `osmosis/imatrix_stream.py` generates llama.cpp-compatible imatrix files from
  safetensors without loading the full model into RAM.
- `osmosis/imatrix_gen.py` generates an imatrix with optional activation
  calibration.
- `osmosis/cerebellum.py` reads measured ablation results and writes a
  `llama-quantize` tensor type override file.
- `osmosis/budget.py` reads weight-sensitivity curves and writes a
  budget-fitting tensor type override file.
- `osmosis/micro_quantizer.py` contains shared sensitivity helpers used by the
  imatrix generators.

The public repo is not meant to ship dashboards, local automation, private
devlogs, credentials, or one-off model-building scripts. Public release
artifacts live here.

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

Install or build llama.cpp separately. The commands below assume
`llama-quantize` is on `PATH`. You can also pass `--quantize-bin` to the
allocator commands.

Generate a streaming imatrix from a Hugging Face model ID or local safetensors
directory:

```bash
python -m osmosis.imatrix_stream \
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

## Released Data

Released model recipes and measurements are kept in public directories:

- `model-data/` contains ablation outputs, tensor override files, PPL notes, and
  other small release files used to build published GGUFs.
- `benchmarks/` contains benchmark summaries and detailed answer artifacts for
  published runs.

To reproduce a release, start from the matching source GGUF and imatrix,
inspect the model's `model-data/<model>/` files, then pass the released tensor
type file to `llama-quantize --tensor-type-file`.

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

The public repository is for release-ready engine files, docs, benchmark
artifacts, ablation outputs, model-data recipes, and reproducible configs.
Exploratory automation, local scripts, unpublished logs, and unfinished
experiments should stay out of public history.

When unsure, keep it out of `origin`.

## License

Apache-2.0
