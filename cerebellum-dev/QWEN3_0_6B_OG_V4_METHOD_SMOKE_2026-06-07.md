# Qwen3 0.6B OG Qwen 3.6 v4 Method Smoke - 2026-06-07

Private checkpoint. Do not publish raw paths/logs to `origin`.

## Target

- Source GGUF: `/var/home/deucebucket/games/models/Qwen3-0.6B-f16.gguf`
- Run root: `/var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607`
- Corpus: `/var/home/deucebucket/games/osmosis-quants/wiki.test.raw`
- CUDA command wrapper: `distrobox enter ai --`

This target was chosen because it is a tiny local F16 GGUF and can exercise the old method quickly. GLM-4.7 Flash is only a smoke/template target because the local GGUF is already quantized.

## Planka Decision

Planka GLM board says:

- Smoke local GLM-4.7 Flash first only for runtime/template/tensor-layout checks.
- Use GLM-4-9B-Chat-HF as the first real GLM Cerebellum source target after conversion.
- GLM Edge 1.5B/4B is only a plumbing probe, not the main research target.

No local GLM-4-9B source directory was found in the quick scan.

## Matched OG Qwen 3.6 v4 Steps

Qwen 3.6 27B v4 used:

- Calibrated imatrix path, not stream-only.
- `ncall=8`.
- Old imatrix dataset style.
- F16 source to quantized candidates with `llama-quantize --imatrix`.
- Single tensor or small group demotion tests.
- PPL gate before benchmark gate.

Smoke command:

```bash
python3 -m cerebellum.cli imatrix \
  --model Qwen/Qwen3-0.6B \
  --output /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/imatrix/cerebellum_imatrix_ncall8.dat \
  --mode calibrated \
  --num-samples 8 \
  --source-gguf /var/home/deucebucket/games/models/Qwen3-0.6B-f16.gguf \
  --family qwen \
  --model-name qwen3-0.6b-og-method \
  --source-name Qwen3-0.6B-f16 \
  -v
```

Validation:

- Entries: `196`
- `ncall_min=8`
- `ncall_max=8`
- Trailer `ncall=8`
- Dataset: `cerebellum-sensitivity`
- Size: `1.1M`

## Watch Wiring

Existing `cerebellum watch` is usable; no new UI code was needed.

Watch command:

```bash
cerebellum watch /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/watch-smoke-blk0-ffn-down-uppercase --once --plain
```

The watch smoke wrote:

- `manifest.json`
- `state.json`
- `cerebellum_events.jsonl`
- `group_types.txt`

The initial group-scan failed before PPL because current `llama-quantize` rejects some generated tensor map values. This is a tooling compatibility issue, not a method failure.

## Compatibility Findings

1. Host llama.cpp binaries fail without the CUDA runtime:

```text
libcudart.so.12: cannot open shared object file
```

Use:

```bash
distrobox enter ai -- /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize ...
distrobox enter ai -- /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-perplexity ...
```

2. Current `llama-quantize --tensor-type-file` rejects `Q4_K_M` as a per-tensor value.

Working per-tensor aliases:

```text
^blk\.0\.ffn_down\.weight$=q2_K
^blk\.1\.ffn_down\.weight$=q4_K
```

So maps generated with `Q4_K_M` entries must be normalized to `q4_K` before use with this binary.

3. `--token-embedding-type f16` also caused the generated group-scan command to fail. The direct working command omitted token embedding override.

## Direct Smoke Result

Run dir:

```text
/var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/direct-smoke
```

Baseline build:

```bash
distrobox enter ai -- /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --allow-requantize \
  --imatrix /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/imatrix/cerebellum_imatrix_ncall8.dat \
  /var/home/deucebucket/games/models/Qwen3-0.6B-f16.gguf \
  /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/direct-smoke/artifacts/qwen3-0.6b-q4km-imatrix.gguf \
  Q4_K_M
```

Candidate build:

```bash
distrobox enter ai -- /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --allow-requantize \
  --imatrix /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/imatrix/cerebellum_imatrix_ncall8.dat \
  --tensor-type-file /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/direct-smoke/tensor_types_blk0_ffn_down_q2_from_q4.txt \
  /var/home/deucebucket/games/models/Qwen3-0.6B-f16.gguf \
  /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/direct-smoke/artifacts/qwen3-0.6b-q4km-blk0-ffn-down-q2.gguf \
  Q4_K_M
```

PPL command:

```bash
distrobox enter ai -- /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-perplexity \
  --model MODEL.gguf \
  --ctx-size 2048 \
  -f /var/home/deucebucket/games/osmosis-quants/wiki.test.raw \
  -ngl 99 \
  --chunks 32
```

Results:

| Artifact | Size | Tensor Mix | PPL |
|---|---:|---|---:|
| `qwen3-0.6b-q4km-imatrix.gguf` | 462 MiB | default Q4_K_M imatrix mix | `19.7289 +/- 0.36559` |
| `qwen3-0.6b-q4km-blk0-ffn-down-q2.gguf` | 409 MiB | `F32=113`, `Q4_K=197`, `Q2_K=1` | `21.0872 +/- 0.39031` |

Interpretation: demoting `blk.0.ffn_down.weight` to Q2_K saves about 53 MiB on the tiny target but hurts PPL materially. This tensor should be rejected/protected in a tiny-model replay.

## Next

## Completed Negative-Control Run: Group-Forward Flow

I also ran the current `cerebellum legacy-flow --execute-forward` path on the
tiny Qwen target. This is useful as a negative control, because it is not the
same method described by the Qwen 3.6 27B v4 card.

Command shape:

```bash
cerebellum legacy-flow \
  --source-gguf /var/home/deucebucket/games/models/Qwen3-0.6B-f16.gguf \
  --output-dir /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/full-legacy-forward/out \
  --run-dir /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/full-legacy-forward \
  --imatrix /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/imatrix/cerebellum_imatrix_ncall8.dat \
  --corpus /var/home/deucebucket/games/osmosis-quants/wiki.test.raw \
  --model-name qwen3-0.6b-og-method \
  --family qwen \
  --source-name Qwen3-0.6B-f16 \
  --profile code \
  --base-type Q4_K_M \
  --start-type q4_K \
  --levels q4_K,q3_K,q2_K \
  --survivability-levels q4_K,q3_K,q2_K \
  --survivability-target-type q2_K \
  --baseline-ppl 19.7289 \
  --quantize-bin /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --perplexity-bin /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-perplexity \
  --distrobox ai \
  --gpu-layers 99 \
  --ctx-size 2048 \
  --chunks 32 \
  --scratch-root /var/home/deucebucket/games/cerebellum-pipeline-tmp/qwen3-0.6b-og-method-full \
  --serial-candidates \
  --execute-forward \
  --token-embedding-type '' \
  --write /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/full-legacy-forward/legacy_flow.json \
  --json
```

Results:

| Group | Tensors | Q2_K PPL | Delta vs 19.7289 | Verdict |
|---|---:|---:|---:|---|
| `attn-q` | 28 | 25.5434 | +5.8145 | protect |
| `attn-k` | 28 | 23.3581 | +3.6292 | protect |
| `attn-v` | 28 | 29.6535 | +9.9246 | protect |
| `attn-output` | 28 | 24.6529 | +4.9240 | protect |
| `ffn-gate` | 28 | 25.6831 | +5.9542 | protect |
| `ffn-up` | 28 | 31.3661 | +11.6372 | protect |
| `ffn-down` | 28 | 57.0379 | +37.3090 | protect |
| `early-blocks` | 35 | 73.6198 | +53.8909 | protect |

Interpretation:

- The group-forward flow crushes whole tensor families and is much harsher than
  the Qwen 27B v4 sparse per-tensor probe + budget allocator.
- It is expected to show large damage on a 0.6B dense model. Do not use this
  result to infer the OG v4 allocator failed.
- This run explains why the current reproduction did not show the old size/quality
  pattern: it exercised a different Cerebellum generation of the pipeline.

## Correct Next Reproduction Step

Patch or wrap `group-scan` map generation so:

- `Q4_K_M` tensor-file entries become `q4_K`.
- Optional token embedding override is omitted unless explicitly needed.

Then run the v4-shaped sparse probe + allocator replay:

```bash
# 1. Select 20-30 single tensor probes from Qwen3-0.6B, mirroring
# osmosis-qwen36-27b/ablation_plan.json roles/reasons.

# 2. For each tensor, build from F16 with base Q2_K or Q4_K_M and one
# tensor-type override to Q2_K; measure PPL with the same imatrix/corpus.

# 3. Write an ablation_results.json in osmosis.cerebellum schema.

# 4. Run the allocator:
python -m osmosis.cerebellum \
  --ablation ABLA_RESULTS.json \
  --plan ABLA_PLAN.json \
  --source-gguf /var/home/deucebucket/games/models/Qwen3-0.6B-f16.gguf \
  --budget-gb 0.42 \
  --base-type Q2_K \
  --imatrix /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/imatrix/cerebellum_imatrix_ncall8.dat \
  --quantize-bin /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --output tensor_types_v4_replay_tiny.txt \
  -v

# 5. Build final:
distrobox enter ai -- /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --allow-requantize \
  --imatrix /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/imatrix/cerebellum_imatrix_ncall8.dat \
  --tensor-type-file tensor_types_v4_replay_tiny.txt \
  /var/home/deucebucket/games/models/Qwen3-0.6B-f16.gguf \
  qwen3-0.6b-cerebellum-v4-replay.gguf \
  Q2_K
```

## Full Qwen 27B v4-Shaped Sparse Replay Result

Run root:

```text
/var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/v4-sparse-replay
```

Runner:

```bash
python3 /var/home/deucebucket/ai-drive/cerebellum/cerebellum-dev/tools/run_tiny_qwen27_v4_replay.py
```

Watch:

```bash
cerebellum watch \
  /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/v4-sparse-replay \
  --plain \
  --measurements-limit 12
```

Artifacts:

- Probe plan: `ablation_plan.json`
- Probe results: `ablation_results.json`
- Watch sidecar: `manifest.json`, `state.json`, `cerebellum_candidates.jsonl`, `cerebellum_events.jsonl`
- 420 MB allocation map: `tensor_types_v4_replay_tiny_420mb.txt`
- 462 MB allocation map: `tensor_types_v4_replay_tiny_462mb.txt`
- Final 420 MB GGUF: `final/qwen3-0.6b-cerebellum-v4-replay-420mb.gguf`
- Final 462 MB GGUF: `final/qwen3-0.6b-cerebellum-v4-replay-462mb.gguf`

### Sparse Probe Findings

Baseline for probe deltas:

- Q4_K_M + imatrix PPL: `19.7289 +/- 0.36559`
- Baseline artifact size: `462M`

Measured 23 single-tensor Q2_K probes.

Demotable / negative-delta:

| Tensor | PPL | Delta |
|---|---:|---:|
| `blk.18.attn_output.weight` | 19.6591 | -0.0698 |
| `blk.7.attn_q.weight` | 19.6894 | -0.0395 |

Neutral:

| Tensor | PPL | Delta |
|---|---:|---:|
| `blk.2.ffn_gate.weight` | 19.7294 | +0.0005 |
| `blk.27.attn_v.weight` | 19.7484 | +0.0195 |

Largest sacred/protected:

| Tensor | PPL | Delta |
|---|---:|---:|
| `blk.27.ffn_down.weight` | 21.6027 | +1.8738 |
| `blk.18.attn_v.weight` | 20.8611 | +1.1322 |
| `blk.1.ffn_gate.weight` | 19.9866 | +0.2577 |
| `blk.0.ffn_down.weight` | 19.9518 | +0.2229 |
| `blk.25.ffn_down.weight` | 19.9398 | +0.2109 |

This reproduces the important Qwen 27B v4 signal shape on a tiny dense model:
some tensors get better when crushed, and a few tensors are disproportionately
fragile.

### Allocation And Final PPL

420 MB target:

```bash
distrobox enter ai -- bash -lc 'cd /var/home/deucebucket/ai-drive/cerebellum && python3 -m osmosis.cerebellum \
  --ablation /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/v4-sparse-replay/ablation_results.json \
  --plan /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/v4-sparse-replay/ablation_plan.json \
  --source-gguf /var/home/deucebucket/games/models/Qwen3-0.6B-f16.gguf \
  --budget-gb 0.42 \
  --base-type Q2_K \
  --imatrix /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/imatrix/cerebellum_imatrix_ncall8.dat \
  --quantize-bin /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --output /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/v4-sparse-replay/tensor_types_v4_replay_tiny_420mb.txt \
  -v'
```

- Overrides: `196`
- Final size: `407M`, `400.56 MiB`, `4.47 BPW`
- Type mix: `F32=113`, `Q2_K=1`, `Q3_K=60`, `Q4_K=89`, `Q5_K=47`, `Q6_K=1`
- Final PPL: `22.8992 +/- 0.43223`
- Interpretation: much better than pure Q2_K, but worse than Q4_K_M. Too tight.

462 MB target:

```bash
distrobox enter ai -- bash -lc 'cd /var/home/deucebucket/ai-drive/cerebellum && python3 -m osmosis.cerebellum \
  --ablation /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/v4-sparse-replay/ablation_results.json \
  --plan /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/v4-sparse-replay/ablation_plan.json \
  --source-gguf /var/home/deucebucket/games/models/Qwen3-0.6B-f16.gguf \
  --budget-gb 0.462 \
  --base-type Q2_K \
  --imatrix /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/imatrix/cerebellum_imatrix_ncall8.dat \
  --quantize-bin /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --output /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/v4-sparse-replay/tensor_types_v4_replay_tiny_462mb.txt \
  -v'
```

- Overrides: `196`
- Promotion budget: `121.0 MB`; spent `120.9 MB`
- Final estimated size: `0.46 GB`
- Final size: `447M`, `440.56 MiB`, `4.92 BPW`
- Type mix: `F32=113`, `Q2_K=1`, `Q4_K=85`, `Q5_K=92`, `Q6_K=20`
- Final PPL: `19.7198 +/- 0.36159`
- Interpretation: closest tiny reproduction of Qwen 27B v4 behavior. It is
  about 15 MB smaller than Q4_K_M and slightly better on PPL.

Baseline comparisons:

| Artifact | Size | PPL |
|---|---:|---:|
| Pure Q2_K + imatrix | 332M | `116.9828 +/- 2.58370` |
| Qwen27-v4-shaped 420 MB replay | 407M | `22.8992 +/- 0.43223` |
| Qwen27-v4-shaped 462 MB replay | 447M | `19.7198 +/- 0.36159` |
| Q4_K_M + imatrix baseline | 462M | `19.7289 +/- 0.36559` |

Conclusion:

- The exact Qwen 27B v4-shaped process does reproduce on tiny Qwen.
- At too small a budget, the allocator repairs Q2_K but cannot beat Q4_K_M.
- At roughly Q4 size, it gets a small size win and a tiny PPL win.
- This validates the process shape; the huge gains on Qwen 27B came from
  larger-model redundancy and a better budget/scale regime, not from the newer
  group-forward flow.

### 2026-06-07 Benchmark Sidecar

Because PPL is not enough, ran 200-question MCQ smoke comparisons against the
Q4_K_M imatrix baseline and the 462 MB sparse replay.

Artifacts:

- Baseline: `/var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/benchmark_sidecar/baseline/`
- Replay: `/var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/benchmark_sidecar/replay/`
- Summary sidecar: `/var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/v4-sparse-replay/benchmark_sidecar_summary.json`

| Benchmark | Q4_K_M baseline | 462 MB replay | Delta |
|---|---:|---:|---:|
| ARC-Challenge 200 | 51.0% | 45.0% | -6.0 |
| HellaSwag 200 | 29.0% | 38.5% | +9.5 |
| MMLU-Redux 200 | 40.0% | 37.5% | -2.5 |

Interpretation:

- The 462 MB replay is not a blanket better tiny model. It is smaller and PPL is
  slightly better, but downstream task behavior is mixed.
- HellaSwag improved materially, while ARC and MMLU-Redux regressed on this
  small smoke subset.
- Treat this as a successful pipeline reproduction and a useful counterexample:
  the OG method transfers mechanically, but tiny dense Qwen does not reproduce
  the 27B-scale downstream win profile.

### 2026-06-07 Well-Rounded PPL Follow-Up

Ran a broader PPL panel on the Q4_K_M baseline, the old 462 MB sparse replay,
and a new balanced-panel allocation. Corpora:

- `/var/home/deucebucket/games/cerebellum-calibration/cerebellum_calibration_wiki.txt`
- `/var/home/deucebucket/games/cerebellum-calibration/cerebellum_calibration_code.txt`
- `/var/home/deucebucket/games/cerebellum-calibration/cerebellum_calibration_math.txt`
- `/var/home/deucebucket/games/cerebellum-calibration/cerebellum_calibration_dialogue.txt`
- `/var/home/deucebucket/games/cerebellum-calibration/cerebellum_calibration_agent.txt`

Artifacts:

- Multidomain ablation: `/var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/well-rounded-ppl/ablation_results_multidomain_23.json`
- Balanced tensor map: `/var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/well-rounded-ppl/tensor_types_balanced_462mb.txt`
- Balanced GGUF: `/var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/well-rounded-ppl/final/qwen3-0.6b-cerebellum-balanced-462mb.gguf`
- Summary JSON: `/var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/well-rounded-ppl/summary_balanced_vs_old.json`

The 23-probe balanced ablation found `16 sacred`, `7 neutral`, and `0 demote`
with weights `wiki:0.20,code:0.25,math:0.20,dialogue:0.20,agent:0.15`.
Human-readable verdicting on relative domain deltas was `17 PROTECT`, `6 MIXED`,
and `0 DEMOTE`.

Key early tensor calls:

- `blk.0.ffn_down.weight`: broad protect.
- `blk.1.ffn_gate.weight`: broad protect.
- `blk.2.ffn_gate.weight`: mixed/protect; WikiText improves, but code, math,
  and dialogue worsen.
- `blk.0.ffn_up.weight`: mixed/neutral, not a free demotion.

Broad PPL deltas vs Q4_K_M baseline:

| Model | Wiki | Code | Math | Dialogue | Agent | Avg rel |
|---|---:|---:|---:|---:|---:|---:|
| Old sparse replay 462 MB | -0.26% | -0.89% | -0.03% | -1.59% | -0.72% | -0.70% |
| Balanced 462 MB | +0.20% | -0.75% | -0.15% | -1.58% | -0.68% | -0.59% |

200-question benchmark smoke:

| Benchmark | Q4_K_M baseline | Old replay 462 MB | Balanced 462 MB |
|---|---:|---:|---:|
| ARC-Challenge 200 | 51.0% | 45.0% | 45.5% |
| HellaSwag 200 | 29.0% | 38.5% | 36.0% |
| MMLU-Redux 200 | 40.0% | 37.5% | 39.0% |

Interpretation:

- Well-rounded PPL confirms the tiny Qwen has many real golden-calf tensors;
  early FFN down/gate tensors are not disposable.
- The old sparse replay is still the best broad-PPL candidate, but the balanced
  candidate recovers some ARC/MMLU while giving back some HellaSwag.
- This supports the working theory that PPL is a quick damage meter, not a
  final quality claim. WikiText may preserve general-language behavior, but a
  release recipe needs domain PPL plus benchmark gates.

### CLI / Watch Integration

Added a public CLI path for this exact replay shape:

```bash
python -m osmosis.hillstep sparse-replay \
  --source-gguf /var/home/deucebucket/games/models/Qwen3-0.6B-f16.gguf \
  --corpus /var/home/deucebucket/games/osmosis-quants/wiki.test.raw \
  --run-dir /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/v4-sparse-replay \
  --baseline-ppl 19.7289 \
  --baseline-gguf /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/direct-smoke/artifacts/qwen3-0.6b-q4km-imatrix.gguf \
  --imatrix /var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/imatrix/cerebellum_imatrix_ncall8.dat \
  --budget-gb 0.462 \
  --model-name qwen3-0.6b \
  --family qwen3 \
  --distrobox ai
```

The existing completed run was normalized to `cerebellum.sparse_replay.v1`,
with final GGUF and benchmark metadata written into `state.json`/`manifest.json`.
`cerebellum watch` now shows it as measurement-style scan findings, not group
verdicts, and `/mobile?run_dir=...` exposes final and benchmark rows.

HF release pipeline gap:

- Prior releases put scores into README/model-card text and benchmark artifact
  directories, but did not populate HF structured Evaluation Results.
- Fix needed for future releases: emit model-card `model-index` metadata and/or
  `.eval_results/*.yaml` Community Evals files from benchmark artifacts.
