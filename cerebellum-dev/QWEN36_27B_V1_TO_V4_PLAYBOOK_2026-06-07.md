# Qwen 3.6 27B Cerebellum v1 -> v4 Playbook

Date: 2026-06-07
Scope: private reconstruction of the original Qwen3.6-27B Cerebellum/Osmosis lineage from local artifacts, Claude session logs, benchmark artifacts, and Spark-agent archaeology.

Do not publish this file to `origin`. It references local paths, private chat logs, and raw experiment state.

## Executive conclusion

The user's memory is mostly right, with one important nuance: early Qwen3.6-27B did have size pressure (`8.5gb`, `10gb`, `12gb` artifacts), but the winning idea was not "fit a normal target-size quant." The winning idea was:

1. Start from BF16/HF source converted to F16 GGUF.
2. Build and use an imatrix.
3. Smash aggressively with a Q2_K base.
4. Identify which tensors survive, improve, or break when crushed.
5. Spend precision back only on tensors that proved they mattered.

The size win emerged because v2 proved the Q2_K-base mixed-precision direction was viable. v4 is the polished 12GB budget spend on top of that discovery.

The key evidence is v1 -> v2:

| Version | Size | BPW | Type shape | Wiki PPL |
|---|---:|---:|---|---:|
| v1 | 14.86 GiB | 4.74 | Q4-ish mixed, many Q4/Q5/Q6 tensors | 7.6713 |
| v2 | 10.67 GiB | 3.41 | Q2_K base, many Q2/Q3 tensors | 7.0868 |
| v3 | 10.14 GiB | 3.24 | budget experiment | 7.3156 |
| v3c | 10.58 GiB | 3.38 | budget experiment, tied v2 | 7.0870 |
| v4 | 11.97 GiB | 3.82 | Q2_K base plus high-bit sacred promotions | 7.0344 |

That is the original "smaller and better" effect: v2 was dramatically smaller than v1 and also lower PPL. v4 then spent back to 12GB and became the release champion.

## Primary evidence

Main local tree:

- `osmosis-qwen36-27b/`
- `benchmarks/qwen36-27b/`
- `cerebellum-dev/OG_CEREBELLUM_RECONSTRUCTION_GUIDES_2026-06-07.md`

Primary chat/session logs:

- `/var/home/deucebucket/.claude/projects/-var-home-deucebucket-ai-drive-osmosis/f866b942-19bd-469c-893c-fc013b0d80c1.jsonl`
- `/var/home/deucebucket/.claude/projects/-var-home-deucebucket-ai-drive-cerebellum/5bddbf71-de3a-4f1e-98a8-e7a5ac013466.jsonl`

Important transcript findings:

- Osmosis/Cerebellum method was framed as crushing tensors to Q2_K and measuring PPL impact.
- Qwen v4 allocator work used `osmosis.cerebellum` with `ablation_results.json`, `ablation_plan.json`, and `/var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf`.
- Branch/history text mentions v3/v4 progression: v3 allocations, v3c tying v2, v4 beating the Q2_K_XL comparison at roughly 12GB.
- Chat logs also show the size-estimation bug work: old estimator used raw BPW and underspent because imatrix-compressed sizes differed from the estimate.

## Initial setup

The initial overnight log shows the actual source path:

- Downloaded Qwen3.6-27B to `/var/tmp/osmosis-qwen36/Qwen3.6-27B`.
- HF snapshot size: `52G`.
- Converted source to F16 GGUF: `/var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf`.
- Converted weights show `torch.bfloat16 --> F16`, so yes, this lineage started from BF16/HF weights via F16 GGUF, not from an already quantized GGUF.
- F16 GGUF size: `51G`.

The original imatrix:

- Path in old logs: `osmosis-qwen36-27b/osmosis_imatrix.dat`.
- Current renamed path: `osmosis-qwen36-27b/cerebellum_imatrix.dat`.
- Size: about `13M`.
- Entries: `496`.
- Chunks / ncall: `8`.
- Dataset metadata: `osmosis-sensitivity`.
- llama.cpp logs load it as old-format legacy imatrix, but valid.

Baseline quantizations were built from the F16 GGUF using that imatrix:

```bash
llama-quantize --imatrix osmosis-qwen36-27b/osmosis_imatrix.dat \
  /var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf \
  osmosis-qwen36-27b/qwen3.6-27b-osmosis-imatrix-Q4_K_M.gguf \
  Q4_K_M

llama-quantize --imatrix osmosis-qwen36-27b/osmosis_imatrix.dat \
  /var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf \
  osmosis-qwen36-27b/qwen3.6-27b-osmosis-imatrix-Q3_K_M.gguf \
  Q3_K_M

llama-quantize --imatrix osmosis-qwen36-27b/osmosis_imatrix.dat \
  /var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf \
  osmosis-qwen36-27b/qwen3.6-27b-osmosis-imatrix-Q2_K.gguf \
  Q2_K
```

Observed baseline PPL anchors:

| Artifact | Size | PPL |
|---|---:|---:|
| Q2_K + imatrix | 9.97 GiB | 7.4996 |
| Q2_K no imatrix | 9.97 GiB | 7.6494 |
| Q3_K_M + imatrix | 12.38 GiB | 7.6413 |
| Q4_K_M + imatrix ablation baseline | n/a | 8.2556 |

The Q2_K imatrix baseline already beat Q3_K_M imatrix on Wiki PPL. That is why the Q2-base path was so important.

## Ablation basis

The saved v4 ablation input is not a complete 851-tensor brute force sweep. Local evidence shows:

- `ablation_plan.json`: 23 planned probes.
- `ablation_results.json`: 23 measured single-tensor PPL probes.
- `interaction_results.json`: 6 interaction tests.
- `tensor_types_v4_12gb.txt`: 181 final overrides.

The public/model-card language says "each tensor" or "full ablation." Treat that as compressed methodology language, not the exact saved local v4 data. The saved local recipe is sparse PPL probes plus allocator extrapolation.

Most sensitive probes from `ablation_results.json`, baseline PPL `8.2556`:

| Tensor | PPL when crushed | Delta | Meaning |
|---|---:|---:|---|
| `blk.63.attn_q.weight` | 8.4178 | +0.1622 | sacred |
| `blk.63.ffn_down.weight` | 8.3933 | +0.1377 | sacred |
| `blk.1.ffn_gate.weight` | 8.2941 | +0.0385 | sensitive |
| `blk.18.ssm_alpha.weight` | 8.2810 | +0.0254 | sensitive |
| `blk.62.ssm_out.weight` | 8.2732 | +0.0176 | sensitive |

Best demotable / regularizing probes:

| Tensor | PPL when crushed | Delta | Meaning |
|---|---:|---:|---|
| `blk.2.ffn_gate.weight` | 8.1089 | -0.1467 | actively better at Q2_K |
| `blk.32.attn_qkv.weight` | 8.1221 | -0.1335 | actively better at Q2_K |
| `blk.34.ffn_down.weight` | 8.1610 | -0.0946 | actively better at Q2_K |
| `blk.0.ffn_up.weight` | 8.1988 | -0.0568 | actively better at Q2_K |
| `blk.0.ffn_down.weight` | 8.2015 | -0.0541 | actively better at Q2_K |

Interaction tests:

- Cross-layer additive demotion: expected delta `-0.5567`, actual `-0.4774`, interaction ratio `0.8575`.
- Same/near-layer ladder step 4: expected `-0.4316`, actual `-0.0568`, interaction ratio `0.1316`.

Interpretation: single-tensor probes worked well enough for cross-layer additive effects, but stacking same-layer FFN demotions could destroy the expected benefit. That is one reason later group-level pipelines can diverge from the OG v4 result.

## Version play-by-play

### v1: conservative mixed precision

Artifact:

- `/var/home/deucebucket/games/osmosis-quants/qwen3.6-27b-cerebellum-v1.gguf`

Loader shape:

- File type: `Q4_K - Medium`
- Size: `14.86 GiB`
- BPW: `4.74`
- Tensor counts: `F32 353`, `Q8_0 9`, `Q3_K 126`, `Q4_K 238`, `Q5_K 88`, `Q6_K 37`
- Wiki PPL: `7.6713 +/- 0.05718`

No saved `tensor_types_v1*.txt` was found. The exact map has to be inferred from loader counts.

Interpretation:

- v1 was still protection-heavy and Q4-ish.
- It was not the magic. It was a working checkpoint that proved mixed precision could run, but it was larger and worse than v2.

### v2: breakthrough Q2_K-base smash map

Artifact:

- `/var/home/deucebucket/games/osmosis-quants/qwen3.6-27b-cerebellum-v2.gguf`

Loader shape:

- File type: `Q2_K - Medium`
- Size: `10.67 GiB`
- BPW: `3.41`
- Tensor counts: `F32 353`, `Q2_K 241`, `Q3_K 187`, `Q4_K 49`, `Q5_K 20`, `Q6_K 1`
- Wiki PPL: `7.0868 +/- 0.04737`

This is the key "we smashed it and it got better" artifact:

- v1: 14.86 GiB, PPL 7.6713
- v2: 10.67 GiB, PPL 7.0868

Historical benchmark files exist for v2/v2b, but they are not final release-quality:

| Variant | ARC | HellaSwag | HumanEval | MMLU-Redux |
|---|---:|---:|---:|---:|
| v2 | 84.81 | 75.02 | 63.40 | 57.29 |
| v2b | 85.67 | 75.34 | 68.30 | 58.42 |

Use these as lineage clues only. They were measured before the later benchmark-audit discipline.

Interpretation:

- v2 made Q2_K the base and only spent precision on selected tensors.
- This is where the original Cerebellum idea became real.
- PPL looked excellent, but downstream benchmark handling was still immature.

### v3 / v3c: budget-map experiments

Explicit tensor maps start here:

| Tensor file | Overrides | Mix |
|---|---:|---|
| `tensor_types_v3_8.0gb.txt` | 41 | 22 Q2_K, 19 Q3_K |
| `tensor_types_v3_10.7gb.txt` | 115 | 22 Q2_K, 79 Q3_K, 8 Q4_K, 6 Q5_K |
| `tensor_types_v3_12gb.txt` | 181 | 22 Q2_K, 119 Q3_K, 22 Q4_K, 18 Q5_K |
| `tensor_types_v3c_11.4gb.txt` | 181 | 22 Q2_K, 119 Q3_K, 22 Q4_K, 18 Q5_K |

PPL:

- v3: `10.14 GiB`, `3.24 BPW`, PPL `7.3156`.
- v3c: `10.58 GiB`, `3.38 BPW`, PPL `7.0870`.

Failed or intermediate routes around this phase:

| Attempt | Size | PPL | Lesson |
|---|---:|---:|---|
| `ppl_budget_12gb.log` | 12.42 GiB | 8.4998 | bad budget path / likely no useful imatrix behavior |
| `ppl_budget_12gb_v4_with_imatrix.log` | 11.17 GiB | 7.9491 | imatrix alone did not fix bad allocation |
| `ppl_demote_12gb.log` | 10.95 GiB | 7.2225 | demotion helped but not enough |
| `ppl_sacred_q8.log` | 11.39 GiB | 7.3797 | just promoting sacred tensors was not enough |
| `ppl_budget_10gb_konly.log` | 10.19 GiB | 7.9066 | narrow K-only policy was weak |

Interpretation:

- This is where size budgeting became explicit.
- Several "reasonable" budget ideas failed.
- v3c found a low-size map that tied v2, but v4 changed the distribution of the same 181 overrides to spend much harder on sacred tensors.

### v4: final 12GB release allocation

Final tensor map:

- `osmosis-qwen36-27b/tensor_types_v4_12gb.txt`
- 181 overrides:
  - 22 `q2_K`
  - 19 `q3_K`
  - 22 `q4_K`
  - 70 `q5_K`
  - 41 `q6_K`
  - 7 `q8_0`

Loader shape:

- File type: `Q2_K - Medium`
- Size: `11.97 GiB`
- BPW: `3.82`
- Tensor counts: `F32 353`, `Q8_0 7`, `Q2_K 253`, `Q3_K 77`, `Q4_K 49`, `Q5_K 70`, `Q6_K 42`
- Wiki PPL: `7.0344 +/- 0.04625`

Reconstructed build command:

```bash
llama-quantize \
  --imatrix /var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/osmosis_imatrix.dat \
  --tensor-type-file /var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/tensor_types_v4_12gb.txt \
  /var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf \
  qwen3.6-27b-cerebellum-v4.gguf \
  Q2_K
```

Important: v4 was not just "v3 with more tensors." It kept 181 overrides but moved spending upward:

- v3c: 119 Q3_K, 18 Q5_K, 0 Q6_K, 0 Q8_0.
- v4: 19 Q3_K, 70 Q5_K, 41 Q6_K, 7 Q8_0.

That is the polished version of the v2 insight: keep the Q2_K base, preserve demotables, and spend precision on the sacred tensors.

## Benchmark correction history

The benchmark history is messy and matters.

Known bugs from `BENCHMARK_CORRECTIONS.md`:

- HumanEval was too low by roughly 7-8 points because fence stripping used `.strip()` and destroyed indentation.
- Temperature sweep produced false lows around 25% before the fix; after the fix, the same model reached 81-83%.
- ARC had numeric-answer label mismatch on 19 questions.
- HellaSwag had 108 empty/whitespace responses from thinking-template handling.
- MMLU/MMLU-Redux did not show the same parser bug.
- An attempted `prefix: true` fix failed because llama.cpp did not support that field in the used path.
- Later memory says benchmark wrong answers must be audited before publishing.

Trust these final release-style v4 numbers from `benchmarks/qwen36-27b/`:

| Benchmark | Score | Detail |
|---|---:|---|
| HumanEval | 81.10% | `133/164` equivalent pass@1 |
| ARC-Challenge | 96.76% | `1134/1172` |
| HellaSwag | 92.21% | `9260/10042` |
| MMLU-Redux | 76.58% | `1838/2400` |
| MMLU full | 82.52% | `11643` questions |
| WikiText-2 PPL | 7.0344 | 2048 ctx PPL log |

Use cautiously:

- `osmosis-qwen36-27b/benchmark_results/cerebellum_v4_fixed_*`: partly fixed set; ARC is lower than the later corrected mirror.
- `cerebellum_v4_humaneval_lcpp_results.json`: 0.0, bad path.
- `cerebellum_v4_tuned_humaneval_results.json`: 29.9, bad/tuned path.
- old public memory row: HumanEval 75.0 and ARC 95.1 were pre-final/correction-era.

## What we did differently from later/recent pipelines

1. The original Qwen27 v4 did not start as a clean group-forward pipeline.
   It used sparse tensor probes, KL/damage-guided selection, interaction tests, and allocator extrapolation.

2. It did not start from an already quantized GGUF.
   It came from BF16 HF weights converted to F16 GGUF, then one quantization pass with imatrix and tensor overrides.

3. The first win was not exact budget optimization.
   v2 won by aggressive Q2_K-base smashing plus selective protection. Budgeting became explicit afterward.

4. The imatrix mattered.
   Q2_K with imatrix was PPL 7.4996; Q2_K without imatrix was 7.6494.

5. "Promote sacred only" was not enough.
   `ppl_sacred_q8.log` was 7.3797, worse than v2/v3c/v4. The magic was both demotion and promotion.

6. Same-layer interactions mattered.
   Cross-layer demotions were about 86% additive, but same-layer stacking could collapse to about 13% of expected improvement.

7. Benchmarks were fixed after the model looked good.
   Some early scores were bad because the harness was bad. The model was not necessarily bad.

## Practical reproduction profile

For a faithful OG-Qwen27-style replay on another model, do not begin by forcing a target budget. Use this sequence:

1. Convert HF/BF16 source to F16 GGUF.
2. Generate or import a valid imatrix with enough coverage.
3. Build uniform imatrix baselines: Q2_K, Q3_K_M, Q4_K_M.
4. Run quick WikiText PPL only as a damage sensor, not as final quality.
5. Build a sparse probe plan:
   - highest/lowest KL or damage candidates,
   - early/mid/late layer samples,
   - role probes for attention, SSM, FFN gate/up/down,
   - suspected sacred final layers,
   - suspected demotable early/regularizing tensors.
6. Crush one selected tensor at a time to Q2_K against a stable baseline and measure PPL.
7. Mark:
   - sacred: positive delta, promote in v4-style allocator,
   - demotable: negative delta, keep Q2_K,
   - tolerant: small positive/near-zero, do not over-trust as demotable.
8. Run interaction tests before stacking same-layer demotions.
9. Build a v2-style aggressive Q2_K-base candidate.
10. Only after v2-style candidate survives, add budget allocation and high-bit sacred promotions.
11. Run benchmark gates against same-size baselines using audited benchmark scripts.

## Bottom line for current work

If the tiny Qwen replay started by forcing a neat budget, that was not the OG discovery process. It replayed the final v4 build shape, not the v1 -> v2 discovery that made Cerebellum work.

The correct next experiment should be a v2-first replay:

- Find what can be smashed.
- Build an aggressive Q2_K-base map.
- Compare it to Q2_K, Q3_K_M, and Q4_K_M.
- Only then spend a v4-style budget on sacred tensors.

For small models, the effect may be weaker or absent because there is less redundancy to exploit. Qwen27's success likely depended on a large model having enough redundant/tolerant capacity that Q2_K regularization could remove damage while preserving useful behavior.
