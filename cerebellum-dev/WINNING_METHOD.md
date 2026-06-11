# Cerebellum Winning Method — Canonical Formula
Date: 2026-06-11
Source forensics: cerebellum-dev/forensics_2026-06-11/claude_sessions.md,
                  cerebellum-dev/forensics_2026-06-11/other_clis_sessions.md,
                  cerebellum-dev/QWEN36_27B_V1_TO_V4_PLAYBOOK_2026-06-07.md,
                  cerebellum-dev/OG_CEREBELLUM_RECONSTRUCTION_GUIDES_2026-06-07.md

---

## The OG Formula (group-first, benchmark-gated)

This is the method that produced every shipped Cerebellum model. The hill-climber
(hillstep.py) is a failed deviation; see DEAD_PATHS.md.

### Step sequence

1. Convert HF/BF16 source to F16 GGUF (single pass, no intermediate quants).
2. Generate or import an imatrix with full tensor coverage (including experts/routers
   for MoE models). Coverage gaps are fatal on MoE architectures (see DEAD_PATHS.md,
   entry: local 205-entry Gemma imatrix).
3. Build uniform baselines: Q2_K+imatrix, Q3_K_M+imatrix, Q4_K_M+imatrix. Measure
   WikiText PPL. For models where Q2_K+imatrix beats Q3_K_M+imatrix on PPL, the
   Q2_K-base path is correct (Qwen 3.6 27B: Q2_K=7.4996 vs Q3_K_M=7.6413).
4. Group ablation (~20-25 test points): crush each named tensor group to Q2_K
   against the chosen baseline, measure PPL delta. Groups are: attn_q/k/v/qkv,
   attn_output/gate, ffn_gate/up/down (dense), ffn_gate_exps/up_exps/down_exps/
   shexp (MoE), router/ffn_gate_inp, ssm_alpha/beta/out (hybrid SSM), norms.
5. Classify each group:
   - PROTECT (>+1.5% PPL delta): never demote.
   - DEMOTABLE (+0.0 to +1.5%): demote in v3/override file.
   - FREE (<= 0%): demote, may improve PPL.
6. Reverse ablation from fully-demoted v1: confirm which groups show real regression
   when restored. If regression is uniform across all layers, no surgical layer
   selection needed; keep all at same quant.
7. Optional per-layer ablation: if a group shows interesting per-layer variation,
   probe key layers (first/last/mid). Result may identify critical layers for
   surgical promotion.
8. Optional router curve: for MoE models, test router tensor at Q2_K, Q3_K, Q4_K,
   Q6_K, Q8_0, F16 to find safe minimum. K-quants may be broken on router tensors
   (Gemma 26B: Q6_K +15.9%, Q2_K +17.2%, Q8_0 safe — see claude_sessions.md §2A).
9. Build override file: list tensor=quant pairs; all unlisted tensors take the base
   quant. Run stock llama-quantize with --imatrix and --tensor-type-file.
10. PPL sanity check on output GGUF.
11. Benchmark gates (ARC, HellaSwag, MMLU-Redux, HumanEval+) against a same-size
    uniform-quant baseline. PPL alone is not sufficient — see DEAD_PATHS.md hillstep
    entry for proof.
12. Audit wrong answers (docs/benchmark_protocol.md audit gate) before recording
    any score.
13. If benchmark gates pass, ship. If not, inspect which tensor group is causing
    regression and selectively promote.

---

## Instantiation A: Gemma 4 26B-A4B Cerebellum v6/v6.1

Source: forensics_2026-06-11/claude_sessions.md §2A.

**Source GGUF:** `/var/home/deucebucket/games/models/gemma-4-26B-A4B-it-bf16.gguf`
**Imatrix:** `osmosis-gemma4-26b/google_gemma-4-26B-A4B-it-imatrix.gguf`
  bartowski/ggml import; 590 tensors (295 entries), 822 chunks; full expert coverage.
**Override file:** `osmosis-gemma4-26b/cerebellum_v6_overrides.txt`
  91 entries, 2636 bytes.

Override distribution (from claude_sessions.md):
- 9 selected `attn_q` tensors -> Q5_K (sensitivity-confirmed)
- Selected `attn_k` -> Q2_K (group ablation: -12.1% PPL)
- 22 layers `ffn_up` -> Q2_K (group ablation: -18.2% PPL, most demotable group)
- All 30 `ffn_gate_up_exps` -> Q2_K (group ablation: -5.5% PPL)
- 4 critical `attn_k` layers (5, 11, 16, 29) -> Q4_K
  (per-layer ablation identified these; below Q4_K is catastrophic: +51.2% PPL,
  -13.4 HumanEval in the early v6 attempt that was discarded)

**Router surgery** (applied after llama-quantize, via gguf_tensor_surgery.py):
- `blk.8.ffn_gate_inp.weight` -> Q8_0
- Evidence: router curve study showed K-quants broken (+15.9% for Q6_K, +17.2% for
  Q2_K); Q8_0 was the only safe option. Layer 8 was identified by ablating all 30
  router layers and finding layer 8's contribution to be highest.

**Build command (reconstructed):**
```bash
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --imatrix /var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/google_gemma-4-26B-A4B-it-imatrix.gguf \
  --tensor-type-file /var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/cerebellum_v6_overrides.txt \
  /var/home/deucebucket/games/models/gemma-4-26B-A4B-it-bf16.gguf \
  out.gguf \
  Q3_K_M
```

**Router surgery (post-quantize):**
```bash
python scripts/gguf_tensor_surgery.py recast \
  --tensor blk.8.ffn_gate_inp.weight \
  --type Q8_0 \
  v5_base.gguf gemma-4-26B-A4B-it-cerebellum-v6.gguf
```

**PPL (shipped GGUF, post-surgery):** 12,054 (confirmed from router surgery ablation;
the 12,894 in v6_ppl.log is stale pre-surgery intermediate — see DEAD_PATHS.md §LF-6).

**Benchmarks (v6 = v6.1 tensor allocation is identical):**
| Benchmark | Score | Source |
|---|---:|---|
| ARC-Challenge | 95.56% (1120/1172) | cerebellum_v6_arc_results.json |
| HellaSwag | 84.55% (10042 questions) | cerebellum_v6_hellaswag_results.json |
| MMLU-Redux | 71.33% (1712/2400) | cerebellum_v6_mmlu_results.json |
| HumanEval+ (chat, correct harness) | ~89-92% | estimated from Heretic v1 rerun; v6 raw-completions score of 35.97% is invalid |

**v6.1:** Same tensor allocation. Only GGUF metadata and runtime chat-template
compatibility updated (commit ded491334, 2026-05-22).
SHA256: `d24229facdef8360a7ffa8b37a50e1de636b9139a5eba0efe899828e45ae7989`
mmproj: `gemma-4-26b-a4b-it.mmproj.gguf` (same dir)

---

## Instantiation B: Qwen 3.6 27B Cerebellum v4

Source: forensics_2026-06-11/claude_sessions.md §2D,
        cerebellum-dev/QWEN36_27B_V1_TO_V4_PLAYBOOK_2026-06-07.md.

**Source GGUF:** `/var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf` (converted from BF16 HF, 51 GB)
**Imatrix:** `osmosis-qwen36-27b/cerebellum_imatrix.dat` (renamed from osmosis_imatrix.dat)
  496 entries, ncall=8, dataset=osmosis-sensitivity; old legacy format but valid.
**Override file:** `osmosis-qwen36-27b/tensor_types_v4_12gb.txt`
  181 overrides: 22 Q2_K, 19 Q3_K, 22 Q4_K, 70 Q5_K, 41 Q6_K, 7 Q8_0.

**Imatrix generation:**
```bash
python -m osmosis.imatrix_stream \
  --model Qwen/Qwen3.6-27B \
  --output osmosis-qwen36-27b/osmosis_imatrix.dat \
  --mode calibrated \
  --num-samples 8 \
  --source-gguf qwen3.6-27b-f16.gguf
```

**Ablation basis:** 23 sparse PPL probes from `ablation_results.json`, baseline PPL 8.2556.
Most sensitive tensors:
- `blk.63.attn_q.weight`: +0.1622 (protect)
- `blk.63.ffn_down.weight`: +0.1377 (protect)
Best demotable:
- `blk.2.ffn_gate.weight`: -0.1467 (actively better at Q2_K)
- `blk.32.attn_qkv.weight`: -0.1335 (actively better at Q2_K)

Key baseline PPL comparison that validated the Q2_K-base approach:
| Config | PPL |
|---|---:|
| Q2_K + imatrix | 7.4996 |
| Q3_K_M + imatrix | 7.6413 |
Q2_K with imatrix already beat Q3_K_M. This made the aggressive-demotion path correct.

**Budget allocator command:**
```bash
python -m osmosis.cerebellum \
  --ablation osmosis-qwen36-27b/ablation_results.json \
  --plan osmosis-qwen36-27b/ablation_plan.json \
  --source-gguf /var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf \
  --budget-gb 12.0 \
  --base-type Q2_K \
  --imatrix osmosis-qwen36-27b/osmosis_imatrix.dat \
  --quantize-bin /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --output osmosis-qwen36-27b/tensor_types_v4_12gb.txt \
  -v
```

**Final build command:**
```bash
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --imatrix /var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/osmosis_imatrix.dat \
  --tensor-type-file /var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/tensor_types_v4_12gb.txt \
  /var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf \
  qwen3.6-27b-cerebellum-v4.gguf \
  Q2_K
```

**v4 benchmarks (corrected, post 2026-05-03 bug fixes):**
| Benchmark | Score | Count |
|---|---:|---:|
| ARC-Challenge | 96.76% | 1134/1172 |
| HellaSwag | 92.21% | 9260/10042 |
| MMLU-Redux | 76.58% | 1838/2400 |
| HumanEval pass@1 | 81.10% | 133/164 equiv. |
| WikiText PPL | 7.0344 | 2048 ctx |

Source: `benchmarks/qwen36-27b/` (post-correction artifacts).
Pre-correction scores (ARC 95.1%, HumanEval 75.0%) are from stale benchmark files;
do not publish.

---

## Instantiation C: Qwen 3.6 35B-A3B Cerebellum v3

Source: forensics_2026-06-11/claude_sessions.md §2E,
        forensics_2026-06-11/RECIPE_heretic_qwen36_35b.md §Q2.

**Source GGUF:** `/var/home/deucebucket/games/models/staging-qwen36-35b/` (65 GB BF16 2-split)
**Imatrix:** `/var/home/deucebucket/games/qwen36-35b-v2/imatrix_unsloth.gguf_file`
  192 MB, 1020 tensors, 76 chunks, Unsloth coder calibration dataset.
**Override file:** `/var/home/deucebucket/games/qwen36-35b-v2/cerebellum_v3_overrides.txt`
  360 entries, 11230 bytes.
  Pattern: blk.0-39 (40 layers), each with 9 tensor types all -> Q2_K:
  `ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_shexp, ffn_up_shexp,
  ffn_down_shexp, attn_gate, ssm_alpha, ssm_beta`
  Protected (stays at Q3_K_M default): `attn_qkv, ssm_out`, all F32 SSM state
  params, router tensors.

**Group ablation results vs Q3_K_M baseline (PPL 7.1758):**
| Group at Q2_K | PPL | Delta | Verdict |
|---|---:|---:|---|
| ssm_out | 7.4132 | +3.3% | PROTECT |
| attn_qkv | 7.2766 | +1.4% | PROTECT |
| ffn_down_exps | 7.3136 | +1.9% | DEMOTABLE (demoted in v3) |
| ffn_up_exps | +1.2% | DEMOTABLE (demoted in v3) | — |
| ffn_gate_shexp | +0.6% | DEMOTABLE | — |
| attn_gate | +0.1% | FREE | — |
| ssm_alpha | -0.01% | FREE | — |
| ssm_beta | +0.06% | FREE | — |

Reverse ablation: restoring attn_qkv, ffn_down_exps, ffn_up_exps showed genuine
improvement, but per-layer ablation found the effect uniform across all 40 layers
(no surgical layers). v3 decision: demote all anyway at Q2_K and rely on the imatrix
to compensate. Result validated by benchmark gates.

**Router surgery:** Zero signal on 256-expert MoE. Not applied.

**Build command:**
```bash
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --imatrix /var/home/deucebucket/games/qwen36-35b-v2/imatrix_unsloth.gguf_file \
  --tensor-type-file /var/home/deucebucket/games/qwen36-35b-v2/cerebellum_v3_overrides.txt \
  <source-bf16.gguf> \
  /var/home/deucebucket/games/qwen36-35b-v2/Qwen3.6-35B-A3B-Cerebellum-v3.gguf \
  Q3_K_M
```

The pipeline script is at `/var/home/deucebucket/games/qwen36-35b-v2/build_v2_pipeline.sh`.

**v3 benchmarks (shipped, vs Q3_K_M baseline):**
| Benchmark | v3 (11 GB) | Q3_K_M (16 GB) |
|---|---:|---:|
| ARC-Challenge | 95.82% | 96.10% |
| HellaSwag | 92.28% | 91.50% |
| MMLU-Redux | 75.00% | 74.12% |
| HumanEval base | 70.73% | 64.02% |
| HumanEval+ | 65.24% | 56.71% |
| Wiki PPL | 7.4307 | 7.1758 |
| Size | 11 GB | 15.6 GB |

Source: `games/qwen36-35b-v2/benchmark_results_v3/` (verified local).

---

## Heretic Transfer Protocol

Source: forensics_2026-06-11/RECIPE_heretic_qwen36_35b.md §Q1,
        forensics_2026-06-11/claude_sessions.md §2C.

**When to use:** Base model has a proven Cerebellum override file. A "heretic" variant
(e.g., coder3101, llmfan46 uncensored) uses the same architecture with modified
activations (orthogonal projection on o_proj/out_proj) but identical tensor layout.

**Protocol:**
1. Verify source architecture: count layers, check tensor naming, confirm no extra
   blocks (e.g., MTP block blk.40 — fatal if present, see DEAD_PATHS.md).
   ```bash
   distrobox enter ai -- llama-gguf --list $SRC_BF16 2>/dev/null | grep "blk\.40" \
     || echo "NO BLK.40 — clean non-MTP"
   ```
2. Transfer override file verbatim. No re-ablation needed: heretic ablation modifies
   activations not weight tensor layout, so tensor sensitivity transfers exactly.
3. Use the same imatrix as the stock build. The imatrix is architecture-tied, not
   checkpoint-tied.
4. Run llama-quantize with the same base quant and override file.
5. Benchmark against the stock heretic's same-size uniform-quant baseline (not the
   Cerebellum stock baseline). Gate criteria: match or exceed stock heretic Q3_K_M/
   Q4_K_M on all four benchmarks.

**Validated example — Gemma 4 26B Heretic v1:**
- Source: `coder3101/gemma-4-26B-A4B-it-heretic` BF16
- Override: identical to v6 (`cerebellum_v6_overrides.txt`, 91 entries)
- Imatrix: same bartowski imatrix used for v6
- All 91 override entries matched (658 tensors loaded, confirmed)
- Result (fresh full rerun): ARC 95.48%, HellaSwag 83.49%, MMLU 71.42%,
  HumanEval base 92.07%, HumanEval+ 89.63%
- Note: refusal rate dropped from 57.8% (regular v6) to 2.2% (heretic v1).
  Tensor allocation does not drive safety behavior; base model ablation training does.

**Contra-example — Qwen 3.6 35B Heretic first attempt (2026-06-03, FAILED):**
- Source used: `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF`
- That source contained block blk.40 (MTP), a different architecture than stock.
- llama.cpp load failure; benchmarks showed -14 HS / -32 HE+ regression.
- The correct non-MTP source is `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF`
  (69.4 GB BF16 plain, no blk.40).

---

## Formula vs Hill-Climber: Summary

The hillstep block-10 checkpoint (Gemma 4 12B, 2026-06-04/06) achieved 35% wiki PPL
improvement over Q4_K_M baseline, was 1.02 GiB larger, and scored -14.03 pts on
HumanEval+. That is the definitive proof that wiki PPL optimization without benchmark
gating produces wrong models. Full evidence in DEAD_PATHS.md.
