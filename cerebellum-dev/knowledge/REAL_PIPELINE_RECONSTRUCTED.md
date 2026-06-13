# THE REAL CEREBELLUM PIPELINE — Reconstructed From Primary Artifacts

Reconstructed 2026-06-13 from the surviving evidence of the **Qwen3.6-27B v4** build —
the real winning method (75.0% → 81.1% HumanEval territory at 12 GB). Every claim below
cites the artifact file it comes from. The point of this document: the reconstruction
guides (`OG_CEREBELLUM_RECONSTRUCTION_GUIDES_2026-06-07.md`, `docs/multi_domain_ablation.md`)
**flattened away an entire phase** — the **coding ablation** — and replaced it with
multi-domain PPL, which does not capture coding damage. This restores the true method.

---

## The end-to-end method, in order

| # | Phase | What it measures | Artifact proof |
|---|-------|------------------|----------------|
| 1 | HF/BF16 → F16 GGUF | — | (standard) |
| 2 | Imatrix, full coverage | — | `osmosis-qwen36-27b/osmosis_imatrix.dat` |
| 3 | Uniform baselines + WikiText PPL | global PPL floor | `ppl_budget_12gb_*.log` |
| 4 | **Group PPL ablation** (cheap sieve) | per-group PPL Δ | `ablation_results.json` (baseline_ppl 8.2556) |
| 5 | **CODING ABLATION — per group** ← LOST | per-group **HumanEval** Δ | `coding_ablation/coding_ablation.log` |
| 6 | **CODING ABLATION — per layer band** ← LOST | per-band **HumanEval** Δ | `coding_ablation_layers/layer_ablation.log` |
| 7 | **Budget allocation** (protects coding-critical) | size under budget | `tensor_types_v4_12gb.txt`, `ppl_budget_12gb_v4_with_imatrix.log` |
| 8 | PPL sanity check | global PPL | `cerebellum_v4_ppl.log` |
| 9 | Benchmark gates (ARC/HellaSwag/MMLU/HumanEval+) | task ability | `benchmark_results/cerebellum_v4_fixed_*` |

Phases 5 and 6 are the lost steps. The PPL sieve (phase 4) is **cheap and necessary but
blind to coding** — it exists to narrow the candidate set the coding ablation then tests
with a real harness.

---

## Phase 5 — CODING ABLATION (per group)

**Source of truth:** `osmosis-qwen36-27b/coding_ablation/coding_ablation.log`

### Exact procedure (verbatim from the log header, lines 1-9)
```
QWEN3.6-27B CEREBELLUM — PIPELINED CODING ABLATION
Base: v4 (12 GB, 75.0% HumanEval baseline)
Strategy: demote each group to Q2_K, measure HumanEval delta
Pipeline: CPU builds to games drive, GPU evaluates as available
Groups: 7 | Parallel slots: 4
Build dir: /var/home/deucebucket/games/coding_ablation_builds
```

For **each** of the 7 tensor groups:
1. Write an override file demoting **only that group** to Q2_K across all 64 layers.
   Proof: `override_demote_attn_qkv.txt` etc. — flat lists `blk.0.attn_qkv.weight=q2_K`
   … `blk.63.attn_qkv.weight=q2_K`. `output` is the single line `output.weight=q2_K`
   (`override_demote_output.txt`).
2. Build the candidate GGUF with stock `llama-quantize` on **CPU**, written to the games
   partition (`/var/home/deucebucket/games/coding_ablation_builds`). Log: `[BUILD] ffn_down
   done: 10.93 GB`.
3. Serve it with `llama-server` on **port 8084**. Log: `[EVAL] Server up (PID 343187),
   running HumanEval...`.
4. Run **REAL HumanEval pass@1**. Log lines 24-26:
   ```
   HumanEval: 164 total, 0 already done, 164 remaining
   API: http://127.0.0.1:8084/v1/completions
   Settings: temp=0, max_tokens=512, parallel=4
   ```
   The harness executes the generated code (`Running evaluation (executing code)... /
   Running test suites... / Passed: 116, Failed: 48`) and prints `HumanEval pass@1: 70.7%`.
   This is the OpenAI `human_eval` evaluator (`evaluate_functional_correctness`), confirmed
   by the recovered runner (`git show 005437c^:scripts/benchmark_humaneval.py`) and the
   `*_humaneval_samples.jsonl_results.jsonl` output convention.
5. Record the delta vs the **75.0% base build baseline**, classify, and **KEEP the
   least-damaging GGUF / PRUNE the rest** for disk hygiene. Log:
   `[KEEP] Keeping ffn_down GGUF (least damage so far)` vs
   `[PRUNE] Deleted attn_qkv GGUF (scored 28.7%)`.
6. **Pipelined**: builds overlap evals. Log interleaves `[BUILD] attn_qkv done` while
   `[EVAL] [1/7] Evaluating ffn_down`. CPU quantize for candidate N+1 runs during GPU
   HumanEval for candidate N.

### The measured result (log lines 325-340) — the whole reason this phase exists
```
Baseline: 75.0%
  ssm_alpha   :  22.6% (-52.4%) [CRITICAL]
  ssm_beta    :  25.6% (-49.4%) [CRITICAL]
  attn_qkv    :  28.7% (-46.3%) [CRITICAL]
  output      :  29.9% (-45.1%) [CRITICAL]
  attn_v      :  30.5% (-44.5%) [CRITICAL]
  attn_output :  42.1% (-32.9%) [CRITICAL]
  ffn_down    :  70.7% ( -4.3%) [MEDIUM]
CODING TENSOR (most critical): ssm_alpha (-52.4%)  -> promote to max precision
DISPOSABLE (least impact): ffn_down (-4.3%)         -> crush to Q2_K to make room
```
**attn_qkv → Q2_K cost 46.3 HumanEval points** while its group PPL delta in
`ablation_results.json` was a few hundredths (≈8.24–8.29 vs baseline 8.2556 — under a
percent). **PPL is blind to this. HumanEval is not.** That gap is the entire thesis.

---

## Phase 6 — CODING ABLATION (per layer band)

**Source of truth:** `osmosis-qwen36-27b/coding_ablation_layers/layer_ablation.log`

### Procedure (header, lines 1-9)
```
QWEN3.6-27B CEREBELLUM — PER-LAYER CODING ABLATION
Strategy: drill into attn_qkv and attn_v to find critical layers
Phase 1: thirds | Phase 2: individual layers in worst third
Build dir: /var/home/deucebucket/games/layer_ablation_builds
```

**Phase 1 — thirds.** For the coding-critical groups (`ssm_alpha, ssm_beta, attn_qkv,
attn_v`), split the 64 layers into **early / middle / late** bands, demote each band to
Q2_K, re-run HumanEval. Proof of the split: `override_attn_qkv_early.txt` = blk.0–21,
`override_attn_qkv_middle.txt` = blk.22–42, `override_attn_qkv_late.txt` = blk.43–63.
12 builds, same build→serve→eval→PRUNE pipeline. Result (log lines 518-539):
```
attn_qkv: early 31.7% / middle 40.2% / late 27.4%   -> worst third: late
attn_v  : early 34.1% / middle 27.4% / late 26.8%   -> worst third: late
ssm_alpha/ssm_beta: all bands ~25% (uniformly critical — SSM hard-fail, force ≥4-bit)
```
The **middle attn_qkv band tolerates Q2_K far better than the late band** (40.2% vs 27.4%)
— so the budget stage can spend bits on late-layer attn and save them on the middle. PPL
never sees this either.

**Phase 2 — individual layers in worst third.** `PHASE 2: INDIVIDUAL LAYERS IN WORST
THIRDS — Testing 84 individual layers...` then per-layer builds (`override_ssm_alpha_layer43.txt`
= single line `blk.43.ssm_alpha.weight=q2_K`). The log is **truncated mid-phase-2** (cuts
off at `Building ssm_alpha_l45...`), so the full per-layer table did not survive — but the
procedure and override convention are intact, and the script reproduces it.

---

## Phase 7 — Budget allocation PROTECTS the coding-critical tensors

**Proof the coding findings fed the budget:** inspect the shipped v4 map
`osmosis-qwen36-27b/tensor_types_v4_12gb.txt`:
- `ssm_alpha` (most critical, -52.4%) → **q5_K / q6_K** (e.g. `blk.18.ssm_alpha.weight=q6_K`).
- `attn_qkv` (critical, -46.3%) → up to **q8_0** (`blk.1.attn_qkv.weight=q8_0`).
- `ffn_down` (disposable, -4.3%) → carries the **q2_K** entries (`blk.0.ffn_down.weight=q2_K`).

The disposable group absorbs the Q2_K crush so the coding-critical groups can be promoted
under the 12 GB budget. That is the coding ablation, executed. Allocator command shape is
in the OG guide (`python -m osmosis.cerebellum … --output tensor_types_v4_12gb.txt`).

---

## THE DIVERGENCE — what `scripts/ablate_multidomain.py` does vs the real method

`scripts/ablate_multidomain.py` is what the 2026-06-12/13 overnight reconstruction ran in
place of the coding ablation. Phase-by-phase, it is **not the same step**:

| Dimension | Real coding ablation (phases 5-6) | `ablate_multidomain.py` |
|-----------|-----------------------------------|--------------------------|
| **What's measured** | **HumanEval pass@1** — code is generated, executed, unit-tested | **Perplexity** on a code-text corpus (`measure_per_domain`, domains `wiki,code,math,dialogue`, lines 95-107) |
| **Signal type** | Task ability (does the code run and pass?) | Next-token likelihood on code *prose* |
| **Engine** | `llama-server` + HumanEval harness (`run_humaneval`) | `llama-perplexity` (`measure_ppl`, lines 72-92) |
| **Captures attn_qkv→Q2_K = -46pts?** | **YES** (28.7%) | **NO** — PPL on code text barely moves |
| **Layer drill** | early/middle/late bands by HumanEval, then per-layer | none (per-tensor PPL only) |
| **Output** | protect/tolerant by *coding* | per-tensor PPL deltas |
| **Pipelining** | CPU build ↔ GPU **HumanEval** (port 8084, parallel 4) | CPU quantize ↔ GPU **PPL** (queue, N=2 workers) |

`ablate_multidomain.py`'s own header sells "code" as a domain — but **"PPL on code text"
is a proxy that does not capture HumanEval damage**. It is the same category error as
hillstep's wiki-PPL-only gating (which lost 14 HumanEval+ points while PPL improved 35%).
Multi-domain PPL is a *better* sieve than single-domain wiki PPL, but it is **still a
sieve, not a coding gate**. Flash C1, North, and 12B were assessed with this proxy and all
"lost coding" because the step that *measures and protects* coding (HumanEval per group,
then per layer band) was never run.

**Verdict:** multi-domain-PPL-only is **INSUFFICIENT** for any model where coding matters.
It may serve as the cheap sieve feeding phase 5; it may never replace it.

---

## The runnable reconstruction

`scripts/coding_ablation.py` reproduces phases 5 and 6 exactly:
- `coding_ablation.py groups` — per-group demote→build→serve→HumanEval→KEEP/PRUNE.
- `coding_ablation.py layers [--phase2]` — early/middle/late drill, then per-layer in the
  worst third.

It mirrors the logged procedure: override-file convention (`blk.N.<group>.weight=q2_K`,
`output.weight=q2_K`), Q2_K base, port 8084, temp 0 / max_tokens 512 / parallel 4,
`/v1/completions`, OpenAI `human_eval` evaluator (evalplus base fallback flagged at
runtime), CPU-build↔GPU-eval pipelining, KEEP-least-damaging/PRUNE, idempotent resume,
nohup-safe `LD_LIBRARY_PATH`. The `normalize_indent` + fence-strip logic is carried
verbatim from the recovered runner.

**Unrecoverable details (flagged in-script, artifact-consistent defaults chosen):**
- The parallel `/v1/completions` HumanEval fork was inlined in the deleted driver — no
  source survives. Reconstructed from the recovered single-thread `/v1/chat/completions`
  runner + a thread pool + the log's `parallel=4 /v1/completions` banner.
- The exact 2026-05-02 `llama-server` flags were never logged (only `Server up (PID N)`).
  Default to the canonical no-think suite invocation (`-ngl 99 --parallel 4 -c 24576`,
  `--reasoning off --reasoning-budget 0`, with a fallback for older servers).
- Phase-2 per-layer results were truncated from the log; the procedure is intact.

---

## Re-run plan: Flash C1 + North through the REAL pipeline

Both were assessed on multi-domain PPL and wrongly judged. They must run phases 5-6 before
any kill verdict (per the CRITICAL_LOST_STEP mandate). **Do not run yet — GPU is on the
fleet.** Queue:

1. **Confirm the base build + its HumanEval baseline.** Measure each candidate's own
   HumanEval pass@1 once (the `--baseline`), with the same temp-0 settings. Without the
   true base baseline the deltas are meaningless.
2. **Identify each model's tensor groups + layer count.** Flash/North group sets differ
   from 27B's hybrid-SSM set — enumerate the GGUF tensor names (`gguf-dump` / the existing
   ablation tensor list) and map to groups (attn_q/k/v or fused attn_qkv, ffn_down/gate/up,
   output, plus any SSM/router/PLE groups for that architecture).
3. **Phase G (per group):**
   ```bash
   python scripts/coding_ablation.py groups \
       --base-gguf   <flashC1-base>.gguf \
       --source-gguf <flashC1-f16>.gguf \
       --imatrix     <flashC1-imatrix>.dat \
       --base-type   Q2_K \
       --groups      <model-specific group list> \
       --n-layers    <N> --baseline <measured base HumanEval %> \
       --model-tag   FLASH-C1 \
       --out-dir     cerebellum-flash-c1/coding_ablation
   ```
4. **Phase L (layer drill)** on whatever phase G flags CRITICAL (`--phase2` for the worst
   third). Repeat steps 3-4 for North.
5. **Re-allocate the budget** protecting the coding-critical groups/layers (promote them,
   crush the disposable group), rebuild, then run the **full benchmark gate suite**
   (ARC/HellaSwag/MMLU/HumanEval+, Gate-3 BigCodeBench), audit wrong answers.
6. Only after phases 5-6 + gates may a "no-ship" verdict stand. A model that "looks
   well-packed" on multi-domain PPL but was never coding-ablated is **un-assessed**, not
   un-improvable.

**Operational notes for the run:** HumanEval here is the *ablation* harness (parallel=4,
fast OpenAI human_eval, 164 problems) — distinct from the *publication* gate (HumanEval+
via evalplus at WORKERS=1, max_tokens 4096). Use the ablation harness for phases 5-6 to
match the artifacts; use the publication harness for the final gate. Builds go to the
games partition; check `df -h` first (F16 source + candidates won't fit on the main drive).
