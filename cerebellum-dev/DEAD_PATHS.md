# Cerebellum Dead Paths Archive
Date: 2026-06-11
Source forensics: cerebellum-dev/forensics_2026-06-11/claude_sessions.md,
                  cerebellum-dev/forensics_2026-06-11/other_clis_sessions.md,
                  cerebellum-dev/QWEN36_27B_V1_TO_V4_PLAYBOOK_2026-06-07.md,
                  cerebellum-dev/OG_CEREBELLUM_RECONSTRUCTION_GUIDES_2026-06-07.md

Entries here are stored, not deleted. Each describes a path that was explored and
abandoned. They are useful for understanding why the OG formula works and for
not repeating mistakes.

---

## DP-1: Wiki-PPL-only per-tensor hill-climb (osmosis/hillstep.py)

**What it was:** A resumable SQLite-journaled per-tensor hill-climber starting from
Q4_K_M baseline. Tests each tensor bidirectionally (demotion q4->q3->q2, promotion
q4->q5->q6->f16). Commits locks when a tensor's optimal precision is found on wiki PPL.

**When:** 2026-06-03 through 2026-06-06, Gemma 4 12B.

**Why it died:** The block-10 checkpoint proved wiki PPL and task benchmarks can
diverge catastrophically. Hard evidence:

| Metric | Q4_K_M base | Block-10 hillstep | Delta |
|---|---:|---:|---:|
| Size GiB | 6.8744 | 7.8955 | +1.02 GiB |
| Wiki PPL | 2504.2787 | 1615.8505 | -35% |
| HumanEval+ | 83.54% | 69.51% | -14.03 pts |
| HumanEval base | 85.98% | 71.95% | -14.03 pts |
| ARC | 93.43% | 93.86% | +0.43 |
| HellaSwag | 81.95% | 81.40% | -0.55 |
| MMLU-Redux | 67.04% | 66.96% | ~0 |
| tok/s (narrative) | 80.69 | 72.72 | -7.97 |

Source: forensics_2026-06-11/claude_sessions.md §4, COMPARISON_SUMMARY.md.

**Additional failure modes found during hillstep development:**
1. Hillstep started from a poisoned base map that already demoted `attn_v` and
   `attn_o` broadly. OG v6 never demoted these (ablation showed +24-31% PPL
   sensitivity). No per-tensor tuning escapes a poisoned starting map.
2. K-quant corruption on router tensors not detected: Q6_K and Q2_K both caused
   +15-17% PPL on Gemma 26B router; Q8_0 was the only safe format. Hillstep's
   demotion chain (q4->q3->q2) passes through K-quant formats that were known broken
   for router tensors.
3. No benchmark gate before committing locks. Locks were committed on PPL evidence
   alone.
4. Five hillstep.py bugs fixed pre-June-4 (break on q3 rejection exited loop
   skipping q2/q5/q6/f16; delta=0 tiebreaker wrong direction; ABS_BEST tracking
   absent causing PPL drift upward; post-f16 fallback missing; quantize target was
   Q3_K_M instead of Q4_K_M baseline).
5. Lock persistence bug: `json.load(open(''))` in hillclimb.sh could not write new
   locks during interrupted runs.

**Status:** hillstep.py remains in osmosis/ as an optional add-on (per its own
documentation: "use the proven group-first, benchmark-gated workflow instead of
exhaustive wiki-only per-tensor hillclimb"). The exhaustive mode is DEPRECATED.
Targeted hillstep AFTER group-first scan is still listed as a valid optional step.

**Evidence pointer:** forensics_2026-06-11/claude_sessions.md §4.

---

## DP-2: MTP-preserved heretic source (Qwen 3.6 35B, 2026-06-03)

**What it was:** Heretic Qwen 3.6 35B-A3B built from
`llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF` (67 GB).
Used v3 overrides (360 entries) + 20 MTP entries at BF16.

**When:** 2026-06-02/03. Triggered by HF user TheodoreH comment (Discussion #3 on
Qwen3.6-35B-A3B-Cerebellum-GGUF, 2026-05-31) requesting a Heretic Qwen build.

**Why it died:** The MTP-preserved source contained extra block blk.40, adding a
different architecture layer. This caused a llama.cpp load failure and severe benchmark
regression (-14 HellaSwag / -32 HumanEval+) vs the stock v3. Both output GGUFs
(13 GB Heretic and the MTP-preserved intermediate) were pruned 2026-06-11.

**Residual on disk:** `games/cerebellum-heretic-qwen36-35b/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf`
(861 MB mmproj) — safe to keep as reference for the mmproj format; the GGUF was
pruned.

**Corrected path:** Use `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF` (plain
BF16, 69.4 GB, no blk.40). Recipe in forensics_2026-06-11/RECIPE_heretic_qwen36_35b.md.

**Evidence pointer:** forensics_2026-06-11/other_clis_sessions.md §IV,
forensics_2026-06-11/RECIPE_heretic_qwen36_35b.md.

---

## DP-3: Raw /v1/completions harness for chat models

**What it was:** HumanEval/EvalPlus benchmark harness that POSTed to `/v1/completions`
(raw completions) instead of `/v1/chat/completions`. Used for Gemma 4 26B v6 and early
Heretic runs.

**When:** 2026-05-01 (v6), 2026-05-18 (Heretic first bad run).

**Why it died:** Gemma 4 is a chat/thinking model. Raw completions bypass the chat
template and produce malformed continuations. Results:
- Heretic v1 run 1: HumanEval base 3.05%, HumanEval+ 3.05% (clearly invalid)
- v6: 35.97% in `cerebellum_v6_humaneval_results.json` (same artifact; file is a
  false low and must not be used as a reference score)

**Fix:** Switch to `/v1/chat/completions` with `enable_thinking: false` and
`thinking_budget_tokens: 0`. Also normalize indentation using `ast.parse(prompt +
body)` validation. Correct Heretic v1 score (fresh rerun): base 92.07%, plus 89.63%.

**Current status:** `scripts/benchmark_evalplus_chat.py` contains the fixed harness.
Raw `/v1/completions` path is retired for any model that requires chat-template
interaction.

**Evidence pointer:** forensics_2026-06-11/claude_sessions.md §3 Incident 4.

---

## DP-4: Local 205-entry Gemma imatrix (actively harmful)

**What it was:** Locally generated imatrix for Gemma 4 26B-A4B from
`osmosis.imatrix_stream`. Path: `osmosis-gemma4-26b/imatrix.dat`.
205 entries covering only 205/658 tensors.

**When:** 2026-04-30, Phase 2 of Gemma 26B work.

**Why it died:** Zero coverage of expert tensors (90), router tensors (60), or norms
(271). Using this imatrix made Q4_K_M WORSE than no imatrix:
- Q4_K_M + bad imatrix: PPL 67,869
- Q3_K_M + bad imatrix: PPL 64,069
- Q4_K_M + no imatrix: PPL 52,961

**Fix:** Switched to bartowski `google_gemma-4-26B-A4B-it-imatrix.gguf` (295 entries,
822 chunks, full expert coverage). First Q3_K_M with correct imatrix: PPL 42,369.
All subsequent Gemma 26B v6 work used the bartowski imatrix.

**File still present:** `osmosis-gemma4-26b/imatrix.dat`. Do not use for 26B MoE.
The bartowski imatrix is `osmosis-gemma4-26b/google_gemma-4-26B-A4B-it-imatrix.gguf`.

**Evidence pointer:** forensics_2026-06-11/claude_sessions.md §3 Incident 8.

---

## DP-5: Wiki-only imatrix on Qwen 3.5 9B v1 (false failure)

**What it was:** First Qwen 3.5 9B build using a wiki-only calibration imatrix
(not code-weighted). v1 produced a model that appeared to fail benchmarks.

**When:** 2026-05 era (Codex sessions, other_clis_sessions.md §I).

**Why it died:** Wiki-only imatrix misalignment + contaminated benchmark artifacts +
too-generous budget produced a false failure signal. The model was not definitively
bad; the imatrix and measurement were.

**Fix:** v2_code used a code-weighted imatrix (4.0 GB output). v2_code is the winner
at 53.0% EvalPlus+.

**Evidence pointer:** forensics_2026-06-11/other_clis_sessions.md §I item 12.

---

## DP-6: 9B v3_rowblock (unresolved anomaly, not a confirmed dead path)

**What it was:** Qwen 3.5 9B v3 with row-block allocation. Passed classical benchmarks
(ARC, HellaSwag, MMLU) but failed a real agent-loop snake task.

**When:** 2026-05 (Codex sessions).

**Why it's here:** Classical benchmark pass + agent task failure is the same pattern
as hillstep block-10. Row-block allocation is suspect. The regression was never
formally diagnosed.

**Status:** Not formally deprecated; root cause unresolved. Treat as an open caution:
row-block allocation strategies require agent-capable tasks in the gate suite, not
just multiple-choice.

**Evidence pointer:** forensics_2026-06-11/other_clis_sessions.md §I item 13.

---

## DP-7: Base HumanEval harness (fence/indentation bug)

**What it was:** The original `scripts/benchmark_humaneval.py` (and early evalplus
variant) using `.strip()` to remove code fences, which also stripped leading
whitespace from the first code line and corrupted indentation normalization.

**When:** All runs pre-2026-05-03.

**Why it died:** Systematic -7 to -8 point undercount on pass@1. Temperature sweep
revealed the bug: anomalously low scores at temp>0 that should not exist for a
calibration sweep. Fixed script at temp=0 immediately gave 82.9% vs prior 75.0% on
Qwen 27B v4.

**Fix:**
```python
# BROKEN:
content = content[len("```python"):].strip()
# FIXED:
content = content[len("```python"):]
content = content.strip("\n")
```

**Impact:** Affected all published HumanEval rows pre-2026-05-03: Qwen 3.6 27B v4,
Gemma 4 26B v4-v6, Qwen 3.5 9B, Granite 4.1 30B.

**Current status:** EvalPlus upstream runner replaced the hand-rolled harness.
Per project memory: default to evalplus.codegen / bigcode-evaluation-harness /
lm-eval-harness; do not roll thin wrappers.

**Evidence pointer:** forensics_2026-06-11/claude_sessions.md §3 Incident 1.

---

## DP-8: Critical-layer demotion below Q4_K in Gemma 26B v6 first attempt

**What it was:** Early v6 experiment that set critical `attn_k` layers (5, 11, 16, 29)
to Q3_K or below, following the pattern used for other layers.

**When:** 2026-05-01 (v6 session, before the shipped v6 build).

**Why it died:** +51.2% wiki PPL, -13.4 HumanEval. Catastrophic. The ablation
identified these four layers as requiring Q4_K minimum; any lower format destroys them.

**Fix:** Set layers 5/11/16/29 to Q4_K in the override file. All other attn_k
tensors demoted to Q2_K as ablation indicated.

**Evidence pointer:** forensics_2026-06-11/claude_sessions.md §2A (critical finding
about CRITICAL layers).

---

## DP-9: Unauditable Brainloop benchmark runs (conch-poc, 2026-06)

**What they were:** A set of HumanEval+/coding benchmark runs for the conch-poc
Brainloop experiment:
`brainloop-code-examples, brainloop-code-trained2, brainloop-rag-trained,
brainloop-combined, brainloop-both-active, brainloop-full-corpus,
qwen3b-refiner-arc, qwen3b-baseline-arc, qwen3b-refiner-hellaswag`

**When:** 2026-06 (Codex/OpenCode sessions).

**Why they died:** All result JSONs omit checkpoint path, git commit, and GGUF path.
Permanently unauditable. Additionally:
- `qwen7b-baseline` and `qwen7b-baseline-real`: byte-identical, both contain 164
  literal `"    pass"` completions, elapsed=0, pass@1=None. Generation never ran.
- `brainloop-best-combo` and `brainloop-fix-13k`: byte-identical files (same MD5);
  one was a file copy.
- `brainloop-rag-coding` and `brainloop-sharp-rag`: 97.6% identical; RAG
  intervention was inert.
- `recall_results_deadblock.json` A vs B: 100% identical despite different GGUF
  paths specified; server ran same model twice.
- Published PyTorch A/B comparison (62.2 vs 56.7 HumanEval+) was confounded:
  `fused_refiners.pt` had `tanh(gate) ≈ -0.005` (scaling contribution 0.5%);
  L17 `inj_proj.weight` was exact identity matrix (never received gradient updates).
  The -5.5-point "logic tax" reflected prompt-formatting difference (instruct wrap),
  not the refiner. Correction posted to RESULTS.md and conch-poc/README.md 2026-06-11.

**Status:** These runs are archived as evidence of what not to do. The conch-poc
project's bench_results/ directory should be treated as untrustworthy until a new
benchmarking pass with proper provenance logging is completed.

**Evidence pointer:** forensics_2026-06-11/claude_sessions.md §3 Incident 9.

---

## Logging Failures (for reference alongside dead paths)

These are not dead methodologies but documentation/logging failures that produced
ambiguous artifacts still on disk.

**LF-1: bench_results/ no checkpoint provenance (conch-poc)**
All 7-field bench_results JSONs omit checkpoint path, git commit, GGUF path. The
7 Brainloop runs listed in DP-9 have no recoverable provenance.

**LF-2: v6 HumanEval result from wrong harness**
`cerebellum_v6_humaneval_results.json` records `pass_at_1: 0.3597` from raw-completions
harness (DP-3). This file is in benchmark_results/ alongside valid results. The 35.97%
figure is invalid; do not use as a reference score.

**LF-3: Qwen 27B v2/v2b benchmark files pre-correction**
v2 benchmark files written before 2026-05-03 bug fixes. `cerebellum_v4_humaneval_lcpp_results.json`
shows 0.0 (bad path); `cerebellum_v4_tuned_humaneval_results.json` shows 29.9 (bad/
tuned path). Both exist in benchmark_results/ alongside correct files.

**LF-4: osmosis-gemma4-26b/imatrix.dat still named as primary**
The harmful 205-entry local imatrix (DP-4) sits in the same directory as the correct
bartowski imatrix. The correct file is `google_gemma-4-26B-A4B-it-imatrix.gguf`.

**LF-5: v6 quantize log targets wrong filename**
`v6_quantize.log` header shows output filename `Gemma4-E4B-Cerebellum-v6-Q3_K_M.gguf`
(wrong model name: E4B vs 26B MoE). File was renamed post-run; log shows wrong destination.

**LF-6: Shipped v6 PPL ambiguous in logs**
`v6_ppl.log` shows 12,894 (pre-surgery intermediate). Shipped v6 GGUF PPL is 12,054
(post-surgery, confirmed from router surgery ablation). No clean `v6_shipped_ppl.log`
entry exists for the final GGUF.
