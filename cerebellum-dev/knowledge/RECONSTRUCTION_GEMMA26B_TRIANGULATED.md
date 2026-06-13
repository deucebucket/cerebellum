# Gemma 4 26B — Triangulated Pipeline Reconstruction (v1 → v6)

**Model:** Gemma-4-26B-A4B-it (MoE, 30 layers, 658 tensors, ~4B active params/token, 128 experts/layer)
**Base dir:** `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/`
**Build window:** 2026-04-30 13:00 → 2026-05-01 13:32 (Heretic/vision/agentic/creative dirs are May 14–22, outside this window)
**Sources triangulated:** artifact inventory + three independent v1–v3 agents + three independent v4–v6 agents, all cross-checked against the live filesystem on 2026-06-13.
**Verification status:** Central question and the load-bearing artifacts below were re-checked against the filesystem at write time. `find`/`grep`/result-JSON contents reproduced as quoted.

---

## ★ MOST IMPORTANT SECTION — Did Gemma 4 26B run a coding ablation?

### Answer: **NO. No HumanEval-per-tensor-group coding ablation was run. Confidence: HIGH, artifact-grounded.**

This is a **definitive negative**, not a gap. All six agents (three on v1–v3, three on v4–v6) independently returned `ran_coding_ablation = no-evidence`, and the filesystem confirms it directly.

**Negative-proof, reproduced at write time (2026-06-13):**

1. **No `coding_ablation*` directory exists anywhere under `osmosis-gemma4-26b/`.** The 27B has a dedicated `coding_ablation/` dir; the 26B does not.
2. `find . -iname "*coding*"` returns **only out-of-window May-22 Heretic agentic-eval files** (`agentic_eval_20260522/*noncoding_agentic_tools*.jsonl`), none of which are tensor-group ablation artifacts.
3. `grep -rilE "coding ablation|coding_ablation|demote.*humaneval|humaneval.*demote"` over the entire model dir returns **zero hits (exit 1)**.
4. **Every group-level and per-layer ablation artifact is PPL-only.** Group stage: `ablation/group_{attn_q,attn_k,expert_gate_up,ffn_gate,ffn_up}_Q2K.txt` + matching `ablation/ppl_group_*_q2k.log`. Per-layer stage: `attn_q_layers/`, `attn_k_layers/`, `ffn_up_layers/`, `attn_v_layers/`, `attn_o_layers/`, `ffn_gate_up_exps_layers/`, `v8_ffn_gate_up_exps/`, `v8_ffn_down_exps/`, `router_test/gate_layers/` — all contain only `ppl_*.log` + `*_override.txt` pairs. **No `*humaneval*` file in any of them.**

### Where did the coding signal actually come from?

HumanEval entered the Gemma 4 26B loop in **exactly two forms, neither a per-group coding ablation:**

**(A) Whole-build, post-quant benchmark gating** — one HumanEval run per shipped/candidate version:
`benchmark_results/cerebellum_v{1..7}_humaneval_*`. This gated each *finished GGUF*, never an individual tensor group during the ablation search.

**(B) One narrow 3-layer MoE-router probe** — the **only** HumanEval-during-ablation artifact in the whole tree, at
`ablation/surgery/road_mapping/`. It runs HumanEval on three **whole-GGUF builds**, each demoting a single router layer (`ffn_gate_inp`, F32→Q8_0 / Q2_K), for layers 8, 10, 12 only. Verified result JSONs (164 problems, temp 0, thinking off):

| Probe build | HumanEval pass@1 | File |
|---|---|---|
| `router_layer8` | **72.0%** | `router_layer8_humaneval_results.json` |
| `router_layer12` | **71.3%** | `router_layer12_humaneval_results.json` |
| `router_layer10_q2k` | **64.6%** | `router_layer10_q2k_humaneval_results.json` |

This is a **3-layer follow-up probe on full models**, not a network-wide per-tensor-group HumanEval sweep. It cannot be equated to the 27B's `coding_ablation/`.

### What this means for the method

The 27B's winner was selected with a **per-group HumanEval ablation** in the loop. **The 26B's winner was not.** The 26B method was:

> group-level Q2_K **PPL-only** ablation → per-layer **PPL-only** refinement → **whole-build** benchmark gating (HumanEval / ARC / HellaSwag / MMLU per version) → one narrow **3-layer router-HumanEval probe** to settle the final router recast.

Plainly: **Gemma 26B's coding-quality decisions were driven by whole-build benchmark gates and PPL, not by a per-tensor-group coding ablation.** If the project's canon assumes "coding ablation is the universal mechanism," the 26B is a **counterexample whose winner arose from a different mechanism than the 27B.** The artifacts that *would* prove a 26B coding ablation do not exist (confirmed by exhaustive `find`/`grep`), so this is not "artifacts were lost" — it is "the step was never run for this model."

---

## CONFIRMED timeline (v1 → v6), in order

All steps below are `artifact-proven` (override file, PPL log, and/or benchmark JSON present) unless flagged.

### Pre-ablation setup (Apr 30)
1. **PLE-protection sweep — FAILED/no-op:** target PLE tensors were already F32 in source; nothing to protect. (`ppl_q3km_ple_Q5/Q6/Q8.log`)
2. **imatrix selection:** in-house osmosis imatrix was **actively harmful** to PPL → switched to **bartowski's** imatrix. Build logs (`cerebellum_v1_run.log`) show **295 imatrix entries used for v1/v2/v3**. (The "205" figure refers to the *rejected* in-house imatrix — see DISPUTED #1.)
3. **Baseline:** Q3_K_M over bf16 source with bartowski imatrix → **PPL 42,369** (the reference baseline used throughout). A separate no-imatrix baseline `ppl_baseline_q3km.log` reads **64,069** — both are primary; deltas depend on which is cited.

### Group-level PPL ablation (Apr 30 14:16–14:40) — **PPL only**
4. Each tensor group crushed to Q2_K, PPL measured. Results (`research_log.md` Phase 4):
   - `attn_q` SACRED (+13.4% PPL when crushed)
   - `ffn_up` MOST DEMOTABLE (−18.2%)
   - `attn_k` −12.1%, `expert_gate_up` −5.5%, `ffn_gate` −1.2%

### Build iterations
5. **v1** — group recipe applied. PPL **20,614**, ~11 GB. HumanEval **65.2%** (post-build). `cerebellum_v1_overrides.txt`.
6. **v2** — `expert_gate_up` reverted to Q3_K. **REGRESSION:** PPL **22,336**, HumanEval **58.5%** (−6.7 pts). `cerebellum_v2_overrides.txt`.
7. **Per-layer attn_q PPL ablation** (PPL only) → promote only the important attn_q layers.
8. **v3** — promote the 9 key attn_q layers (6/9/11/12/13/18/27/28/29). New best: PPL **19,826**, HumanEval **67.1%**. `cerebellum_v3_overrides.txt`.
9. **v4 (FIRST SHIPPED)** — 91 tensor-level overrides: 9 attn_q→Q5_K, 30 attn_k→Q2_K, 22 ffn_up→Q2_K, 30 ffn_gate_up_exps→Q2_K, PLE F32. PPL **12,614**, HumanEval **69.5%** (`results.json`/README). `cerebellum_v4_overrides.txt`, `README_v4.md`, `hf_v4_upload.log`.
10. **v5 (NOT shipped)** — v4 + 7 attn_k layers (1,6,17,18,23,24,28) Q2_K→Q3_K. Large PPL win but HumanEval + MMLU regress → rejected. PPL **71.3%** HumanEval gate; `v5_ppl.log` reads 9,937.81 (vs 12,356 used as the v5/baseline elsewhere — see DISPUTED #5). `cerebellum_v5_overrides.txt`.
11. **`cerebellum_v6_overrides.txt` (the override-file "v6") — DEAD ON ARRIVAL:** v4 + 4 attn_k (5,11,16,29) Q2_K→Q4_K → PPL **+51%** regression. This override file would rebuild the *dead* model. (See DISPUTED #2 — naming collision.)
12. **Router PPL surgery sweep** (Apr 30→May 1): all 30 router layers `ffn_gate_inp` F32→Q8_0 tested individually via `gguf_tensor_surgery.py`. Layer 10 best on PPL (baseline 12,356 → 11,872). **Router stacking fails** — demoting even the top 3 simultaneously worsens PPL (MoE routing-compensation effect). (`ablation/surgery/results/router_surgery_ablation.log`)
13. **3-layer router HumanEval probe** (`road_mapping/`, May 1 12:32–13:12): layers 8/10/12 whole-build HumanEval (table above). Layer 8 chosen as shipped router recast.
14. **v6 (SHIPPED winner per README)** — single MoE router surgery on layer 8: `blk.8.ffn_gate_inp` F32→Q8_0 on top of the v4/v5 tensor map. README: PPL **12,054**, HumanEval **72.0%**, "no regressions on any benchmark." `gguf_tensor_surgery.py` recast (llama-quantize ignores `--tensor-type-file` for `ffn_gate_inp`).
15. **v6.1** — metadata/chat-template fix over v6, **zero tensor change** (sha256 `d24229fa…`). The shipped chat-template-fixed GGUF.

---

## DISPUTED — flagged, not smoothed

**1. imatrix entry count (apparent conflict, reconcilable).** v1–v3 agents: in-house osmosis imatrix ≈205 entries (rejected) vs bartowski 295 entries (used). One agent phrased bartowski as "covers all 658 tensors" (coverage scope, not entry count). v4–v6 agents split further: one cites a **self-generated** imatrix (295 entries, 822 chunks, `calibration_datav5.txt`, from `v6_quantize.log`); two cite **bartowski's external** imatrix (`README.md`, `ppl_q4km_bartowski.log`). **Resolution:** build logs prove 295 entries used for v1–v3; these may be different imatrices at different pipeline stages, but the self-gen-vs-bartowski attribution for the v6-era build is **not fully reconciled.**

**2. Which "v6" was uploaded — the Q4_K override build or the router-surgery build? (naming collision, UNRESOLVED.)** Two distinct objects share the name "v6": (a) `cerebellum_v6_overrides.txt` = the DEAD Q4_K-attn_k build (HumanEval 61.6% in `FULL_EXPERIMENT_LOG`, git-tracked, defined by `v6_quantize.log` + `run_v6_benchmarks.sh`); (b) the **router-surgery-on-v5 build** = the README's shipped winner. Two agents treat (b) as the genuine ship and (a) as a naming collision. One agent flags this as MAJOR-unresolved: the git-tracked override file + quantize log + bench script all define v6 as the **Q4_K build**, and the actual shipped `cerebellum-v6.gguf` was **never independently hash-verified** (only the v6.1-templatefix sha256 is on record). **Do not assume the git-tracked v6 override file reproduces the shipped model.**

**3. Which router layer is in the shipped v6 — layer 8 or layer 10? (genuinely DISPUTED.)**
- **README is explicit: layer 8** ("Final Precision Map (v6)": `ffn_gate_inp layer 8 → Q8_0`; "Why layer 8 and not layer 10"; PPL 12,054, HumanEval 72.0%). `road_mapping/layer8_full_bench.log` + the full layer-8 ARC/HellaSwag/MMLU bench set exist.
- **The only router-surgery BUILD log present is `surgery_v6_router10.log`** (Input v5.gguf → Output v6-router10.gguf, RECAST blk.10.ffn_gate_inp F32→Q8_0, 11.746 GB, PPL 11,872). There is **no `surgery_v6_router8` build log / GGUF** in the tree. The 72.0% layer-8 figure traces to a `road_mapping/` **probe** build, not a named shipped-build artifact.
- **Net:** README claims layer 8; the only surgery build artifact on disk is layer 10. No artifact conclusively ties the *shipped* GGUF to either recast. **Left unresolved.**

**4. v6 HumanEval score (four values, none reconciled).**
- **36.0%** — `cerebellum_v6_humaneval_results.json` (2026-05-04), raw `/v1/completions`, the known **Gemma-4 chat-template-bypass artifact** (project benchmark rules flag this class explicitly).
- **~41.5%** — corrected chat-template reruns (`benchmark_v6_humaneval_final.log` / `_thinking.log`, May 4).
- **61.6%** — `FULL_EXPERIMENT_LOG.md`, attached to the **Q4_K override build** (regression vs v4).
- **72.0%** — README + `router_layer8` road_mapping probe.
- README itself carries a **"Numbers under audit (2026-05-08)"** banner: the v6 table is preliminary pending a clean re-run. **The true shipped-v6 HumanEval is unresolved.**

**5. v5 / baseline PPL inconsistency.** `v5_ppl.log` = 9,937.81; router sweep + README use 12,356 as the v5/baseline. Unreconciled across docs.

**6. v4 HumanEval (minor).** `results.json` + README = **69.5%**; `FULL_EXPERIMENT_LOG` narrative tables = 75.0% (unsourced to any results JSON). Primary artifact favors 69.5%.

---

## GAPS — verification could not reach (not disputes)

- **`gguf_tensor_surgery.py` not in-hand.** The tool behind every router surgery + road_mapping build lived at `/var/home/deucebucket/ai-drive/osmosis/scripts/` (per run scripts and the README GitHub link), **outside this model dir**; none of the six agents verified it present. The exact recast code is not in this evidence set.
- **Shipped output GGUFs not hash-verified.** The actual shipped GGUFs (and bf16 source) live under `/var/home/deucebucket/games/`, not the inspected dir. Only the **v6.1-templatefix sha256 (`d24229fa…`)** is on record; `cerebellum-v6.gguf` was never independently hashed.
- **v7 provenance is thin.** Only `v7_quantize.log`; no override→build argv log, and v7 is **not git-tracked** (only v1–v6 in `model-data/gemma4-26b-a4b/` at commit `75f315b`). Confidence the v7 overrides built the benchmarked GGUF: **medium.** v7 HumanEval: 63.4% (`results.json`), rerun 60.4%.
- **v2 build→GGUF command line not argv-verified.** v1 and v3 override→bf16→Q3_K_M chains were argv-confirmed; v2's quantize invocation is reconstructed from captured stdout, not an echoed argv. Confidence: high-but-not-line-verified.
- **ARC / HellaSwag / MMLU per-version scores** read from `research_log.md` narrative tables, **not raw result JSONs** (only HumanEval + PPL are raw-artifact-confirmed for v1–v3). The layer-8 v6 ARC/HellaSwag/MMLU JSONs *do* exist under `road_mapping/`.

---

## Bottom line

Gemma 4 26B's published quant (**v6 / v6.1**) was produced by **group-PPL ablation → per-layer-PPL refinement → whole-build benchmark gating → a single MoE-router recast settled by a 3-layer HumanEval probe.** It **did not** run a 27B-style network-wide per-tensor-group coding (HumanEval) ablation — confirmed by exhaustive `find`/`grep` returning empty, agreed by all six agents at artifact-proven confidence. The two genuinely open questions for any future canon or rebuild are **(a)** which GGUF was actually uploaded as "v6" (Q4_K override build vs router-surgery build — naming collision) and **(b)** whether the shipped router recast was layer 8 (README) or layer 10 (the only surgery build log on disk). Neither is resolvable from the current artifact set.
