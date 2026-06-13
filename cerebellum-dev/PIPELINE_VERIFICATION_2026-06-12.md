# Pipeline Verification — Third Independent Pass — 2026-06-12

Private. Do not push to `origin`.

Scope: triple-verify every load-bearing claim in
`cerebellum-dev/OG_CEREBELLUM_RECONSTRUCTION_GUIDES_2026-06-07.md` (the "OG guide") and
`cerebellum-dev/QWEN36_27B_V1_TO_V4_PLAYBOOK_2026-06-07.md` (the "playbook")
against PRIMARY evidence: override files, ablation JSONs/logs, PPL/quantize logs,
benchmark JSONs, and the LIVE HF model cards (curled 2026-06-12, ~03:40 CDT).

Verdict counts: **43 VERIFIED, 5 MISMATCH/INCONSISTENCY, 2 UNVERIFIABLE.**
Mismatches are listed first because they are the point.

---

## MISMATCHES AND INCONSISTENCIES (read these first)

### M1. LIVE 27B card publishes a wrong "Q2_K (no imatrix)" PPL — PUBLIC CARD ERROR

| | |
|---|---|
| Live card says | "Perplexity vs Size" table: `Q2_K (no imatrix) | 9.98 GB | 8.256` (https://huggingface.co/deucebucket/Qwen3.6-27B-Cerebellum-GGUF, README line ~139) |
| Disk evidence says | `osmosis-qwen36-27b/ppl_q2k_no_imatrix.log`: `file size = 9.97 GiB`, `Final estimate: PPL = 7.6494 +/- 0.05067` |
| What 8.256 actually is | The ablation baseline PPL (`ablation_results.json` → `baseline_ppl: 8.2556`), a different artifact entirely |
| Verdict | **MISMATCH — live public card error.** The playbook (7.6494 for Q2_K-no-imatrix, 8.2556 as ablation baseline) matches disk; the live card conflated the two. The card's own "Step 1" section even cites 8.256 as the ablation baseline, contradicting its own size table. Fix on next card update. |

### M2. start.md "448-tensor ablation" vs disk (23 probes)

| | |
|---|---|
| OG guide says (line ~197) | "`start.md` records the completed 448-tensor ablation" |
| Disk evidence says | `start.md` lines 80, 144 do claim 448 tensors, but `osmosis-qwen36-27b/ablation_results.json` contains exactly **23** measured probes (`tests` dict, len 23), `ablation_plan.json` 23 planned tensors, `interaction_results.json` 6 tests |
| Verdict | **MISMATCH in start.md itself** (448 is not what the saved JSON holds). Both guides handle this correctly later (both say 23 probes + allocator extrapolation), but the OG guide's evidence bullet repeats the 448 claim without an inline qualifier. The live card's "Each tensor is individually crushed" is the same compression — known, flagged by both guides. |

### M3. The two guides disagree on the 27B v4 MMLU-Redux anchor (76.875 vs 76.58)

| | |
|---|---|
| OG guide says | Corrected anchor "MMLU-Redux: 76.875" |
| Playbook says | "MMLU-Redux 76.58% (1838/2400)" — trust `benchmarks/qwen36-27b/` |
| Disk evidence says | `cerebellum_v4_fixed_mmlu_redux_results.json` = 76.875 (1845/2400, 05:06); `cerebellum_v4_corrected_mmlu_redux_results.json` = 76.583 (1838/2400, 05:14, later run). The public mirror `benchmarks/qwen36-27b/cerebellum_v4_mmlu_redux_results.json` and the LIVE card (76.6 / frontmatter 0.766) both carry the **corrected** 76.58. |
| Verdict | **INCONSISTENCY between guides.** Both numbers exist on disk; the published/canonical one is 76.58 ("corrected", later timestamp). Same trap on ARC: the OG guide's Gates section points at `cerebellum_v4_fixed_arc_results.json` (94.71, 1110/1172) but the published 96.76 (1134/1172) comes from `cerebellum_v4_corrected_arc_results.json`. The OG guide should point at `corrected_*` (or the `benchmarks/` mirror), not `fixed_*`. |

### M4. LIVE 35B card "6 of 10 groups perform better at Q2_K" — disk says 7 of 10

| | |
|---|---|
| Live card says | "Key finding from reverse ablation: 6 of 10 groups perform better at Q2_K than Q3_K_M" |
| Disk evidence says | `osmosis-qwen36-35b/ablation/reverse_ablation_results.log` (v1 PPL 7.8484): un-demoting made PPL **worse** for 7 groups (ffn_gate_exps 7.8902, attn_gate 7.9429, ssm_alpha 8.0034, ssm_beta 8.0184, ffn_gate_shexp 7.9646, ffn_down_shexp 7.9270, ffn_up_shexp 7.9865); un-demoting **helped** 3 (attn_qkv 7.7109, ffn_down_exps 7.7889, ffn_up_exps 7.7909) |
| Verdict | **MISMATCH (minor, undersells).** By the log it is 7 of 10, not 6. The card understates its own result. OG guide's reverse-ablation narrative (attn_qkv/ffn_down_exps/ffn_up_exps benefit from un-demotion) matches disk exactly. |

### M5. LIVE 35B card "outperforming it on 5 of 6 benchmarks"

| | |
|---|---|
| Live card says | "v3 at 11 GB is 29% smaller than stock Q3_K_M (15.6 GB) while outperforming it on 5 of 6 benchmarks" |
| Disk evidence says | v3 vs Q3_K_M baseline: HellaSwag 92.28 vs 91.49 (win), MMLU-Redux 75.0 vs 74.12 (win), HumanEval base 70.73 vs 64.02 (win), HumanEval+ 65.24 vs 56.71 (win), ARC 95.82 vs 96.08 (**loss**). That is 4 wins of 5 measured. The 6th row (vision smoke 100%) has no baseline number in the card ("—"). |
| Verdict | **MISMATCH (minor, oversells).** "4 of 5 measured, ARC inside noise, vision not baselined" is the defensible phrasing. All the underlying numbers themselves verify exactly against the JSONs. |

### Provenance ambiguity worth recording (not a guide error)

`DEVLOG_2026-05-01_qwen36_35b_start.md` line 70 says "Base quant: Q3_K_M with **bartowski** imatrix" while line 225 names the file `Qwen_Qwen3.6-35B-A3B-imatrix.gguf (184 MB)`. The on-disk file `games/qwen36-35b-v2/imatrix_unsloth.gguf_file` is 192,223,904 bytes (≈183 MiB ≈ "184 MB") and its embedded dataset is `unsloth_calibration_Qwen3.6-35B-A3B.txt`. The LIVE card says "unsloth coder imatrix." File metadata is primary: the dataset is unsloth calibration; the devlog's "bartowski" wording is the loose attribution (likely bartowski-style repo naming). OG guide's attribution (imported unsloth) matches the primary evidence.

---

## UNVERIFIABLE

| Claim | Why |
|---|---|
| Exact final build shell commands for all three models (27B v4, 35B v3, 26B v6) | Both guides already mark these as reconstructions ("exact final shell line was not found"). No quantize log for the 35B v3 build survives in `games/qwen36-35b-v2/` (only `download.log`). The 27B and 26B command shapes are consistent with GGUF metadata, imatrix load lines, and override files, but the literal command lines remain reconstructed, not recovered. |
| 35B v3 build actually consumed `imatrix_unsloth.gguf_file` (vs the local `cerebellum_imatrix.dat`) | No surviving quantize log in the build dir. Supported by: devlog file-size match (184 MB), live card statement ("unsloth coder imatrix"), and the OG guide's own hedge ("evidence points to"). Consistent, but not proven from a primary build log. |

---

## VERIFIED CLAIMS — Qwen 3.6 27B v4

All paths relative to `/var/home/deucebucket/ai-drive/cerebellum/` unless absolute.

| # | Claim | Guide says | Disk evidence says | Verdict |
|---|---|---|---|---|
| 1 | v4 override count | 181 entries | `osmosis-qwen36-27b/tensor_types_v4_12gb.txt`: 181 lines | VERIFIED |
| 2 | v4 type distribution | q2_K 22, q3_K 19, q4_K 22, q5_K 70, q6_K 41, q8_0 7 | awk on same file: 22/19/22/70/41/7 — exact | VERIFIED |
| 3 | v3 ladder files | v3_8.0gb 41 (22 Q2/19 Q3); v3_10.7gb 115 (22/79/8/6); v3_12gb 181 (22/119/22/18); v3c 181 (same) | All four files present, line counts and distributions exact | VERIFIED |
| 4 | Ablation basis | 23 measured probes, 23 planned, 6 interaction tests, baseline PPL 8.2556 | `ablation_results.json` 23 tests, baseline_ppl 8.2556; `ablation_plan.json` 23 tensors; `interaction_results.json` 6 tests (additive_all_demote + ladder_step_1..5) | VERIFIED |
| 5 | Sacred probes | blk.63.attn_q 8.4178 (+0.1622), blk.63.ffn_down 8.3933, blk.1.ffn_gate 8.2941, blk.18.ssm_alpha 8.2810, blk.62.ssm_out 8.2732 | JSON (HF names): layer_63.self_attn.q_proj 8.4178, layer_63.mlp.down_proj 8.3933, layer_1.mlp.gate_proj 8.2941, layer_18.linear_attn.in_proj_a 8.281, layer_62.linear_attn.out_proj 8.2732 | VERIFIED |
| 6 | Demotable probes | blk.2.ffn_gate 8.1089, blk.32.attn_qkv 8.1221, blk.34.ffn_down 8.1610, blk.0.ffn_up 8.1988, blk.0.ffn_down 8.2015 | layer_2.mlp.gate_proj 8.1089, layer_32.linear_attn.in_proj_qkv 8.1221, layer_34.mlp.down_proj 8.161, layer_0.mlp.up_proj 8.1988, layer_0.mlp.down_proj 8.2015 | VERIFIED |
| 7 | Imatrix provenance | `cerebellum_imatrix.dat`, ~13M, 496 entries, ncall=8, dataset `osmosis-sensitivity` | File 13,582,641 bytes; parsed: 496 entries, all ncall=8, m_last_call=8, trailing dataset string `osmosis-sensitivity` | VERIFIED |
| 8 | Base type Q2_K, loader shape | F32 353, Q8_0 7, Q2_K 253, Q3_K 77, Q4_K 49, Q5_K 70, Q6_K 42; 11.97 GiB; 3.82 BPW; PPL 7.0344 | `benchmarks/qwen36-27b/cerebellum_v4_wiki_ppl.log`: exact tensor counts, `Q2_K - Medium`, 11.97 GiB (3.82 BPW), `Final estimate: PPL = 7.0344 +/- 0.04625`, 851 tensors total | VERIFIED |
| 9 | Baseline PPL anchors | Q2_K+imatrix 9.97 GiB / 7.4996; Q2_K no imatrix 7.6494; Q3_K_M+imatrix 12.38 GiB / 7.6413 | `ppl_q2k_uniform.log` 9.97/7.4996; `ppl_q2k_no_imatrix.log` 9.97/7.6494; `ppl_q3km_imatrix.log` 12.38/7.6413 | VERIFIED |
| 10 | Failed-route PPLs | budget_12gb 8.4998; budget_12gb_v4_with_imatrix 7.9491; demote_12gb 7.2225; sacred_q8 7.3797; budget_10gb_konly 7.9066 | All five logs present in `osmosis-qwen36-27b/`, values exact | VERIFIED |
| 11 | Published bench numbers | HumanEval 81.1 (133/164), ARC 96.76 (1134/1172), HellaSwag 92.21 (9260/10042), MMLU-Redux 76.58 (1838/2400), MMLU full 82.52 (11643 q) | `benchmarks/qwen36-27b/` mirror JSONs: 0.81097→81.1, 96.7577/1134, 92.2127/9260, 76.583/1838, 82.5217/11643 | VERIFIED (note M3 for which file is canonical) |
| 12 | Live card matches audited JSONs | — | LIVE card: PPL 7.034, HumanEval 81.1, ARC 96.8, HellaSwag 92.2, MMLU 82.5, MMLU-Redux 76.6; frontmatter 0.968/0.922/0.766/0.811/7.034 — all match `benchmarks/qwen36-27b/` | VERIFIED (except M1 size-table row) |
| 13 | Card "vs Q2_K imatrix" comparison | — | LIVE card: HumanEval 47.0, ARC 95.0, HS 90.8, MMLU-R 74.3 ↔ `osmosis_q2k_*_results.json`: 47.0 / 94.97 / 90.79 / 74.33 | VERIFIED |
| 14 | Interaction ratios | cross-layer 0.8575 additive; same-layer ladder step 4 collapses to 0.1316 | `interaction_results.json` (6 tests) + card states 0.86/0.13 | VERIFIED |
| 15 | Bad-file warnings | `cerebellum_v4_humaneval_lcpp_results.json` = 0.0; `cerebellum_v4_tuned_humaneval_results.json` = 29.9 | Both files exist in `benchmarks/qwen36-27b/` as warned | VERIFIED (existence; values not re-opened) |

## VERIFIED CLAIMS — Qwen 3.6 35B-A3B v3

| # | Claim | Guide says | Disk evidence says | Verdict |
|---|---|---|---|---|
| 16 | v3 override file | 360 entries, all Q2_K, 40 layers x 9 groups | `games/qwen36-35b-v2/cerebellum_v3_overrides.txt`: 360 lines, 360x Q2_K, exactly 40 each of ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp, attn_gate, ssm_alpha, ssm_beta | VERIFIED |
| 17 | v1/v2 evolution | v1 400 lines (10 groups incl. attn_qkv); v2 280 lines (7 groups; attn_qkv, ffn_up_exps, ffn_down_exps kept at baseline) | `osmosis-qwen36-35b/cerebellum_v1_overrides.txt` 400 all-Q2_K, 10 groups; `cerebellum_v2_overrides.txt` 280 all-Q2_K, 7 groups, missing exactly attn_qkv/ffn_up_exps/ffn_down_exps | VERIFIED |
| 18 | Forward ablation | baseline 7.1758; ssm_out 7.4132 (+0.2374); attn_qkv 7.2766; ffn_down_exps 7.3136; ssm_alpha 7.1753; ssm_beta 7.1803 | `osmosis-qwen36-35b/ablation/group_ablation_results.log`: all values exact | VERIFIED |
| 19 | Reverse ablation | from v1 PPL 7.8484; attn_qkv/ffn_down_exps/ffn_up_exps benefit from un-demotion | `reverse_ablation_results.log`: v1 7.8484; un-demote attn_qkv 7.7109, ffn_down_exps 7.7889, ffn_up_exps 7.7909 (all improve) | VERIFIED (see M4 for the card's 6-vs-7 count) |
| 20 | Base quant Q3_K_M | — | LIVE card: `Qwen3.6-35B-A3B-Cerebellum-v3-Q3_K_M.gguf`; protected groups listed at Q3_K_M | VERIFIED (as published claim; build log lost, see UNVERIFIABLE) |
| 21 | Imatrix file properties | imported, 1020 tensors, 76 chunks, dataset `unsloth_calibration_Qwen3.6-35B-A3B.txt` | GGUF header parse of `imatrix_unsloth.gguf_file`: general.type=imatrix, 1020 tensors, chunk_count 76; strings: dataset name exact | VERIFIED |
| 22 | Local alt imatrix | `cerebellum_imatrix.dat` 470 entries / ncall 1 exists alongside | File present, 108,152,016 bytes (entry parse not redone this pass) | VERIFIED (existence/size) |
| 23 | v3 bench anchors | ARC 95.82, HumanEval 70.73/65.24, HellaSwag 92.28, MMLU-Redux 75.0 | `games/qwen36-35b-v2/benchmark_results_v3/*.json`: exact | VERIFIED |
| 24 | Baseline anchors | ARC 96.08, HumanEval 64.02/56.71, HellaSwag 91.49, MMLU-Redux 74.12 | `benchmark_results_baseline/*.json`: exact | VERIFIED |
| 25 | Live card matches audited JSONs | — | LIVE card table: 95.8/92.3/75.0/70.7/65.2 and frontmatter 0.958/0.923/0.75/0.652 — match the JSONs | VERIFIED (modulo M5 phrasing) |
| 26 | Final GGUF exists | `Qwen3.6-35B-A3B-Cerebellum-v3.gguf` | Present, 11,955,464,480 bytes (≈11.1 GiB; card says "11 GB") | VERIFIED |
| 27 | "PPL got worse, benchmarks improved" framing | v3 PPL-vs-bench compromise; benchmark-selected | Forward/reverse logs + v3-beats-baseline-on-4 pattern support it; consistent with org-card failure story | VERIFIED (interpretive, evidence-consistent) |

## VERIFIED CLAIMS — Gemma 4 26B-A4B v6/v6.1

| # | Claim | Guide says | Disk evidence says | Verdict |
|---|---|---|---|---|
| 28 | v6 override count/distribution | 91 entries: Q2_K 78, Q4_K 4, Q5_K 9 | `osmosis-gemma4-26b/cerebellum_v6_overrides.txt`: 91 lines, 78/4/9 exact | VERIFIED |
| 29 | v1–v5 evolution table | v1 120 (90 Q2/30 Q5); v2 90 (60/30); v3 99 (90/9); v4 91 (82/9); v5 91 (75 Q2/7 Q3/9 Q5) | All five files present; every count and distribution exact | VERIFIED |
| 30 | v6 override families | 9 attn_q@Q5_K; many attn_k@Q2_K with selected@Q4_K; 22 ffn_up@Q2_K; all ffn_gate_up_exps@Q2_K; no attn_v/attn_output demotion | Family breakdown: 30 ffn_gate_up_exps=Q2_K, 26 attn_k=Q2_K, 22 ffn_up=Q2_K, 9 attn_q=Q5_K, 4 attn_k=Q4_K; zero attn_v/attn_output lines | VERIFIED |
| 31 | Group ablation deltas | expert_gate_up -5.5%, attn_q +13.4% sacred, attn_k -12.1%, ffn_gate -1.2%, ffn_up -18.2% | `osmosis-gemma4-26b/research_log.md` lines 94–98: exact table (PPL 40,035 / 48,056 / 37,262 / 41,874 / 34,655) | VERIFIED |
| 32 | Imatrix switch story | local `imatrix.dat` 205 entries, valid but MoE-incomplete; shipped builds used bartowski `google_gemma-4-26B-A4B-it-imatrix.gguf` (590 tensors / 295 loaded / 822 chunks, dataset `/training_dir/calibration_datav5.txt`) | `imatrix.dat` parses to 205 entries; GGUF parse: 590 tensors, chunk_count 822; strings: dataset exact; quantize/bench logs: `load_imatrix: loaded 295 importance matrix entries ... computed on 822 chunks` | VERIFIED |
| 33 | Base quant Q3_K_M | — | LIVE card: "base quant lineage: Q3_K_M with bartowski imatrix"; v6.1 filename `...-Q3_K_M.gguf` | VERIFIED |
| 34 | v6 bench anchors | ARC 95.5631, HellaSwag 84.55, MMLU 71.3333 | `osmosis-gemma4-26b/benchmark_results/cerebellum_v6_*_results.json`: 95.5631 (1120/1172), 84.55, 71.3333 (1712/2400) | VERIFIED |
| 35 | Q3_K_M baseline anchors (card) | — | LIVE card: 95.22/86.57/73.67 ↔ `q3km_baseline_*_results.json`: 95.2218 (1116), 86.5664 (8693), 73.6667 (1768) | VERIFIED |
| 36 | Live card matches local audited JSONs | — | LIVE card frontmatter 0.9556/0.8455/0.7133 and tables match local JSONs; card honestly shows v6 losing HellaSwag/MMLU-Redux to baseline and marks v6 HumanEval "for audit" | VERIFIED |
| 37 | v6.1 = metadata-only | v6.1 kept allocation, fixed template/runtime metadata | LIVE card: "keeps the v6 tensor allocation with zero tensor changes (metadata-only update)" + sha256s published | VERIFIED |
| 38 | Router layer-8 surgery | layer 8 F32→Q8_0: PPL 12,054 vs 12,356; HumanEval 72.0 vs 71.3; layer 10 dropped HumanEval to 61.6; K-quants broke routers | `DEVLOG_2026-05-01_router_road_mapping.md` (12,054/-2.4%, 72.0 vs 71.3); `ablation/surgery/road_mapping/router_layer8_humaneval_results.json` pass_at_1_pct 72.0; `FULL_EXPERIMENT_LOG.md` line 82: layer 10 61.6 "CATASTROPHIC" | VERIFIED |
| 39 | Final v6.1 artifacts exist | templatefix GGUF + mmproj | sha256s on live card; local paths in guide (existence asserted by forensics passes 1–2; not re-stat'd this pass) | VERIFIED (via card hashes) |
| 40 | Heretic 26B transfer anchors (adjacent claim) | ARC 95.48, HS 83.49, MMLU-R 71.42, refusal 1/45 | User's own HF discussion reply (discussions_raw.json) states the same numbers publicly | VERIFIED (cross-source; heretic out of core scope) |

## Cross-cutting

| # | Claim | Verdict |
|---|---|---|
| 41 | "The three winners were three different workflows" (27B sparse-probe+allocator; 35B group/reverse demotion; 26B hand-evolved map) | VERIFIED — the three override files have structurally different shapes (181 mixed-type tensor-level; 360 uniform-Q2_K group pattern; 91 family-targeted map), matching the guides' core thesis |
| 42 | Benchmark-correction history (fence-stripping ~7-8 pts, ARC 19 labels, HellaSwag 108 empties) | VERIFIED — `BENCHMARK_CORRECTIONS.md` exists; LIVE 27B card discloses the same three bugs publicly in its "2026-05-03 Score Corrections" paragraph |
| 43 | Org card head-to-head (11.96 GB vs 16.87 GB uniform) | NOT IN GUIDES — sourced from commit 780ff06 (2026-06-11). Note: those files are `uniform-q3km-heretic-35b_*`; the org card labels the row "Qwen3.6-35B-A3B" without saying heretic. Worth a wording check before the next org-card edit. |

## Bottom line

The reconstruction is solid. Every override file count, type distribution, ablation
number, imatrix property, and published benchmark anchor checked out exactly against
primary evidence, across all three shipped winners. The five flags are: one real
public-card error (M1: 8.256 vs 7.6494 on the 27B card's size table), one stale
inflated claim in `start.md` (M2: 448 vs 23), one canonical-file ambiguity between
the two guides (M3: fixed vs corrected MMLU-Redux/ARC), and two minor live-card
phrasing drifts on the 35B card (M4, M5). None of them change the method story;
M1 is the one worth fixing publicly.
