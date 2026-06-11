# Forensic Catalog: Other CLI Session Histories — Cerebellum Quantization Project
Generated: 2026-06-11 (agent-mined from codex/opencode/gemini session stores; saved by orchestrator)

## I. Codex CLI — dominant non-Claude agent (May 1 – Jun 7, 384 relevant sessions)

Phases: (1) May 1–5 crash recovery + benchmark debugging; (2) May 11–20 9B v2_code winner, Heretic Gemma build, hill-climber dev; (3) Jun 1–7 Heretic Qwen 35B build, hillstep investigation, process comparison.

### Build commands found
- **Qwen 3.6 35B v1 (released)**: all 40 layers x 10 groups -> Q2_K via cerebellum_v1_overrides.txt (400 lines); `llama-quantize --imatrix imatrix.dat --tensor-type-file cerebellum_v1_overrides.txt f16.gguf out.gguf Q2_K`
- **Qwen 3.6 35B v2 (Jun 1)**: cerebellum_v2_overrides.txt (280 lines) — protects attn_qkv + routed experts at Q3_K_M. Benches: ARC 96.08, HellaSwag 92.08, MMLU-Redux 76.29, EvalPlus 66.46/60.37, vision 100%, RealWorldQA 78.0. File: games/models/qwen3.6-35b-a3b-cerebellum-v2.gguf (15 GB).
- **Heretic Qwen 3.6 35B (Jun 2–3)**: source llmfan46 **MTP-Preserved** BF16 (KL=0.0015). v3 overrides (360 entries) + 20 MTP entries at BF16. Output 13 GB (since pruned — failed build). Triggered by HF user TheodoreH comment (May 31, Discussion #3 on Qwen3.6-35B-A3B-Cerebellum-GGUF) requesting a Heretic Qwen.
- **Gemma 4 26B v6.1 (May 22)**: chat-template metadata fix ONLY, same weights as v6. SHA256 d24229fac... Companion Heretic v1.1-templatefix SHA256 103f97331... Release checkpoint: osmosis-gemma4-26b/hf_release_v6_1_templatefix_20260522.md
- **Qwen 3 30B v3**: coder imatrix, benchmark-gated. tensor_types_v3.txt protects attn_q/v/ffn_down at Q3_K_M, demotes attn_k/output/ffn_gate/up to Q2_K. ARC 92.66, HellaSwag 83.83, MMLU 66.62, EvalPlus 75.00/70.73.

### Errors and bugs catalog
1. **v6 HumanEval false 36.0%** (May 4 rerun overwrote May 1 result): original 61.6% at 115 compl/min; rerun 36.0% at 3/min — wrong/throttled server. Flagged May 5, never fully traced. The 36.0% in cerebellum_v6_humaneval_results.json is false.
2. **v7 HellaSwag false 0%** (May 1): harness counted API transport failures as wrong answers against a stale server. Fix: benchmark_utils.py patched to abort on API errors. Rerun ~84.05%.
3. **MMLU abort at 600/2400**: --ctx-size 4096 --parallel 4 = 1024 tok/slot; MMLU prompts up to ~1070 tok -> HTTP 400. Fix: >=2048 tok/slot; resumed from checkpoint.
4. **ARC TypeError crash** (May 5): benchmark_arc.py:130 query_model() returned tuple, pool.map expected string. Patched.
5. **EvalPlus null pass_at_1_plus**: summary parser bug; patched May 5.
6. **HumanEval fence/indentation bug**: chat wrapper stripped fences without normalizing indentation -> IndentationError -> systematic underscore. Base HumanEval RETIRED, replaced by EvalPlus.
7. **enable_thinking empty completions** (Qwen 3.5 9B): server defaulted to thinking -> "content":"" for all requests. Fix: chat_template_kwargs {"enable_thinking": false} + thinking_budget_tokens 0 in every request.
8. **Qwen 35B v1 "vision damage" claim was FALSE**: all variants scored 100% on 36-image smoke + 78% RealWorldQA. The "75.0% HumanEval" README figure was inflated by the fence bug; corrected v2 EvalPlus+ = 60.37%. v2 still kept for real gains (ARC +1.3, HS +0.5, MMLU +2.4).
9. **Gemma 26B chat template bug** -> v6.1/v1.1 templatefix releases (metadata only).
10. **libcudart.so.12 missing** for CUDA llama-quantize outside distrobox -> use build-cpu/bin/llama-quantize.
11. **Legacy imatrix format warning**: accepted with warnings; code-weighted imatrix used for 9B rebuild.
12. **9B v1 false failure**: wiki-only imatrix misalignment + contaminated benchmarks + too-generous budget. v2_code (code-weighted imatrix, 4.0 GB) = winner at 53.0% EvalPlus+.
13. **9B v3_rowblock anomaly**: passed ARC/HellaSwag/MMLU but failed real agent-loop snake task. Unresolved; row-block allocation suspect.

## II. Hill-climber (hillstep) vs earlier formula — definitive comparison
Source: ~/.codex/sessions/2026/06/06/rollout-2026-06-06T01-02-35-*.jsonl

**Old process (Qwen 3.6 27B — the formula that worked)**: PPL + benchmark feedback; group-first ablation (23 test points, not 616-tensor walk); groups attn_qkv/attn_v/ffn_down/ssm/output; multiple baselines; protected tensors attn_qkv=q8_0/q6_K, attn_v=q8_0; release gates ARC 96.76 / HS 92.21 / MMLU 76.58 / HumanEval 81.10; v3->v4 used coder imatrix + per-group HumanEval deltas.

**New hillclimber (Gemma 4 12B, Jun 4–6)**: wiki PPL ONLY (benchmark_suite=null); exhaustive 616-tensor walk; Q4_K_M baseline; no benchmark gate; unstable PPL (2504.28 -> 1615.85, flagged as broken calibration); stopped externally (reason benchmark-block10-vs-base-q4).

Codex verdict quote: "Wiki PPL alone is not a safe objective. Prior artifacts show PPL wins can destroy HumanEval/tool behavior... old Qwen used group/layer ablations, ladder/additive checks, and benchmark feedback."

hillstep.py itself warns: line 315 "Use the proven group-first, benchmark-gated workflow instead of exhaustive wiki-only per-tensor hillclimb"; line 378 "run coarse group Q2_K survivability ablations before any per-tensor hillclimb"; --with-targeted-hillstep is an optional add-on AFTER group-first scan.

Also: osmosis-gemma4-e2b/README.md — v1 improved PPL but destroyed benchmarks (HumanEval 17.7% vs baseline 46.3%); v2 surgical layer demotion kept benchmarks intact.

## III. Gemma 26B build chain
v6 (new recipe, 11 GB) -> v6.1-templatefix (same weights) -> Heretic v1 (v6 recipe on coder3101 Heretic base) -> Heretic v1.1-templatefix. Heretic corrected EvalPlus: 92.07/89.63 (raw-completions false-low was 3.05).

## IV. Qwen 3.6 35B version map (as of Jun 3)
| Version | File | Size | Status |
|---|---|---|---|
| v1 released | qwen36-35b-v1-public/Qwen3.6-35B-A3B-Cerebellum.gguf | 12 GB | HF |
| v2 | models/qwen3.6-35b-a3b-cerebellum-v2.gguf | 15 GB | benched Jun 1 |
| Q3_K_M baseline | models/qwen3.6-35b-a3b-q3km-baseline.gguf | 16 GB | reference |
| v3 | qwen36-35b-v2/Qwen3.6-35B-A3B-Cerebellum-v3.gguf | 12 GB | Jun 2 |
| Heretic v3 (Qwen3.5 base) | (pruned 2026-06-11 — failed) | 12 GB | failed |
| Heretic MTP-preserved | (pruned 2026-06-11 — failed) | 13 GB | failed, regressed |

## V. OpenCode CLI (deepseek-v4-pro, Jun 1–9, 19 sessions)
All read-only subagent sessions, no code modifications. Key: Jun 1 TheodoreH HF comment investigation -> decision to build Heretic Qwen; Jun 1 override-file audit (v1=400 lines all Q2_K; v2=280 lines protective); Jun 1 finding that v1 "vision damage" was not real; Jun 3 full artifact survey (15 benchmark dirs, 265 summary JSONs, 55 tensor_types TXTs, 41 GGUFs, 136 docs). Jun 7–9 sessions = conch-poc (separate, not quant).

## VI. Gemini CLI
Zero cerebellum quant work (only carl, scrithub, brief conch-poc).

## Key source pointers
- ~/.codex/sessions/2026/06/06/rollout-2026-06-06T01-02-35-019e9b86-af18-75f0-af7c-a651ee5482cb.jsonl (process comparison)
- osmosis-gemma4-26b/hf_release_v6_1_templatefix_20260522.md (v6.1 release)
- ~/.local/share/opencode/opencode.db (session + part tables)
- osmosis/hillstep.py lines 305-384, 1369, 1812, 5117 (group-first warnings)
