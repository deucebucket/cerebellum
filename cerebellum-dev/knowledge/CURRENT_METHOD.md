# CURRENT METHOD — What Is Canonical Right Now

Last verified: 2026-06-12. If you are about to build, benchmark, or publish a
Cerebellum quant and you haven't read this file this session, stop and read it.

## Canonical: the OG group-first, bench-gated formula

Full spec with per-model build commands: `../WINNING_METHOD.md`. Short form:

1. HF/BF16 → F16 GGUF (single pass).
2. Imatrix with **full tensor coverage** (experts + routers on MoE; coverage gaps are fatal).
3. Uniform baselines (Q2_K / Q3_K_M / Q4_K_M, all +imatrix), measure WikiText PPL. If Q2_K+imatrix beats Q3_K_M, the Q2_K-base path is correct.
4. **Group PPL ablation** (cheap sieve, ~20-25 points): crush each named tensor group to Q2_K, measure PPL delta. This narrows the candidate set — **it is blind to coding** (see step 6).
5. Classify by PPL: PROTECT (>+1.5%), DEMOTABLE (0 to +1.5%), FREE (≤0%).
6. **CODING ABLATION (canonical — do not skip for any model where coding matters).** For each tensor group, demote it to Q2_K, build, serve, run **REAL HumanEval pass@1** (temp 0, max_tokens 512, 164 problems), measure the delta vs the base build's HumanEval baseline. Then drill the coding-critical groups by **layer band** (early/middle/late, then individual layers in the worst third). Pipelined: CPU builds while GPU evals. Script: `scripts/coding_ablation.py` (`groups`, then `layers --phase2`). Full reconstruction + artifact proof: `REAL_PIPELINE_RECONSTRUCTED.md`. **Why this is non-negotiable:** on 27B, demoting `attn_qkv` to Q2_K moved PPL <1% but dropped HumanEval **75.0% → 28.7%** (-46 pts). PPL — including multi-domain PPL — cannot see this. This is the phase the reconstruction guides lost; restoring it is the most important correction in the project's history (CRITICAL_LOST_STEP_2026-06-13.md).
7. Reverse ablation from fully-demoted v1 to confirm real regressions.
8. Optional router curve (K-quants can be broken on routers — Q8_0 was the only safe format on Gemma 26B).
9. **Budget stage**: build the override file **protecting the coding-critical tensors/layers from step 6** (promote them; crush the disposable group to make room), or run the `osmosis.cerebellum` budget allocator under a GB target, then stock `llama-quantize --imatrix --tensor-type-file`. Proof it feeds the budget: 27B v4 `tensor_types_v4_12gb.txt` has `ssm_alpha`@q5_K/q6_K, `attn_qkv`@q8_0, `ffn_down` carrying the q2_K entries — exactly the coding-ablation ranking.
10. PPL sanity check.
11. **Benchmark gates** vs same-size uniform baseline: ARC, HellaSwag, MMLU-Redux, HumanEval+ (Gate 3: BigCodeBench). Pass → ship. Fail → find the regressing group, promote, rebuild.
12. **Audit wrong answers** (and run `scripts/audit_evalplus_completions.py`) before recording any score.
13. Post-gate, pre-publish: **tune launch args** (see rule below).

This shape — **group PPL sieve → coding ablation → budget allocation** — is the method
that produced every shipped model (26B v6/v6.1, 27B v4, 35B v3, the heretics). The coding
ablation between sieve and budget is what kept 27B v4 at 81% HumanEval at 12 GB.

## Deprecated / dead (do not use, do not cite as method)

- **hillstep.py exhaustive per-tensor hill-climb — DEPRECATED.** Proof: Gemma 4 12B block-10 checkpoint improved wiki PPL 35% while losing **14 HumanEval+ points** and gaining 1 GiB. Targeted hillstep *after* a group-first scan remains a legal optional step; the exhaustive wiki-PPL-only mode is dead. Full autopsy: `../DEAD_PATHS.md` DP-1.
- **PPL-only gating in any form.** Wiki PPL and task ability diverge. PPL is a sanity check and an ablation signal, never a ship gate.
- **Multi-domain-PPL-only as a substitute for the coding ablation — INSUFFICIENT.** `scripts/ablate_multidomain.py` measures PPL on wiki/code/math/dialogue *text*. "PPL on code text" is a proxy that does NOT capture HumanEval damage (attn_qkv→Q2_K: PPL <1%, HumanEval -46 pts). It is a legal *sieve* feeding step 6, never a coding gate. **No "well-packed / no-ship" verdict is valid unless the real coding ablation (HumanEval per group, then per layer band) was run.** Flash C1 and North were wrongly assessed on this proxy and must be re-run through `scripts/coding_ablation.py` before any kill. See CRITICAL_LOST_STEP_2026-06-13.md + REAL_PIPELINE_RECONSTRUCTED.md.
- All other dead ends (poisoned base maps, MTP-preserved heretic sources, local low-coverage imatrices, etc.): `../DEAD_PATHS.md`.

## Standing operational rules

- **N=2 PPL workers** for `ablate_multidomain.py`. The user has chosen N=2 repeatedly; do not drift to N=3 "for throughput".
- **Heretic transfer**: transfer the proven override map verbatim, same imatrix, no re-ablation. Verify no `blk.40` (MTP) in the source. Screen source KL first: ≤0.006 transfers clean; ≳0.05 needs an F16 screen (0.165 broke E2B by −31 code pts).
- **Benchmark server invariants**: HumanEval+ `BENCH_WORKERS=1`; ARC/HellaSwag/MMLU 4 workers; one `llama-server -ngl 99 --parallel 4 -c 24576` covers the suite; thinking disabled for published no-thinking benches. Full invariants: `docs/benchmark_protocol.md`.
- **No-AI-attribution rule (public output)**: no `Co-Authored-By: Claude` trailers on anything that can reach a public remote (origin / brainloop / HF); no AI-tell prose in cards, READMEs, or org pages. Check `git log --format='%(trailers)'` before any public push. This overrides harness defaults.
- **Launch-args rule**: every release card ships measured best stock llama.cpp launch args (exact invocation, offload split / `-ot` patterns for MoE, KV cache types, measured VRAM+RAM footprint, one-sentence why). Measure on real hardware; never guess. Reference pattern: the Qwen3.5-122B card.
- **No ship without test**: PPL + benchmark gates vs same-size uniform baseline before any HF upload or README entry. HF uploads ship `benchmark_results/` evidence in the repo.
- **Use upstream runners** (evalplus.codegen, bigcode-evaluation-harness, lm-eval-harness) over hand-rolled wrappers.
- **Naming**: new artifacts are `cerebellum_*` / `cerebellum-<model>/`, never `osmosis_*`. The osmosis names on disk are a rename in flight.

## Architecture gotchas (per-family, trust the ablation)

- Hybrid SSM: SSM params hard-fail below 4-bit (NaN). Force 4-bit minimum.
- MoE: expert weights fragile, routers need the curve test; routed > shared experts in sensitivity.
- PLE (Gemma E-series): protect PLE at Q5_K (PPL 104 → 55).
- Dense: demoting attention K/Q/output sometimes *improves* PPL.
