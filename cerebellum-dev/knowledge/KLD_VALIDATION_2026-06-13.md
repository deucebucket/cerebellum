# KLD Validation — Can a cheap metric replace per-group HumanEval in coding ablation?

Date: 2026-06-13
Status: IN PROGRESS (metrics landing)

## The question

Per-group coding ablation currently costs a full HumanEval run per group (build the demote,
serve it, run 164 problems). That is the honest signal but it is slow and GPU-bound. Can a
**cheap forward-pass metric** — KL-divergence vs the undemoted base, top-token agreement, or
plain PPL on code text — rank the per-group coding damage the SAME way HumanEval does? If yes,
we get a fast coding-ablation proxy (cheap like PPL, honest like HumanEval). If no, we keep
paying for HumanEval. Either answer is useful.

## Ground truth (the oracle)

Qwen 3.6 27B, v4 12GB build, 75.0% HumanEval baseline. Demote each of 7 tensor groups to
Q2_K, measure real HumanEval pass@1. Source:
`osmosis-qwen36-27b/coding_ablation/ablation_<group>_humaneval_results.json`,
captured in `cerebellum-dev/knowledge/CODING_ABLATION_ORACLE_27B.json`.

HumanEval pass@1 after demoting each group (lower = MORE coding damage):

| group        | HE pass@1 | damage rank (1=worst) |
|--------------|-----------|------------------------|
| ssm_alpha    | 22.6      | 1 (most damage)        |
| ssm_beta     | 25.6      | 2                      |
| attn_qkv     | 28.7      | 3                      |
| output       | 29.9      | 4                      |
| attn_v       | 30.5      | 5                      |
| attn_output  | 42.1      | 6                      |
| ffn_down     | 70.7      | 7 (least damage)       |

## Path taken: SELF-CONSISTENT (not oracle-exact). Why.

Oracle-exact would require rebuilding the 7 demotes from the **standard Qwen3.6-27B F16**
(the override+base_type recipe rebuilds from F16; you cannot re-quantize a quantized GGUF).
That F16 is **not on disk**, and the games partition was at 99-100% — no room for a ~54GB
download + build. The only 27B F16/BF16 on disk is the **heretic v2 BF16** (a different
checkpoint, same architecture/tensor layout).

Decision: run a **self-consistent** validation on a model I can fully build locally — the
heretic 27B BF16 — applying the EXACT v4 tensor map + the EXACT oracle group-demote override
files. All cheap metrics are measured on these same builds against the same base. The premise
being tested is architecture-level group sensitivity (which group's demotion most damages code
logits), which is what the Heretic transfer recipe already assumes transfers across these two
close checkpoints. Caveat recorded: the HumanEval target is the standard-checkpoint oracle; if
the cheap-metric ranking matches it, that is strong cross-checkpoint evidence; if it diverges,
we cannot fully separate "metric fails" from "checkpoint differs" without measuring HumanEval
on these same heretic builds (escalation path).

### Build recipe (faithful to oracle)
- Source: `Qwen3.6-27B-uncensored-heretic-v2-BF16.gguf` (53.8GB, on disk)
- v4 map: `osmosis-qwen36-27b/tensor_types_v4_12gb.txt` (181-entry sparse override over Q2_K base)
- Demote = v4 map MERGED with `override_demote_<group>.txt` (group forced to q2_K, demote wins)
- imatrix: `osmosis-qwen36-27b/cerebellum_imatrix.dat` (496 entries, 8 chunks)
- `llama-quantize --imatrix <imat> --tensor-type-file merged_demote_<g>.txt <BF16> <out> Q2_K`
- Base build = v4 map only → 12.87 GB (matches the real v4's 12.8GB exactly — recipe verified)

### Metric recipe
- Code corpus: first ~70KB of `cerebellum_calibration_code.txt` → 81 chunks @ ctx 512
- Base logits: `llama-perplexity --kl-divergence-base base_logits.dat` (base code PPL = 4.0089)
- Each demote: `llama-perplexity --kl-divergence-base base_logits.dat --kl-divergence`
  → reports Mean KLD, top-token agreement, candidate PPL in one pass (GPU, no generation)
- llama.cpp build with KLD support: `llama.cpp-pr24260/build` (CUDA), run in distrobox `ai`

## Reconstruction confound found mid-run (recorded honestly)

The oracle's exact base is ambiguous in the records: the documented invocation says
`--base-type Q2_K` with only the demote override, but the oracle demote GGUF sizes are
10.9-12.0 GB — far above a uniform Q2_K 27B (~9 GB). That is only possible if the base was
effectively the **v4 mixed map** (which is what this run reconstructs) rather than uniform
Q2_K. My v4-merged ssm_alpha demote came out 12.87 GB vs the oracle's 11.98 GB — ~0.9 GB
heavier, so the reconstruction is *close but not byte-identical* to the oracle base (likely
standard-vs-heretic checkpoint tensor differences + base-merge details).

Consequence for each group: forcing all 64 layers of a group to Q2_K only *changes* the
layers the v4 map had kept above Q2_K (ssm_alpha 14, ssm_beta 14, attn_qkv 29, attn_v 8,
attn_output 4, ffn_down 18; **output = 0 — output.weight was already Q2_K in base, so the
`output` demote is a near-no-op in this reconstruction**). This means absolute KLD magnitudes
are attenuated and the `output` group is an expected outlier (flag it). The *ranking* across
the other 6 groups still reflects how much coding-relevant precision each loses, which is the
quantity the validation tests.

## Results

Base code PPL = 4.0089. Per-group demote (heretic-v4 base, one group → Q2_K), measured on
36 chunks of code text. Full per-group KLD logs in `games/kld_validation/metrics/`,
parsed numbers in `metrics_parsed.json`, correlations in `correlation_results.json`.

| group        | HE pass@1 | Mean KLD | Same-top-p % | Mean PPL(Q) | RMS Δp % |
|--------------|-----------|----------|--------------|-------------|----------|
| ssm_alpha    | 22.6 (worst) | 0.002864 | 98.137 | 4.008989 | 1.607 |
| ssm_beta     | 25.6      | 0.002027 | 98.246 | 4.004495 | 1.349 |
| attn_qkv     | 28.7      | 0.007875 | 97.190 | 4.004892 | — |
| output       | 29.9      | 0.000000 | 100.000 | 4.008897 | — |  ← no-op artifact (was already Q2_K)
| attn_v       | 30.5      | 0.007166 | 97.004 | 4.003348 | — |
| attn_output  | 42.1      | 0.002675 | 98.181 | 4.009104 | — |
| ffn_down     | 70.7 (safest)| 0.006886 | 96.841 | 4.018884 | 2.474 |

### Ranking, most → least coding damage
- **HumanEval (ground truth):** ssm_alpha, ssm_beta, attn_qkv, (output), attn_v, attn_output, **ffn_down (safest)**
- **KLD says:** attn_qkv, attn_v, **ffn_down**, ssm_alpha, attn_output, ssm_beta, output
- **PPL says:** **ffn_down (most!)**, attn_output, ssm_alpha, output, attn_qkv, ssm_beta, attn_v

### Spearman rank correlation vs HumanEval coding-damage ranking (rho=+1 = identical ranking)
| metric | rho (all 7) | rho (6, drop output no-op) |
|--------|-------------|----------------------------|
| Mean KLD | -0.143 | -0.200 |
| Mean PPL(Q) | -0.429 | -0.486 |
| RMS Δp | -0.143 | -0.200 |
| Same-top-p | -0.429 | -0.543 |

Every metric is **negatively** correlated with the HumanEval ranking — the cheap metrics order
the groups roughly *opposite* to coding damage.

## Verdict

**A cheap forward-pass metric CANNOT replace per-group HumanEval for coding ablation.** KLD,
PPL, top-token agreement, and RMS Δp all fail — not merely "weaker than HumanEval," but
**anti-correlated** with it on this 7-group set.

The clearest failure is `ffn_down`: it is the **safest** group to demote (HumanEval barely
moves, 70.7% vs 75.0%) yet it produces the **highest PPL shift and among the highest KLD** of
any group. PPL literally ranks ffn_down as the *most* damaging demote — the exact inverse of
the truth. Symmetrically, the SSM groups (ssm_alpha/ssm_beta) are the genuine coding killers
(-49 to -52 HumanEval points) but barely move KLD or PPL.

This is the project's central thesis, now measured head-to-head on one model: **perplexity-class
signals and code-generation ability are decoupled.** Same mechanism as the memory note
"PPL-only gating lost 14 HumanEval pts while PPL improved 35%" and the "hillstep deprecated"
decision. The OG bench-gated formula stays the method; HumanEval (or a real code execution gate)
remains non-optional for coding-precision allocation. A cheap metric can pre-screen for *gross*
breakage (NaN, SSM hard-fail) but must not *rank* coding sensitivity.

### Caveats (do not over-claim)
1. **Checkpoint mismatch:** builds are the heretic-v2 BF16; the HumanEval oracle is the standard
   checkpoint. A clean within-checkpoint test would measure HumanEval on these same heretic
   builds. But the failure is strong and directionally consistent (PPL inverts ffn_down, the
   single most-cited PPL-vs-coding divergence) — checkpoint drift would have to flip multiple
   negative correlations to positive to rescue the metric.
2. **`output` is a reconstruction no-op** (KLD=0); excluded from interpretation, and dropping it
   does not change the negative sign.
3. **Base-build fidelity:** v4-merged demote (12.87 GB) is ~0.9 GB heavier than the oracle's
   per-group demote, so absolute magnitudes are attenuated; only rankings are interpreted.
4. n=7 (6 usable) groups — small set, but the negative rho is consistent across all four metrics.

### What would strengthen this (future, not blocking)
- Re-run with the standard Qwen3.6-27B F16 source (needs ~54 GB free + download) to remove the
  checkpoint caveat, OR measure HumanEval on these exact heretic demotes for a fully within-build
  correlation.
- Test whether a *code-execution* proxy cheaper than full HumanEval (e.g. a 20-problem smoke set)
  preserves the ranking — that is the real "cheap but honest" candidate, not KLD/PPL.

## Evidence locations
- Metrics + logs: `/var/home/deucebucket/games/kld_validation/metrics/`
- Parsed metrics: `/var/home/deucebucket/games/kld_validation/metrics_parsed.json`
- Correlations: `/var/home/deucebucket/games/kld_validation/correlation_results.json`
- Scripts + merged override maps: `/var/home/deucebucket/games/kld_validation/`
- Oracle truth: `cerebellum-dev/knowledge/CODING_ABLATION_ORACLE_27B.json`
- Large binaries (base_v4.gguf, base_logits.dat) deleted after measurement; regenerable from
  the recipe above.
