> SUPERSEDED 2026-06-13 by METHOD_TRUTH_2026-06-13.md — triangulation showed coding ablation was POST-HOC, not the build method. The real lever is the multi-pass promotion allocator (cerebellum/budget.py), which the overnight runs bypassed. Read METHOD_TRUTH first.

# CRITICAL: the reconstruction dropped the CODING ABLATION phase

Found 2026-06-13 (Jerry's instinct: "the math worked, we just lost it"). Verified
against primary artifacts in osmosis-qwen36-27b/coding_ablation/ + coding_ablation_layers/.

## What the REAL winning method did (27B v4, the artifacts prove it)

The OG bench-gated formula is NOT group-PPL-ablation + budget. It has a phase the
reconstruction guides (OG_CEREBELLUM_RECONSTRUCTION_GUIDES, multi_domain_ablation)
FLATTENED AWAY:

1. Group PPL ablation (cheap sieve) — ablation_results.json
2. **CODING ABLATION** (coding_ablation/coding_ablation.log): for EACH tensor group,
   demote it to Q2_K, build the GGUF, run **REAL HumanEval pass@1** (temp 0, 164
   problems), measure the delta vs baseline. Pipelined: CPU builds while GPU evals.
3. **CODING ABLATION BY LAYER** (coding_ablation_layers/): for the coding-critical
   groups, split into early/middle/late layer bands, demote each band, run HumanEval
   again — protect only the layers that actually hold code.
4. Budget allocation PROTECTS the coding-critical tensors/layers found above.

## The proof it matters

Demoting attn_qkv on 27B: PPL delta was a few percent (looks tolerable). REAL
HumanEval: 75.0% baseline -> **28.7%**. PPL completely fails to see coding collapse.
That is the entire reason v4 kept 81% HumanEval at 12GB and every 2026-06-12/13
overnight build "lost coding."

## What the overnight reconstruction ran instead (WRONG)

scripts/ablate_multidomain.py — PPL on wiki/code/math/dialogue TEXT. PPL-on-code-text
is a proxy that does NOT capture HumanEval damage. Flash C1, North, 12B all "lost
coding" because the step that measures+protects coding was never run.

## Mandate

NO "model is already well-packed / no-ship" verdict is valid unless it ran the real
coding ablation (HumanEval per group, then per layer band) — not multi-domain PPL.
Re-run Flash C1 and North candidates through the real pipeline before any kill.
The reconstruction guides must be corrected to restore this phase as canonical.
