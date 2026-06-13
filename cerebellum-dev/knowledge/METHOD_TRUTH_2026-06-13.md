# THE definitive method finding (triangulation-verified, supersedes earlier claims)

2026-06-13. A 17-agent triangulation (3 agents/version x4 + consensus, primary
artifacts only, mtime-checked, forbidden from smoothed guides) OVERTURNED the
single-agent "coding ablation was the lost secret" finding. Trust this doc over
CRITICAL_LOST_STEP and the earlier devlog where they conflict.

## What actually produced the 27B winners (v1-v4) — PPL-gated end to end
1. sensitivity-proxy imatrix (osmosis-sensitivity, 496 entries — NOT wikitext activation)
2. SINGLE-TENSOR PPL ablation (23 individual tensors, baseline Q4_K_M PPL 8.2556) — NOT group
3. **MULTI-PASS PROMOTION budget allocator** = cerebellum/budget.py, PROMOTION_ORDER
   q2_K->q3_K->q4_K->q5_K->q6_K->q8_0. v4 = 181 overrides (Q2x22/Q3x19/Q4x22/Q5x70/Q6x41/Q8x7).
   THIS is v4's distinguishing lever. It survives in code, never deleted.
4. stock llama-quantize -> PPL 7.0344 (beat Unsloth Q2_K_XL 7.040), HumanEval ~81%.

## The coding ablation was POST-HOC, not the method
Every coding_ablation artifact is dated May 2-3, headed "Base: v4". v4's override froze
Apr 29. The coding ablation ran 3 DAYS AFTER v4 shipped — a diagnostic to plan v5, never
a build input. The per-layer drill was truncated, drove nothing. (The attn_qkv 75->28.7%
finding is TRUE information about coding sensitivity, just not how v4 was built.)

## The 26B took a different road (Gemma triangulation, HIGH conf)
PPL group ablation + PLE protection sweep + ITERATIVE build/bench: build v1 -> full
HumanEval -> adjust map -> rebuild, v1 through v7. No per-group coding ablation.

## The actual overnight regression (Flash/North/12B)
They (a) used GROUP multi-domain PPL instead of SINGLE-TENSOR PPL, and (b) BYPASSED
cerebellum/budget.py entirely — hand-rolling flat "group-verdict translation" because
the group output did not fit the allocator's per-tensor input. They had the winning
allocator and did not use it. Their no-ship verdicts are INVALID.

## THE FIX (matches both winners)
single-tensor (or fine-grained) PPL ablation -> cerebellum/budget.py multi-pass
promotion allocator -> build -> bench-gate -> ITERATE if it loses (26B style).
NOT group-verdict translation. Coding ablation optional as a diagnostic/gate, not required.

## Why this was recoverable only by triangulation
One agent saw coding_ablation/ and called it the method (didn't check mtimes). Three
agents cross-checking caught the dates. Jerry's "small, scoped, multiple agents on the
same version" methodology is the reason the truth surfaced. Log this as the standard.
