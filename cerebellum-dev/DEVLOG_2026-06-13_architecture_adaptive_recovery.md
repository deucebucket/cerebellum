# DEVLOG 2026-06-13 — architecture-adaptive method recovered; logging is why

## What Jerry had to re-reason from scratch (and shouldn't have had to)

The Cerebellum method is NOT one generic pipeline. It is ARCHITECTURE-ADAPTIVE.
Each model family has its own sacred-tensor-finding phase. This was never written
down as a first-class rule, so when the runner scripts were git-deleted in the
rebrand, the reconstruction smoothed every architecture's phase into one generic
"group ablation + budget", and ablate_multidomain.py (PPL proxy) replaced all of
them. Jerry felt the wrongness in the overnight numbers and re-derived the
architecture difference by hand. That re-derivation should have been a doc read,
not a 4am rescue.

## The rule, now logged so it cannot be lost again

| architecture | sacred-tensor-finding phase | proof |
|---|---|---|
| PLE (Gemma 4 E-series, 26B) | PLE protection sweep | recovered pipeline_gemma4_26b.sh Ph4+Ph6; PLE@Q5_K PPL 104->55 |
| hybrid SSM (Qwen 3.6 27B) | per-GROUP CODING ABLATION (HumanEval per tensor group, then per layer band) | osmosis-qwen36-27b/coding_ablation/; attn_qkv demote 75->28.7% HumanEval |
| PLE/MoE (Gemma 4 26B) | per-BUILD ITERATION: build candidate -> full HumanEval -> adjust map -> rebuild (v1->v7) + PLE sweep + router surgery | Gemma triangulation 2026-06-13 HIGH conf; NO per-group coding ablation (definitive negative); v1-v7 each a full build w/ HumanEval |
| dense (some) | trust PPL ablation, attention sometimes improves when demoted | Granite findings |
| any model where coding matters | coding ablation is MANDATORY, multi-domain PPL is INSUFFICIENT | hillstep -14 HE pts; overnight Flash/North false no-ships |

## THE UNIFYING PRINCIPLE (the actual lost step, stated correctly)

Real HumanEval ALWAYS drove the override map. The granularity was architecture-specific:
- 27B: HumanEval measured per tensor GROUP (fine-grained ablation)
- 26B: HumanEval measured per BUILD VERSION, iterated v1->v7 (coarse-grained, hand-evolved)
Either way: REAL HumanEval, never PPL alone, and ALWAYS iterated/measured before shipping.
"Coding ablation" is one form (27B); "build-bench-adjust-rebuild iteration" is the other
(26B). Both are valid; the model's structure picks which. Multi-domain PPL is NEITHER.

## Why the overnight builds "failed"
Flash, North, 12B got ablate_multidomain.py (PPL on wiki/code/math/dialogue text)
and STOPPED after one allocation. They never iterated with real HumanEval at EITHER
granularity (per-group like 27B, or per-build like 26B). Their
"already packed / no-ship" verdicts are INVALID. PPL-on-code-text cannot see
coding collapse (27B: attn_qkv PPL moved <1%, HumanEval fell 46 points).

## The logging mandate (Jerry, 2026-06-13)
"this is why devlog and logging everything is important." Going forward: every
method decision, every architecture-specific reasoning, every why — gets a dated
devlog entry AT THE TIME, not reconstructed later. Scripts get deleted; logs are
forever. The method's intelligence must live in prose, not only in code that a
cleanup commit can drop. Recovered scripts: cerebellum-dev/knowledge/recovered_scripts/.
