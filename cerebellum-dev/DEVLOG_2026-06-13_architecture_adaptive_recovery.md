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
| hybrid SSM / MoE coder (Qwen 3.6 27B) | CODING ABLATION (HumanEval per group, then per layer band) | osmosis-qwen36-27b/coding_ablation/; attn_qkv demote 75->28.7% HumanEval |
| dense (some) | trust PPL ablation, attention sometimes improves when demoted | Granite findings |
| any model where coding matters | coding ablation is MANDATORY, multi-domain PPL is INSUFFICIENT | hillstep -14 HE pts; overnight Flash/North false no-ships |

## Why the overnight builds "failed"
Flash, North, 12B are MoE/coder-class. They got ablate_multidomain.py (PPL on
wiki/code/math/dialogue text), never the coding ablation their class needs. Their
"already packed / no-ship" verdicts are INVALID. PPL-on-code-text cannot see
coding collapse (27B: attn_qkv PPL moved <1%, HumanEval fell 46 points).

## The logging mandate (Jerry, 2026-06-13)
"this is why devlog and logging everything is important." Going forward: every
method decision, every architecture-specific reasoning, every why — gets a dated
devlog entry AT THE TIME, not reconstructed later. Scripts get deleted; logs are
forever. The method's intelligence must live in prose, not only in code that a
cleanup commit can drop. Recovered scripts: cerebellum-dev/knowledge/recovered_scripts/.
