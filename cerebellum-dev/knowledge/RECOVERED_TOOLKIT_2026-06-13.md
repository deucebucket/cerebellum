# The canonical pipeline scripts were git-DELETED, not lost — all recovered

2026-06-13. Jerry was right twice: (1) the rebrand should not have deleted anything,
(2) the data is on disk / in git. The original pipeline scripts WERE deleted by a
cleanup commit (scripts/ is gitignored from origin) and are fully recoverable.

Recovery method: `git log --all --diff-filter=D --format='%H' -- <path>` gives the
deletion commit; `git show <deletion_commit>^:<path>` reads the pre-deletion version.

Recovered into cerebellum-dev/knowledge/recovered_scripts/ (deletion commit 005437c8):
- pipeline_gemma4_26b.sh (336 lines) — the Gemma 26B canonical runner
- ablate_gemma4.sh (109) — E4B PPL ablation sweep
- apply_ablation.py (172) — turns ablation results into override files
- run_overnight.sh, run_phase2.sh, run_sensitivity_multi.sh, run_repair_lora.sh, run_targeted_repair.sh

## THE KEY TRUTH THE SMOOTHING DESTROYED: the method is ARCHITECTURE-ADAPTIVE

There was never one generic "ablation + budget" pipeline. Each architecture has its
own sacred-tensor-finding phase, proven by the recovered runners:

- **PLE models (Gemma 4 E-series, 26B)**: the critical phase is the **PLE PROTECTION
  SWEEP** (pipeline_gemma4_26b.sh Phase 4 PLE override gen + Phase 6 PLE protection
  sweep). PLE@Q5_K took PPL 104 -> 55.
- **Hybrid SSM / MoE (Qwen 3.6 27B)**: the critical phase is the **CODING ABLATION**
  (HumanEval pass@1 measured per tensor group, then per layer band) — see
  [[CRITICAL_LOST_STEP_2026-06-13]] and scripts/coding_ablation.py.
- **Multi-domain PPL (scripts/ablate_multidomain.py)**: weaker than BOTH. PPL-on-code
  -text cannot see coding collapse (27B attn_qkv demote: PPL <1%, HumanEval 75->28.7%).
  This is what the 2026-06-12/13 overnight Flash/North/12B runs used. Their "no-ship /
  already-packed" verdicts are INVALID — they never ran the architecture's real phase.

## Mandate
- Before any model is called "already packed", run its architecture's real phase:
  PLE sweep for PLE models, coding ablation for MoE/SSM coders.
- The recovered scripts use old /ai-drive/osmosis paths — they are EVIDENCE/ground-truth
  for the triangulation workflows, update paths before re-running.
- Two triangulation workflows (27B v1-v4, Gemma 26B v1-v6) are verifying this from
  primary artifacts independently. The recovered scripts are the answer key.
