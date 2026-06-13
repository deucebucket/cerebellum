# Brainloop A/B Wiring Forensic Audit
Generated: 2026-06-11

---

## Verdicts Table

| Benchmark / Pair | Identical% | Verdict | Key Evidence |
|---|---|---|---|
| gguf_baseline vs gguf_deadblock | 98.2% (161/164) | WIRED-CORRECTLY | Different model files loaded per log: baseline=`qwen2.5-3b-brainloop.gguf`, deadblock=`cerebellum-deadblock-python.gguf`. 161/164 known-expected (eval script comment says ~161). 3 differ = models are distinct. |
| pytorch_baseline vs pytorch_conch | 45.1% differ (74/164) | WIRED-CORRECTLY | Published 62.2 vs 56.7 scores confirmed in eval_base.log / eval_conch.log. 74/164 completions differ — active intervention. bench_humaneval.py loads `checkpoints-fusion-13k/fused_refiners.pt` explicitly. |
| qwen7b-baseline vs qwen7b-baseline-real | 100.0% (164/164) | MISWIRED / DEGENERATE | Both files contain 164 completions of literally `"    pass"` (8 bytes each). Both results files show pass@1=None, elapsed=0. These are broken runs with generation failure — the "baseline" was never actually generated. File hash identical. |
| brainloop-rag-coding vs brainloop-sharp-rag | 97.6% (160/164) | MISWIRED | Both runs show identical scores (32.32/36.59 pass@1_plus). 4 completions differ only in whitespace/trivial variable rename. Elapsed times differ (280s vs 294s) suggesting two real runs, but with functionally identical models or identical RAG config — the intervention (sharp vs non-sharp RAG) was effectively inert. |
| brainloop-best-combo vs brainloop-fix-13k | 100.0% (164/164) | MISWIRED | Files are byte-identical (same MD5). Both results files report identical scores and elapsed_seconds=134.9. One is a file copy masquerading as a separate run. |
| brainloop-both-active vs brainloop-combined | 70.7% differ | WIRED-CORRECTLY (sufficient divergence) | 48 completions differ, distinct scores, not a copy. |
| brainloop-code-examples vs brainloop-code-trained2 | 17.7% identical | WIRED-CORRECTLY | 82% differ — clearly distinct models. |
| qwen7b-true-baseline vs qwen7b-rag | 13.4% identical | WIRED-CORRECTLY | Strongly distinct. |
| qwen7b-true-baseline vs qwen7b-full-13k | 13.4% identical | WIRED-CORRECTLY | Strongly distinct. |
| recall_results_deadblock.json A vs B | 100.0% (200/200) | MISWIRED | model_a=`qwen2.5-3b-brainloop.gguf`, model_b=`cerebellum-deadblock-python.gguf`. Both models returned identical completions on all 200 prompts AND identical recall scores (hits=20/200). The recall bench cannot distinguish the two models — either the server ran the same model twice, or both models produce identical recall behavior on this task. |

---

## Checkpoint Forensics

**checkpoints-fusion-13k/fused_refiners.pt** (PyTorch conch bench):
- l17.gate: tanh(-0.005) = **-0.005** → NEAR-INERT (scales output by 0.5%)
- l30.gate: tanh(-0.00485) = **-0.00485** → NEAR-INERT
- l17.rag_scale: sigmoid(0.0) = 0.5; l30.rag_scale: sigmoid(0.00412) ≈ 0.501
- l17.inj_proj.weight: **L2 norm(W - I) = 0.000** → identity, did NOT train
- l30.inj_proj.weight: **L2 norm(W - I) = 4.48** → did train
- Conclusion: refiners were near-inert at bench time. The published -5.5% "logic tax" (62.2→56.7) is confounded — the score drop likely reflects the prompt-formatting difference between the base bench and the conch bench (instruct prompt wrapping), not the refiner intervention.

**checkpoints-deadblock-13k/dead_blocks.pt** (GGUF deadblock bench):
- down_proj rows[:1536] L2 norm: **0.000** (l17 and l30) → correctly zeroed
- down_proj rows[1536:] L2 norm: 1.39 / 1.60 → trained and non-zero
- PARITY CHECK in train log: `max_abs_diff = 0.0 PASSED` confirms zero-initialization
- Conclusion: dead block architecture is correctly implemented.

**checkpoints-liveblock/live_block_best.pt**:
- All weight norms >> 0 (e.g., mlp.gate_proj norm=125, down_proj norm=18.5)
- meta: epoch=1, val_ppl=8.76, timestamp=2026-06-11 04:13:08
- Conclusion: trained, non-trivial weights.

---

## Provenance Gaps

All results files in `bench_results/` use the same schema with 7 fields: `benchmark, model, pass_at_1_plus, pass_at_1_base, total_problems, elapsed_seconds, timestamp`. **No file records the checkpoint path, git commit, GGUF path, or model revision that produced it.**

Unauditable results (no recoverable checkpoint/config provenance):
- brainloop-code-examples, brainloop-code-trained2, brainloop-rag-trained, brainloop-combined, brainloop-both-active, brainloop-full-corpus — no log files found, no checkpoint ref in results JSON
- qwen3b-refiner-arc, qwen3b-baseline-arc, qwen3b-refiner-hellaswag — no log files found

---

## Most Damning Finding

**The published PyTorch A/B comparison (62.2 vs 56.7 pass@1) is confounded by near-inert gates.** At bench time, `fused_refiners.pt` had `tanh(gate) ≈ -0.005` on both injected layers (L17, L30) — scaling the refiner's output contribution by only 0.5%. L17's `inj_proj.weight` was an exact identity matrix (norm(W-I)=0.0), meaning it never received gradient updates. The -5.5 point drop between baseline and conch likely reflects the instruct-prompt wrapping (added `<|im_start|>user\n...`) rather than the refiner injection itself. The intervention was geometrically present but functionally ~zero.
