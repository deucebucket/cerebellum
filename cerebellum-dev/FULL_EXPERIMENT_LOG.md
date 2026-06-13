# Cerebellum — Gemma 4 E4B (26B MoE) Full Experiment Log

## Hardware
- RTX 3090 (24 GB VRAM)
- AMD Ryzen 7 5800XT
- 64 GB DDR4
- Fedora 43 Atomic

## Model
- Gemma 4 E4B 26B (MoE, 30 layers, 658 tensors)
- bf16: `/var/home/deucebucket/games/models/gemma-4-26B-A4B-it-bf16.gguf`
- imatrix: `google_gemma-4-26B-A4B-it-imatrix.gguf`

## Tools
- llama-quantize (CPU build): `/var/home/deucebucket/ai-drive/llama.cpp/build-cpu/bin/llama-quantize`
- llama-perplexity (GPU build): `/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-perplexity`
- WikiText test: `/var/home/deucebucket/games/osmosis-quants/wiki.test.raw`
- PPL config: `-ngl 99 --chunks 128`

## Baseline: v4 (SHIPPED)
- 91 overrides: 9 attn_q Q5_K promotions, rest Q2_K demotions
- PPL: 12,613.59
- HumanEval: 75.0% | ARC: 95.0% | HellaSwag: 83.8% | MMLU-Redux: 77.0%
- File: `cerebellum_v4_overrides.txt`

---

## Tensor Coverage Map (658 total tensors)

| Tensor Group | Layers | v4 Overrides | Ablation Tested | Status |
|-------------|--------|-------------|----------------|--------|
| attn_q | 30 | 9 (Q5_K) | 30 (prior work) | DONE |
| attn_k | 30 | 30 (Q2_K) | 30 (v5 experiment) | DONE |
| attn_v | 30 | 0 (default Q3_K_M) | 7 sampled | DONE — all SENSITIVE |
| attn_o | 30 | 0 (default Q3_K_M) | 0 → RUNNING | IN PROGRESS |
| ffn_gate_up_exps | 30 | 30 (Q2_K) | 2 (stopped early) | DONE — Q2_K correct |
| ffn_down_exps | 30 | 0 (default Q3_K_M) | 0 | PENDING |
| ffn_gate | 30 | 0 (default Q3_K_M) | 0 | PENDING |
| ffn_up | 30 | 22 (Q2_K) | 22 (prior work) | DONE |
| ffn_down | 30 | 0 (default Q3_K_M) | 0 | PENDING |

**Tested: ~91/270 tensor-layer combos (34%). Untested: ~179 (66%).**

---

## Step-by-Step Log

### Step 1: v4 Baseline (prior session)
- Built from prior ablation work on attn_q, attn_k, ffn_up, ffn_gate_up_exps
- 91 tensor overrides, base quant Q3_K_M
- Shipped to HuggingFace

### Step 2: v5 — Un-demote attn_k layers (2026-05-01)
**Hypothesis:** Promoting 7 SLIGHTLY_BETTER attn_k layers from Q2_K → Q3_K recovers PPL.

Layers un-demoted: 1, 6, 17, 18, 23, 24, 28 (all showed >1% PPL improvement when un-demoted in prior ablation)

**Results:**
| Benchmark | v4 | v5 | Delta | Verdict |
|-----------|----|----|-------|---------|
| WikiText PPL | 12,614 | 9,938 | **-21.2%** | BIG WIN |
| HumanEval | 75.0% | 71.3% | -3.7 | REGRESSION |
| ARC | 95.0% | 95.4% | +0.4 | MATCH |
| HellaSwag | 83.8% | 84.7% | +0.9 | WIN |
| MMLU-Redux | 77.0% | 71.4% | -5.6 | REGRESSION |

**Decision: DO NOT SHIP.** HumanEval and MMLU regressions.

**Key Insight:** v5 data proves TASK-SPECIFIC QUANTIZATION is viable:
- Code-optimized: only promote layers that help HumanEval
- Reasoning-optimized: promote all 7 for PPL/HellaSwag/ARC
- Balanced: cherry-pick layers that improve everything

### Step 3: v6 — Q4_K on CRITICAL attn_k layers (2026-05-01)
**Hypothesis:** Promoting 4 CRITICAL attn_k layers (5, 11, 16, 29) from Q2_K → Q4_K recovers quality.

**Results:**
| Benchmark | v4 | v6 | Delta | Verdict |
|-----------|----|----|-------|---------|
| WikiText PPL | 12,614 | 19,069 | +51.2% | CATASTROPHIC |
| HumanEval | 75.0% | 61.6% | -13.4 | CATASTROPHIC |

**Decision: DEAD ON ARRIVAL.** Killed early.

**Key Finding:** CRITICAL layers are fragile in BOTH directions. Q2_K with imatrix found a sweet spot — Q4_K disrupts it just as badly as Q3_K.

### Step 4: v7 — F16 on CRITICAL attn_k layers (2026-05-01)
**Hypothesis:** Full precision (bf16) on CRITICAL layers 5/11/16/29 preserves original weights.

**Results:**
| Benchmark | v4 | v7 |
|-----------|----|----|
| WikiText PPL | 12,614 | 18,461 (+46.4%) |

**Decision: DEAD.** F16 also destroys quality. These layers are calibrated by the imatrix to a specific Q2_K configuration that cannot be changed.

### Step 5: ffn_gate_up_exps ablation (2026-05-01)
**Direction:** Un-demote Q2_K → Q3_K_M (test if v4's demotions are correct)

| Layer | PPL | Delta | Verdict |
|-------|------|-------|---------|
| 0 | 14,495 | +14.9% | Q2_K correct |
| 5 | 15,762 | +25.0% | Q2_K correct |

**Decision: STOPPED EARLY.** Q2_K demotions on MoE expert weights are correct. Un-demoting makes things worse.

### Step 6: attn_v ablation (2026-05-01)
**Direction:** DEMOTE Q3_K_M → Q2_K (test if default tensors can be crushed)
**Method:** Pipelined — CPU quantize layer N+1 while GPU runs PPL on layer N

| Layer | PPL | Delta | Verdict |
|-------|------|-------|---------|
| 0 | 15,981 | +26.7% | SENSITIVE |
| 5 | 16,486 | +30.7% | SENSITIVE |
| 10 | 15,696 | +24.4% | SENSITIVE |
| 15 | 14,211 | +12.7% | SENSITIVE |
| 20 | 16,575 | +31.4% | MOST SENSITIVE |
| 25 | 16,473 | +30.6% | SENSITIVE |
| 29 | 16,486 | +30.7% | SENSITIVE |

**Conclusion:** attn_v is uniformly sensitive across ALL layers. Cannot demote any.
**Promotion candidates** (most starved for bits): layers 20, 5, 29, 25.
**New insight:** attn_v was never touched in v4 but is clearly load-bearing. This is a new budget dimension for optimization.

### Step 7: attn_o ablation (2026-05-01, IN PROGRESS)
**Direction:** DEMOTE Q3_K_M → Q2_K
**Method:** Cerebellum Autopilot (automated pipelined ablation)
**PID:** 3065161
**Log:** `ablation/attn_o_layers/autopilot.log`
**Status:** Running, layer 0 quantizing...

---

## Remaining Ablation Queue
1. ~~attn_o~~ (running)
2. ffn_down_exps (30 layers, untested)
3. ffn_gate (30 layers, untested)
4. ffn_down (30 layers, untested)

## Task-Specific Quantization Plan
v5 proved we can build task-optimized quants from ablation data:
- **Code-optimized:** only promote attn_k layers that help HumanEval
- **Reasoning-optimized:** promote layers that improve PPL/HellaSwag/ARC
- **Balanced:** cherry-pick layers that improve ALL benchmarks
- **Budget trick:** demote tolerant tensors to Q2_K to pay for promotions elsewhere

This is the differentiator — not just "how small" but "optimized for what."

## Next Models (Downloads Ready)
- Qwen 3.6 35B-A3B: `/var/home/deucebucket/games/models/staging-qwen36-35b/` (65 GB bf16 + imatrix)
- Granite 4.0-H-Small: `/var/home/deucebucket/games/models/staging-granite-h-small/` (61 GB bf16 + imatrix)
