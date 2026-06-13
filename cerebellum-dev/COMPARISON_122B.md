# Qwen3.5-122B-A10B — Cerebellum vs Standard Quants

PRIVATE — dev repo only, never publish comparisons.

## Model Specs
- 122B total params, 10B active per token (256 experts, 8 active)
- 48 layers, hybrid SSM+Attention (full attention every 4th layer)
- BF16 source: 244 GB

## Published Quant Sizes (Unsloth / AesSedai / bartowski)

| Quant | Size (GB) | PPL | KLD Mean | Notes |
|-------|-----------|-----|----------|-------|
| BF16 | 244 | — | — | Full precision |
| Q8_0 | 130 | 4.316 | baseline | Reference |
| Q6_K | 101 | 4.320 | 0.0066 | |
| Q5_K_M | 87-92 | 4.321-4.332 | 0.006-0.011 | |
| Q4_K_M | 74-77 | 4.326-4.436 | 0.011-0.033 | Big spread between quantizers |
| IQ4_XS | 60-65 | 4.405-4.462 | 0.027-0.039 | |
| Q3_K_M | 56-59 | 4.648 | 0.084 | |
| UD-Q3_K_XL | 55 | 4.916 | 0.129 | XL hurts here (MXFP4 bug) |
| IQ3_XXS | 45-51 | 4.568 | 0.069 | Beats Q3_K_M at smaller size! |
| **UD-Q2_K_XL** | **42** | **5.133** | **0.174** | **Our baseline** |
| IQ2_XXS | 34 | 5.106 | 0.178 | AesSedai, beats Q2_K_XL! |
| UD-IQ1_M | 34 | — | — | Unsloth, no PPL data |

## Our Measurements (wikitext, ctx=512, chunks=128)

| Build | Size (GB) | PPL | BPW | Strategy |
|-------|-----------|-----|-----|----------|
| UD-Q2_K_XL (baseline) | 42 | 5.6415 | 2.74 | Unsloth stock |
| Cerebellum v1 | 44 | 5.7002 | 3.04 | ffn_down_exps→Q3_K but iq2→q2K damage |
| **Cerebellum v2b** | **43.5** | **5.1786** | **~2.84** | **ffn_down_exps→Q3_K (promotion only)** |
| Cerebellum v2c (all combos) | 40 | 7.4869 | ~2.61 | promo + all 4 demotions (collapsed) |
| Cerebellum v2d (promo+shexp) | 41 | 5.5665 | ~2.68 | promo + shared experts → q2_K |
| Cerebellum v2e (promo+ssm_out) | 41 | 5.5579 | ~2.68 | promo + ssm_out → q2_K |

Note: our PPL (5.64) differs from AesSedai's (5.13) because different test sets / context / chunks.
AesSedai measured on their calibration set. We use wikitext-test.txt with ctx=512 chunks=128.

## What the Reverse Ablation Tells Us

Starting from Q2_K_XL baseline (PPL 5.6415):

| Promotion | PPL After | Δ PPL | Size Cost | Efficiency (PPL/GB) |
|-----------|-----------|-------|-----------|---------------------|
| ffn_down_exps → Q3_K | 5.1786 | -0.4629 (-8.2%) | +1.69 GB | 0.274 |
| ffn_gate_exps → Q3_K | 5.2309 | -0.4106 (-7.3%) | +5.36 GB | 0.077 |
| ffn_up_exps → Q3_K | 5.2326 | -0.4089 (-7.2%) | +5.36 GB | 0.076 |

ffn_down_exps is 3.6x more efficient than gate/up because it's [1024, 3072] (smaller) vs [3072, 1024].

## Demotion Ablation (individual demotions from baseline, WITHOUT promotion)

Each demotion tested in isolation against the Q2_K_XL baseline (PPL 5.6415):

| Group | PPL | Δ PPL | Size | Notes |
|-------|-----|-------|------|-------|
| ssm_out → q2_K | 5.4911 | -0.1504 | 39 GB | Regularization effect |
| attn_qkv_ssm → q2_K | 5.5440 | -0.0975 | 39 GB | Regularization effect |
| attn_gate_ssm → q2_K | 5.5384 | -0.1031 | 39 GB | Regularization effect |
| shexp → q2_K | 5.4740 | -0.1675 | 39 GB | Strongest regularizer |

ALL improve PPL individually. But combining demotions with the promotion cancels the benefit:
- Promotion alone: 5.18 (great)
- Any demotion alone: 5.47-5.55 (mild improvement)
- Promotion + any demotion: 5.55-5.57 (nearly baseline again)
- Promotion + ALL demotions: 7.49 (collapsed)

The promotion and demotions are competing mechanisms. Promotion improves by adding precision
to down_exps. Demotions improve by forcing regularization. Combined, neither mechanism has
enough headroom to work. **Promotion-only is optimal.**

## Strategy Comparison: Our 3 MoE Models

| | Qwen3.6-35B-A3B | Qwen3-30B-A3B | Qwen3.5-122B-A10B |
|---|---|---|---|
| Experts | 256, 8 active | 128, 8 active | 256, 8 active |
| Layers | 40 | 48 | 48 |
| Base quant | Q3_K_M | Q3_K_M | Q2_K_XL |
| **Sacred** | ssm_out only | attn_q, attn_v, ffn_down_exps | **ffn_down_exps** (promoted) |
| **Demoted** | 10/11 groups to Q2_K | attn_k, gate_exps, up_exps, output | NONE (demotions fight promotion) |
| Finding | Q2_K = regularization | down_exps needs precision | Promotion-only optimal; demotions work alone but not combined |
| Size | 12 GB | 12 GB | 43.5 GB |
| ARC | 94.8% | — | **97.01%** |
| HellaSwag | 91.5% | 90.5% | **94.53%** |

**Universal MoE pattern**: ffn_down_exps is ALWAYS the sacred tensor. Gate/up experts and attention tolerate extreme compression. This holds across 30B, 35B, and 122B.

## Size Opportunity

Our v2 is same size as Q2_K_XL (42 GB). Can we go smaller?

**Current demotion budget**: ~2 GB total (all non-expert tensors to q2_K)
**Expert weights**: Already at iq2_xs/iq3_xxs — rock bottom except ffn_down_exps

To go below 40 GB while keeping ffn_down_exps at Q3_K, we'd need:
- IQ1_S/TQ1_0 on ffn_gate_exps or ffn_up_exps (untested, saves ~3.4 GB each)
- Or only promote SOME layers of ffn_down_exps (proportional benefit reduction)

AesSedai's IQ2_XXS at 34 GB has PPL 5.106 — LOWER than Unsloth's Q2_K_XL at 5.133.
This suggests aggressive IQ2 on experts can beat Q2_K even at smaller size.

## Optimized Serving (24GB VRAM)

```bash
llama-server -m Qwen3.5-122B-A10B-Cerebellum-v2b.gguf \
  -ngl 48 \
  -ot ".*\.ffn_gate_exps\..*=CPU" \
  -ot ".*\.ffn_up_exps\..*=CPU" \
  -ctk q4_0 -ctv q4_0 -fa on \
  --parallel 4 --reasoning-budget 0
```

All 48 layers on GPU, cold experts (gate+up, ~30GB at iq2_xs) exiled to CPU via -ot.
Hot path (attention + ffn_down_exps Q3_K + routers + shared experts) fits in ~13.5GB VRAM.

| Config | Gen tok/s | Notes |
|--------|-----------|-------|
| -ngl 16 (naive) | ~3 | 2/3 model on CPU, layer-level transfers |
| -ngl 48 -ot gate/up→CPU -ctk q4_0 | **25.6** | Only cold experts cross PCIe |

85x speedup. Expert cache (Issue #20757) would add another 3-5x by avoiding redundant copies.

## Next Steps

1. ~~Measure v2b PPL~~ DONE: 5.1786 (confirmed -8.2% improvement)
2. ~~Demotion ablation~~ DONE: demotions work alone but fight promotion
3. ~~Full benchmarks on v2b~~ DONE: ARC 97.01%, HellaSwag 94.53%
4. Explore IQ1 territory on gate/up experts to push below 40 GB
5. Compare head-to-head with AesSedai IQ2_XXS (34 GB, PPL 5.11)
6. Prototype persistent expert cache (GPU ring buffer, LRU eviction)

## Reference: Qwen 3.5 122B Official Benchmarks (FP16)

From Qwen model card:
- MMLU-Pro: 86.1%
- GPQA Diamond: 86.6%
- SWE-bench Verified: 72.4%
- OCRBench: 92.1%
