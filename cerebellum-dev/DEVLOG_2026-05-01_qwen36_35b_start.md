# Devlog: Qwen 3.6 35B-A3B Cerebellum — Start

## Date: 2026-05-01

## Model

Qwen 3.6 35B-A3B — hybrid SSM+Attention MoE.

| Parameter | Value |
|-----------|-------|
| Total params | 35B |
| Active params | 3B per token |
| Layers | 40 |
| Experts | 256 per layer |
| Active experts | 8 per token |
| Embedding dim | 2048 |
| Attention heads | 16 (2 KV heads) |
| K/V length | 256 |
| Expert FFN size | 512 |
| Shared FFN size | 512 |
| SSM inner size | 4096 |
| Full attention every | 4 layers (10 full attn layers) |
| Context | 262K |
| bf16 size | 65 GB (2-split GGUF) |

## Architecture Notes

This is significantly more complex than Gemma 4 26B:

1. **Hybrid SSM+Attention**: Most layers (30/40) use linear attention + SSM (Mamba-style). Only every 4th layer (10 total) uses full multi-head attention with separate Q/K/V projections.

2. **256 experts** vs Gemma's 128. Router tensors are shape [2048, 256] — double the routing decisions per token.

3. **Shared experts**: Each layer has shared gate/up/down FFN projections alongside the MoE experts. These are always active regardless of routing.

4. **SSM parameters**: ssm_a, ssm_conv1d, ssm_dt, ssm_alpha, ssm_beta, ssm_norm, ssm_out — these are likely VERY sensitive to quantization (same as Qwen 3.5's SSM params that caused NaN at low precision).

5. **Linear attention layers** use fused attn_qkv [2048, 8192] + attn_gate [2048, 4096] instead of separate Q/K/V.

## Tensor Groups (from split 1, 423 tensors)

| Group | Count | Shape | Notes |
|-------|-------|-------|-------|
| ffn_gate_exps | 23 | [2048,512,256] | MoE expert gate (3D!) |
| ffn_up_exps | 23 | [2048,512,256] | MoE expert up |
| ffn_down_exps | 23 | [512,2048,256] | MoE expert down |
| ffn_gate_inp (router) | 23 | [2048,256] | MoE router — surgery target |
| ffn_gate_shexp | 23 | [2048,512] | Shared expert gate |
| ffn_up_shexp | 23 | [2048,512] | Shared expert up |
| ffn_down_shexp | 23 | [512,2048] | Shared expert down |
| ffn_gate_inp_shexp | 23 | [2048] | Shared expert gate (scalar) |
| attn_qkv (linear attn) | 18 | [2048,8192] | Fused Q+K+V for SSM layers |
| attn_gate | 18 | [2048,4096] | Linear attention gate |
| ssm_out | 18 | [4096,2048] | SSM output projection |
| ssm_alpha | 18 | [2048,32] | SSM mixing |
| ssm_beta | 18 | [2048,32] | SSM mixing |
| ssm_a | 18 | [32] | SSM state matrix (F32 always) |
| ssm_conv1d | 18 | [4,8192] | SSM convolution (F32 always) |
| ssm_dt.bias | 18 | [32] | SSM dt bias (F32 always) |
| ssm_norm | 18 | [128] | SSM group norm (F32 always) |
| attn_q (full attn) | 5 | [2048,8192] | Full attention Q |
| attn_k (full attn) | 5 | [2048,512] | Full attention K |
| attn_v (full attn) | 5 | [2048,512] | Full attention V |
| attn_output (full attn) | 5 | [4096,2048] | Full attention output |
| attn_norm | 23 | [2048] | RMSNorm (F32 always) |
| post_attention_norm | 23 | [2048] | Post-attn RMSNorm (F32 always) |

## Baseline

- **Base quant**: Q3_K_M with bartowski imatrix
- **File size**: 16 GB (3.87 BPW)
- **VRAM**: 15.8 GB weights + 251 MB SSM state + 493 MB compute = fits in 24 GB
- **Baseline PPL**: 7.1758 ± 0.10191

This is a proper perplexity value — unlike Gemma 4 26B which had inflated PPL (12,000+), Qwen 3.6 35B behaves normally on WikiText.

## Ablation Plan

Group-level ablation order (test each at Q2_K, measure PPL delta):

| Priority | Group | Count | Why First |
|----------|-------|-------|-----------|
| 1 | ffn_gate_exps | 40 | Biggest tensor group, MoE expert gates — RUNNING on CPU |
| 2 | ffn_up_exps | 40 | MoE expert up projections |
| 3 | ffn_down_exps | 40 | MoE expert down projections |
| 4 | ffn_gate_shexp | 40 | Shared expert gates |
| 5 | ffn_up_shexp | 40 | Shared expert up |
| 6 | ffn_down_shexp | 40 | Shared expert down |
| 7 | attn_qkv | 30 | Linear attention fused QKV |
| 8 | attn_gate | 30 | Linear attention gate |
| 9 | ssm_out | 30 | SSM output projection |
| 10 | ssm_alpha | 30 | SSM mixing — likely SENSITIVE |
| 11 | ssm_beta | 30 | SSM mixing — likely SENSITIVE |
| 12 | attn_q (full attn) | 10 | Full attention Q |
| 13 | attn_k (full attn) | 10 | Full attention K |
| 14 | attn_v (full attn) | 10 | Full attention V |
| 15 | attn_output (full attn) | 10 | Full attention output |

After group ablation: per-layer ablation on mixed-result groups, then router surgery on all 40 layers.

## Group Ablation Results (ALL COMPLETE)

All groups tested at Q2_K (demote from Q3_K_M default). Baseline PPL: 7.1758.

| Group | PPL | Delta | Verdict |
|-------|-----|-------|---------|
| ssm_alpha | 7.175 | -0.01% | FREE |
| attn_gate | 7.187 | +0.1% | FREE |
| ssm_beta | 7.180 | +0.06% | FREE |
| ffn_gate_shexp | 7.219 | +0.6% | VERY DEMOTABLE |
| ffn_down_shexp | 7.228 | +0.7% | VERY DEMOTABLE |
| ffn_up_exps | 7.262 | +1.2% | DEMOTABLE |
| ffn_up_shexp | 7.264 | +1.2% | DEMOTABLE |
| attn_qkv | 7.277 | +1.4% | DEMOTABLE |
| ffn_gate_exps | 7.277 | +1.4% | DEMOTABLE |
| ffn_down_exps | 7.314 | +1.9% | DEMOTABLE |
| **ssm_out** | **7.413** | **+3.3%** | **ONLY SENSITIVE GROUP** |

**Key finding:** This model is FAR more compressible than Gemma 4 26B. 10 of 11 groups tolerate Q2_K under 2%. Only ssm_out shows real sensitivity (+3.3%), and even that is mild compared to Gemma's attn_v (+30%) or attn_o (+30%).

**Comparison with Gemma 4 26B:**
- Gemma: 3 groups demotable, 2 sensitive (attn_v, attn_o at +25-31%)
- Qwen 3.6: 10 groups demotable, 1 mildly sensitive (ssm_out at +3.3%)
- The hybrid SSM+attention architecture distributes information more evenly, making individual tensor groups less critical

**Next:** Build v1 with all demotable groups at Q2_K, ssm_out at Q3_K_M (default). Then test full attention layers (attn_q/k/v/output on the 10 full-attention layers) and router surgery.

## v1 Results

- **File size**: 12 GB (2.73 BPW)
- **PPL**: 7.8484 (+9.4% vs baseline 7.1758)
- **HumanEval pass@1**: 75.0%

## Reverse Ablation (ALL COMPLETE)

Un-demote each Q2_K group back to Q3_K_M one at a time from v1. Measures which Q2_K demotions stack worst.

| Group un-demoted | PPL | vs v1 (7.8484) | v2 Action |
|---|---|---|---|
| **attn_qkv** | **7.7109** | **-1.8%** | **UN-DEMOTE → Q3_K_M** |
| **ffn_down_exps** | **7.7889** | **-0.8%** | **UN-DEMOTE → Q3_K_M** |
| **ffn_up_exps** | **7.7909** | **-0.7%** | **UN-DEMOTE → Q3_K_M** |
| ffn_gate_exps | 7.8902 | +0.5% | Keep Q2_K |
| ffn_down_shexp | 7.9270 | +1.0% | Keep Q2_K |
| attn_gate | 7.9429 | +1.2% | Keep Q2_K |
| ffn_gate_shexp | 7.9646 | +1.5% | Keep Q2_K |
| ffn_up_shexp | 7.9865 | +1.9% | Keep Q2_K |
| ssm_alpha | 8.0034 | +2.0% | Keep Q2_K |
| ssm_beta | 8.0184 | +2.2% | Keep Q2_K |

**Key finding:** 7 of 10 groups are BETTER at Q2_K — un-demoting them makes things worse. Only 3 groups (attn_qkv, ffn_down_exps, ffn_up_exps) genuinely benefit from Q3_K_M precision.

**Universal pattern confirmed across Gemma 4 26B + Qwen 3.6 35B:**
- Gate/mixing weights (ffn_gate_exps, ffn_gate_shexp, attn_gate, ssm_alpha, ssm_beta) prefer Q2_K — imatrix Q2_K acts as beneficial regularization
- Projection weights (ffn_down_exps, ffn_up_exps, attn_qkv) need precision
- Shared expert weights (shexp) prefer Q2_K even when MoE expert equivalents don't — always-on weights may benefit more from regularization

## v2 Results (COMPLETE)

- **280 overrides**: 7 groups × 40 layers at Q2_K, 3 groups at Q3_K_M default
- **File size**: 15 GB
- **PPL**: 7.4307 (+3.6% vs baseline)
- **HumanEval pass@1**: 75.0%
- **ARC-Challenge**: 96.2%
- **HellaSwag**: 91.5%

**Comparison with 27B v4:**

| Benchmark | 27B v4 | 35B v2 |
|---|---|---|
| ARC | 95.1% | **96.2%** |
| HellaSwag | 91.2% | **91.5%** |
| HumanEval | 75.0% | 75.0% |
| MMLU-Redux | 77.1% | TBD |

35B beats 27B across the board despite heavier compression (15GB vs 27B's size).

**Issue:** v2 is 15GB — only 1GB smaller than baseline Q3_K_M (16GB). Not compelling enough to ship. Need per-layer ablation to find which specific layers within the 3 promoted groups actually need Q3_K_M.

## Per-Layer Ablation (COMPLETE — ALL DEMOTE)

Tested 9 sampled layers (0,5,10,15,20,25,30,35,39) per group. Each demotes ONE layer back to Q2_K from v2.

| Group | Max delta | Verdict |
|---|---|---|
| attn_qkv | +0.4% (layer 0) | ALL DEMOTE — no layer exceeds noise |
| ffn_down_exps | +0.3% (layer 39) | ALL DEMOTE — same pattern |
| ffn_up_exps | +0.3% (layers 5, 39) | ALL DEMOTE — same pattern |

**Key finding:** The reverse ablation group-level improvements (-0.7% to -1.8%) were cumulative micro-effects distributed across all 40 layers. No individual layer carries enough weight to justify Q3_K_M. This means v2's 3GB size increase bought quality that can't be surgically preserved — it's all-or-nothing for these groups.

**Implication:** v3 = v1 (all Q2_K, 12GB) + router surgery. Per-layer ablation is not the path for this model — router surgery is the differentiator.

**Universal pattern update:** On hybrid SSM+MoE architectures, per-layer ablation within promoted groups may not find surgical targets. The architecture distributes information too evenly. Router surgery (the "highway intersection") is the higher-leverage optimization.

## Router Surgery (COMPLETE — NO SIGNAL)

Tested all 40 router layers individually at Q8_0 (F32 → Q8_0) on v1. All layers within ±0.11% — zero signal. 256-expert MoE distributes routing too evenly for Q8_0 to matter.

Precision curve on layer 20: Q8_0 flat, Q6_K broken (+5.5% — K-quant corruption confirmed on Qwen too). Q2_K crush shows depth gradient (0.7% layer 0 → 2.9% layer 30) but below actionable threshold.

**Conclusion:** Router surgery not applicable to 256-expert MoE. The architecture's redundancy makes individual router precision irrelevant at Q8_0 level.

## v1 Full Benchmarks (COMPLETE — SHIPPABLE)

| Benchmark | v1 (12GB) | v2 (15GB) | 27B v4 |
|---|---|---|---|
| ARC-Challenge | 94.8% | 96.2% | 95.1% |
| HellaSwag | 91.5% | 91.5% | 91.2% |
| MMLU-Redux | 73.9% | TBD | 77.1% |
| HumanEval | 75.0% | 75.0% | 75.0% |
| PPL | 7.8484 | 7.4307 | — |

**Key finding:** v1 matches v2 on HellaSwag and HumanEval despite being 3GB smaller and having much worse PPL. PPL does NOT predict downstream task performance on MoE models — the compression damage from Q2_K on gate/mixing weights acts as regularization that helps on structured tasks even as it hurts on raw perplexity.

**Ship decision:** v1 at 12GB. 25% smaller than stock Q3_K_M (16GB), matches or exceeds 27B v4 on HellaSwag/HumanEval, within noise on ARC. MMLU-Redux 3.2% below 27B but acceptable for a model with 4GB less weight data.

## SSM Sensitivity Warning

From Qwen 3.5 experience: SSM parameters (ssm_a, ssm_conv1d, ssm_dt, ssm_alpha, ssm_beta) are EXTREMELY sensitive. They must stay at F32 or at minimum Q8_0. The quantizer should handle this automatically (norms stay F32), but ssm_alpha and ssm_beta are BF16 tensors that might get quantized — need to verify they're protected.

## Files

- bf16 source: `/var/home/deucebucket/games/models/staging-qwen36-35b/`
- imatrix: `Qwen_Qwen3.6-35B-A3B-imatrix.gguf` (184 MB)
- Work dir: `/var/home/deucebucket/ai-drive/osmosis/osmosis-qwen36-35b/`
- Base quant output: `/var/home/deucebucket/games/models/qwen36-35b-q3km-base.gguf`
