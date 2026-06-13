# Devlog: Granite 4.0-H-Small Cerebellum — Start

## Date: 2026-05-02

## Model

Granite 4.0-H-Small — hybrid Mamba-2 + Transformer MoE with shared experts.

| Parameter | Value |
|-----------|-------|
| Total params | 32B |
| Active params | 9B per token |
| Layers | 40 |
| Experts | 72 per layer |
| Active experts | 10 per token |
| Hidden dim | 4096 |
| Attention heads | 32 (8 KV heads) |
| Expert FFN size | 768 (intermediate_size) |
| Shared FFN size | 1536 (shared_intermediate_size) |
| Mamba heads | 128 |
| Mamba state dim | 128 |
| Mamba d_head | 64 |
| Mamba d_conv | 4 |
| Full attention every | 10 layers (layers 5, 15, 25, 35) |
| Context | 128K |
| License | Apache 2.0 |

## Architecture Notes

Very similar to Qwen 3.6 35B — hybrid SSM+Attention MoE. Key differences:

1. **72 experts** (vs Qwen's 256). 10 active per token (vs Qwen's 8). Higher ratio of active experts = each routing decision matters more.

2. **Mamba-2** (not linear attention). Uses A_log, D, dt_bias, conv1d, in_proj, out_proj, norm. 36/40 layers are Mamba.

3. **4 full attention layers** at positions 5, 15, 25, 35 (every 10 layers). Qwen had 10 attention layers every 4.

4. **Shared MLP** at every layer (always active, size 1536). Like Qwen's shared experts.

5. **Expert FFN size only 768** — smaller than Qwen's 512×2. But 72 experts × 768 = significant capacity.

6. **9B active** per token (vs Qwen's 3B). More compute per token = potentially more sensitive to quantization.

## Tensor Groups (586 total tensors)

| Group | Count | Notes |
|-------|-------|-------|
| block_sparse_moe.input_linear | 40 | MoE expert input (3D: [72, hidden, expert_ffn]) |
| block_sparse_moe.output_linear | 40 | MoE expert output |
| block_sparse_moe.router.layer | 40 | MoE router — surgery target |
| shared_mlp.input_linear | 40 | Shared expert input |
| shared_mlp.output_linear | 40 | Shared expert output |
| mamba.in_proj | 36 | Mamba input projection |
| mamba.out_proj | 36 | Mamba output projection |
| mamba.A_log | 36 | Mamba state matrix (F32 always) |
| mamba.D | 36 | Mamba skip connection (F32 always) |
| mamba.conv1d.weight | 36 | Mamba convolution (F32 always) |
| mamba.conv1d.bias | 36 | Mamba conv bias (F32 always) |
| mamba.dt_bias | 36 | Mamba dt bias (F32 always) |
| mamba.norm | 36 | Mamba group norm (F32 always) |
| input_layernorm | 40 | RMSNorm (F32 always) |
| post_attention_layernorm | 40 | Post-attn RMSNorm (F32 always) |
| self_attn.q_proj | 4 | Full attention Q |
| self_attn.k_proj | 4 | Full attention K |
| self_attn.v_proj | 4 | Full attention V |
| self_attn.o_proj | 4 | Full attention O |
| embed_tokens | 1 | Token embeddings |
| norm | 1 | Final RMSNorm |

## Ablation Plan

Based on Qwen 3.6 35B patterns (same architecture family):

| Priority | Group | Count | Hypothesis |
|----------|-------|-------|------------|
| 1 | block_sparse_moe.input_linear | 40 | Expert gates — likely demotable (Qwen: +1.4%) |
| 2 | block_sparse_moe.output_linear | 40 | Expert down — likely demotable (Qwen: +1.9%) |
| 3 | shared_mlp.input_linear | 40 | Shared input — likely Q2_K friendly (Qwen: +0.6%) |
| 4 | shared_mlp.output_linear | 40 | Shared output — likely Q2_K friendly (Qwen: +0.7%) |
| 5 | mamba.in_proj | 36 | Mamba input — corresponds to Qwen's attn_qkv (+1.4%) |
| 6 | mamba.out_proj | 36 | Mamba output — corresponds to Qwen's ssm_out (+3.3%) — LIKELY SENSITIVE |
| 7 | self_attn.q_proj | 4 | Full attention Q |
| 8 | self_attn.k_proj | 4 | Full attention K |
| 9 | self_attn.v_proj | 4 | Full attention V |
| 10 | self_attn.o_proj | 4 | Full attention O |

**Prediction from Qwen universal patterns:**
- MoE expert weights (input/output) and shared MLP: demotable to Q2_K
- mamba.out_proj: likely the only sensitive group (like Qwen's ssm_out)
- Router surgery: may show signal with 72 experts (fewer than Qwen's 256, more like Gemma's 128)

## Files

- Base quant: bartowski Q3_K_M with imatrix (downloading)
- Imatrix: bartowski (downloading)
- Work dir: `/var/home/deucebucket/ai-drive/osmosis/osmosis-granite4-h-small/`
