# Mamba Hybrid Quantization Findings — Qwen 3.5 9B

## Key Discovery: SSM Tensor Type Overrides Break Inference

llama.cpp's `--tensor-type` and `--tensor-type-file` overrides cause catastrophic PPL degradation
when applied to SSM (Mamba) tensors, even when *promoting* to higher precision.

### Proof

Single-tensor override test on Qwen 3.5 9B (Mamba hybrid, 32 layers):

| Build | Override | PPL | Size |
|-------|----------|-----|------|
| Q4_K_M vanilla | none | 7.769 | 5.62GB |
| Q4_K_M + imatrix | none | 7.724 | 5.36GB |
| Q4_K_M + imatrix | `blk.7.ffn_up=q6_K` (non-SSM) | 7.775 | 5.37GB |
| Q4_K_M + imatrix | `blk.0.ssm_out=q8_0` (SSM) | **8.402** | 5.37GB |
| Q4_K_M no imatrix | `blk.0.ssm_out=q8_0` (SSM) | **8.290** | 5.37GB |
| Q4_K_M + imatrix | `blk.0.ssm_out=q6_K` (SSM) | **8.239** | 5.36GB |

Promoting `blk.0.ssm_out` from Q4_K to Q8_0 should only *improve* quality — it's strictly
higher precision. Instead, PPL increases by +0.5 to +0.7. The effect is independent of imatrix.
Non-SSM tensor overrides (ffn_up) show no such degradation.

### Hypothesis

The Mamba inference kernels in llama.cpp may have specific dequantization paths or memory layout
expectations for SSM tensors (ssm_out, ssm_alpha, ssm_beta, ssm_conv1d, ssm_dt, ssm_norm).
Overriding the quant type disrupts these assumptions even when the override is to a higher-precision
format.

### Implication for Cerebellum

Cerebellum's allocator was designed for pure transformers (Qwen 3.6 27B) where every tensor can be
freely re-typed. For Mamba hybrid models:

1. **Never override SSM tensor types** — leave them at the base quant's default assignment
2. **Never override attn_gate types** — these may also be SSM-pathway-related
3. **The Q2_K base strategy doesn't work** — SSM tensors at Q2_K cause severe degradation
4. **Promote-only (non-SSM) gives marginal benefit** — 22 promotions: 7.724 → 7.818 (slightly worse)
5. **imatrix alone is the best strategy** — 7.769 → 7.724 with zero tensor overrides

### `--tensor-type` Substring Matching Gotcha

The `--tensor-type` flag uses **substring matching**, not exact matching:

```
--tensor-type "output=q8_0"
```

This matches `output.weight` AND every `blk.*.attn_output.weight` — promoting all attention output
projections, not just the output head. The `--tensor-type-file` format has the same behavior.
Always use fully-qualified tensor names (e.g., `blk.7.attn_output.weight=q6_K`).

## Ablation Scan Results

Full per-tensor ablation data: `osmosis-qwen35-9b/ablation_results.json`
Scan script: `osmosis-qwen35-9b/cerebellum_scan_9b.sh`

### Architecture

Qwen 3.5 9B: 32 layers, hybrid Mamba + self-attention
- Self-attention layers: 3, 7, 11, 15, 19, 23, 27, 31 (with attn_q/k/v/output)
- Mamba layers: all others (with attn_gate, attn_qkv, ssm_alpha, ssm_beta, ssm_out)
- All layers: MLP (ffn_gate, ffn_up, ffn_down)

### 202 Tensors Tested

| Category | Count | Description |
|----------|-------|-------------|
| Demote | 30 | Negative ablation delta — tensor improves at Q2_K |
| Neutral | 93 | Delta within ±0.02 noise threshold |
| Sacred | 79 | Positive delta — needs protection |

### Top 5 Most Sensitive (Sacred)

| Tensor | Delta | Component |
|--------|-------|-----------|
| blk.0.ssm_out | **+6.342** | SSM output, layer 0 — catastrophically sensitive |
| output | +0.720 | Output head |
| blk.8.attn_qkv | +0.288 | Self-attention QKV, layer 8 |
| blk.11.attn_v | +0.225 | Self-attention V, layer 11 |
| blk.6.ffn_up | +0.161 | MLP up projection, layer 6 |

### Top 5 Most Crushable (Demote)

| Tensor | Delta | Component |
|--------|-------|-----------|
| blk.3.attn_v | **-0.425** | Self-attention V, layer 3 — better at Q2_K |
| blk.3.attn_q | -0.268 | Self-attention Q, layer 3 |
| blk.2.attn_qkv | -0.265 | Mamba QKV, layer 2 |
| blk.10.attn_gate | -0.217 | Mamba gate, layer 10 |
| blk.3.attn_output | -0.161 | Self-attention output, layer 3 |

### Confirmed External Findings

- **Unsloth's SSM output warning**: blk.0.ssm_out at +6.342 confirms SSM outputs should not be
  aggressively quantized. Ground-truth PPL measurement vs Unsloth's KL-divergence proxy.
- **Early self-attention layers are crushable**: layers 2-3 have consistently negative deltas,
  matching general transformer quantization folklore.

## Best Available Quant for Qwen 3.5 9B

For this model, standard Q4_K_M with our Osmosis imatrix is the best approach:

```bash
# Generate Osmosis imatrix
python -m osmosis.imatrix_stream \
    --model Qwen/Qwen3.5-9B \
    --output osmosis_imatrix.dat -v

# Quantize
llama-quantize --imatrix osmosis_imatrix.dat \
    model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```

PPL: 7.724 at 5.36GB — the imatrix does the heavy lifting, and the Q4_K_M recipe already
handles Mamba tensor types correctly.
