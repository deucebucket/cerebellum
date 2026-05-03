# Gemma 4 E4B — PLE Quantization Sensitivity Results

Date: 2026-04-30
Hardware: RTX 3090 (24 GB), AMD Ryzen 7 5800XT, 64 GB DDR4
Dataset: WikiText-2 raw test, 2048 ctx, all 142 chunks
Tool: llama-perplexity (llama.cpp mainline build)

## Results

| Quant | Size | PPL | ± | vs BF16 |
|-------|------|-----|---|---------|
| BF16 | 15 GB | 54.58 | — | baseline |
| Q4_K_M | 5.0 GB | 55.74 | 0.67 | +2.1% |
| Q4_K_M+PLE Q8_0 | 5.7 GB | 58.33 | 0.72 | +6.9% |
| Q3_K_M+PLE Q8_0 | 5.2 GB | 55.82 | 0.66 | +2.3% |
| Q3_K_M | 4.6 GB | 104.74 | 1.34 | +91.9% |
| Q2_K+PLE Q8_0 | 4.8 GB | 62.72 | 0.71 | +14.9% |
| Q2_K | 4.1 GB | 7296.76 | 122.31 | +13,268% |

## Key Findings

1. PLE (Per-Layer Embedding) tensors are the single point of failure.
   174 tensors at Q8_0 adds ~0.6 GB overhead.

2. The precision cliff is between Q4_K and Q3_K.
   Q4_K_M handles PLE fine. Q3_K_M destroys them.

3. Q3_K_M+PLE Q8_0 (5.2 GB) matches Q4_K_M (5.0 GB) quality.
   Best value point on the curve.

4. PLE protection at Q4_K_M slightly hurts (58.33 vs 55.74).
   Default K-quant profile is already adequate at Q4_K.

5. Q2_K+PLE Q8_0 (4.8 GB) rescues the model from PPL 7297 to 62.7.
   Non-PLE tensor damage at Q2_K still causes 15% degradation.

6. BF16 baseline PPL is 54.58 — inherently high due to 262K multimodal vocab.
   Community reports of "broken Q4" were comparing against text-only Gemma 3.

## PLE Override File

174 tensor overrides in ple_overrides.txt:
- per_layer_token_embd.weight (global, shape 262144x256x42-ish)
- per_layer_model_proj.weight (global, shape 10752x2560)
- blk.{0-41}.inp_gate.weight (per-layer gate)
- blk.{0-41}.proj.weight (per-layer projection)
- plus post_norm and layer_output_scale tensors

## Imatrix

Weight-sensitivity imatrix (L2_norm x max_abs x variance per column).
380 tensors, 4698.2 KB, generated in 45.2 seconds on CPU.
Used for all quantizations via --imatrix flag.
