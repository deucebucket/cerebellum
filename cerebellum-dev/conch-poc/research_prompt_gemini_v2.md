# Research Log Entry: Cartridge KV Injection — Manual Per-Head Attention

## 2026-06-09 (continued): Manual Attention for 1-Token Cartridge

### Key Insight

For a single cartridge token, flash_attn is overkill. The attention math
collapses: Q @ K^T with K having 1 token gives [n_tokens, 1] scores.
Softmax over the single token dimension gives [1.0] for all positions.
Output = 1.0 * V_cart, broadcast to all Q tokens.

So cartridge attention IS just V_cart repeated everywhere. No Q @ K^T needed.
No softmax needed. Just expand V from 2 KV heads to 16 Q heads, project
through wo, and add to hidden state.

### Simplified Cartridge Pipeline

1. Load per-layer K/V from real forward pass (36 layers × 256 dim)
2. For each layer >= 18: extract row, reshape to [128, 2, 1]
3. Expand V from 2 heads to 16 via 8x concat on dim 1
4. Reshape to [2048, 1] (flatten heads × head_dim)
5. wo @ V → [2048, 1]
6. Add to hidden state at scale 0.5

No flash_attn needed. No ggml_repeat. Just concat + matmul + add.
