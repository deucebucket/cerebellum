# Cerebellum Memory Controller — Three-Tier Tensor Architecture

## The Idea

LLM inference should work like the brain: most neurons are dormant, the
cerebellum activates only what's needed. Three-tier memory hierarchy for
tensor micro-shards:

```
VRAM (L1)  ←→  System RAM (L2)  ←→  NVMe SSD (L3)
 0ms access      0.1ms/shard         0.2ms/shard
 sacred pins     working set         model library
```

## Hardware Reality (RTX 3090 + 64GB RAM + NVMe)

| Tier | Capacity | Bandwidth | Shard Latency | Role |
|------|----------|-----------|---------------|------|
| VRAM | 24GB (4GB usable*) | 936 GB/s internal | 0 ms | Sacred shards + scratch buffers |
| RAM | 64-128GB | 14 GB/s (PCIe 3.0) | 0.1 ms/1.4MB | Full model, warm shards |
| NVMe | 1-4TB | 7 GB/s (Gen4) | 0.2 ms/1.4MB | Multiple models/fine-tunes |

*4GB VRAM = user's target. Rest for gaming/ComfyUI.

## Micro-Shard Spec

Each tensor split into N row-blocks (default: 128 per tensor for large MLP weights).

Example: `blk.63.ffn_down.weight` (17408 × 5120 = 170MB fp16)
- 128 shards: 136 rows × 5120 = 1.4 MB each
- 64 shards: 272 rows × 5120 = 2.8 MB each
- Smaller tensors (attn_k, 1024×5120 = 10MB): 8-16 shards

Shard ID format: `blk.{layer}.{tensor}.shard.{idx}` 
e.g. `blk.63.ffn_down.weight.shard.042`

## Three-Tier Paging Protocol

### Startup: Cold → Warm
```
1. Cerebellum reads shard manifest from disk
2. Sacred shards (from ablation heatmap) → DMA directly to VRAM
3. Remaining shards → mmap into RAM (lazy, pages fault in on first access)
4. Model is "loaded" — all shards addressable, most still on disk
```

### Inference: Token Processing
```
For each token:
  1. Router MLP (always in VRAM, ~2MB) sees current hidden state
  2. Router predicts: for layer N+1, which shards activate?
     Output: binary mask per tensor, ~100 bits per tensor
  3. Prefetch engine DMAs predicted shards:
     - If in VRAM already → no-op (sacred or recently used)
     - If in RAM → PCIe DMA to VRAM scratch buffer (0.1ms)
     - If only on disk → NVMe read → RAM → VRAM (0.3ms, rare)
  4. Compute layer N using currently-loaded shards
  5. Eviction: LRU shards in VRAM scratch buffer get overwritten
```

### Eviction Policy
```
Priority (highest = last evicted):
  PIN    — sacred shards, never evict (ablation delta > +0.05)
  HOT    — activated in last 10 tokens
  WARM   — activated in last 100 tokens  
  COLD   — not activated recently → evict first
  
VRAM budget:
  sacred_pool:  fixed allocation for pinned shards
  scratch_buf:  2 double-buffers for DMA prefetch
  compute_buf:  KV cache + activation workspace
```

## Router Architecture

Tiny MLP that runs per-token, always resident in VRAM:

```
Input:  hidden_state at layer boundary (5120 dims)
        + positional encoding (layer index)
        + token type hints (optional)
        
Hidden: 2 layers, 256 dims each (ReLU)

Output: per-tensor shard activation mask
        Shape: [n_tensors_in_layer × n_shards_per_tensor]
        Sigmoid → threshold at 0.5
        
Size:   ~2MB total (negligible VRAM)
```

Training data: recorded activation magnitudes from calibration runs.
For each token in calibration set, record which row-blocks contributed
>1% of the output magnitude. That's the ground truth mask.

## Multi-Model Fusion (Disk Tier)

The disk tier enables something no existing system does: **per-shard model selection**.

```
Disk layout:
  /models/base-70b/          # 89,600 shards, 140GB
  /models/code-70b/          # 89,600 shards, 140GB  
  /models/math-70b/          # 89,600 shards, 140GB
  /models/creative-70b/      # 89,600 shards, 140GB

Shard manifest includes provenance:
  blk.15.ffn_down.shard.042:
    versions:
      - source: base-70b, path: /models/base-70b/blk.15.ffn_down.042.bin
      - source: code-70b, path: /models/code-70b/blk.15.ffn_down.042.bin
    active: code-70b  # cerebellum's current selection
```

The cerebellum's router doesn't just predict WHICH shards activate — it
predicts which VERSION of each shard to use. Code tokens → code-tuned
shards. Math tokens → math-tuned shards. Creative writing → creative shards.

This is MoE across entire models. The "experts" are fine-tune variants
stored on disk. Switching cost: 0.2ms per shard (NVMe read).

## Implementation Phases

### Phase 1: Static Sharding (works with current llama.cpp)
- Split GGUF into shard files post-quantization
- `--tensor-backend-file` assigns shards to VRAM/CPU based on ablation
- No router, no dynamic paging — just smart static placement
- Proves the memory savings

### Phase 2: Dynamic Paging (requires llama.cpp patches)
- Double-buffer DMA prefetch in ggml_backend_sched
- Shard-level eviction in VRAM scratch pool
- Layer-ahead prefetch triggered by compute pipeline
- Proves the latency model

### Phase 3: Learned Router
- Train router MLP on calibration activation data  
- Per-token shard activation prediction
- Integrate into llama.cpp forward pass
- Proves the accuracy (no quality loss from selective activation)

### Phase 4: Multi-Model Fusion
- Disk-tier shard manifest with model provenance
- Router predicts shard version selection
- Hot-swap between fine-tune variants per-token
- The piano plays notes from different instruments

## Key Metrics to Prove

1. **Quality**: PPL with 20% shard activation vs 100% — must be <1% degradation
2. **Latency**: Per-token overhead from paging — target <5ms on PCIe 3.0
3. **VRAM savings**: 70B model in 4GB VRAM — currently impossible
4. **Throughput**: Tokens/sec with paging vs full VRAM load

## Comparison to Existing Approaches

| Approach | Granularity | Dynamic? | Multi-model? |
|----------|-------------|----------|--------------|
| -ngl (llama.cpp) | Per-layer | No | No |
| MoE (Mixtral) | Per-expert | Yes (trained) | No |
| Offloading (HF) | Per-layer | No | No |
| **Cerebellum** | **Per-shard (1.4MB)** | **Yes (learned)** | **Yes** |

The closest existing work is MoE routing, but MoE is baked into the model
architecture. Cerebellum retrofits MoE-like efficiency onto ANY dense model,
at inference time, with no retraining of the base model.
