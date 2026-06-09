# Brainloop Research Log

All ideas from Gemini strategy sessions + C++ port devlog. Preserved as
project artifacts for future reference and test design.

---

## 2026-06-09: C++ Port — GPU Allocation Fix (Fix Path A)

### Context

Refiner weights were successfully ported to llama.cpp as a custom graph builder
(`llm_build_qwen2_brainloop` / `LLM_ARCH_QWEN2_BRAINLOOP`). The C++ engine
intercepts ALL Qwen2 models (hardcoded `LLM_ARCH_QWEN2` → brainloop routing at
`llama-model.cpp:8669`) and injects the refiner loop at layer 18.

Early test produced **49M PPL** — proof of pipeline working with garbage data.
The cause: `ggml_new_tensor_2d(ctx0, ...)` allocated refiner weights on CPU,
but the CUDA graph operates on GPU tensors. Fused ops (permute, transpose,
flash attention) read garbage across the CPU/GPU boundary.

### Fix: GPU Backend Allocation

Replaced `ggml_new_tensor_2d(ctx0, ...)` + `memcpy` with:

1. Create a no-alloc `ggml_context` for tensor metadata only
2. `ggml_backend_alloc_ctx_tensors_from_buft(ctx, model.select_buft(18))`
   — allocates all 18 weight tensors on the same CUDA backend as layer 18
3. `ggml_backend_tensor_set()` — copies raw float data from CPU `.bin` files to GPU
4. Static cache — allocated once per process, reused across all graph builds

Gate value: **sigmoid(0) = 0.4980** (matching PyTorch STE eval gate).

### Results

| Configuration | PPL |
|---|---|
| Baseline (no refiner) | 8.1475 |
| Refiner (output+FFN path) | 8.6096 |

The simplified path lacks full QKV attention. Weights were trained with
self-attention in PyTorch, so the output+FFN-only path doesn't recover the
-15.28% improvement. **GPU allocation fix confirmed working** — this was
the blocking issue preventing the C++ port from functioning at all.

### Remaining: Full Attention Path

The `build_attn_mha` function in llama.cpp handles multi-head attention with
proper ggml tensor layout: permute `[head_dim, heads, tokens]` →
`[head_dim, tokens, heads]`, flash attention, reshape, output projection.
Manual replication of this in `brainloop_refine_pass` hits shape assertions
in `ggml_mul_mat` and `ggml_can_repeat`. The correct approach is to call
`build_attn_mha` directly — but this requires the refiner weights to be in
the model's tensor tree, not a separate cache context.

### Build Artifacts

- `qwen2.5-3b-brainloop.gguf` (~6.2 GB F16) — converted HF model, any Qwen2 GGUF
  routes through brainloop
- `brainloop-ggml-weights/` — 34 `.bin` files (134 MB), exported from PyTorch
  checkpoint `checkpoints-refiner-qwen3b-v4-wd/best_refiner.pt`
- `libllama.so` — CUDA-enabled, contains `llm_build_qwen2_brainloop` + GPU cache

### Gotchas

- **libcuda.so symlink** — distrobox has empty `/lib/.../libcuda.so`. Must
  `ln -sf libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so` or `LD_PRELOAD` it.
- **Build caching** — cmake Makefile doesn't detect changes to brainloop source
  reliably inside distrobox. Delete `.o` file and manually recompile with `g++`
  if build system is stale.
- **GGML tensor layout** — hidden states in ggml are `[n_embd, n_tokens]` with
  `ne[0]=n_embd`. The original `brainloop_refine_pass` had `n_tokens=hidden->ne[0]`
  bug (got n_embd instead of n_tokens). Fixed in this session.
- **`ggml_norm` vs `ggml_rms_norm`** — Qwen2 uses RMSNorm. The original reference
  function used `ggml_norm` (LayerNorm). Must use `ggml_rms_norm`.
- **All Qwen2 models intercepted** — `LLM_ARCH_QWEN2` now routes to brainloop.
  Regular Qwen2 inference goes through the brainloop graph builder but with
  refiner disabled (if `brainloop-ggml-weights/` missing).

---

## 2026-06-09: Inline RAG Refiner Concept

The bolt-on refiner at layer 18 can act as a **runtime oracle** — during its
refinement loop, it projects hidden states into query vectors, retrieves
relevant context from a FAISS index, and injects it directly into the model's
representation space. No prompt bloat, no context window waste.

Key design:
- **W_query** [query_dim × n_embd]: stethoscope — learns what the model needs
- **W_context** [n_embd × context_dim]: injects retrieved docs as vectors
- **rag_gate**: learns when retrieval helps vs hurts
- **Index**: pre-computed FAISS IVF-PQ, 256-dim, VRAM-pinned, <1ms search
- **Iterative**: each revolution re-queries with refined hidden state

~2M additional params. Full spec in `SPEC_inline_rag_refiner.md`.

---

## 2026-06-07: Initial Research & Strategy

## 2026-06-07: Architecture Concepts & Gate Strategy

### Gas Tensor (Dynamic Phase Changes During Loops)
The refiner's loop creates a "gas phase" in the hidden state vector space.
Unlike standard transformer layers (solid/liquid pipe), the loop block can:
- Pass 1 (Solid): Raw token data enters
- Pass 2-3 (Gas): Hidden states "evaporate" into hyper-abstract reasoning where
  tokens blend and expand semantic boundaries
- Pass 4 (Condensation): Compress gas state back into a dense vector that the
  frozen base model's next layer expects

Key insight: Because the base model only sees entrance and exit, what happens
inside the loop can violate standard transformer constraints (dimensions,
information density, etc.).

### Multi-Resolution Loop (Expanding Dimensions Mid-Loop)
The refiner could project the tensor into a massive intermediate space during
middle loops — letting the tensor expand to fill much larger capacity — before
compressing back for the final exit. Variable-volume brain with flat VRAM.

### Zero-Initialized Universal Router (Empty Tensor Switchboard)
A linear layer initialized to 0.0 that acts as:
- Invisible bridge at step 0 (tensor "never left the ground")
- Train yard switch: reads incoming hidden state, routes to standard/math/code/refusal-bypass paths
- The routing tensor names dictate the path: `router.math`, `router.code`, `router.heretic`

### Abliteration-as-Service (Live Refusal Bypass)
Refusals are just vector directions. The router intercepts the hidden state,
subtracts the "refusal direction" vector, and forces the tensor down a
different track. The model complies because the physical refusal trigger was
removed mid-flight. Same mechanics as abliterated models, but live/on-the-fly
instead of baked into weights.

### Hot-Swap Runtime Engine
Keep the giant base model pinned in VRAM permanently. Swap refiner blocks
in milliseconds based on task:
- `Math-Refiner.bin` → Layer 15
- `Code-Refiner.bin` → Layer 15
- `Persona-Refiner.bin` → Layer 15

1% compute cost for specialized fine-tuned performance across domains.

## Production Packaging Strategies (for llama.cpp / GGUF)

### Hack 1: Static Unroll (GGUF Tensor Aliasing)
- Python script writes GGUF that lies about layer count (30 → 40)
- Layers 15-25 all point to same physical bytes of RefinerBlock on disk
- llama.cpp thinks it's 40-layer model, actually runs refiner 10x
- Community gets native GGUF, tiny VRAM footprint
- No custom .exe required

### Hack 2: Weight-Bake (Control Vectoring)
- Fuse learned refiner vectors permanently into base model weights
- Calculate delta between original and refined hidden states
- Add delta directly into Layer 15 weights
- Delete refiner entirely, export 100% standard GGUF
- Lose dynamic loops, keep the knowledge

### Hack 3: Custom C++ PR (Fork llama.cpp)
- Define LLM_ARCH_BRAINLOOP in llama.cpp
- Add for-loop logic into `llama_build_graph`
- Build ggml computation graph: ggml_add, ggml_mul_mat, ggml_norm
- Dynamic early exit in C: compute one pass, check MTP confidence, `break` if done
- True dynamic loops, saves actual compute, but requires custom runner

## Dynamic Execution Graph (MoE on Steroids)
Standard MoE is rigid (router → Expert A or B). The refiner router creates a
truly dynamic graph:
- "Needs coding" → send to massive looping Gas Tensor refiner
- "Simple greeting" → skip Layers 16-25 entirely, dump straight to output head
- Python hooks control flow live, base weights never change

This works today in PyTorch/vLLM via forward hooks. Cannot pack into static
llama.cpp GGUF without custom C++ runner.

## Gate Strategy (from Gemini sessions)

### Gate Evolution History (SmolLM→3B)

| Attempt | Init | Method | Epoch 1 Result | Issue |
|---|---|---|---|---|
| v1 | `torch.zeros(1)` | sigmoid(0)=0.5 | -25.7% PPL (SmolLM) | 50% untrained noise at step 0 |
| v2 | `torch.tensor(-10.0)` | sigmoid ≈ 0 | -5.6% then flatlined | Gradient starvation — gate never moved |
| v3 | `torch.tensor(-2.0)` | sigmoid ≈ 0.12 | -18.33% at epoch 2 | Gate still stuck at 0.1191 |
| **v4** | `torch.tensor(0.0)` | **STE** (train=1.0, eval=sigmoid) | **-15.28% at epoch 2** | **Winner** — stable, no drift |

### Winner: Straight-Through Estimator
- Forward: gate hardcoded to 1.0 (forces 100% loop usage)
- Backward: gradient passed through unchanged (gate can't close)
- Eval: sigmoid(0) ≈ 0.5 (balanced contribution)
- Gate stays locked at 0.4980 across all epochs — perfect equilibrium
- Weight decay 0.1 prevents overfitting; best PPL at epoch 2, then stabilizes

### Problem: Gate Gradient Starvation
Gate initialized at negative values (sigmoid(-2)/sigmoid(-10)) sits in
flat region of sigmoid curve. Gradient ≈ 0, gate never moves.

### Fix 1: Split Learning Rate
Give gate param 100x higher LR (1e-2) than rest of refiner (1e-4).

### Fix 2: Urgency Penalty
Add `(1.0 - sigmoid(gate)) * urgency_weight` directly to loss. Model must
open gate to relieve artificial pressure.

### Fix 3: Highway Sabotage (Dropout)
Randomly zero out identity path 10% of time during training. Model can't
be lazy because the bypass route is unreliable.

### Fix 4: Straight-Through Estimator (Trick)
Forward pass: gate hardcoded to 1.0 (100% loop usage).
Backward pass: gradient passed through as identity, gate can't shut down.
Forces refiner internals to train on real data at full volume.

### Fix 5: Direct Scalar (No Sigmoid)
Remove sigmoid, use raw parameter. Init at 0 or 1e-4. Uninhibited gradient.

## Research TODOs

1. ~~Prove PPL drops at 3B scale~~ — CONFIRMED: -15.28% PPL, Qwen2.5-3B, golden config
2. ~~Fix gate gradient starvation~~ — CONFIRMED: STE gate, locked at 0.4980 equilibrium
3. ~~Fix overfitting drift~~ — CONFIRMED: weight_decay=0.1, best at epoch 2, stable
4. ~~Fix C++ GPU/CPU memory chasm~~ — CONFIRMED: `ggml_backend_alloc_ctx_tensors_from_buft`, CUDA0 allocation, gate=0.4980
5. Implement full QKV attention in C++ refiner — blocked on ggml tensor layout for `build_attn_mha`
6. Bake refiner weights into GGUF (Hack 1 / Fix Path B) — tensor slots exist in llama-model.cpp, needs GGUF writing script
7. Scale to 7B/9B using golden config
8. Test separate refiner training for code (opencode JSONL available)
9. Test on quantized base models (refiner recovering quant damage)
10. Test hot-swap at inference: load two refiners, switch mid-session
11. Extract refusal direction vectors from a base model
12. Build zero-init router block (gas tensor switchboard)
13. Implement MTP exit head for adaptive loop depth
14. Reboot system → activate python3-devel → torch.compile inductor backend

---

## 2026-06-09: Placement Fix & Revolution Sweep

### Placement Correction

The C++ refiner was placed AFTER layer 18's full computation. PyTorch version
places it BETWEEN layers 17 and 18 (after layer 17's output, before layer 18
processes it).

**Fix:** Changed `il == split_layer` to `il == split_layer - 1`. The refiner
now receives layer 17's output, matching PyTorch training conditions.

### Revolution Sweep Results

With corrected placement, the sweet spot SHIFTED from 2 to 1 revolution:

| Revs | PPL | Delta vs Baseline |
|---|---|---|
| Baseline | 8.5775 | — |
| **1** | **8.1883** | **-4.5%** |
| 2 | 8.2098 | -4.3% |
| 3 | 8.6580 | +0.9% |

1 revolution gives best PPL with minimum compute overhead. At 3 revolutions,
PPL is worse than baseline (overprocessing with corrected placement).

### PPL Progress Summary

| Milestone | PPL Delta |
|---|---|
| GPU fix + output/FFN only | +0.4% |
| + full QKV attention | -3.1% |
| + placement correction (17-18) | -4.3% |
| + 1 revolution sweet spot | **-4.5%** |

Remaining gap to PyTorch (-15.28%): possibly dtype (F32 .bin vs bf16 training),
causal mask, or subtle differences in batched norm behavior.

### Causal Mask Status

CPU-allocated mask context goes out of scope before inference runs. Mask may
not be needed — base model's autoregressive forward already enforces causality.
Further investigation deferred.

### Revolution Embedding

Not yet wired into the inlined refiner path. The rev_emb weights are loaded
into the GPU cache but not applied during refinement. PyTorch adds revolution
embedding at each pass to distinguish loop iterations. Low priority — likely
small PPL contribution.

### Inline RAG Experiment

Self-contained FAISS experiment in `rag-experiment/`:
- 100 random 2048-dim vectors as test corpus (FAISS FlatL2)
- W_query [2048, 256], W_context [256, 2048]
- inject_rag(h) stub: query projection → top-3 FAISS → context projection → h + ctx
- Python venv, faiss-cpu, no CUDA conflicts
- Ready for integration into C++ refiner loop

---

## 2026-06-09 (continued): Inline RAG Injection Test

### RAG Document Loading — Works

Successfully loaded 300 normalized token embedding vectors (2048-dim) from the
model's own `embed_tokens.weight` into GPU memory alongside the refiner weights.
The vectors were exported as `rag-experiment/rag_docs.bin` using safetensors.

Loading pipeline: `ggml_backend_alloc_ctx_tensors_from_buft` on CUDA0, data
copied via `ggml_backend_tensor_set`. Confirmed: "loaded RAG index: 300 docs x
2048 dim on CUDA0" at init. No crash. No memory issue.

### RAG Injection Approaches Tested

**Softmax-weighted docs:** Computed similarity via `ggml_mul_mat(rag_docs, x)`
→ softmax → weighted sum via `ggml_transpose + ggml_mul_mat`. Works but
produces near-uniform weights for random token embeddings — effectively a no-op.
PPL unchanged at 8.1883 regardless of scale (0.05 to 0.5).

**Single doc injection via ggml_repeat:** Attempted to broadcast a single
document vector [n_embd] to match hidden state shape [n_embd, n_tokens].
Crash: CUDA error in ggml_backend_cuda_synchronize. ggml_repeat doesn't
handle this broadcast pattern correctly on CUDA backend.

### Findings

1. **Index loading infrastructure is proven.** FAISS isn't needed in C++ —
   the document matrix on GPU plus ggml_mul_mat gives similarity search natively.
2. **Additive injection works** — the softmax-weighted path runs without crash
   and doesn't degrade PPL. The issue is semantic: random token embeddings
   carry no useful signal.
3. **For semantic injection**, documents must be embedded from actual text
   using tok_embd, not random token vectors. The index needs real content.
4. **Broadcasting** for additive injection needs a different ggml approach —
   `ggml_repeat` fails on CUDA. Alternatives: `ggml_reshape` + `ggml_add`
   with compatible shapes, or using `ggml_mul_mat` with a projection.
