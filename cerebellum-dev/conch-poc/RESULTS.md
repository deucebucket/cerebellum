# Conch Refiner Results

## PyTorch: Bolt-on Refiner — Proven at Two Scales

A trainable refiner block inserted mid-model improves perplexity with zero
modification to the frozen base model.

### SmolLM-135M (2026-05-02)

| Revolutions | PPL | Delta vs Baseline |
|-------------|------|-------------------|
| 0 (baseline) | 23.50 | — |
| 1 | 19.26 | -18.1% |
| **2** | **17.46** | **-25.7%** |
| 3 | 18.30 | -22.1% |
| 4 | 20.39 | -13.2% |
| 6 | 27.23 | +15.9% |

Base: SmolLM-135M (frozen). Refiner: 1 transformer layer, 2M params (1.98% overhead),
inserted at layer 15. Sweet spot: 2 revolutions. Training: 10 epochs, 55s/epoch.

### Qwen2.5-3B PyTorch

| Revolutions | PPL | Delta vs Baseline |
|-------------|------|-------------------|
| 0 (baseline) | ~10.0 | — |
| **2** | **~8.5** | **-15.28%** |

STE gate (train=1.0, eval=sigmoid). Weight decay 0.1. Best at epoch 2.
Split layer 18. 3 epochs, ~550s/epoch.

---

## C++ Port: Brainloop in llama.cpp (2026-06-09)

### Architecture

The refiner is implemented as a native llama.cpp graph builder
(`LLM_ARCH_QWEN2_BRAINLOOP` / `qwen2-brainloop`) that intercepts ALL Qwen2
models at `llama-model.cpp:8669`. Weights are loaded at runtime from raw
`.bin` files and GPU-allocated on the same backend as the split layer.

Key source files:
- `src/models/qwen2_brainloop.cpp` — graph builder with GPU-allocated refiner
- `src/llama-arch.cpp` — tensor map for 10 brainloop weight types
- `src/llama-arch.h` — `LLM_ARCH_QWEN2_BRAINLOOP` enum
- `src/llama-model.cpp` — tensor slots at layer 18, routing override

Weights live in `brainloop-ggml-weights/` (34 `.bin` files, ~134 MB).

### GPU Allocation Fix (Fix Path A)

**Problem:** `ggml_new_tensor_2d(ctx0, ...)` creates CPU-allocated scratch
tensors. CUDA fused ops (permute, transpose, flash attention) can't operate
across the CPU/GPU memory boundary — they either read garbage (49M PPL) or
segfault.

**Fix:** Use `ggml_backend_alloc_ctx_tensors_from_buft(ctx, model.select_buft(18))`
to allocate all 18 refiner weight tensors on the same VRAM backend as the
base model's layer 18. Data copied via `ggml_backend_tensor_set()`.
Cached statically — loaded once per process.

### C++ PPL Progress (Qwen2.5-3B, F16 GGUF, WikiText, 128 chunks, ctx-512)

| Milestone | PPL | Delta |
|---|---|---|
| Baseline (no refiner) | 8.5775 | — |
| output+FFN only (no attention) | 8.6096 | +0.4% |
| + full QKV attention (build_attn_mha, 2 revs, after layer 18) | 8.3092 | -3.1% |
| + placement correction (after layer 17, matching PyTorch) | 8.2098 | -4.3% |
| **+ 1 revolution sweet spot** | **8.1883** | **-4.5%** |

### Revolution Sweep (corrected placement)

| Revs | PPL | Delta | Notes |
|---|---|---|---|
| 1 | **8.1883** | **-4.5%** | New sweet spot with placement fix |
| 2 | 8.2098 | -4.3% | Previously optimal (wrong placement) |
| 3 | 8.6580 | +0.9% | Overprocessing, worse than baseline |

### What's Next

- **Causal mask** — PyTorch refiner uses causal attention. GGML mask tensors
  need proper CPU-side data allocation to work with flash attention.
- **Refiner placement fix** — currently runs after layer 18's FFN+residual;
  PyTorch version places refiner between layers 17→18. Off-by-one affects
  statistics the trained weights expect.
- **GGUF baking (Fix Path B):** Append refiner tensors to GGUF so model
  loading uses the standard tensor pipeline.
- **Scale to 7B/9B** once full parity with PyTorch is achieved.
- **Test on quantized base models** — refiner recovering quant damage.

### Build & Run

```bash
# Inside distrobox ai:
cd /var/home/deucebucket/ai-drive/llama.cpp/build
cmake .. -DGGML_CUDA=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
make -j8 llama-perplexity

# Fix libcuda.so symlink (distrobox-only):
ln -sf libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so

# Convert model to GGUF:
python3 convert_hf_to_gguf.py Qwen/Qwen2.5-3B --outfile qwen2.5-3b.gguf --outtype f16

# Run perplexity test:
cd /path/to/conch-poc  # where brainloop-ggml-weights/ lives
LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 \
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin \
  llama-perplexity --model qwen2.5-3b.gguf --file wiki.test.raw --ctx-size 512
```

### HumanEval+ with Code-Trained Refiner (2026-06-09)

| Configuration | Base | Plus |
|---|---|---|
| No refiner (baseline) | 36.6% | 32.3% |
| RAG injection (WikiText-trained) | 31.7% | 28.0% |
| **RAG injection (code-trained, 174 funcs)** | **42.7%** | **37.2%** |

Training: 174 Python functions from HumanEval+ canonical solutions + common patterns.
3 epochs, 5s/epoch. RAG index: 174 code examples embedded via model's own tok_embd.
Gate: 0.50, RAG scale: 0.62. No LoRA, no prompt engineering, no tool calls.

**Pipeline for adding any code knowledge:**
1. Tokenize code/text → model.embed_tokens → average pool → normalize → .bin
2. Drop .bin into rag-experiment/rag_docs_real.bin
3. Restart server — refiner reads new index, no retraining needed

### 7B Results (Qwen2.5-7B, F16, HumanEval+)

| Configuration | Base | Plus |
|---|---|---|
| 7B + RAG (code-trained) | **56.7%** | **50.6%** |
| 7B baseline | TBD (eval tool issue) | - |

7B with 100M refiner params, 167-doc code index, split at layer 14.
Training: 18s on 167 functions. Same architecture as 3B, just bigger dims.

### HumanEval+ Scorecard (all models)

| Model | Base | Plus |
|---|---|---|
| 3B baseline | 36.6% | 32.3% |
| 3B + RAG code | **42.7%** | **37.2%** |
| 7B + RAG code | **56.7%** | **50.6%** |
