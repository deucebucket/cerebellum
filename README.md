# Cerebellum — Ablation-Informed Model Surgery for LLMs

Surgical precision allocation and neuron-level behavior removal for GGUF quantization. Beats uniform quant methods at the same file size by putting bits where they matter.

## What It Does

### Tensor Cerebellum (v1 — working now)

Instead of applying the same quant level uniformly across all tensors, Cerebellum:

1. **Ablates individual tensors** — crushes each to Q2_K one at a time, measures perplexity impact
2. **Classifies tensors** — sacred (high sensitivity), neutral, or demotable (improves when crushed)
3. **Allocates precision surgically** — sacred tensors get Q6_K/Q8_0, safe zones stay at Q2_K
4. **Multi-pass promotion** — tensors climb multiple quant levels (Q2_K → Q3_K → ... → Q8_0) across passes until the size budget is exhausted

The result: better perplexity at the same file size compared to uniform quantization.

### Neuron Cerebellum (v2 — planned)

Same sweep-measure-classify methodology, finer scalpel:

- **Dead neuron pruning** — find neurons that never activate, zero them out (free size savings)
- **Behavior removal** — find directions responsible for censorship/refusal/hedging, subtract them from weights
- **Thinking mode control** — locate and optionally suppress chain-of-thought trigger circuits
- **Activation-informed quantization** — neurons that fire hard get preserved, neurons that barely fire get crushed

Pipeline: `Base f16 → Neuron Surgery → Tensor Cerebellum → GGUF`

## Results

### Cerebellum v4 — Qwen3 27B

| Method | Size | PPL (wiki) | Notes |
|--------|------|------------|-------|
| **Cerebellum v4** | **11.98 GB** | **7.034** | Multi-pass, 181 tensor overrides |
| Unsloth Q2_K_XL | 12.0 GB | 7.040 | Uniform quant |
| Cerebellum v2 | 10.68 GB | 7.087 | Single-pass, 115 overrides |
| Q2_K + imatrix | 9.98 GB | 7.500 | Standard baseline |

Cerebellum v4 beats Unsloth's dynamic quant at the same 12GB file size. The v4 GGUF is available on HuggingFace: [deucebucket/Qwen3.6-27B-Cerebellum-v4-GGUF](https://huggingface.co/deucebucket/Qwen3.6-27B-Cerebellum-v4-GGUF)

### Tensor Allocation (v4, 12GB budget)

- **7 tensors at Q8_0** — sacred attention/FFN in the most sensitive layers
- **41 tensors at Q6_K** — high-sensitivity layers
- **70 tensors at Q5_K** — moderate sensitivity
- **22 tensors at Q4_K** — mild sensitivity
- **19 tensors at Q3_K** — low sensitivity
- **22 tensors at Q2_K** — explicitly demoted (improve when crushed)
- Remaining ~670 tensors at Q2_K base

### Key Findings

- **Layer 63 is sacred** — q_proj (+0.162 PPL) and ffn_down (+0.138 PPL) need maximum precision
- **7 tensors actively improve at Q2_K** — crushing them helps (negative ablation delta)
- **Same-layer interaction effects are destructive** — crushing two FFN tensors in the same layer simultaneously causes regression (interaction ratio 0.13)
- **Cross-layer effects are ~86% additive** — single-tensor ablation deltas predict multi-tensor outcomes with ~14% attenuation
- **23 vs 18 tensors of ablation data** — extrapolation is already near-optimal; more data doesn't change allocation much

### Chat Benchmark (v4, RTX 3090)

Speed: **71 prompt tok/s**, **36.5 gen tok/s** (full GPU offload, 4096 context)

6 chat prompts across technical, coding, math, creative, analysis, and debugging categories — all producing coherent, accurate responses. Full results in `osmosis-qwen36-27b/benchmark_results/`.

## Tools

### Fast Imatrix Generation (Osmosis)

Generate importance matrices for llama.cpp quantization in **seconds**, not hours. Standard imatrix generation requires running calibration text through the full model. Osmosis computes importance directly from weight statistics in ~60 seconds on CPU.

```bash
# Generate imatrix — any model size, ~4GB RAM
python -m osmosis.imatrix_stream \
    --model Qwen/Qwen3.6-27B \
    --output osmosis_imatrix.dat -v

# Then quantize with llama.cpp
llama-quantize --imatrix osmosis_imatrix.dat model-f16.gguf model-Q2_K.gguf Q2_K
```

### Tensor Ablation Sweep

```bash
# Run ablation sweep — crush each tensor to Q2_K, measure PPL
python -m osmosis.cerebellum ablate \
    --base-gguf model-Q2_K.gguf \
    --tensors ablation_plan.json \
    --output ablation_results.json

# Generate optimal tensor type allocation for a size budget
python -m osmosis.cerebellum allocate \
    --ablation ablation_results.json \
    --budget 12.0 \
    --output tensor_types.txt

# Build the GGUF with per-tensor overrides
llama-quantize --imatrix imatrix.dat \
    --tensor-type @tensor_types.txt \
    model-f16.gguf model-cerebellum.gguf Q2_K
```

## Installation

```bash
pip install -e .
```

Requires PyTorch and Transformers (for loading HuggingFace models).

## Architecture Support

Auto-detects model architecture:

- **Qwen 3.6** (dense transformer) — self_attn + MLP projections
- **Qwen 3.5** (hybrid SSM + attention) — linear_attn + self_attn + MLP
- **Standard transformers** — any model with self_attn q/k/v/o + MLP gate/up/down

The technique is model-agnostic — give it any GGUF, ablation data, and a size budget, and it finds the optimal per-tensor precision allocation.

## Project Structure

```
osmosis/
├── cerebellum.py      # Ablation-informed precision allocator (v1)
├── imatrix_stream.py  # Streaming imatrix generation (any model size)
├── imatrix_gen.py     # Standard imatrix generation (with optional activation calibration)
└── imatrix_format.py  # llama.cpp imatrix binary format writer

osmosis-qwen36-27b/    # Qwen3 27B findings
├── ablation_results.json        # Full tensor ablation data (23 tensors)
├── interaction_results.json     # Multi-tensor interaction effects
├── tensor_types_v4_12gb.txt     # v4 allocation (181 overrides)
└── benchmark_results/           # Chat quality + speed benchmarks
```

## Roadmap

- [x] Fast imatrix generation (Osmosis)
- [x] Tensor ablation sweep
- [x] Multi-pass precision allocator (Cerebellum v1)
- [x] Beat Unsloth Q2_K_XL at same size (v4: 7.034 vs 7.040)
- [ ] Activation recorder for neuron-level analysis
- [ ] Dead neuron pruning
- [ ] Contrastive behavior analysis (censorship/refusal removal)
- [ ] Neuron classifier (sacred/dead/censorship/thinking)
- [ ] Full pipeline: neuron surgery + tensor allocation → GGUF

## Attribution

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — imatrix quantization system and tensor type override support
- **[AWQ](https://arxiv.org/abs/2306.00978)** — channel-level weight sensitivity insights
- **[Abliteration](https://huggingface.co/blog/mlabonne/abliteration)** — proving that refusal behavior is directionally localized and removable
- **[GPTQ](https://arxiv.org/abs/2210.17323)** — foundational post-training quantization work
- **[Unsloth](https://unsloth.ai/)** — dynamic quantization benchmarks that pushed the field forward
- **[Qwen Team](https://huggingface.co/Qwen)** — open-weight models used in development
- **[HuggingFace](https://huggingface.co/)** — Transformers library and model distribution

## License

Apache 2.0
