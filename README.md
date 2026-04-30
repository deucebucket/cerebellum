# Cerebellum — Ablation-Informed Quantization for LLMs

Surgical precision allocation for GGUF quantization. Instead of applying the same quant level uniformly across all tensors, Cerebellum measures the actual sensitivity of each tensor and allocates bits where they matter most.

## How It Works

1. **Ablate** — crush each tensor to Q2_K individually, measure the perplexity impact
2. **Classify** — sacred (high sensitivity), neutral, or demotable (improves when crushed)
3. **Allocate** — sacred tensors get Q6_K/Q8_0, demotable tensors stay at Q2_K, everything else fills in to meet the size budget
4. **Multi-pass promotion** — tensors climb quant levels (Q2_K → Q3_K → ... → Q8_0) across passes until the budget is exhausted

The result: lower perplexity at the same file size compared to uniform quantization.

## Results

### Qwen 3.6 27B — Cerebellum v4 (12 GB)

| Metric | Score |
|--------|-------|
| **Perplexity** (WikiText-2, 2048 ctx) | **7.034** |
| **MMLU** (11,643 questions) | **82.5%** |
| **MMLU-Redux** (2,400 questions) | **77.1%** |
| **ARC-Challenge** (1,172 questions) | **95.1%** |
| **HellaSwag** (10,042 questions) | **91.2%** |
| **File size** | **11.98 GB** |
| **Tensor overrides** | 181 |

Speed on RTX 3090 (full GPU offload, 4096 context): **71 tok/s prompt**, **36.5 tok/s generation**

#### Perplexity Comparison

| Method | Size | PPL (wiki) | Notes |
|--------|------|------------|-------|
| **Cerebellum v4** | **11.98 GB** | **7.034** | 181 tensor overrides |
| Cerebellum v2 | 10.68 GB | 7.087 | 115 overrides |
| Q2_K + imatrix | 9.98 GB | 7.500 | Standard imatrix baseline |
| Q2_K (no imatrix) | 9.98 GB | 8.256 | Vanilla baseline |

#### Tensor Allocation (v4, 12 GB budget)

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
- **Same-layer interaction effects are destructive** — crushing two FFN tensors in the same layer simultaneously causes 87% regression (interaction ratio 0.13)
- **Cross-layer effects are ~86% additive** — single-tensor ablation deltas predict multi-tensor outcomes with ~14% attenuation

### Qwen 3.5 9B — Mamba Hybrid (202-tensor ablation)

| Method | Size | PPL (wiki) | Notes |
|--------|------|------------|-------|
| Q4_K_M + imatrix | 5.36 GB | **7.724** | Best for this architecture |
| Q4_K_M vanilla | 5.62 GB | 7.769 | Standard baseline |
| Cerebellum promote-only | 5.47 GB | 7.818 | 22 non-SSM tensor promotions |
| Cerebellum Q2_K base | 5.21 GB | 9.827 | SSM tensors at Q2_K = disaster |

**Finding:** Cerebellum tensor overrides don't work for Mamba hybrid models. llama.cpp's SSM inference kernels break when tensor types are overridden — even *promoting* to higher precision degrades PPL. Full analysis in [docs/mamba_hybrid_findings.md](docs/mamba_hybrid_findings.md).

The 202-tensor ablation dataset is the only ground-truth PPL ablation data for Qwen 3.5 9B. Key finding: `blk.0.ssm_out` has +6.342 PPL delta, confirming SSM output sensitivity.

## Usage

### Ablation Sweep

```bash
# Run ablation sweep — crush each tensor to Q2_K, measure PPL
python -m osmosis.cerebellum ablate \
    --base-gguf model-Q2_K.gguf \
    --tensors ablation_plan.json \
    --output ablation_results.json
```

### Budget Allocation

```bash
# Generate optimal tensor type allocation for a size budget
python -m osmosis.cerebellum allocate \
    --ablation ablation_results.json \
    --budget 12.0 \
    --output tensor_types.txt
```

### Build the GGUF

```bash
# Quantize with per-tensor overrides
llama-quantize --imatrix imatrix.dat \
    --tensor-type @tensor_types.txt \
    model-f16.gguf model-cerebellum.gguf Q2_K
```

### Imatrix Generation

Cerebellum includes a fast imatrix generator that computes importance directly from weight statistics in ~60 seconds on CPU (no calibration data, no GPU required):

```bash
python -m osmosis.imatrix_stream \
    --model Qwen/Qwen3.6-27B \
    --output osmosis_imatrix.dat -v
```

## Installation

```bash
pip install -e .
```

Requires PyTorch and Transformers (for loading HuggingFace models). llama.cpp required for quantization and perplexity measurement.

## Architecture Support

Auto-detects model architecture:

- **Qwen 3.6** (dense transformer) — self_attn + MLP projections
- **Qwen 3.5** (hybrid SSM + attention) — linear_attn + self_attn + MLP
- **Standard transformers** — any model with self_attn q/k/v/o + MLP gate/up/down

Works best on pure transformers. Mamba hybrid models have SSM tensors whose quant types cannot be safely overridden in llama.cpp — see [docs/mamba_hybrid_findings.md](docs/mamba_hybrid_findings.md).

## Project Structure

```
osmosis/
├── cerebellum.py      # Ablation-informed precision allocator
├── imatrix_stream.py  # Streaming imatrix generation (any model size)
├── imatrix_gen.py     # Standard imatrix generation
├── imatrix_format.py  # llama.cpp imatrix binary format writer
├── refusal_direction.py  # Differential activation analysis
└── refusal_prompts.py    # Contrastive prompt datasets

osmosis-qwen36-27b/    # Qwen 3.6 27B results (pure transformer)
├── ablation_results.json        # Tensor ablation data (23 tensors)
├── interaction_results.json     # Multi-tensor interaction effects
├── tensor_types_v4_12gb.txt     # v4 allocation (181 overrides)
└── benchmark_results/           # MMLU, ARC, HellaSwag, speed

osmosis-qwen35-9b/     # Qwen 3.5 9B results (Mamba hybrid)
├── ablation_results.json        # Full tensor ablation data (202 tensors)
├── cerebellum_scan_9b.sh        # Ablation scan script
└── tensor_types_promote_only.txt # Best non-SSM promotions (22 overrides)

docs/
├── mamba_hybrid_findings.md     # SSM quantization constraints + analysis
└── llama_cpp_tensor_backend_patch.md
```

## Models

- [Qwen3.6-27B-Cerebellum-v4-GGUF](https://huggingface.co/deucebucket/Qwen3.6-27B-Cerebellum-v4-GGUF) — 12 GB, PPL 7.034, 181 overrides
- [Qwen3.6-27B-Osmosis-Q2_K-GGUF](https://huggingface.co/deucebucket/Qwen3.6-27B-Osmosis-Q2_K-GGUF) — 10 GB, PPL 7.500, imatrix baseline

## Test Hardware

| Component | Spec |
|-----------|------|
| **GPU** | NVIDIA RTX 3090 (24 GB) |
| **CPU** | AMD Ryzen 7 5800XT |
| **RAM** | 64 GB DDR4 |
| **OS** | Fedora Linux 43 (Atomic) |

## Attribution

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — imatrix quantization and tensor type override support
- **[AWQ](https://arxiv.org/abs/2306.00978)** — channel-level weight sensitivity insights
- **[GPTQ](https://arxiv.org/abs/2210.17323)** — foundational post-training quantization
- **[Qwen Team](https://huggingface.co/Qwen)** — open-weight models
- **[HuggingFace](https://huggingface.co/)** — Transformers library and model distribution

## License

Apache 2.0
