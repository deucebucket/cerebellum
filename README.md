# Osmosis — Fast Imatrix for llama.cpp

Generate importance matrices for [llama.cpp](https://github.com/ggerganov/llama.cpp) quantization in **seconds**, not hours.

Standard imatrix generation (`llama-imatrix`) requires running calibration text through the full model — hours of GPU time for large models. Osmosis computes importance directly from weight statistics, producing a compatible imatrix file in ~20 seconds on CPU.

## How It Works

For each weight tensor in the model:

1. Compute per-channel sensitivity: `L2_norm × max_abs × variance`
2. Normalize to importance scores
3. Write in llama.cpp's legacy imatrix binary format

The quantizer (`llama-quantize --imatrix`) uses these scores to allocate more bits to important weight blocks within each tensor. No calibration data needed — the sensitivity signal comes from the weights themselves.

## Installation

```bash
pip install -e .
```

Requires PyTorch and Transformers (for loading HuggingFace models).

## Usage

```bash
# Generate imatrix from any HuggingFace model
python -m osmosis.imatrix_gen \
    --model Qwen/Qwen3.6-27B \
    --output osmosis_imatrix.dat \
    --no-calibrate \
    -v

# Then quantize with llama.cpp as usual
llama-quantize --imatrix osmosis_imatrix.dat model-f16.gguf model-Q2_K.gguf Q2_K
```

### With activation calibration (optional)

If the model fits in GPU memory, you can blend weight sensitivity with activation statistics for potentially better results:

```bash
python -m osmosis.imatrix_gen \
    --model Qwen/Qwen3.5-9B \
    --output osmosis_imatrix.dat \
    --num-samples 8 \
    -v
```

This runs 8 calibration prompts through the model and blends activation sensitivity (50%) with weight sensitivity (50%).

## Architecture Support

Auto-detects model architecture:

- **Qwen 3.6** (dense transformer) — self_attn + MLP projections
- **Qwen 3.5** (hybrid SSM + attention) — linear_attn + self_attn + MLP
- **Standard transformers** — any model with self_attn q/k/v/o + MLP gate/up/down

## Results

Qwen 3.6 27B quantized with osmosis imatrix, perplexity on wikitext-2-raw (2048 ctx, full test):

| Quant | Size | PPL |
|---|---|---|
| Q4_K_M | 16 GB | 7.44 |
| Q3_K_M | 13 GB | 7.45 |
| Q2_K | 10 GB | 8.31 |

The Q2_K GGUF is available on HuggingFace: [deucebucket/Qwen3.6-27B-Osmosis-Q2_K-GGUF](https://huggingface.co/deucebucket/Qwen3.6-27B-Osmosis-Q2_K-GGUF)

## How the Imatrix Format Works

llama.cpp's imatrix is a binary file mapping tensor names to per-channel importance scores. During quantization, tensors with higher importance scores get allocated more bits within the quant type's budget. This is a standard llama.cpp feature — osmosis just provides a faster way to generate the importance data.

The binary format:
- Header: number of entries
- Per entry: tensor name, call count, importance values (float32 array)
- Footer: call count, dataset name

## Attribution

This project builds on the work of:

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** by Georgi Gerganov — the imatrix quantization system that makes importance-weighted quantization possible. The imatrix binary format and `--imatrix` flag in `llama-quantize` are llama.cpp features.
- **[AWQ (Activation-aware Weight Quantization)](https://arxiv.org/abs/2306.00978)** by Lin et al. — the insight that channel-level weight sensitivity (computed from activations or weight statistics) can guide quantization precision allocation. Our per-channel sensitivity formula is inspired by AWQ's approach.
- **[GPTQ](https://arxiv.org/abs/2210.17323)** by Frantar et al. — foundational work on post-training quantization showing that second-order weight information improves quantized model quality.
- **[Qwen Team](https://huggingface.co/Qwen)** — for the open-weight models used in development and testing.
- **[Unsloth](https://unsloth.ai/)** — their Dynamic quantization work and public benchmarks on GGUF quality pushed the field forward and motivated this exploration of faster imatrix alternatives.
- **[HuggingFace](https://huggingface.co/)** — Transformers library for model loading and the Hub for distribution.

## License

Apache 2.0
