# Conch POC — Bolt-On Refiner for Small Models

A trainable refiner block (~2% params) inserted mid-model that improves
perplexity by iterating on hidden states. Base model stays frozen.

## Results

| Model | PPL Delta | Status |
|---|---|---|
| SmolLM-135M | -25.7% | PyTorch |
| Qwen2.5-3B | -15.3% | PyTorch |
| Qwen2.5-3B (C++ llama.cpp port) | -3.1% | In progress |

Qwen2.5-3B benchmarks (C++ port, F16, full runs):

| Benchmark | Baseline | Refiner |
|---|---|---|
| ARC-Challenge (1172q) | 4.44% | 4.35% |
| HellaSwag (10042q) | — | 7.60% |

## Files

| File | Description |
|---|---|
| `refiner.py` | RefinerBlock + ConchRefinerModel (PyTorch) |
| `train_refiner.py` | Training script for bolt-on refiner |
| `retrain_modelnorm.py` | Retrain with frozen norms + export .bin for C++ |
| `force_loop_qwen3b.py` | Force-loop diagnostic without training |
| `model.py`, `train.py` | Original conch shell (v1-v3, dead end) |
| `evaluate.py` | PPL evaluation script |
| `brainloop-ggml-weights/` | Exported .bin weights for C++ port |
| `bench_results/` | ARC and HellaSwag benchmark results |
| `checkpoints-refiner/` | SmolLM-135M trained refiner |
| `checkpoints-refiner-qwen3b-v4-wd/` | Qwen2.5-3B best refiner (weight decay) |

C++ port lives in a llama.cpp fork at the sibling repo.
`SPEC_inline_rag_refiner.md` sketches a proposed inline RAG extension.

## Training

```bash
# SmolLM-135M
python train_refiner.py --base-model HuggingFaceTB/SmolLM-135M --epochs 10

# Qwen2.5-3B
python train_refiner.py --base-model Qwen/Qwen2.5-3B --epochs 3 --batch-size 4 \
  --split-layer 18 --revolutions 2 --lr 1e-4
```

Single 3090, bfloat16. Gate uses straight-through estimator (train=1.0, eval=sigmoid).
Weight decay 0.1 prevents overfitting after epoch 2.
