# RTX 3090 LoRA/QLoRA Training Speed Notes

Private working notes for speeding up local fine-tuning on the 24 GB RTX 3090.

## Current Hardware Baseline

- GPU: NVIDIA GeForce RTX 3090
- VRAM: 24,576 MiB
- CUDA capability: 8.6
- Driver observed: 580.142
- Power limit observed: 420 W
- `torch.cuda.is_bf16_supported()` currently returns `True` in this environment.

NVIDIA's RTX 3090 spec page lists 24 GB GDDR6X, 10,496 CUDA cores, third-generation Tensor Cores, and Ampere architecture.

## Practical Ranking Of Speed Levers

1. Use Unsloth for single-GPU LoRA/QLoRA when supported.
2. Pack short samples so each 4096-token sequence is mostly useful training text.
3. Lower `max_seq_length` to the smallest value that covers the useful distribution.
4. Keep the model fully resident on GPU; avoid CPU offload unless the alternative is OOM.
5. Use Flash Attention / SDPA / optimized attention where compatible.
6. Reduce LoRA rank and target modules before reducing dataset quality.
7. Prefer one clean epoch over multiple noisy epochs.
8. Track tokens/sec, not examples/sec.

## What Axolotl Is Good For

Axolotl is mainly an orchestration and reproducibility win:

- YAML-driven training configs.
- Built-in knobs for sample packing, attention backends, LoRA/QLoRA, gradient checkpointing, layer offload, Cut Cross Entropy, Liger kernels, FSDP, and DeepSpeed.
- Easier multi-run and multi-GPU experiments.

It is not automatically faster than the current Unsloth path on one RTX 3090. A naive Axolotl port may be equal or slower. Axolotl becomes attractive if we want systematic experiment configs, multipack controls, or distributed training.

## What Unsloth Is Good For

Unsloth is still the default choice for fast single-GPU LoRA/QLoRA:

- It advertises roughly 2x faster training with much lower VRAM than standard Hugging Face/TRL paths.
- It already powers the BoltHands v3 script.
- It exports LoRA/merged/GGUF conveniently.

The current BoltHands script already uses the right broad class of tool:

- `unsloth/Qwen3.5-9B`
- 4-bit loading
- LoRA rank 32
- attention + MLP target modules
- sequence length 4096
- batch size 2, gradient accumulation 8
- 41,735 training examples, 2 epochs

That is a large real workload for a 3090. The 18 hour estimate is plausible.

## Sample Packing Is The First Thing To Test

TRL's SFTTrainer supports packing multiple examples into one sequence. Axolotl calls this sample packing / multipack and recommends it for utilization when examples are short.

For BoltHands v3, the JSONL line-length distribution is:

- rows: 41,735
- average: 3,061 chars
- p50: 2,820 chars
- p90: 4,328 chars
- p95: 5,052 chars
- p99: 13,514 chars
- max: 17,371 chars

This strongly suggests many samples are below a 4096-token training window. Without packing, those shorter examples are padded or under-fill compute. Packing should improve useful tokens/sec, especially for the bulk of the dataset.

Test plan:

1. Measure current baseline tokens/sec on a 500-1000 step run.
2. Enable packing with the same model, rank, LR, and data.
3. Compare tokens/sec, peak VRAM, train loss curve, and a small behavioral eval.

## Sequence Length Strategy

Do not default to 4096 unless the task needs it.

Recommended sweep:

- 2048 packed
- 3072 packed
- 4096 packed

Pick the shortest setting that preserves tool-call quality and multi-turn behavior. Context length has a large cost because attention and activations scale hard with sequence length.

For long-tail samples, options:

- Keep 4096 and pack.
- Truncate only known low-value sections.
- Split long conversations into shorter windows.
- Bucket long examples separately and train a small long-context pass after the main short-context pass.

## LoRA Rank And Target Modules

The current BoltHands rank is `r=32`, `alpha=64`, targeting:

- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `gate_proj`, `up_proj`, `down_proj`

That is strong but expensive. For behavior/style/tool routing, test:

- `r=16`, `alpha=32`, same target modules.
- `r=16`, attention-only.
- `r=8`, attention + MLP for tiny/domain-specific adapters.

Use quality gates before assuming lower rank is worse. A cleaner dataset at `r=16` can beat a noisy `r=32`.

## Precision And Optimizer

Use QLoRA for 7B-14B class models on the 3090. QLoRA freezes the 4-bit base model and trains LoRA adapters through it, using NF4/double quantization/paged optimizers to reduce memory.

Use:

- 4-bit base loading for 9B/14B work.
- bf16 if the local stack supports it and loss is stable.
- fp16 fallback if bf16 causes issues.
- paged/8-bit optimizer only when needed for memory spikes.

Avoid CPU offload for speed. Offload is a survival tool, not a speed tool.

## Batch Size

Think in tokens:

```text
effective_tokens_per_step = max_seq_length * per_device_batch * grad_accum * packing_efficiency
```

The current BoltHands setup is:

```text
4096 * 2 * 8 = 65,536 nominal tokens per optimizer step
```

Without packing, useful tokens may be far lower. With packing, the same nominal step can train much more real content.

## Data Beats Extra Epochs

For local 3090 runs, prefer:

- deduped examples
- consistent chat template
- clean tool-call JSON
- assistant-only loss where appropriate
- domain-balanced sampling
- hard negative examples for "do not call a tool"

Then train fewer epochs. Two full epochs over noisy or duplicated data is slow and can overfit the format.

## Recommended BoltHands Next Experiment

Keep Unsloth. Run a controlled speed experiment:

1. Baseline current config for 500-1000 steps; record tokens/sec, VRAM, loss.
2. Enable packing at 4096; same everything else.
3. Try 2048 packed with `r=16`, `alpha=32`.
4. Compare tool-call validity, JSON validity, domain routing, and refusal/no-tool cases.

Success criteria:

- at least 1.3x useful tokens/sec improvement, or
- same speed with lower VRAM and equal quality, or
- same quality in materially fewer wall-clock hours.

## Source Links

- NVIDIA RTX 3090 specs: https://www.nvidia.com/en-my/geforce/graphics-cards/30-series/rtx-3090-3090ti/
- Unsloth docs: https://unsloth.ai/docs
- Axolotl optimizations: https://docs.axolotl.ai/docs/optimizations.html
- Axolotl multipack: https://docs.axolotl.ai/docs/multipack.html
- Hugging Face TRL SFTTrainer: https://huggingface.co/docs/trl/v0.23.0/en/sft_trainer
- QLoRA paper: https://arxiv.org/abs/2305.14314
- LoRA paper: https://arxiv.org/abs/2106.09685
- bitsandbytes optimizers: https://huggingface.co/docs/bitsandbytes/v0.43.0/optimizers
