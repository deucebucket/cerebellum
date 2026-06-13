# Qwen3.5-9B MoE v1 Residual Bridge Status

Timestamp: 2026-05-29T06:30:04-05:00

## Current Decision

Keep pursuing v1 as the active dense-to-MoE path. Do not continue v0.

v1 is not ready for imatrix, ablation, GGUF quantization, or public benchmarks.
It is ready for more engineering work around runtime and trainability.

## Main Checkpoint

```text
/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v1-residual-dense-bridge
```

Manifest:

```text
/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v1-residual-dense-bridge/cerebellum_sparse_upcycle_manifest.json
```

Facts:

- 888 tensors
- 7 safetensor shards
- 28,487,768,032 bytes
- 32 converted MLP layers
- 0 dense MLP leftovers
- 0 MTP leftovers
- routed experts are zero residual tensors
- shared expert is full dense FFN
- shared expert down projection is scaled by 2
- shared expert gate is zero, so sigmoid(0)=0.5

## Code Added/Changed

```text
cerebellum-dev/sparse-upcycling/src/cerebellum_sparse_upcycling/qwen_moe_upcycle.py
cerebellum-dev/sparse-upcycling/src/cerebellum_sparse_upcycling/expert_lora.py
cerebellum-dev/sparse-upcycling/scripts/diagnose_upcycle_math.py
cerebellum-dev/sparse-upcycling/scripts/probe_generation.py
cerebellum-dev/sparse-upcycling/scripts/dump_next_logits.py
cerebellum-dev/sparse-upcycling/scripts/compare_cached_decode.py
cerebellum-dev/sparse-upcycling/scripts/train_stage1_expert_lora.py
cerebellum-dev/tool_tests/test_qwen_moe_upcycle.py
```

## Validation Artifacts

Math equivalence:

```text
cerebellum-dev/sparse-upcycling/runs/qwen35_9b_moe_v1_residual_bridge_layer0_math.json
cerebellum-dev/sparse-upcycling/runs/qwen35_9b_moe_v1_residual_bridge_layer15_math.json
cerebellum-dev/sparse-upcycling/runs/qwen35_9b_moe_v1_residual_bridge_layer31_math.json
cerebellum-dev/sparse-upcycling/runs/qwen35_9b_moe_v1_residual_bridge_layer31_math_bf16.json
```

Result:

- fp32 sampled layers 0/15/31: half-shared vs dense = 0.0
- bf16 layer 31: half-shared vs dense = 0.0

Generation/logit probes:

```text
cerebellum-dev/sparse-upcycling/runs/probe_v1_residual_dense_bridge.jsonl
cerebellum-dev/sparse-upcycling/runs/probe_v1_residual_dense_bridge_no_cache.jsonl
cerebellum-dev/sparse-upcycling/runs/dense_qwen35_9b_next_logits_code.json
cerebellum-dev/sparse-upcycling/runs/v1_residual_bridge_next_logits_code.json
cerebellum-dev/sparse-upcycling/runs/dense_qwen35_9b_next_logits_code_return.json
cerebellum-dev/sparse-upcycling/runs/v1_residual_bridge_next_logits_code_return.json
```

Cached-vs-full-prefix decode:

```text
cerebellum-dev/sparse-upcycling/runs/v1_cached_decode_compare_code.json
cerebellum-dev/sparse-upcycling/runs/dense_cached_decode_compare_code.json
cerebellum-dev/sparse-upcycling/runs/v1_cached_decode_compare_code_cpu_bf16.json
```

Findings:

- Dense source cached decode matches full-prefix decode over the 24-token code probe.
- v1 with `device_map=auto` diverges at step 13, after the requested function is complete.
- v1 CPU BF16 cached decode matches full-prefix decode through step 13.
- Therefore the bad cached behavior is from HF `device_map=auto` offload, not the checkpoint tensors and not the MoE cache logic in general.

## Training Smoke Attempt

Public data sample:

```text
/var/home/deucebucket/games/cerebellum-pipeline-tmp/qwen35_9b_moe/v1_smoke_smol_8.jsonl
```

Source:

```text
HuggingFaceTB/smol-smoltalk
```

Attempted command shape:

```bash
distrobox enter ai -- python3 cerebellum-dev/sparse-upcycling/scripts/train_stage1_expert_lora.py \
  /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v1-residual-dense-bridge \
  --data /var/home/deucebucket/games/cerebellum-pipeline-tmp/qwen35_9b_moe/v1_smoke_smol_8.jsonl \
  --output-dir /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v1-residual-lora-smoke-cpu \
  --steps 1 \
  --max-seq-len 24 \
  --device cpu \
  --dtype bfloat16 \
  --lora-rank 2 \
  --lora-alpha 4 \
  --lora-target both \
  --lora-gate-up-b-init-std 0.001 \
  --router-noise-std 0.01 \
  --lr 1e-4 \
  --log-every 1 \
  --save-adapter
```

Result:

- Model loaded on CPU BF16.
- LoRA attached.
- Trainable tensors: 192.
- Trainable params: 12,976,128.
- No adapter was saved.
- The one-step CPU train was manually killed after more than 7 minutes without completing.

Decision:

- CPU training is non-viable for v1 except tiny diagnostics.
- Next training work needs GPU/offload-aware strategy, QLoRA-style loading, FSDP/ZeRO, or a smaller bridge/training scaffold.

## Important LoRA Note

Zero expert banks make ordinary LoRA initialization dead if both LoRA B matrices
start at zero. `expert_lora.py` now supports nonzero B initialization. For v1,
use at least:

```text
--lora-gate-up-b-init-std 0.001
```

Keep down B at zero initially if the goal is zero initial residual output.

## Resume Point

Recommended next step:

1. Do not retry CPU training.
2. Decide GPU training strategy:
   - quantized/frozen shared expert loading,
   - FSDP/ZeRO/offload,
   - or a compact bridge checkpoint built specifically for residual-expert training.
3. Once a training path fits, run a 1-8 step residual-LoRA smoke with router noise.
4. Gate on:
   - loss decreases or at least backprop completes,
   - adapter saved,
   - no-cache probe still matches sane dense behavior,
   - routed residual norms become nonzero.

