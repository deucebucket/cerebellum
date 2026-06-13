# Cerebellum Sparse Upcycling

Private experiment workspace for converting a popular dense base model into a
Cerebellum-friendly sparse/MoE model.

Initial target:

- Source: `Qwen/Qwen3.5-9B`
- Goal: preserve the Qwen3.5-9B backbone, replace selected dense MLP blocks with
  sliced routed experts, then apply Cerebellum mixed-precision quantization.
- First proof: parameter/shape planner and a minimal upcycler scaffold before any
  large checkpoint rewrite.

This lives in `cerebellum-dev` because it is exploratory training and conversion
work. Do not push this to the public `origin` remote.

## Current Shape Hypothesis

Qwen3.5-9B text backbone:

- 32 layers
- hidden size 4096
- dense MLP intermediate size 12288
- SwiGLU MLP weights: `gate_proj`, `up_proj`, `down_proj`

First MoE target:

- 16 routed experts
- top-2 routing
- expert intermediate size 768
- optional shared expert size 1536
- routed expert storage per layer stays comparable to the original dense MLP:
  `16 * 768 = 12288`

This gives active routed MLP width of `2 * 768 = 1536` per token, plus shared
expert width if enabled. The point is not to make the HF checkpoint smaller
before quantization; it is to create structured sparsity that Cerebellum and
runtime paging can exploit.

## Current Artifacts

Base upcycled checkpoint:

```text
/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0
```

Stage-0 mixed router delta:

```text
/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage0-router-mixed-256step/router_delta.safetensors
```

Stage-1 rank-4 fused expert-LoRA adapter:

```text
/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step/adapter_delta.safetensors
```

Merged normal HF checkpoint overlay for conversion/eval:

```text
/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-merged
```

F16 GGUF:

```text
/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf
```

Osmosis imatrix:

```text
/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat
```

Cerebellum prior quant artifacts:

```text
cerebellum-dev/sparse-upcycling/runs/cerebellum_moe_prior_v0_q2k.txt
cerebellum-dev/sparse-upcycling/runs/cerebellum_moe_prior_v0_q2k_manifest.json
cerebellum-dev/sparse-upcycling/runs/cerebellum_moe_prior_v0_q3k.txt
cerebellum-dev/sparse-upcycling/runs/cerebellum_moe_prior_v0_q3k_manifest.json
cerebellum-dev/sparse-upcycling/runs/cerebellum_moe_prior_qwenrole_v1_q2k.txt
cerebellum-dev/sparse-upcycling/runs/cerebellum_moe_prior_qwenrole_v1_q2k_manifest.json
cerebellum-dev/sparse-upcycling/runs/cerebellum_moe_prior_qwenrole_v1_q3k.txt
cerebellum-dev/sparse-upcycling/runs/cerebellum_moe_prior_qwenrole_v1_q3k_manifest.json
/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-cerebellum-prior-q2k.gguf
/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-cerebellum-prior-q3k.gguf
/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-q4km-control.gguf
```

Heldout public loss at 16 rows/domain, max sequence length 96:

| Domain | Base | Stage-0 router | Stage-1 LoRA 256 |
|---|---:|---:|---:|
| Smol-SmolTalk | 9.0816 | 7.3584 | 4.5995 |
| OpenThoughts3 | 8.9547 | 7.0567 | 4.9385 |
| OpenCodeReasoning | 8.2693 | 6.9915 | 4.6621 |

One-chunk Wikitext GGUF smoke, context 512, full GPU offload:

| Variant | GGUF size | CUDA model buffer | PPL |
|---|---:|---:|---:|
| F16 stage-1 | 18G | 16300.55 MiB | 428.5950 |
| Cerebellum prior Q2_K base | 4.6G | 4112.01 MiB | 1161.4534 |
| Cerebellum prior Q3_K base | 4.9G | 4373.26 MiB | 743.5131 |
| Q4_K_M control | 5.6G | 5169.51 MiB | 661.6246 |

These are smoke numbers, not publishable benchmarks. They show that flat/tiny
base quantization is too aggressive for the current stage-1 checkpoint, while a
Q3_K base with Qwen3.5 tensor priors gets close to the Q4_K_M control for
roughly 700 MiB less VRAM.

Four-chunk public heldout diagnostic PPL, context 512, full GPU offload:

| Variant | Smol-SmolTalk | OpenThoughts3 | OpenCodeReasoning |
|---|---:|---:|---:|
| F16 stage-1 | 180.4969 | 135.8197 | 70.8036 |
| Cerebellum prior Q3_K base | 297.4809 | 287.4752 | 119.1644 |
| Q4_K_M control | 247.8093 | 187.8024 | 86.4087 |

Historical note: before direct generation probes, Q4_K_M looked like a plausible
benchmark control from PPL alone. That is now superseded by the generation gate
failure below. The current v0 line is not benchmarkable until a fresh candidate
can answer basic prompts.

## MoE Ablation Smoke

First fresh MoE-specific Q4_K_M -> Q3_K tensor ablation, context 512, two
chunks/domain:

Baseline PPL:

| Chat | Reasoning | Code |
|---:|---:|---:|
| 249.5304 | 171.2806 | 97.7624 |

Per-tensor PPL deltas:

| Tensor | Chat | Reasoning | Code |
|---|---:|---:|---:|
| `blk.0.ffn_gate_up_exps.weight` | +3.1127 | +2.2973 | +0.1144 |
| `blk.0.ffn_down_exps.weight` | +3.0687 | +6.9374 | -0.0763 |
| `blk.11.ffn_gate_up_exps.weight` | +1.7102 | +0.0424 | -0.8680 |
| `blk.11.ffn_down_exps.weight` | +0.7275 | +1.8281 | -0.7538 |
| `blk.31.ffn_gate_up_exps.weight` | +0.4870 | -0.0304 | -0.3588 |
| `blk.31.ffn_down_exps.weight` | -1.3311 | +0.1339 | +0.4578 |
| `blk.11.ffn_gate_shexp.weight` | +0.4533 | -0.0460 | +0.2476 |
| `blk.11.ffn_down_shexp.weight` | +2.5547 | +0.1550 | -0.2024 |

Initial read:

- Dense Qwen3.5 FFN priors are not enough for the upcycled MoE expert bank.
- The larger native-MoE prior transfers better at the role level: routed expert
  tensors are generally more fragile than shared expert gate tensors.
- Sensitivity is strongly depth-dependent. Early routed experts, especially
  `blk.0.ffn_down_exps`, are fragile; mid/late routed experts are much more
  demotable.
- Next allocation should use fresh MoE ablation data as the deciding signal,
  with dense Qwen3.5 data kept as SSM/attention/output guardrails.

## Generation Gate Failure

The current v0 line is not benchmarkable. Standard benchmark scoring failed
because the model emitted empty, whitespace-only, or repeated punctuation/token
outputs. This reproduced in three places:

- Q4_K_M ablation candidate in `llama-server`
- Q4_K_M control and F16 GGUF in `llama-server`
- merged stage-1 HF checkpoint through Transformers

Dense source sanity check passed on the same prompts, so the break is in the
dense-to-MoE surgery path, not Qwen3.5-9B itself, GGUF conversion, llama.cpp, or
Cerebellum quantization.

Reusable probes:

```bash
python cerebellum-dev/sparse-upcycling/scripts/probe_generation.py \
  /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0 \
  --max-new-tokens 48 \
  --output cerebellum-dev/sparse-upcycling/runs/probe_base_upcycled.jsonl
```

```bash
python cerebellum-dev/sparse-upcycling/scripts/diagnose_upcycle_math.py \
  /var/home/deucebucket/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0 \
  --layer 0 \
  --output cerebellum-dev/sparse-upcycling/runs/diagnose_upcycle_math_layer0.json
```

Current math diagnosis:

| Layer | All Experts Sum vs Dense | HF Zero-Router Top-2 + Half Shared vs Dense | Scaled Top-2 vs Dense |
|---:|---:|---:|---:|
| 0 | 0.0000006 | 0.9224 | 1.3012 |
| 15 | 0.0000007 | 0.9317 | 1.3523 |
| 31 | 0.0000007 | 0.9380 | 1.3493 |

Interpretation: tensor slicing is correct; summing all 16 expert slices
reconstructs the dense FFN. The actual top-2 MoE path is the destructive step.
Do not run more imatrix, ablation, quantization, or public benchmarks on this
line until a candidate passes HF generation probes.

## v1 Direction

Research note:

```text
cerebellum-dev/sparse-upcycling/docs/dense_to_moe_upcycling_research_20260529.md
cerebellum-dev/sparse-upcycling/docs/cerebellum_moe_best_practices_20260529.md
```

Next candidate is the residual dense bridge:

```text
cerebellum-dev/sparse-upcycling/configs/qwen35_9b_moe_v1_residual_dense_bridge.json
```

## v1 Residual Dense Bridge

Built checkpoint:

```text
/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v1-residual-dense-bridge
```

Shape/manifest status:

- 888 tensors, 7 safetensor shards, 28,487,768,032 bytes
- 32 converted MLP layers
- zero dense MLP leftovers
- zero MTP leftovers
- routed experts initialized to zero residual tensors
- shared expert contains the full dense FFN with `down_proj * 2`
- `shared_expert_gate.weight = 0`, so the runtime multiplier is sigmoid(0)=0.5

Math gate:

| Layer | Dtype | Half Shared vs Dense | HF Zero-Router + Half Shared vs Dense |
|---:|---|---:|---:|
| 0 | fp32 | 0.0 | 0.0 |
| 15 | fp32 | 0.0 | 0.0 |
| 31 | fp32 | 0.0 | 0.0 |
| 31 | bf16 | 0.0 | 0.0 |

Generation gate:

- Raw HF no-cache generation passes the dense-equivalent code probe:
  `return x + 1`
- Raw HF cached generation with `device_map=auto` is not a valid gate for this
  oversized bridge. Cached-vs-full-prefix decode diverged after the requested
  function was complete. Dense cached decode does not diverge, while v1 CPU BF16
  cached decode also matches full-prefix decode through the same point. Current
  read: HF auto offload is the unreliable piece, not the checkpoint tensors.
- Chat-template generation is not yet acceptable; do not run imatrix,
  ablation, GGUF conversion, quantization, or public benchmarks from v1 until
  the runtime/generation path is hardened.

Reusable artifacts:

```text
cerebellum-dev/sparse-upcycling/runs/qwen35_9b_moe_v1_residual_bridge_layer0_math.json
cerebellum-dev/sparse-upcycling/runs/qwen35_9b_moe_v1_residual_bridge_layer15_math.json
cerebellum-dev/sparse-upcycling/runs/qwen35_9b_moe_v1_residual_bridge_layer31_math.json
cerebellum-dev/sparse-upcycling/runs/qwen35_9b_moe_v1_residual_bridge_layer31_math_bf16.json
cerebellum-dev/sparse-upcycling/runs/probe_v1_residual_dense_bridge_no_cache.jsonl
cerebellum-dev/sparse-upcycling/runs/dense_qwen35_9b_next_logits_code.json
cerebellum-dev/sparse-upcycling/runs/v1_residual_bridge_next_logits_code.json
cerebellum-dev/sparse-upcycling/runs/dense_qwen35_9b_next_logits_code_return.json
cerebellum-dev/sparse-upcycling/runs/v1_residual_bridge_next_logits_code_return.json
cerebellum-dev/sparse-upcycling/runs/v1_cached_decode_compare_code.json
cerebellum-dev/sparse-upcycling/runs/dense_cached_decode_compare_code.json
cerebellum-dev/sparse-upcycling/runs/v1_cached_decode_compare_code_cpu_bf16.json
cerebellum-dev/sparse-upcycling/runs/v1_residual_bridge_status_20260529.md
```

Training note: CPU BF16 residual-LoRA smoke loaded and attached 12.98M trainable
params, but one 24-token step did not finish after more than 7 minutes and was
killed. Do not retry CPU training except for diagnostics; v1 needs a
GPU/offload-aware training plan.

The intended iteration-0 behavior is exact dense preservation through the shared
expert path: copy the dense FFN into `shared_expert`, set shared gate weights to
zero so `sigmoid(0) = 0.5`, and scale the shared down projection by 2. Routed
experts start as zero/small residual capacity. This should answer normally
before training.

Secondary candidate:

```text
cerebellum-dev/sparse-upcycling/configs/qwen35_9b_moe_v1_virtual_group_bridge.json
```

This follows the virtual-group fine-grained upcycling idea: shard the dense FFN,
replicate shards, initialize routing so top-k selects one replica per shard, and
then sparsify after the model passes generation gates.

## Commands

Run the shape planner:

```bash
python cerebellum-dev/sparse-upcycling/scripts/plan_sparse_upcycle.py \
  cerebellum-dev/sparse-upcycling/configs/qwen35_9b_moe_v0.json
```

Train stage-1 expert LoRA from the stage-0 router delta:

```bash
python cerebellum-dev/sparse-upcycling/scripts/train_stage1_expert_lora.py \
  /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0 \
  --data /var/home/deucebucket/games/cerebellum-pipeline-tmp/qwen35_9b_moe/mixed_router_train_1024.jsonl \
  --router-delta /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage0-router-mixed-256step/router_delta.safetensors \
  --steps 64 \
  --max-seq-len 96 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --lora-target both \
  --gradient-checkpointing \
  --save-adapter
```

Merge a saved adapter into a normal checkpoint overlay:

```bash
python cerebellum-dev/sparse-upcycling/scripts/create_expert_lora_merged_checkpoint.py \
  /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0 \
  --adapter-delta /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step/adapter_delta.safetensors \
  --output-dir /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-merged
```

Convert merged checkpoint to F16 GGUF:

```bash
python /var/home/deucebucket/ai-drive/llama.cpp/convert_hf_to_gguf.py \
  /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-merged \
  --outfile /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf \
  --outtype f16 \
  --fuse-gate-up-exps
```

Note: this local llama.cpp converter was patched to recognize the
upcycled Qwen3.5 base tokenizer pre-tokenizer hash as `qwen35`.

Generate the imatrix from the merged HF checkpoint:

```bash
python -m osmosis.imatrix_stream \
  --model /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-merged \
  --output /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat \
  -v
```

The streamer has local support for Qwen3.5 MoE fused expert tensors and writes
entries compatible with llama.cpp's `tensor->ne[0] * tensor->ne[2]` imatrix
shape rule.

Synthesize Qwen3.5-9B dense tensor priors for the upcycled MoE:

```bash
python cerebellum-dev/sparse-upcycling/scripts/synthesize_moe_cerebellum_prior.py \
  --base-quant Q3_K \
  --output cerebellum-dev/sparse-upcycling/runs/cerebellum_moe_prior_v0_q3k.txt \
  --manifest cerebellum-dev/sparse-upcycling/runs/cerebellum_moe_prior_v0_q3k_manifest.json
```

Synthesize the native-Qwen-MoE role-prior variant:

```bash
python cerebellum-dev/sparse-upcycling/scripts/synthesize_moe_cerebellum_prior.py \
  --base-quant Q3_K \
  --moe-ablation-prior osmosis-qwen35-122b/ablation/ablation_results.json \
  --output cerebellum-dev/sparse-upcycling/runs/cerebellum_moe_prior_qwenrole_v1_q3k.txt \
  --manifest cerebellum-dev/sparse-upcycling/runs/cerebellum_moe_prior_qwenrole_v1_q3k_manifest.json
```

The native MoE prior is transferred by tensor role, not exact layer number.
`ffn_gate_exps` and `ffn_up_exps` map onto this model's fused
`ffn_gate_up_exps`; `ffn_down_exps` maps directly. Demotion priors are recorded
in the manifest but not applied automatically.

Build a conservative Q4_K_M candidate from live ablation analyses:

```bash
python cerebellum-dev/sparse-upcycling/scripts/synthesize_q4km_demotions.py \
  --analysis cerebellum-dev/sparse-upcycling/runs/moe_ablation_routed_q4km_to_q3_analysis.json \
  --analysis cerebellum-dev/sparse-upcycling/runs/moe_ablation_routed_q4km_to_q3_reason_profile_analysis.json \
  --candidate-name q4km_ablation_balanced_reason_intersection_q3 \
  --output-dir cerebellum-dev/sparse-upcycling/runs \
  --imatrix /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat \
  --source-gguf /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf \
  --output-gguf /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-q4km-ablation-intersection-q3.gguf
```

The candidate planner defaults to the conservative intersection of `demotable`
tensors across all supplied analyses and removes any tensor that is
`sacred`/`critical` in any profile.

Dataset policy:

- Reuse public datasets first; do not recreate a dataset locally.
- Keep local data work to sampling manifests, schema adapters, and small smoke
  caches.
- See `docs/dataset_research_20260529.md` and
  `configs/dataset_manifest_v0.json`.
