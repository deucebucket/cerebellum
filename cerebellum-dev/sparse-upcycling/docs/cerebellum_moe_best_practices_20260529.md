# Cerebellum MoE Best Practices

Date: 2026-05-29

Scope: what the broader MoE/upcycling literature implies for Cerebellum model
surgery, quantization, and small-footprint runtime design.

## Baseline Rule

MoE is not a quantization trick by itself. MoE only becomes useful for Cerebellum
after the model is generation-safe, routing is stable, expert sensitivity is
measured, and runtime/offload can exploit expert sparsity.

Do not run imatrix, ablation, quantization, or public benchmarks until direct
generation probes pass.

## What People Have Tried

### Native Sparse MoE Training

- Switch Transformer simplified sparse routing and showed large sparse models
  could train efficiently, but also made clear that routing/training stability
  is a core problem.
- GLaM showed sparse MoE can scale parameter count while lowering training and
  inference compute relative to dense alternatives.
- ST-MoE focuses on stable and transferable sparse expert models; it treats MoE
  stability and fine-tuning behavior as first-class design issues.
- OLMoE is a fully open sparse MoE release with model weights, data, code, logs,
  and routing analysis. It is valuable as a reference for telemetry and
  reporting, not just architecture.
- DeepSeekMoE adds fine-grained experts plus shared experts, showing that common
  knowledge and routed specialization should be separated.

### Dense-to-MoE Upcycling

- Sparse Upcycling copies dense checkpoints into sparse MoE structures and
  continues training. The classic safe version copies full dense FFNs into
  experts, not arbitrary top-2 slices.
- NVIDIA's upcycling study identifies the exact failure mode of naive
  fine-grained sharding: segmented experts plus top-k are not dense-equivalent
  unless routing selects the right shards and output scaling is corrected. Their
  remedies are virtual group initialization and weight scaling.
- LLaMA-MoE confirms decoder-only SwiGLU models do not have natural ReLU-style
  sparsity. Their successful models require continual pretraining and careful
  expert construction.
- Drop-Upcycling argues identical dense expert copies can slow specialization
  over long training; partial reinitialization helps after dense knowledge is
  safely transferred.
- UpIT and BTX use intermediate checkpoints, parameter merging, or domain-trained
  branches to create diverse experts before MoE routing.
- DS-MoE trains with dense/all-expert computation and uses sparse inference
  later, directly addressing under-trained sparse experts.
- MoEfication partitions FFNs into experts and trains routers, but its success
  relies on functional partitions and activation sparsity; it should not be read
  as permission to top-2 slice SwiGLU decoder FFNs without a bridge.

### Runtime, Compression, and Deployment

- QMoE shows MoE expert parameters can be extremely compressible when the
  compression format and execution kernels are co-designed.
- MoQE/GEMQ-style work treats expert quantization as expert-aware and
  mixed-precision, not flat layer quantization. GEMQ adds global allocation and
  router fine-tuning after quantization.
- Expert pruning/skipping papers show not all experts matter equally. Router
  hints, activation statistics, and task/domain specialization can guide pruning.
- MoE-Pruner uses router weights and activation information for one-shot pruning
  and expert-wise distillation.
- REAP-style router-weighted expert activation pruning explicitly combines
  router gate values and expert activation norms. This is aligned with
  Cerebellum's sensitivity-first approach, but it should be validated on
  generative tasks and rare domains before removing experts.
- MoE-I2 combines inter-expert pruning with intra-expert low-rank decomposition,
  which suggests a second-stage compression pass after expert-level Cerebellum
  quantization.
- HybriMoE and related offload work make the runtime lesson explicit: expert
  activation patterns are unstable, so CPU/GPU scheduling needs dynamic caching,
  prefetching, and impact-aware placement.
- fMoE-style serving uses fine-grained expert selection patterns and semantic
  prompt hints to guide prefetching/caching/offload. This maps directly to a
  future Cerebellum runtime profiler.
- MobileMoE shows that on-device MoE should be designed around active/total
  parameter tradeoffs, moderate sparsity, and shared/common expert paths rather
  than assuming server-scale MoE layouts transfer cleanly to small devices.

## Cerebellum Rules

### 1. Preserve Dense Behavior Before Sparsity

For dense upcycling, the first candidate must be dense-equivalent or
dense-preserving:

- full dense FFN duplicated into experts
- residual dense bridge through shared experts
- virtual-group sharded bridge
- dense/all-expert training before sparse inference

Do not start with raw top-2 vertical slicing unless there is a large continual
pretraining budget and a reason to accept severe initial degradation.

### 2. Treat SwiGLU Differently From ReLU FFNs

SwiGLU decoder FFNs do not naturally expose the same simple activation sparsity
used by older MoEfication-style work. Splitting intermediate channels and routing
top-2 can destroy the dense function even when tensor slicing is numerically
correct.

### 3. Use Shared/Common Experts Deliberately

Shared experts are useful for common knowledge. For Cerebellum upcycling, the
shared path can also be a safety bridge:

- start with shared expert equal to dense FFN
- train routed residual experts without breaking base behavior
- later quantify how much shared path can be quantized, gated down, or replaced

### 4. Stage Sparsity

Move through a ladder:

1. dense-equivalent bridge
2. dense/all-expert or high-top-k bridge
3. top-8/top-4
4. top-2
5. Cerebellum quantization
6. CPU/GPU expert offload

Each step needs generation, heldout loss, and router telemetry gates.

### 5. Router Telemetry Is Required

Record per-layer:

- expert activation counts
- router entropy
- top expert concentration
- co-activation pairs
- domain-conditioned routing
- routing drift after quantization

Without this, quantization/offload decisions are guesswork.

### 6. Expert Quantization Must Be Expert-Aware

Cerebellum should not use flat precision on MoE expert banks. Use:

- per-expert activation frequency
- per-expert/task sensitivity
- routed vs shared expert separation
- router/shared gate protection
- post-quant router drift checks

Low-activity experts are good low-bit/offload candidates only after sensitivity
confirms they are not rare-but-critical.

### 7. Prune and Merge Only After Routing Stabilizes

Pruning/skipping/merging is a deployment phase, not an upcycle bootstrap. Use it
after:

- generation passes
- routing is stable across domains
- expert importance has been measured
- distillation or repair data is available

### 8. Runtime Layout Matters

Pi-class or small-GPU MoE only works if the runtime can exploit expert locality:

- keep always-active dense/shared/attention tensors resident
- cache hot experts
- prefetch likely next-layer experts
- keep cold experts CPU/storage-backed
- measure hit/miss and latency by layer

If GGUF stores tensors layer-wise in a way that blocks expert-granular placement,
runtime layout work may be required before MoE offload reaches the goal.

### 9. Keep A Frontier Watchlist, But Do Not Build On Unverified Claims

Recent papers and preprints explore expert merging, buddy experts, speculative
expert prefetch, cacheless edge MoE, bit-sliced expert caches, and dynamic expert
quantization. These are promising for Cerebellum's long-term small-device goal,
but the immediate pipeline should only depend on mechanisms we can test locally:

- router activation telemetry
- expert sensitivity
- expert-wise quantization
- expert pruning/skipping
- CPU/GPU hit-rate measurement
- generation-gated staged sparsity

## Current Qwen3.5-9B Implications

The failed v0 proves one thing clearly: top-2 vertical slicing is not a safe
iteration-0 conversion for this model.

Best next path:

1. implement residual dense bridge
2. prove HF generation matches dense-source sanity
3. train routed residual experts with shared dense path frozen
4. collect router telemetry
5. only then quantize or ablate

Second path:

1. implement virtual-group fine-grained bridge
2. validate top-8 dense-equivalent behavior
3. train
4. stage down toward top-4/top-2

## Source Map

- Sparse Upcycling: https://arxiv.org/abs/2212.05055
- NVIDIA Upcycling LLMs into MoE: https://arxiv.org/abs/2410.07524
- LLaMA-MoE: https://arxiv.org/abs/2406.16554
- Drop-Upcycling: https://arxiv.org/abs/2502.19261
- UpIT: https://arxiv.org/abs/2410.01610
- Branch-Train-MiX: https://arxiv.org/abs/2403.07816
- DS-MoE: https://arxiv.org/abs/2404.05567
- MoEfication: https://arxiv.org/abs/2110.01786
- Expert Choice Routing: https://arxiv.org/abs/2202.09368
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
- Fine-Grained MoE Scaling Laws: https://arxiv.org/abs/2402.07871
- Switch Transformer: https://arxiv.org/abs/2101.03961
- GLaM: https://arxiv.org/abs/2112.06905
- ST-MoE: https://arxiv.org/abs/2202.08906
- OLMoE: https://arxiv.org/abs/2409.02060
- Not All Experts Are Equal: https://arxiv.org/abs/2402.14800
- REAP: https://arxiv.org/abs/2510.13999
- MoE-Pruner: https://arxiv.org/abs/2410.12013
- MoE-I2: https://arxiv.org/abs/2411.01016
- QMoE: https://arxiv.org/abs/2310.16795
- fMoE: https://arxiv.org/abs/2502.05370
- HybriMoE: https://arxiv.org/abs/2504.05897
- GEMQ: https://arxiv.org/abs/2605.23078
- MobileMoE: https://arxiv.org/abs/2605.27358
