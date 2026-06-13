# Dense-to-MoE Upcycling Research Notes

Date: 2026-05-29

Scope: dense Qwen3.5-9B to Cerebellum-friendly MoE. This note records primary
source findings and the resulting v1 direction after the failed top-2 sliced v0.

## Current Local Finding

The v0 checkpoint sliced each dense FFN into 16 routed experts and immediately
used top-2 routing. This is a fine-grained vertical split, not a dense-preserving
bridge.

Local math diagnostics show the slicing itself is correct:

- all 16 expert slices summed together reconstruct dense FFN output at about
  `6e-7` relative L2 on layers 0, 15, and 31
- the actual HF zero-router top-2 path is about `0.92-0.94` relative L2 away
  from dense

So the break is not tensor orientation. The break is that top-2 activation drops
most of the dense FFN function at iteration 0.

## Primary Source Takeaways

### Sparse Upcycling

Komatsuzaki et al. introduce sparse upcycling: initialize sparse MoE models from
dense checkpoints and continue training instead of training sparse models from
scratch. Their results show upcycled T5/ViT models outperform dense baselines and
sparse-from-scratch under reduced compute budgets.

Source: https://arxiv.org/abs/2212.05055

Relevance: upcycling is valid, but it assumes a continued training budget and a
safe initialization. It does not justify an immediate destructive top-2 sliced
FFN path.

### NVIDIA Upcycling LLMs Into MoE

He et al. explicitly identify the fine-grained upcycling failure mode we hit:
segmented experts are not functionally equivalent to dense unless the router
selects exactly one copy of each dense shard, and router scaling can shrink the
output severely. They propose virtual group initialization plus weight scaling.
They also find softmax-then-topK routing and higher granularity can improve
accuracy, and they report an upcycled Nemotron-4 15B result above continued dense
training on the same token budget.

Source: https://arxiv.org/abs/2410.07524

Relevance: our 16-expert top-2 model is the naive fine-grained case. A safer
fine-grained bridge is something like `G=8` dense shards, `E=2` replicas per
shard, `T=8` active experts, virtual-group router init, and scaling so the first
forward pass matches dense.

### LLaMA-MoE

Zhu et al. build MoE models from LLaMA-2 7B by expert construction plus
continual pre-training. They emphasize that changing dense FFNs to sparse MoE
causes an immediate performance decline, especially because SwiGLU FFNs do not
have the natural ReLU sparsity exploited by older MoEfication work. Their
successful models use large continual-pretraining budgets, e.g. 200B tokens.

Source: https://arxiv.org/abs/2406.16554

Relevance: short router/LoRA warmups are not enough to repair a broken sparse
SwiGLU surgery. If we do a vertical split, it needs either a dense-equivalent
start or a serious training budget.

### Dense Training, Sparse Inference

Pan et al. propose training MoE with dense computation across all experts and
only using sparse computation at inference. They report dense-comparable quality
with lower inference compute and active parameter fractions around 30-40%.

Source: https://arxiv.org/abs/2404.05567

Relevance: this supports a staged bridge. First train with dense/all-expert or
dense-preserving behavior, then sparsify inference. That is a better match for
our failed top-2 transition than more immediate top-2 LoRA.

### Drop-Upcycling

Nakamura et al. show that copying dense weights into all experts helps early but
can slow long-term specialization; partial re-initialization can improve
long-run MoE learning.

Source: https://arxiv.org/abs/2502.19261

Relevance: once we have a generation-safe bridge, expert diversity matters. For
our immediate rescue, do not reinitialize the dense-preserving path. Consider
partial reset only for residual routed experts after the model answers normally.

### DeepSeekMoE

DeepSeekMoE combines fine-grained expert segmentation with shared experts to
capture common knowledge and reduce routed expert redundancy.

Source: https://arxiv.org/abs/2401.06066

Relevance: shared/common expert design is not a hack. It is a known way to keep
common behavior stable while routed experts specialize. For us, the shared expert
can initially preserve the dense FFN exactly.

### Expert Choice Routing

Expert-choice routing lets experts select tokens, improving load balance and
training convergence compared with token-choice top-k in their experiments.

Source: https://arxiv.org/abs/2202.09368

Relevance: Qwen3.5 MoE runtime uses token-choice top-k, so expert choice is not a
drop-in for llama.cpp. But its lesson matters: early routing/load balance is a
training stability problem, not a detail.

### MoEfication

MoEfication splits FFNs into functional partitions and builds routers, reporting
large FFN compute reductions while retaining much of original downstream
performance.

Source: https://arxiv.org/abs/2110.01786

Relevance: it supports functional partitioning, but the paper's setting benefits
from activation sparsity not naturally present in SwiGLU decoder FFNs. LLaMA-MoE
explicitly calls this out.

## v1 Candidate Designs

### Candidate A: Residual Dense Bridge

Use the existing Qwen3.5 MoE structure to preserve dense behavior exactly at
iteration 0:

- shared expert stores the original dense FFN
- shared expert gate weights are zero, so sigmoid gate is `0.5`
- shared expert down projection is scaled by `2`, so `0.5 * shared == dense`
- routed experts are residual capacity initialized to zero or very small values
- router can start random because routed residual output is initially zero

Pros:

- should pass generation immediately if implemented correctly
- uses Qwen3.5 MoE shared expert machinery
- gives a safe place to train routed residual experts without damaging base
  language behavior

Cons:

- active compute initially includes dense FFN plus routed residuals
- it is a bridge, not the final small-footprint sparse model
- later training must reduce dependence on the dense shared path

### Candidate B: Virtual-Group Fine-Grained Bridge

Implement the NVIDIA-style fine-grained dense-equivalent start:

- split dense FFN into `G` shards
- create `E` replicas per shard
- set top-k `T=G`
- initialize router so top-k selects exactly one replica from every shard
- apply weight scaling so output magnitude matches dense

For 16 experts, a plausible bridge is:

- `G=8`
- `E=2`
- `num_experts=16`
- `top_k=8`
- `expert_intermediate_size=1536`
- active routed width = `8 * 1536 = 12288`, matching dense

Pros:

- directly addresses our measured top-2 failure
- becomes sparse by lowering top-k after training
- closer to literature on fine-grained upcycling

Cons:

- requires new upcycler/router initialization logic
- initial active compute is dense-equivalent, not smaller
- Qwen3.5 MoE/llama.cpp top-k/runtime assumptions need checking for top-k 8

## Recommendation

Build Candidate A first. It is the fastest generation-safe path because it uses
the existing shared expert pathway to make the first forward pass dense
equivalent. After it passes generation probes, train residual routed experts and
measure whether the dense shared path can be quantized, gated down, or partially
replaced.

Build Candidate B second if Candidate A proves too dense at inference or if we
need a cleaner sparse end state.

## Hard Gate

Every candidate must pass:

1. dense source prompt probe
2. HF candidate prompt probe
3. merged HF prompt probe
4. F16 GGUF prompt probe

Only then run imatrix, ablation, quantization, or public benchmarks.
