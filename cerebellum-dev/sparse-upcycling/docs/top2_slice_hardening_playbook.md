# Top-2 Slice Hardening Playbook

Date: 2026-05-29

This is the recovery sequence for Qwen3.5-9B dense-to-MoE after the failed v0
top-2 sliced checkpoint. The goal is to avoid repeating the mistake of running
imatrix, ablation, quantization, or benchmarks before the model can generate.

## What Failed

The v0 path:

1. split each dense SwiGLU FFN into 16 intermediate-channel slices
2. exposed those slices as routed experts
3. activated only top-2 experts per token

That preserves stored dense weights, but it does not preserve the dense function.
Local math diagnostics prove all 16 slices reconstruct dense, while the actual
top-2 path is about `0.92-0.94` relative L2 away from dense.

## Hardening Sequence

### 1. Dense Source Control

Run `probe_generation.py` or an equivalent dense-source probe against the
original Qwen3.5-9B snapshot.

Pass condition:

- semantic non-empty completions
- no whitespace-only output
- no single-token punctuation loops

### 2. Dense-Equivalent MoE Bridge

Pick one bridge:

- residual dense bridge: shared expert exactly equals dense FFN; routed experts
  start as zero residuals
- virtual-group bridge: top-k selects one replica from every dense FFN shard at
  iteration 0
- dense/all-expert bridge: train with all experts active, sparsify later

Pass condition:

- HF generation looks like dense source before training
- upcycle math reports near-zero dense equivalence where applicable

### 3. Router Warmup Is Not Enough Alone

Loss improvement is useful but not a pass. v0 improved heldout loss and still
failed generation.

Pass condition:

- heldout loss does not regress badly
- generation probe still passes
- router usage is not collapsed

### 4. Expert Training

Prefer training routed residual capacity first while freezing the dense-preserved
path. If using duplicated or virtual-group experts, use enough batch/tokens to
avoid under-trained experts.

Candidate tricks from primary sources:

- virtual group initialization and weight scaling for fine-grained upcycling
- dense training then sparse inference
- load-balancing loss
- partial reinitialization only after a safe dense-preserving path exists

Pass condition:

- generation passes at every saved checkpoint
- repetition metrics do not spike
- no special-token/newline collapse

### 5. Staged Sparsification

Do not jump directly to top-2. Move in stages:

1. dense-equivalent/all-expert or residual bridge
2. top-8 or dense+residual top-2
3. top-4
4. top-2

Each stage gets a generation gate before any downstream work.

### 6. Only Then Cerebellum

Once a candidate answers normally:

1. generate imatrix
2. run MoE ablation
3. build Q4 control
4. build Cerebellum mixed-precision candidate
5. run direct probes
6. run public benchmarks

## First Implementation Target

Implement `qwen35_9b_moe_v1_residual_dense_bridge`:

- copy dense FFN into `shared_expert`
- scale shared `down_proj` by `2`
- leave `shared_expert_gate.weight = 0`
- initialize routed residual experts so their output is zero or near-zero
- verify HF generation before training

This uses the existing Qwen3.5 MoE shared expert structure and should give the
fastest path to a generation-safe checkpoint.
