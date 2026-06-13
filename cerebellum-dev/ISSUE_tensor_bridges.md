# Research Idea: Tensor Bridges / Tract Shortcuts

Status: spitball / research backlog

## Idea

Cerebellum row-block scans are starting to map behavior-specific tensor tracts:
coding, repair, tool JSON, completion, refusal, etc. If two distant tensor
regions repeatedly support the same behavior, maybe a trimmed model could add a
small learned bridge between them instead of preserving all surrounding weight
mass.

Mental model: a direct "cat5 cable" between two useful tracts. Not a metaphorical
prompt trick; a real added path in the computation graph.

## Concrete Version

Given activation slice A in layer X and activation slice B in layer Y:

1. read activation slice A
2. project it through a tiny learned matrix or low-rank adapter
3. add/gate that projection into slice B
4. train only the bridge on coding/agent traces
5. keep the bridge only if it beats simply preserving the original bands at
   higher precision

Possible forms:

- cross-layer low-rank adapter
- gated residual side path
- sparse activation injection
- MoE router/gate bias path
- backend-level custom bridge in llama.cpp

## Why It Might Matter

Quantization and pruning can weaken long-range task-state coherence. Coding and
agent repair often fail as drift: the model loses the plan, rewrites too much,
or stops using the right tool format. A bridge could carry the useful state from
one mapped tract to another after dead/noisy chunks are cut.

## Required Prerequisites

- row/block/voxel map of coding and repair-sensitive tracts
- proof that a candidate cut loses behavior
- proof that a small bridge restores behavior better than preserving the full
  original tensor region
- custom runtime path or adapter format capable of cross-layer injection

## First Experiment Shape

1. Use Qwen 3.5 9B row-block scans to identify repeated coding/repair tracts.
2. Build a trimmed candidate that intentionally loses one repair behavior.
3. Train a tiny adapter/bridge on strict agent traces.
4. Compare:
   - baseline Z
   - trimmed without bridge
   - trimmed with bridge
   - same size spent on preserving precision instead of adding bridge
5. Gate on strict agent loop, EvalPlus audit, and generic PPL.

## Risk

This changes model architecture/runtime assumptions, so it is not a near-term
GGUF-only shipping feature. It belongs after the tract map is real.
