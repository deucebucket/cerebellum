# RESEARCH PROMPT: Brainloop Speed Bumps & Experimental Paths

**Directive:** Treat all current architectural "blockers" purely as speed bumps. We are looking for unconventional, mathematically sound workarounds to achieve our dual goals: 1) Zero degradation to base reasoning, and 2) 100% Vanilla `llama.cpp` compatibility (no custom C++ required).

## Speed Bump 1: The Vanilla Compatibility Paradox
**Context:** We want to bake our 38-layer unrolled model into a standard GGUF. However, vanilla `llama.cpp` executes standard residual additions (`h = h + layer(h)`). It does not natively support our `tanh` gate or our "Subspace Hijacking" (which restricts modifications to only the last 25% of the hidden dimensions).
**The Question:** How can we mathematically trick a standard transformer architecture into performing Subspace Hijacking and gated identity-priors without altering the C++ inference code?

**Experimental Paths to Explore:**
1. **Weight-Level Subspace Masking:** Instead of using Python/C++ logic to isolate the Knowledge Lane (Dims 1536-2048), can we surgically zero out the top 75% of the rows/columns in the Refiner's `o_proj` (output projection) and `down_proj` matrices? If the weights for the reasoning lanes are literally `0.0`, the standard `h = h + layer(h)` operation will mathematically leave the reasoning lane untouched.
2. **Implicit Gating via Pre-Scaling:** Vanilla `llama.cpp` doesn't have a dynamic gate parameter. Can we emulate a `gate = 0.01` identity prior by statically scaling down the baked weights of the refiner's `o_proj` by `0.01` before GGUF export? Will this allow the network to train with a high learning rate but execute silently in production until triggered?
3. **Norm Shifting:** Does inserting a cloned layer alter the running RMSNorm statistics of the residual stream? Do we need to bake a compensating inverse-norm into the base model's Layer 18 to account for the Refiner's presence?

## Speed Bump 2: The 5% Logic Degradation
**Context:** We fixed the Causal Masking bug by using a perfect `deepcopy` of the `Qwen2DecoderLayer`, but any modification to the residual stream still risks a slight "logic tax" (~5% drop on HumanEval+).
**The Question:** If the architecture is identical, why does an identity-prior clone still degrade logic, and how do we achieve absolute 100% parity before knowledge injection?

**Experimental Paths to Explore:**
1. **True Zero-Init (The Dead Block):** Instead of cloning Layer 17 (Identity Prior), what if we insert a "Dead Block" where `o_proj` and `down_proj` are initialized to absolute zero? A zero-block guarantees 100% logic parity on day one. Can we successfully train a Dead Block to predict Delta Vectors, effectively waking it up only for knowledge retrieval?
2. **Attention Deactivation:** Does the refiner even need Self-Attention? Factual recall has been shown to be predominantly an FFN (Feed-Forward Network) mechanism. What happens if we zero out the `q_proj`, `k_proj`, and `v_proj` in the Refiner, leaving only the SwiGLU FFN to act as a pure associative memory lookup for our Delta Vectors?
3. **Prompt-Conditional Routing:** Can we train the refiner's internal attention heads to act as a router, attending *only* to the specific RAG-injected tokens and ignoring the native context window entirely?

**Execution Mandate:** Do not accept "impossible" as an answer. Design PyTorch scripts to empirically test Weight-Level Subspace Masking and True Zero-Init immediately.

## Roadmap: Vanilla First, Fork Later, Upstream Last

Sequencing for the injection work, in order of dependency:

1. **Now — vanilla dead blocks.** Attention-dead, subspace-masked FFN refiner blocks that are mathematically exact under standard `h = h + layer(h)` execution. No gate approximation, no custom C++. Every published artifact must load and run on stock llama.cpp, and every published number must be measured there (PyTorch-path numbers are iteration-only and labeled as such).
2. **Next — a measured llama.cpp fork for live inline data.** Runtime retrieval injection and per-layer bias wires need engine support. Verified 2026-06-11: vanilla qwen2 in llama.cpp neither loads nor applies optional FFN/attention bias tensors, so there is no code-change-free path today. A fork is the experiment bed, not the product.
3. **Later — upstream.** The qwen2 graph builder already consults `attn_output.bias` if present; only the loader line is missing, and the llama/Granite arch already treats these biases as optional. If the dead-block models prove the use case, a small parity PR to llama.cpp would make bias wires vanilla-legal in future releases. Adoption first, then the ask.
