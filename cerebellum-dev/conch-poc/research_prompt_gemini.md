# Research Prompt: Making Frozen Models Internalize Injected Knowledge

**WHAT WORKS (skip these):**

Architecture: Frozen Qwen2.5-3B. 33M-param refiner block (attention+FFN+gated
residual) at layer 17. 1-2 loop passes. Straight-through estimator gate
(train=1.0, eval=sigmoid to 0.5). RAG index: pre-embedded document vectors
injected via softmax-weighted retrieval. Refiner trained with AdamW, lr=1e-4,
weight_decay=0.1, 3 epochs, batch_size=4. Training takes 27 seconds on 3090.

Proven results:
- HumanEval+: 36.6% to 42.7% (3B), 56.1% to 56.7% (7B). Index: 173 curated Python solutions.
- PPL: 8.58 to 8.19 (-4.5%) on WikiText with refiner but no RAG.
- Architecture scales cleanly from 3B (2048-dim) to 7B (3584-dim) with zero code changes.
- C++ port in llama.cpp: native GGML graph builder, GPU-allocated refiner weights.
- RAG scale learns to 0.62 across ALL training runs (WikiText, code, 3B, 7B). Gate settles at 0.50.

**WHAT FAILS (focus here):**

Injecting novel facts (XR-777 canary test). Single canary doc in index. Model
queried about XR-777. Always hallucinates generic answers regardless of:

- Injection point tested: embedding level, layer 8, layer 17, before LM head
- Scale tested: 0.5, 2.0, 10.0, 20.0
- Index type tested: token embeddings (embed_tokens), layer-17 hidden states (full forward pass)
- Document count tested: 1, 3, 6

Code injection works because Python syntax was in the model's training distribution.
The model naturally outputs Python, so injected Python patterns reinforce existing
tendency. Novel facts have no distributional support and get "corrected" by the
model's 3B trained parameters.

**WHAT I NEED:**

A technique that makes a frozen model accept injected hidden-state knowledge as
truth, without full training on that knowledge domain. The refiner training proved
the model CAN learn to use injections (+6% HumanEval). I need the same outcome for
facts but without training on every fact I want to inject.
