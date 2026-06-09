# Research Prompt v2: Forcing Frozen Models to Internalize Injected Knowledge

Skip what works. We have code injection proven (+6% HumanEval, 3B: 36.6 → 42.7%,
7B: 0 → 56.7%). Architecture is stable: gate=0.50, rag_scale=0.62, training=27s.

The 7B base model outputs "pass" for every HumanEval+ problem without RAG.
The RAG injection is the ONLY reason it works. We control a 33M-param refiner
block at the model's midpoint and have free reign over the hidden state at
every layer. We are not passive observers.

## The Wall

Single novel fact injection (XR-777 canary) fails regardless of injection point
(embedding, layer 8, layer 17, layers 20-26, pre-LM head), scale (0.5 through 100),
index type (token embeddings, layer-17 hidden states), or document count (1-13K).
The model's trained parameters override every injection.

## What We Can Do That We Haven't Tried

We own a 33M-param trainable block that sits between layers 17-18. We can
modify any of its weights. We can inject at any layer. We control GGML ops
directly. We can do things no API-based system can:

- Inject directly into the KV cache (the model MUST attend to KV entries)
- Modify the LM head projection to recognize injected concepts
- Set specific token biases in the logits layer
- Replace entire layer outputs with our own computation
- Run a secondary forward pass through the refiner and merge results
- Hijack attention to force the model to look at injected context

## Questions (outside the box)

1. What if our injection doesn't look like noise? What if we inject tokens the
   model already knows — like "Dr. Elena Vasquez" as token embeddings directly
   into the residual stream — making the model "remember" saying it?
   
2. What if we don't inject facts at all? What if the refiner learns to DIRECTLY
   project hidden states to logit biases for specific tokens? Skip the hidden
   state entirely and go straight to token-level steering?

3. What if the refiner doesn't ADD to the hidden state but REPLACES it at
   certain token positions? Like "position 0 is now Dr. Elena Vasquez's
   biographical vector" and the remaining tokens self-attend to it?

4. What if we store the RAG index inside the refiner's OWN attention weights?
   Instead of an external index, bake the knowledge into the refiner's parameters
   through training. The refiner BECOMES the knowledge base.

5. What if we run the model TWICE - once without RAG to see what it would say,
   compute the difference, and inject a CORRECTION vector that counteracts the
   hallucination direction?

6. Could we use the model's own logit lens to verify injection success? Store
   target token IDs, project hidden states to logits at each layer, and inject
   a correction if the target token probability is below threshold?

7. What if we don't inject facts as text concepts at all? What if we treat
   knowledge as a shift in the model's ATTENTION PATTERNS? Inject query/key
   biases that make the model attend to fact-related token positions?

8. What if we train the refiner to be a "universal translator" that can convert
   ANY injected vector into the model's native representation space, regardless
   of domain? One refiner to rule them all?

## Hypothetical Architecture Sketches

### KV Cache Hijacking
Inject 1-4 fact entries directly into the KV cache. The model's subsequent
tokens MUST self-attend to them. The fact becomes part of the autoregressive
history with zero token cost (4 slots out of 131K). The refiner controls WHAT
goes in. No prompt needed.

### Logit Lasso
At each generation step, project hidden state to logits. If target fact tokens
have low probability, compute a gradient-descent correction to the hidden state
that increases their probability. Inject that correction. Like classifier
guidance in diffusion models, but for autoregressive decoding.

### Refiner-as-Extended-Memory
Train the refiner on a massive fact corpus with the RAG index active. The refiner's
attention weights learn to map "what the model is asking about" to "where the
answer lives in the index." At inference, the refiner retrieves and injects facts
as naturally as if they were in the model's training data. One-time training cost,
lifetime of knowledge injection.

### Token-Level Forcing
Inject the target text ("Dr. Elena Vasquez at Zurich Quantum Institute") as a
sequence of TOKEN EMBEDDINGS at specific positions in the residual stream. The
model sees known token vectors appearing in its hidden state. It treats them as
part of its own computation and continues generating from them.

## Constraints

Single 3090, 24GB VRAM. C++ GGML graph builder in llama.cpp. Frozen base model.
33M trainable refiner. No API calls, no cloud, no prompt engineering.
Knowledge must feel "ingrained" — invisible to the user, no context window bloat.
