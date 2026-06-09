# Inline RAG Refiner — Architecture Sketch

## Concept

The bolt-on refiner at layer 18 doesn't just polish hidden states. It acts as a
**runtime oracle** — during its 2-4 revolution loop, it reaches out to an
external knowledge index, retrieves relevant context, and injects it directly
into the model's representation space. No prompt stuffing. No context window
burn. The model asks itself what it needs, mid-thought.

## Architecture

```
Layer 0-17 (frozen) → [Refiner × N revolutions] → Layer 18-35 (frozen) → Output
                            │
                            ▼
                    ┌───────────────────────┐
                    │  Revolution Loop       │
                    │                        │
                    │  1. hidden             │
                    │  2. query = Wq @ h     │──→ FAISS index
                    │  3. docs = top-k       │←── (VRAM-pinned)
                    │  4. ctx = Wc @ docs    │
                    │  5. h = h + ctx        │
                    │  6. attn + FFN (refine)│
                    │  7. gate residual      │
                    │  8. goto 2 (if needed) │
                    └───────────────────────┘
```

### New Components (added to brainloop_gpu_cache)

```
W_query   [query_dim, n_embd]    ~512×2048 = 1.0M params
W_context [n_embd, context_dim]  ~2048×512 = 1.0M params
rag_gate  scalar                   1 param
```

Total: ~2M additional params. The full refiner goes from 33M → 35M. Still <2% of a
7B base model.

### The Query "Stethoscope"

The query projection isn't a text encoder. It's a **stethoscope** — it listens to
what the model's hidden state is struggling with and translates that into a
retrieval key.

At revolution 0, the hidden state is the raw output of layer 17. The model hasn't
"thought" yet. The query captures: *what does this token-in-progress need to know?*

At revolution 1, after attention+FFN refinement, the hidden state has already
absorbed context from the first retrieval. The new query captures: *now that I
know X, what additional detail would help?*

This is fundamentally different from RAG-at-the-prompt because the model is
querying from **inside its own reasoning loop**. Context from retrieval 0 changes
the question for retrieval 1. It's a conversation between the model and the index.

### The Index

Any corpus can be indexed. Embeddings are pre-computed using the same W_query
projection (trained jointly with the refiner) or a frozen encoder.

```
Index format:
  - vectors: [N_docs, query_dim] fp16     (1M docs × 256-dim = 512 MB)
  - ids:     [N_docs] int32               (4 MB)
  - metadata: e.g. file paths, line numbers
```

256 dimensions. 1 million documents. 512 MB VRAM. FAISS IVF-PQ index. <1ms
retrieval time. The index lives in pinned VRAM alongside the base model.

For code: index every function/class/method with its docstring and source.
For docs: index every paragraph/section with its context.
For math: index every theorem/lemma with its proof.

### Gate-Controlled Retrieval

The existing refiner gate already controls how much refinement to apply. The
RAG gate adds retrieval control:

```
h = h + sigmoid(gate_refine) * (refined - h)  // existing
h = h + sigmoid(gate_rag) * W_context @ docs   // new: retrieval injection
```

During training, both gates use STE (forced to 1.0, gradient passes through).
At inference: sigmoid(0) ≈ 0.5 for each.

But here's the kick: the RAG gate can be context-dependent. If the model is
confident (low entropy), it closes the gate → no retrieval needed. If uncertain
(high entropy), it opens the gate → "I need to look this up."

## Training

### Phase 1: Joint Index + Projection Training
- Freeze base model + existing refiner
- Train W_query, W_context, rag_gate only
- Loss: standard cross-entropy + retrieval-aware auxiliary loss
- Aux: does the retrieved context improve next-token prediction confidence?
- Corpus: WikiText for general, codebase for code, etc.
- Epochs: 3, batch_size: 4, LR: 1e-4, weight_decay: 0.1

### Phase 2: Full Refiner Retraining (optional)
- Unfreeze existing refiner weights
- Jointly train everything including W_query + W_context
- Lower LR (1e-5) to preserve existing refinement skill
- This teaches the refiner to USE retrieved context effectively

### Retrieval-Aware Loss
```
loss_rag = -log P(token | hidden + rag_context) + log P(token | hidden)
```
If the retrieved context makes the model MORE confident about the correct token,
this loss is negative → encourages retrieval. If context is irrelevant, the loss
is neutral → the gate can learn to ignore it.

## Inference Flow

```
rev 0:
  1. h = layer_17_output
  2. q = W_query @ h                    // [query_dim]
  3. indices = FAISS.search(q, k=4)     // top-4 docs
  4. docs = index[indices]              // [4, query_dim]
  5. ctx = W_context @ docs             // [4, n_embd] → aggregate to [n_embd]
  6. h = h + sigmoid(gate_rag) * ctx
  7. h = h + attn(ln1(h))               // refine with context
  8. h = h + ffn(ln2(h))
  9. h = hidden_original + gate * (h - hidden_original)

rev 1 (iterative refinement):
  10. q = W_query @ h                   // re-query after refinement
  11. indices = FAISS.search(q, k=4)
  12. ... same as above, but query is now informed by rev 0's retrieval
```

## Why This Wins

| | Prompt RAG | Inline RAG |
|---|---|---|
| Context cost | N tokens × d_model | d_model (fixed 2048) |
| Query mechanism | User writes prompt | Model's internal state |
| Retrieval point | Before generation | Mid-thought, per-layer |
| Iterative lookup | One shot | Per-revolution refinement |
| Training | Prompt engineering | Learned via gradient descent |
| Index latency | Before first token | During refinement (<1ms) |

The model can retrieve 4 documents per revolution, 2-4 revolutions, = 8-16
documents per token — all in the model's native representation space. No text
serialization. No prompt bloat. The model learns what information is worth
retrieving and how to integrate it.

## C++ Implementation Path

1. **Add W_query + W_context + rag_gate to brainloop_gpu_cache** (3 new tensors)
2. **Add FAISS index loading** — read pre-built index file at startup
3. **In the refiner loop:** compute query vector, call FAISS, inject context
4. **FAISS C++ API:** `faiss::IndexIVFPQ` with `faiss::METRIC_INNER_PRODUCT`
5. **Index search:** `index->search(nq=1, query_vec, k=4, distances, indices)`

The CPU/GPU boundary: query vector is computed on GPU, copied to CPU for FAISS
search (<1μs for 256-dim array), FAISS search takes <1ms on CPU, indices sent
back to GPU, embeddings looked up from VRAM. Total overhead: <2ms per revolution,
for a batch of 512 tokens = <4μs per token. 

## The Endgame

With the index in VRAM and the query projection in the refiner, you get:

- **Code-aware model**: Index your entire codebase. The refiner retrieves relevant
  functions/APIs as you code. No more hallucinated function signatures.
- **Domain expert**: Index scientific papers, legal docs, medical guidelines.
  The refiner pulls in relevant facts during generation.
- **Self-improving**: The index can be updated without retraining. Add new docs,
  rebuild the FAISS index, the model immediately benefits.
- **Multi-index routing**: Different W_query projections for different indices.
  The refiner learns which index to query based on the task.

A 7B model with a 5GB codebase index, retrieving inline during refinement, could
outperform a 70B model on code generation — because it has perfect recall of
every function signature, every API, every pattern in the codebase.
