# Native RAG: Model's Own Embeddings as Document Index Space

## Why This Works

The central insight: **document embeddings are in the model's native language because they're built from the model's token embedding layer (`tok_embd`)**. There is no "translation" problem.

### The Translation Problem (What We Avoid)

In a standard RAG adapter, retrieved document vectors live in some external space (e.g., 768-dim BERT embeddings) while hidden states live in the model's internal space (2048-dim for Qwen2.5-3B). You need projection matrices:

```
W_query  [2048, 256]  → project hidden → query space
W_context [256, 2048]  → project retrieved → hidden space
```

These matrices must be learned. They introduce a bottleneck, add parameters, and create an information loss pathway where the adapter must learn to "translate" between incompatible vector spaces.

### The Native Approach

Instead, we build document embeddings directly from the model's own `embed_tokens.weight`:

```
doc_vector = mean(embed_tokens[token_ids_of_document])
```

This produces a 2048-dim vector that lives in **exactly the same space** as the model's hidden states at any layer. The retrieval query is simply a hidden state (also 2048-dim). No projection matrices needed.

### The Injection Path

```
hidden_state (layer L, 2048-dim)
    → direct FAISS query (no W_query)
    → retrieve top-k document vectors (already 2048-dim, already in model space)
    → add: h_new = h + scale * doc_vector  (no W_context)
```

The retrieved vectors are **additive signals** that directly modify the hidden state representation in the model's own language. The scale factor acts as a learned gate during training, but for inference, a fixed scale like 0.1 works as a proof of concept.

### Design Properties

| Property | Standard RAG | Native RAG |
|----------|-------------|------------|
| Doc embedding space | External (e.g., BERT) | Model's tok_embd |
| Query projection | Learned W_query [D→d] | None (identity) |
| Context projection | Learned W_context [d→D] | None (identity) |
| Information loss | At both projections | Zero |
| Additional parameters | ~1M (for 2048/256) | 0 |
| Training required | Yes (projections) | Gate/scale only |

## Files

- `native_rag.py` — Self-contained implementation and smoke test
- `rag_experiment.py` — The original stub with random projections (for comparison)

## Running

```bash
source venv/bin/activate
python native_rag.py
```

The script extracts `model.embed_tokens.weight` from Qwen2.5-3B's safetensors without loading the full model (~600 MB memory for the embedding table in float32).

## Smoke Test Results

The test builds a 50-document FAISS index over synthetic domain-diverse paragraphs. Retrieval quality is verified: queries about "transformer attention" retrieve documents about attention mechanisms; "mitochondria" retrieves the mitochondrion document; "quantum computing" retrieves the quantum computing document.

All shape contracts verified:
- Retrieval: `[batch, seq, top_k, 2048]`
- Injection: `h_new = h + scale * retrieved_doc`
- Self-retrieval: each document retrieves itself as top-1
