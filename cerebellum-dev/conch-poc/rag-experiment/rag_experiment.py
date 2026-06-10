"""
Conch Brainloop RAG Experiment — self-contained FAISS retrieval stub.

This module builds a tiny FAISS index and defines the forward-only
projection path that a brainloop adapter would call: split a hidden
state into a query, search nearest neighbors, project the results
back, and inject them as an additive context signal.

Everything is self-contained here. No external config, no GPU
dependency (faiss-cpu). The stub can be imported and called with
arbitrary hidden states to verify the shapes are correct before
wiring it into the actual brainloop inference loop.
"""

from __future__ import annotations

import faiss
import numpy as np

HIDDEN_DIM = 256
CONTEXT_DIM = 2048
NUM_CORPUS = 100
TOP_K = 3


def _build_test_corpus(n: int = NUM_CORPUS, d: int = CONTEXT_DIM) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, (n, d)).astype(np.float32)


def build_index(corpus: np.ndarray | None = None) -> faiss.IndexFlatL2:
    if corpus is None:
        corpus = _build_test_corpus()
    corpus = np.ascontiguousarray(corpus.astype(np.float32))
    index = faiss.IndexFlatL2(corpus.shape[1])
    index.add(corpus)
    return index


def query_index(index: faiss.IndexFlatL2, q: np.ndarray, k: int = TOP_K) -> np.ndarray:
    q = np.ascontiguousarray(q.astype(np.float32))
    if q.ndim == 1:
        q = q.reshape(1, -1)
    distances, indices = index.search(q, k)
    return np.ascontiguousarray(index.reconstruct_batch(indices.ravel()).reshape(q.shape[0], k, -1))


class RAGStub:
    """Stateless stub that carries a FAISS index and the projection matrices."""

    def __init__(
        self,
        index: faiss.IndexFlatL2 | None = None,
        hidden_dim: int = HIDDEN_DIM,
        context_dim: int = CONTEXT_DIM,
        top_k: int = TOP_K,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.index = index if index is not None else build_index()
        self.W_query = rng.normal(0, 0.02, (context_dim, hidden_dim)).astype(np.float32)
        self.W_context = rng.normal(0, 0.02, (hidden_dim, context_dim)).astype(np.float32)
        self.top_k = top_k

    def inject_rag(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Project hidden states into query space, retrieve top-k context vectors,
        project them back to hidden space, and add them to the original states.

        Args:
            hidden_states: [..., HIDDEN_DIM] float32 tensor.

        Returns:
            hidden_states + context signal, same shape as input.
        """
        h = np.ascontiguousarray(hidden_states.astype(np.float32))
        original_shape = h.shape

        h_flat = h.reshape(-1, h.shape[-1]).T

        q = (self.W_query @ h_flat).T  # [batch_size, CONTEXT_DIM]

        retrieved = query_index(self.index, q, self.top_k)  # [batch_size, TOP_K, CONTEXT_DIM]

        retrieved_flat = retrieved.reshape(-1, retrieved.shape[-1]).T  # [CONTEXT_DIM, batch_size*TOP_K]
        ctx_flat = (self.W_context @ retrieved_flat).T  # [batch_size*TOP_K, HIDDEN_DIM]
        ctx = ctx_flat.reshape(retrieved.shape[0], retrieved.shape[1], -1)  # [batch_size, TOP_K, HIDDEN_DIM]

        ctx_aggregated = ctx.mean(axis=1)  # [batch_size, HIDDEN_DIM]

        ctx_full = ctx_aggregated.reshape(original_shape)

        return h + ctx_full


def smoke_test() -> None:
    """End-to-end test with random vectors — verifies shape contract only."""
    print("=== Conch Brainloop RAG Stub Smoke Test ===")

    corpus = _build_test_corpus()
    print(f"Corpus: {corpus.shape} (float32)")

    index = build_index(corpus)
    print(f"Index: {index.ntotal} vectors, dim={index.d}")

    stub = RAGStub(index=index)
    print(f"W_query: {stub.W_query.shape}")
    print(f"W_context: {stub.W_context.shape}")
    print(f"top_k: {stub.top_k}")

    hidden = np.random.default_rng(7).normal(0, 1, (1, 4, HIDDEN_DIM)).astype(np.float32)
    print(f"\nInput hidden_states shape: {hidden.shape}")

    result = stub.inject_rag(hidden)
    print(f"Output shape: {result.shape}")

    assert result.shape == hidden.shape, f"Shape mismatch: {result.shape} != {hidden.shape}"
    assert result.dtype == np.float32

    diff = result - hidden
    print(f"Context signal norm (mean): {np.linalg.norm(diff, axis=-1).mean():.6f}")
    print("SUCCESS: RAG stub runs end-to-end with correct shapes.")


if __name__ == "__main__":
    smoke_test()
