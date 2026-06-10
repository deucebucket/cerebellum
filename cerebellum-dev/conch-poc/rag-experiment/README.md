# Conch Brainloop — RAG Experiment Stub

Self-contained FAISS-based retrieval experiment for the conch brainloop project.

## What it does

- Builds a small FAISS flat-L2 index (100 random 256-dim vectors as a test corpus).
- Defines two projection matrices:
  - `W_query [2048, 256]` — maps hidden states into query space.
  - `W_context [256, 2048]` — maps retrieved vectors back to hidden state space.
- Provides `inject_rag(hidden_states)` that runs the full pipeline:
  1. Project hidden states → query vectors.
  2. FAISS top-3 search over the corpus.
  3. Project results back → context signal.
  4. Return `hidden_states + ctx` (additive injection).

The stub is importable so it can be wired into actual brainloop hidden state tensors later.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Smoke test

```bash
source venv/bin/activate
python rag_experiment.py
```

## Import

```python
from rag_experiment import RAGStub, build_index, query_index

stub = RAGStub()
output = stub.inject_rag(your_hidden_states)
```
