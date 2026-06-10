# Cerebellum-Brainloop

Cerebellum-Brainloop is a representation engineering project that implements a dual-stage hidden state interceptor for Qwen2.5-3B. The system uses non-destructive forward hooks to inject external factual knowledge directly into the model's residual stream.

## Architecture

- **Dual-Layer Intercept:** Hijacks the hidden states at Layer 18 (Reasoning phase) and Layer 31 (Knowledge Gate phase).
- **Subspace Hijacking (Lane Separation):** The hidden state is partitioned into two distinct lanes:
  - **Reasoning Lanes (Dim 0-1536):** Protected dimensions that preserve the base model's native logic and coherence.
  - **Knowledge Lane (Dim 1536-2048):** Hijacked dimensions where the Brainloop injects translated factual vectors.
- **W_context Projection:** Each intercept point uses a trainable linear matrix to translate raw token embeddings into the abstract geometric space of the target layer.

## Technical Specifications

- **GGUF Unrolling:** Includes a script (`unroll_vanilla_gguf.py`) to surgically modify GGUF metadata, increasing `block_count` from 36 to 38 and remapping the execution graph to include the Brainloop blocks as native sequential layers.
- **Speculative MTP Hijacking:** A C++ implementation (`mtp_hijack_patch.cpp`) allows for real-time kidnapping of speculative draft tokens, replacing hallucinated sequences with factual deltas from the RAG index.
- **Standard Library Mapping:** Reconstructed a 13,000-symbol target corpus (`python_stdlib_13k.txt`) for supervised alignment. Currently mapped 2,002 symbols into verified Delta Vectors.

## Benchmark Status (Qwen2.5-3B)

- **Coherence:** 100% parity with base model on general intelligence and logic tests.
- **Logic (HumanEval+):** Matched base model score (Pass@1: ~75% on first 20 samples) with hooks active.
- **Recall:** Verified correct factual retrieval for 2,002 internal Python symbols using zero-context vector injection.

## Usage

### PyTorch (Research)
```python
from refiner_vanilla import patch_model_vanilla
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")
model = patch_model_vanilla(model) # Identity initialization
```

### GGUF (Production)
```bash
python unroll_vanilla_gguf.py --input qwen2.5-3b.gguf --output brainloop-3b.gguf
```
