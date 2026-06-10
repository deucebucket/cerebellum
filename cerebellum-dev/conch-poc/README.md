# Cerebellum-Brainloop

Cerebellum-Brainloop implements a dual-stage hidden state interceptor for Qwen2.5-3B. The system uses non-destructive forward hooks to inject external factual knowledge directly into the model's residual stream.

## Technical Facts

- **Architecture:** Dual-Layer Interceptor at Layer 18 and Layer 31.
- **Subspace Routing:** HID (Hidden Dimension) partitioning ensures that only 25% of dimensions (Dim 1536-2048) are modified, preserving the base model's native logic in the remaining 75%.
- **Zero-Context Injection:** Knowledge is fused via Delta Vectors extracted from "Knowing" vs "Ignorant" model states.
- **Vanilla Compatibility:** GGUF surgery script (`unroll_vanilla_gguf.py`) enables execution on standard `llama.cpp` releases without custom C++ forks.

## Verified Benchmarks (Qwen2.5-3B)

### Logic & Coherence
| Configuration | HumanEval (164-sample) | HumanEval+ (164-sample) | Coherence Test |
|---|---|---|---|
| Qwen2.5-3B Baseline | 62.2% | 56.1% | Pass |
| **Brainloop (Hooks Active)** | **56.7% (-5.5%)** | **51.2% (-4.9%)** | **Pass** |

*Note: While the intercept mechanism prevents catastrophic token looping ("lobotomy"), a full 164-sample evaluation reveals a minor degradation (~5%) in native reasoning capabilities when the identity-prior hooks are active.*

### Perplexity (WikiText-2)
| Milestone | PPL | Improvement |
|---|---|---|
| Baseline | 8.5775 | — |
| **Brainloop (1 Revolution)** | **8.1883** | **-4.5%** |

### Knowledge Recall
- **Status:** Verified recall of 2,002 Python symbols using zero-context vector injection.
- **Capacity:** Mathematically feasible to map 13,000+ symbols within the 66M parameter Refiner space.

## Implementation Status

- [x] **Subspace Routing:** 100% functional.
- [x] **Forward Hook Patching:** 100% functional.
- [x] **GGUF Unrolling:** Verified metadata surgery for 38-layer unrolls.
- [x] **13k Mapping:** Extraction pipeline completed; 2k symbol proof-of-concept verified.
- [ ] **13k Full Training:** Supervised delta-alignment for complete stdlib is in progress.

## Usage

### PyTorch Intervention
```python
from refiner_vanilla import patch_model_vanilla
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")
model = patch_model_vanilla(model) # Identity-prior initialization
```

### GGUF Creation
```bash
python unroll_vanilla_gguf.py --input qwen2.5-3b.gguf --output cerebellum-brainloop-python.gguf
```
