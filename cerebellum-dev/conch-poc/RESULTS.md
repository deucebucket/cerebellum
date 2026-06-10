# Cerebellum-Brainloop Results

## Architecture: Multi-Stage Intercept

Cerebellum-Brainloop implements a non-destructive intercept strategy for Qwen2.5 models. 

### Golden Config
- **Split Layers:** L18 (Reasoning), L31 (Knowledge Gate)
- **Refinement:** 2 Revolutions per intercept
- **Optimizer:** AdamW, LR 1e-4, **Weight Decay 0.1**
- **Gate:** tanh-gated residual (Identity prior at 0.0)

### Coherence & Logic (Qwen2.5-3B)

By utilizing identity-initialized forward hooks and subspace hijacking, we achieved stable coherence, though with a minor penalty to native logic.

| Configuration | Coherence | HumanEval (164-sample) | HumanEval+ (164-sample) |
|---|---|---|---|
| Qwen2.5-3B Baseline | Pass | 62.2% | 56.1% |
| **Brainloop (Hooks Active)** | **Pass** | **56.7% (-5.5%)** | **51.2% (-4.9%)** |

*Note: While the intercept mechanism prevents the catastrophic looping ("lobotomy") seen in earlier unrolled builds, a full 164-sample evaluation reveals a minor degradation (~5%) in native reasoning capabilities when the identity-prior hooks are active.*

### Knowledge Recall

Verified recall of 2,002 Python symbols using zero-context vector injection at Layer 31.

| Metric | Baseline | Brainloop |
|---|---|---|
| Symbol Recall Accuracy | ~12% | **94%+** |
| Perplexity (WikiText-2) | 8.5775 | **8.1883 (-4.5%)** |

---

## Technical Feasibility

- **13k Scaling:** Mathematically verified that 66M parameters can map the 13,529 symbols in the standard library.
- **Vanilla Compatibility:** GGUF surgery script enables execution on standard llama.cpp releases.
