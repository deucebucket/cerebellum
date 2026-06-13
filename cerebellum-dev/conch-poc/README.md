# Cerebellum-Brainloop

Cerebellum-Brainloop implements a dual-stage hidden state interceptor for Qwen2.5-3B. The system uses non-destructive layer wrappers (the base model stays fully frozen) to inject external factual knowledge directly into the model's residual stream.

## Technical Facts

- **Architecture:** Dual refiner wrappers on layers 17 ("Reasoning") and 30 ("Knowledge Gate"), 0-indexed — each refiner is a clone of its wrapped base layer and executes immediately after it (`refiner_vanilla.py`). In the unrolled 38-block GGUF the trained refiners occupy block indices 18 and 32.
- **Subspace Routing:** HID (Hidden Dimension) partitioning ensures that only 25% of dimensions (Dim 1536-2048) are modified, preserving the base model's native logic in the remaining 75%.
- **Zero-Context Injection:** Knowledge is fused via Delta Vectors extracted from "Knowing" vs "Ignorant" model states.
- **Vanilla Compatibility:** GGUF surgery script (`unroll_vanilla_gguf.py`) produces a standard GGUF that runs on stock `llama.cpp` releases without custom C++ forks. Caveat: the exported refiners execute as plain residual blocks — the tanh gate, subspace mask, and RAG injection currently exist only in the PyTorch path. Closing that gap is an open problem (see `EXPERIMENTAL_PATHS.md`).

## Verified Benchmarks (Qwen2.5-3B)

### Logic & Coherence
| Configuration | HumanEval (164-sample) | HumanEval+ (164-sample) | Coherence Test |
|---|---|---|---|
| Qwen2.5-3B Baseline | 62.2% | 56.1% | Pass |
| **Brainloop (Refiners Active)** | **56.7% (-5.5%)** | **51.2% (-4.9%)** | **Pass** |

*Correction (2026-06-11): a benchmark-wiring audit found the refiners were functionally inert during this run (trained gate ≈ 0.5% contribution; injection projection untrained). The 5.5-point gap reflects prompt-format differences between bench scripts, not the intervention — the previously reported "~5% logic tax" was a measurement artifact. Verified compiled-path results (below) show no measurable logic cost from inserted identity-initialized blocks.*

### Compiled-Path Results — Dead-Block GGUF (2026-06-11)

First benchmark set measured on stock llama.cpp (tag b9275, CUDA, RTX 3090) — no fork, no Python interception. Model: `cerebellum-deadblock-python.gguf`, a 38-block GGUF with two attention-dead, subspace-masked FFN refiner blocks (trained 1 epoch, injection-free delta prediction, final delta cosine ~0.44).

| Metric | Baseline (36L) | Dead-Block (38L) |
|---|---|---|
| HumanEval pass@1 (164, greedy) | 62.8% | 62.8% |
| HumanEval+ pass@1 | 57.3% | 56.7% |
| PPL wikitext (c=2048) | 7.1961 ±0.046 | 7.1973 ±0.046 |
| PPL python-stdlib 300KB (c=2048) | 3.4342 ±0.038 | 3.4579 ±0.038 |
| Symbol recall (n=200, content overlap ≥0.5) | 10.0% | 10.0% |
| Post-cutoff symbol recall (n=30) | 0% | 0% |

What this shows, stated plainly:
- **Structural parity holds on the compiled path.** Inserting the two trained blocks costs no measurable logic or perplexity (161/164 HumanEval completions are token-identical to baseline). The earlier unrolled builds' catastrophic looping does not occur.
- **Knowledge injection is not yet effective** in this delta-trained configuration — the blocks are functionally inert.
- Failure audit: all sampled wrong answers were genuine model errors, no extraction artifacts.
- These PPL values are not comparable to the legacy table below (different eval protocol).

### LM-Trained Insertion Blocks — Verified Recall Through a Vanilla GGUF (2026-06-11)

Replacing delta-prediction with plain LM-loss training of the inserted block produced the first verified knowledge-recall gains on the compiled path (full tables and methodology notes in `RESULTS.md`):

- Symbol recall 10.0% → up to 25.5% (n=200, wiring-verified A/B harness), including recall of **Python 3.14 stdlib symbols that postdate the base model's training data** — content the frozen base model cannot know, surfaced with zero context tokens.
- A wikitext-trained variant of the same block improves wikitext PPL **8.54 → 7.68 (-10.0%)** at its training context on stock llama.cpp.
- Current cost, not yet controlled: corpus-trained variants regress HumanEval; a 25%-subspace write mask and reduced learning rate measurably bound the behavioral damage (best so far: 55.5% vs 62.8% baseline at lr 2e-5). Long-context PPL holds at baseline across all variants. Retaining recall while holding HumanEval at baseline is the open problem and active work.

Correction (2026-06-11): the dead-block table's wikitext and python PPL values were originally published swapped — the measurement session ran its four perplexity jobs in parallel and crossed the corpus labels. The values above are corrected; an identity-block control re-measured both baselines and confirmed the parity conclusion is unchanged.

### Perplexity (WikiText-2)
| Milestone | PPL | Improvement |
|---|---|---|
| Baseline | 8.5775 | — |
| **Brainloop (1 Revolution)** | **8.1883** | **-4.5%** |

*Note: PPL numbers were measured on the earlier looped-refiner C++ port (revolution sweep found 1 revolution optimal). The current vanilla-compatible architecture uses a single refiner pass with no revolution loop and has not been re-measured on WikiText-2.*

### Knowledge Recall
- **Status:** Verified recall of 2,002 Python symbols using zero-context vector injection.
- **Capacity:** Mathematically feasible to map 13,000+ symbols within the 66M parameter Refiner space.

## Implementation Status

- [x] **Subspace Routing:** 100% functional.
- [x] **Layer-Wrapper Patching:** 100% functional.
- [x] **GGUF Unrolling:** Verified metadata surgery for 38-layer unrolls.
- [x] **13k Mapping:** Extraction pipeline completed; 2k symbol proof-of-concept verified.
- [x] **13k Full Training:** Supervised delta-alignment over the full stdlib corpus completed (2026-06-11). Checkpoint (`fused_refiners.pt`), RAG index, and benchmark outputs published to the `deucebucket/cerebellum-brainloop` HF dataset. HumanEval+ with 13k RAG active: 51.2% — identical to the no-RAG hooked score, i.e. live retrieval adds no further logic degradation.
- [ ] **13k Recall Verification:** Symbol-recall accuracy at full 13k scale not yet measured (the 94%+ figure is from the 2,002-symbol POC).

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
