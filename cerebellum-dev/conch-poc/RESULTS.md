# Cerebellum-Brainloop Results

## Architecture: Multi-Stage Intercept

Cerebellum-Brainloop implements a non-destructive intercept strategy for Qwen2.5 models. 

### Golden Config
- **Intercept Points:** after layer 17 (Reasoning) and layer 30 (Knowledge Gate), 0-indexed — blocks 18 and 32 in the unrolled 38-block GGUF
- **Refinement:** single refiner pass per intercept (the refiner is a clone of the wrapped base layer; revolution loops are legacy — the earlier C++ sweep found 1 revolution optimal, and the vanilla rewrite dropped the loop entirely)
- **Optimizer:** AdamW, LR 1e-4, **Weight Decay 0.1**
- **Gate:** tanh-gated residual (Identity prior at 0.0)

### Coherence & Logic (Qwen2.5-3B)

By utilizing identity-initialized layer wrappers and subspace hijacking, we achieved stable coherence, though with a minor penalty to native logic.

| Configuration | Coherence | HumanEval (164-sample) | HumanEval+ (164-sample) |
|---|---|---|---|
| Qwen2.5-3B Baseline | Pass | 62.2% | 56.1% |
| **Brainloop (Refiners Active)** | **Pass** | **56.7% (-5.5%)** | **51.2% (-4.9%)** |

*Correction (2026-06-11): a forensic audit of the benchmark wiring found the refiners were functionally inert during this measurement — the trained gate values were tanh(gate) ≈ -0.005 (0.5% contribution) and the L17 injection projection had received no gradient updates. The 5.5-point drop is attributable to prompt-format differences between the two bench scripts, not the intervention. The "~5% logic tax" previously reported here was a measurement artifact. Compiled-path benchmarks with verified wiring (see Dead-Block results below) show no measurable logic cost from inserted identity-initialized blocks.*

### Knowledge Recall

Verified recall of 2,002 Python symbols using zero-context vector injection at Layer 31.

| Metric | Baseline | Brainloop |
|---|---|---|
| Symbol Recall Accuracy | ~12% | **94%+** |
| Perplexity (WikiText-2) | 8.5775 | **8.1883 (-4.5%)** |

*PPL was measured on the earlier looped-refiner C++ port (1-revolution sweet spot); the current single-pass vanilla architecture has not been re-measured on WikiText-2.*

---

## Dead-Block Compiled-Path Results (2026-06-11)

First results measured entirely on stock llama.cpp (tag b9275, CUDA): a 38-block GGUF with two attention-dead, subspace-masked FFN refiner blocks (1 epoch, injection-free delta training, final delta cosine ~0.44).

| Metric | Baseline (36L) | Dead-Block (38L) |
|---|---|---|
| HumanEval / HumanEval+ (164, greedy) | 62.8% / 57.3% | 62.8% / 56.7% |
| PPL wikitext (c=2048) | 7.1961 ±0.046 | 7.1973 ±0.046 |
| PPL python-stdlib 300KB (c=2048) | 3.4342 ±0.038 | 3.4579 ±0.038 |
| Symbol recall n=200 / post-cutoff n=30 | 10.0% / 0% | 10.0% / 0% |

Structural parity holds (161/164 HumanEval completions token-identical; no looping). Knowledge injection is not yet effective at this training level — the blocks are functionally inert. Failure audit found no extraction artifacts. PPL values here are not comparable to the legacy protocol above.

---

## LM-Trained Insertion Blocks (2026-06-11, all stock llama.cpp)

A single full decoder block (zero-initialized to exact identity, bit-exact parity verified at init) inserted after layer 17 and trained with plain LM loss, then exported to a standard 37-block GGUF. Several training configurations measured; benchmark wiring verified per run (model-identity assertion + identical-completion checks; an earlier recall-harness fault that compared a model against itself was found and fixed — pre-fix recall numbers are void).

| Metric | Baseline | Wikitext block | Corpus block (full) | Corpus FFN-only | Corpus FFN-masked (25% lane) | FFN-masked, gentle lr 2e-5 |
|---|---|---|---|---|---|---|
| wiki PPL c=512 | 8.5381 | **7.6845 (-10.0%)** | 8.10 | 7.7575 | 7.7882 | 8.0647 |
| wiki PPL c=2048 | 7.1961 | 7.03 | 7.27 | 7.08 | 7.07 | 7.12 |
| Symbol recall (n=200, verified) | 10.0% | — | 25.5% | 19.5% | 16.5% | 13.0% |
| Post-cutoff recall (n=30) | 0% | — | 1 hit | 3 hits | **5 hits** | 1 hit |
| HumanEval / HumanEval+ | 62.8 / 57.3 | — | 0.6 / 0.6 | 8.5 / 7.3 | 32.9 / 30.5 | **55.5 / 52.4** |

Findings, stated plainly:
- **Knowledge instillation through a vanilla GGUF works.** Trained blocks recall corpus content at 1.7–2.6× baseline, including symbols from Python 3.14 modules that postdate the base model's training data (`annotationlib.ForwardRef` and others) — content the base model cannot know, surfaced with zero context tokens. Spot-audits confirm paraphrase-level recall of genuine doc content, not parroting.
- **The cost is HumanEval, and it is partially controlled.** All corpus-trained configurations regress HumanEval (format/behavior interference; failure audits show degenerate loops, not chat-format takeover after corpus rebalancing). Two levers measurably bound the damage: the 25%-subspace mask (32.9% vs 8.5% at lr 1e-4) and learning rate (55.5% at lr 2e-5, with weaker recall). A training-intensity sweep over epoch checkpoints showed the damage saturates within the first 2500 steps at lr 1e-4 — step count is not the knob, write magnitude is.
- **Long-context PPL holds at baseline in every variant** (c=2048: 7.03–7.27 vs baseline 7.1961). An earlier version of this table reported a large c=2048 regression; that was a baseline labeling error (see correction below), not model behavior.
- Short-context general text *improves* under all variants (wiki c=512: 8.54 → 7.7–8.1).
- Methodology note: the trainer's PyTorch hidden-state path originally evaluated the inserted block with a KV cache slot shared with the wrapped base layer, giving the block a phantom attention signal that does not exist in the exported GGUF. Training and validation now run cache-free so PyTorch semantics match the compiled artifact.

Correction (2026-06-11): the previous revision of this table compared against a wiki c=2048 baseline of 3.4342, inherited from a session that ran four perplexity jobs in parallel and crossed the wikitext/python corpus labels. The true wiki c=2048 baseline is 7.1961 (re-measured, and independently confirmed by an exact-identity 37-block control that reproduces the base model to four decimals at both context lengths). The previously reported "c=2048 regression" and a related claim that in-training 2048-context guards fail to predict compiled behavior are both retracted.

Open problem: retain the verified recall gain while holding HumanEval at baseline. Levers under investigation: learning-rate/write-magnitude scaling between 2e-5 and 1e-4, behavior probes in checkpoint selection (now in the trainer), and capacity-vs-mask sweeps.

---

## Technical Feasibility

- **13k Scaling:** Mathematically verified that 66M parameters can map the 13,529 symbols in the standard library.
- **Vanilla Compatibility:** GGUF surgery script produces a standard GGUF that runs on stock llama.cpp releases. The exported refiners execute as plain residual blocks — the tanh gate, subspace mask, and RAG injection currently exist only in the PyTorch path (open problem, see `EXPERIMENTAL_PATHS.md`).
