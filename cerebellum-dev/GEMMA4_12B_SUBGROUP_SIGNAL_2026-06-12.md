# Gemma 4 12B — sub-group signal mining (2026-06-12)

Question: after the group-level NO-SHIP, does the existing campaign data contain
per-tensor/per-layer signal that justifies a sub-group campaign?

**Answer: NOT-ENOUGH-DATA on the variance question, NO-GO on a full sub-group
campaign at current economics. One cheap (~2 h) band-ablation would settle it.**

## 1. Data granularity: GROUP-LEVEL ONLY

`ablation_results_multidomain.json` contains exactly 7 tests, one per tensor
*type* — each key is a regex (`^blk\.\d+\.attn_q\.weight$` etc.) that demoted
**all 48 (or 40) layers of that type simultaneously**, giving one PPL per
domain per group. `logs/ablation_multidomain.log` confirms: 7 quantize+PPL
cycles, ~283 s/variant, 1984 s total. There are **zero per-tensor or per-layer
deltas anywhere in the campaign dir** (JSON, logs, TSVs all checked).

Consequences:
- "Rank attn_v tensors by code delta" — **impossible from existing data.**
- "Which blocks were code-negative" — **impossible.** Only group aggregates exist.
- Depth / sliding-vs-global correlation — **impossible**, with one structural
  exception noted in §4.

## 2. The mean/max "variance hint" is a misreading

The mean/max columns in `allocation_report.md` are computed **across the 4
calibration domains, not across layers**. ffn_gate "mean +5.48% / max +27.53%"
decomposes as:

| group | wiki | code | math | dialogue |
|---|---|---|---|---|
| ffn_gate | **+27.5%** | **−7.1%** | +4.6% | −3.1% |
| ffn_up | +28.0% | −0.3% | +1.8% | +21.8% |
| attn_q | +28.0% | +4.7% | +4.2% | +2.0% |
| attn_v | −4.2% | −6.5% | +1.7% | +2.6% |

That gap is **domain disagreement** (wiki screams, code/math tolerate), not
evidence of within-group layer variance. The data contains no within-group
variance signal at all — the premise that the report "hints" at demotable
individual layers is not supported by what was measured.

## 3. Code-negative deltas are a TRAP on this model, not headroom

The one group we actually demoted on a code-negative PPL signal:

| build | size | code PPL Δ | HumanEval+ |
|---|---|---|---|
| uniform Q4_K_M | 7.38 GB | baseline | **83.5** |
| v2 (attn_v→Q2_K) | 7.27 GB | **−6.5% (improved!)** | **77.4** (−6.1 pts) |
| uniform Q3_K_M | 6.09 GB | +44.3% | **84.8** |

Code-domain PPL *improved* while the code benchmark collapsed; meanwhile
Q3_K_M's code PPL is +44% worse and it *outscores everything* on HumanEval+.
On Gemma 4 12B, code-PPL sign is **anti-correlated** with coding ability in
both directions. So ffn_gate's −7.1% code delta — the most tempting sub-group
target — carries the exact signature that already burned v2. Per the canon:
PPL never gates alone; here PPL barely even *correlates*.

## 4. Sliding-vs-global: already controlled for, and it didn't help

attn_v exists only in the 40 sliding-window layers (the 8 global layers
5,11,17,23,29,35,41,47 share K=V and have no attn_v tensor at all). So v2's
attn_v demotion **was already a "sliding-layers-only" intervention by
construction** — and it still cost 7.4 pts vs the same-size Q3_K_M. The
global/sliding split cannot rescue attn_v; it was never in the blast radius.
Depth structure (early vs late) remains unmeasured.

## 5. The prize, honestly

Per-group sizes at Q4_K_M base (from allocation_report.md): attn_q 0.50,
attn_k 0.19, attn_v 0.22, attn_output 0.50, ffn_up 1.59, ffn_gate 1.59,
ffn_down 1.96 GB. Hard off-the-table by every domain: ffn_down (min +30%),
attn_k (min +27.6%) = 2.15 GB. Measured saving rates: attn_v Q→Q2_K saved
0.12/0.22 = ~55% of group mass; Q4_K→Q3_K ≈ 24%.

| scenario (sub-group success level) | demoted | GB saved | final size |
|---|---|---|---|
| conservative: 1/3 of ffn_gate layers → Q3_K | 0.53 GB | 0.13 | 7.25 |
| moderate: 1/2 of ffn_gate+ffn_up → Q3_K | 1.59 GB | 0.38 | 7.00 |
| optimistic: 1/2 gate+up → Q2_K, 1/2 attn_q → Q3_K | 1.84 GB | 0.93 | 6.45 |

Even the **optimistic** case lands at 6.45 GB — 0.36 GB *above* uniform
Q3_K_M (6.09 GB), which already scores **higher** on HumanEval+ (84.8 vs
83.5) and competitive ARC/HellaSwag/MMLU. A sub-group build must beat Q3_K_M
on quality at ≤6.09 GB to have a reason to exist; that requires harvesting
~1.3 GB from groups screaming +27–80%, against a PPL screen we now know
misleads on this model. The prize window is ~1 GB wide and Q3_K_M already
owns it.

## 6. Recommendation: NO-GO (full campaign) / one cheap decision experiment

**NO-GO** on a per-layer sub-group campaign as currently conceivable:
1. No sub-group data exists; the supposed variance hint was domain variance.
2. The prize ceiling (~0.4–0.9 GB) lands inside territory uniform Q3_K_M
   already wins.
3. The only measured "safe" demotion cost 7.4 code pts; every sub-group
   candidate would need a full build + HumanEval+ gate, not just PPL.
4. QAT-trained family — the flat group landscape is plausibly flat at layer
   granularity too (architecture research doc, same date).

**If the door stays open**, the cheapest experiment that settles within-group
variance (measured throughput: ~283 s/variant, N=2 PPL workers, 4 domains):

| experiment | variants | wall time (3090+CPU) |
|---|---|---|
| **band scan**: ffn_gate/ffn_up/attn_q in 6 bands of 8 layers + attn_v in 5 bands | 23 | **~1.8 h** |
| full per-layer of those 4 groups | 184 | ~14.5 h |
| full per-layer, all 328 tensors | 328 | ~25.8 h |

Decision rule for the band scan: if **no band** of ffn_gate/ffn_up is flat
(≤+1% mean across all 4 domains, wiki included), the door closes for good
with 2 hours spent. If a band is flat, it still needs a build + HumanEval+ +
BigCodeBench gate before believing it (≈45 min/candidate) — and the size
math in §5 still has to clear uniform Q3_K_M.

## Sources (read-only)

- `cerebellum-gemma4-12b/ablation_results_multidomain.json` (7 group tests; bogus-noop variant ignored)
- `cerebellum-gemma4-12b/allocation_report.md`, `stage3_candidates.json`, `stage1_baselines.tsv`
- `cerebellum-gemma4-12b/logs/ablation_multidomain.log`, `logs/driver.log` (granularity + timing)
- `cerebellum-gemma4-12b/benchmark_results/*_evalplus_chat_results.json`, `SUMMARY_FOR_HUMAN.md`
- `cerebellum-dev/GEMMA4_12B_ARCH_RESEARCH_2026-06-12.md` (gemma4_unified structure, global layers 5,11,…,47)
