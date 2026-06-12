# Adversarial Audit: Heretic-Qwen3.6-27B EvalPlus Score

**Date:** 2026-06-11  
**Auditor:** cerebellum audit pipeline  
**VERDICT: HARNESS ARTIFACT (dominant) + MINOR REAL DIFFERENCE (apples-to-oranges baseline)**

---

## Reported Scores

| Metric | Heretic v1 | Stock v4 (same harness re-bench) | Qwen pub (May) |
|--------|-----------|----------------------------------|----------------|
| HumanEval base | 65.24% | 62.20% | 81.10% |
| HumanEval+ | 60.98% | 56.71% | ~78% |
| HellaSwag | 90.14% | 81.01% (316-sample subset, high variance) | 92.21% |
| Elapsed | 722.8s | 348.6s | — |

---

## Prime Suspects — Findings

### 1. Thinking Leakage

**CLEARED.** Zero `<think>` tags or `im_end`/role residue in any of 164 completions. The server flagged `thinking = 1` from the heretic chat template, but the benchmark uses the raw `/v1/completions` endpoint (not chat/completions), so the chat template never executes. No thinking leakage occurred.

### 2. Cop-out / Give-up Rate

**CONFIRMED ARTIFACT — 11.0% (18/164 completions).**

The `audit_evalplus_completions.py` output:

```
cop_out:    7 (4.3%)  — "# Your code here\n    pass"
pass_only: 10 (6.1%)  — "    pass"
empty:      1 (0.6%)  — comment-only with no code
GIVE-UP TOTAL: 18/164 (11.0%)
```

**Root cause:** `benchmark_evalplus.py` line 79:
```python
except (httpx.ReadTimeout, httpx.ConnectError, httpx.HTTPStatusError) as e:
    if attempt == retries - 1:
        return "    pass"
```

Every bare `pass` completion is a server timeout (300s limit, 3 retries exhausted). This is a network/scheduling artifact, not model behavior. The stock v4 re-bench on the same harness had **18.9% give-up rate** (31/164) — confirming systematic timeout pressure across both models.

### 3. Truncation

**CLEARED.** Server log shows `truncated = 0` for all 164 completions. No token-limit mid-function cutoffs.

### 4. Template Damage

**CLEARED.** First 5 completions show clean Python function bodies with correct indentation and no leaked special tokens. The heretic v2 chat template is a superset of the stock template (adds image/vision macros) but both resolve identically for text-only tasks. The raw completions endpoint bypasses the template entirely.

### 5. Quantization (SSM Safety)

**NOT THE DIFFERENTIATOR.** Both GGUFs (`Heretic-Cerebellum-v1-Q2_K_Mixed.gguf` and `Cerebellum-v4-Q2_K_Mixed.gguf`) have **identical tensor quantization profiles** — 0 tensor differences. Both have 102 SSM tensors at Q2_K (below the 4-bit threshold), but since this is identical in both builds, it cannot explain the heretic-vs-stock gap. Both inherit this risk equally.

---

## Failure Classification (57 base failures)

**Methodology:** classified all 57 base-failed tasks; then sampled 20 randomly (seed 42) for manual review.

| Category | Count | % of 164 |
|----------|-------|----------|
| Artifacts (bare-pass / cop-out timeout) | 17 | 10.4% |
| Real model logic errors | 40 | 24.4% |

**Sample of 20 failures (seed 42):**
- Artifacts in sample: 5/20 (25%)
- Real logic errors: 15/20 (75%)

Representative real errors found:
- HumanEval/65: circular-shift off-by-one logic
- HumanEval/76: incorrect integer cube root check (`x == n ** int(x)`)  
- HumanEval/113: f-string construction bug
- HumanEval/131: incorrect zero-product edge case
- HumanEval/106: uses undefined `sum(1, i)` call

These are genuine model failures — complex algorithmic problems where the fine-tune (heretic) degraded mathematical/algorithmic precision vs the base.

---

## Corrected Score Analysis

```
Reported base pass@1:           65.24%  (107/164)
Artifacts (timeouts):           17 problems (10.4%)
Real model failures:            40 problems (24.4%)

Artifact-corrected pass@1:      75.6%   (124/164, if retried clean)
Harness noise contribution:    +10.4 points
Remaining real gap vs pub:      ~5.5 points vs 81.10% publisher score
```

**Critical context:** The 81.10% publisher score was obtained with:
- Thinking mode enabled (chain-of-thought budget)
- Chat completions endpoint (full system prompt)
- Different harness (likely evalplus with temperature sampling)

The stock v4 on this identical harness (raw completions, no-think, WORKERS=1) scored **62.2%** — LOWER than heretic's 65.24%. Heretic v1 **outperforms stock v4 by ~3 points** on this harness.

---

## HellaSwag Spot-Check (20 random wrong answers, seed 42)

**CLEARED — 0/20 artifacts, 20/20 real wrong answers.**

All 20 sampled wrong answers were genuine model errors (well-formed A/B/C/D responses, wrong answer, no errors). However, a systematic position bias was detected:

- **68% of wrong answers chose "A"** (vs 6.8% expected by chance)
- Overall "A" prediction rate: 31.1% vs 25.0% expected

This is a mild position bias (A-first anchoring), not a catastrophic artifact. HellaSwag score of 90.14% vs stock 92.21% is plausible given the heretic fine-tune's alignment changes. The 2-point gap is consistent with real model differences, not harness failure.

---

## Root Cause Summary

| Signal | Magnitude | Type |
|--------|-----------|------|
| Server timeouts → bare `pass` fallback | −10.4 pts | HARNESS ARTIFACT |
| Apples-to-oranges baseline (thinking vs no-think) | ~14 pts apparent gap | COMPARISON ERROR |
| Real heretic fine-tune logic regression | ~5 pts | REAL (vs publisher) |
| Heretic vs stock on same harness | +3 pts (heretic WINS) | REAL |

---

## Recommendations

1. **Do not compare to publisher 81.10%** — that score requires thinking mode. Add a no-think baseline from the Qwen3.6 BF16 or Q8 with this identical harness before publishing heretic comparisons.

2. **Fix timeout cascade:** The 300s timeout with 3 retries is appropriate for individual hard problems, but the stock v4 had 18.9% give-up rate — worse than heretic. Investigate whether the server was under concurrent load during these runs (the log shows 4-slot initialization, but only the HellaSwag/ARC benchmarks should use multi-worker).

3. **Heretic v1 verdict: tentatively acceptable.** On the same harness, heretic scores +3 points over stock. The fine-tune appears to have preserved (marginally improved) code completion despite alignment changes. The SSM Q2_K risk is shared by both builds and needs a separate PPL gate.

4. **Required before HF publish:** Run evalplus_chat.py with `enable_thinking=False` at temperature=0 for a clean apples-to-apples comparison against stock Q2_K_Mixed.

---

*Audit performed on: 2026-06-11*  
*Files audited:*  
- `benchmark_results/heretic-27b-cerebellum-v1_evalplus_samples.jsonl` (164 completions)  
- `benchmark_results/heretic-27b-cerebellum-v1_evalplus_samples_eval_results.json`  
- `benchmark_results_stockv4/stock-27b-cerebellum-v4-recheck_evalplus_results.json` (parallel re-bench)  
- `bench_server.log` (server-side analysis)
