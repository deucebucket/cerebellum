# Adversarial Benchmark Audit: Heretic-Qwen3.6-27B (ARC + MMLU + Chat-EvalPlus)

**Model:** heretic-27b-cerebellum-v1
**Auditor:** adversarial / automated (same methodology as 35B audit)
**Audit date:** 2026-06-11
**Audited files:**
- `heretic-27b-cerebellum-v1_arc_detailed.jsonl` (1172 entries)
- `heretic-27b-cerebellum-v1_mmlu_redux_detailed.jsonl` (2400 entries)
- `heretic-27b-chat-nothink_evalplus_chat_samples.jsonl` + eval_results (164 problems)
- `stock-27b-v4-chat-nothink_evalplus_chat_samples.jsonl` + eval_results (164 problems, comparison run)

---

## Verdict Summary

| Benchmark | Verdict | Reported | Recount | Artifacts |
|-----------|---------|----------|---------|-----------|
| ARC-Challenge | **TRUSTWORTHY** | 96.93% | 96.9283% | 0/36 wrong |
| MMLU-Redux | **TRUSTWORTHY** | 76.21% | 76.2083% | 0/50 sampled wrong |
| Chat EvalPlus (heretic) | **TRUSTWORTHY** | 89.63 / 84.76 | 89.63 / 84.76 | 0 give-ups |
| Chat EvalPlus (stock) | **TRUSTWORTHY** | 90.85 / 87.20 | 90.85 / 87.20 | 0 give-ups |

**FINAL: SHIP** (no blocking findings across any check)

---

## 1. ARC-Challenge

### 1a. Aggregate recount

| Field | Value |
|-------|-------|
| Total entries | 1172 |
| Correct (recount) | 1136 |
| Wrong (recount) | 36 |
| Recount accuracy | 96.9283% |
| Reported accuracy | 96.93% |
| Match | YES (delta < 0.01pp) |

### 1b. Data quality checks

| Check | Result |
|-------|--------|
| Empty `raw_response` | 0 / 1172 |
| `raw_response != predicted` (parse mismatches) | 0 / 1172 |
| `correct=True` but `predicted != expected` | 0 |
| `correct=False` but `predicted == expected` | 0 |
| Raw response length | All exactly 1 char (A/B/C/D) |

### 1c. Answer distribution

| Choice | Predicted | Gold | Delta |
|--------|-----------|------|-------|
| A | 268 | 266 | +2 |
| B | 311 | 311 | 0 |
| C | 309 | 310 | -1 |
| D | 284 | 285 | -1 |

Deltas <= 2. No evidence of parser defaulting to any single choice. A-overselection (the known fallback bug signature) is absent.

### 1d. Wrong-answer classification: ALL 36 wrong entries

| Class | Count |
|-------|-------|
| REAL_ERROR (model chose wrong letter, valid choice) | **36** |
| ARTIFACT_EMPTY | 0 |
| ARTIFACT_UNPARSEABLE | 0 |
| ARTIFACT_PARSE_MISMATCH | 0 |
| ARTIFACT_NUMERIC_LABEL | 0 |

**0 artifacts out of 36 wrong answers.** Every miss is a genuine model error: a valid A/B/C/D response that matched the wrong answer key. No parser fallback, no truncation, no empty-response fill-in.

Wrong answers predicted distribution among the 36:
- A: 6 (16.7%), B: 11 (30.6%), C: 8 (22.2%), D: 11 (30.6%)

If the parser were defaulting to any single choice, one would spike to 50%+. Not the case here.

---

## 2. MMLU-Redux

### 2a. Aggregate recount

| Field | Value |
|-------|-------|
| Total entries | 2400 |
| Correct (recount) | 1829 |
| Wrong (recount) | 571 |
| Recount accuracy | 76.2083% |
| Reported accuracy | 76.21% |
| Match | YES (delta < 0.01pp) |

### 2b. Data quality checks

| Check | Result |
|-------|--------|
| Empty `raw_response` | 0 / 2400 |
| `raw_response != predicted` (parse mismatches) | 0 / 2400 |
| `correct=True` but `predicted != expected` | 0 |
| `correct=False` but `predicted == expected` | 0 |
| Raw response length | All exactly 1 char (A/B/C/D) |

### 2c. Answer distribution

| Choice | Predicted | Gold | Delta |
|--------|-----------|------|-------|
| A | 524 | 537 | -13 |
| B | 540 | 600 | -60 |
| C | 654 | 606 | +48 |
| D | 682 | 657 | +25 |

The model under-picks B and over-picks C/D relative to the gold distribution. This is a model-level tendency (consistent with Qwen 3.6 family behavior), not a parser artifact. A-defaulting (the known fallback bug) would produce massive A over-selection; instead A is under-selected by 13.

Wrong answers predicted distribution among 571:
- A: 119 (20.8%), B: 117 (20.5%), C: 185 (32.4%), D: 150 (26.3%)

C/D over-selection in wrong answers mirrors the overall distribution skew. No first-option collapse.

### 2d. Wrong-answer classification: 50-sample (seed=42) of 571 wrong entries

| Class | Count |
|-------|-------|
| REAL_ERROR (valid letter, wrong answer) | **50** |
| ARTIFACT_EMPTY | 0 |
| ARTIFACT_UNPARSEABLE | 0 |
| ARTIFACT_PARSE_MISMATCH | 0 |
| ARTIFACT_NUMERIC_LABEL | 0 |

**0 artifacts out of 50 sampled wrong answers.** At 0/50 artifact rate, the 95% Wilson CI for artifact prevalence in the full 571-wrong population is 0-7%. Most pessimistic corrected score: 76.21 + (40/2400)*100 = 77.9%. The floor cannot drop below reported.

Sample spanned: college_mathematics, business_ethics, global_facts, formal_logic, college_computer_science, machine_learning, virology — broad subject coverage, no subject clustering in errors.

---

## 3. Chat-Harness EvalPlus: Give-Up Census

### 3a. Give-up audit (audit_evalplus_completions.py)

```
heretic-27b-chat-nothink:
  audited 164 completions
  GIVE-UP signals: 0 (0.0%)
  REAL-ATTEMPT signals: normal=133 (81.1%), one_liner=31 (18.9%)

stock-27b-v4-chat-nothink:
  audited 164 completions
  GIVE-UP signals: 0 (0.0%)
  REAL-ATTEMPT signals: normal=130 (79.3%), one_liner=34 (20.7%)
```

**Zero give-up events in either run.** The patched chat harness has fully resolved the timeout-fabricated `pass` stub issue documented in the prior non-chat bench run (which had 11.0% give-up rate from the raw completions endpoint + timeout fallback). The chat endpoint handles retries cleanly.

No timeout-fabricated pass stubs were found passing unit tests in either run (which would be physically impossible anyway — confirmed 0 such entries).

### 3b. Score verification from eval_results.json

| Model | Base recount | Reported | Plus recount | Reported | Match |
|-------|-------------|----------|-------------|----------|-------|
| heretic-27b | 147/164 = 89.63% | 89.63% | 139/164 = 84.76% | 84.76% | YES |
| stock-27b | 149/164 = 90.85% | 90.85% | 143/164 = 87.20% | 87.20% | YES |

Both summary JSONs are arithmetically exact against their eval_results.json ground truth.

### 3c. Gap analysis

| Metric | Heretic | Stock | Delta |
|--------|---------|-------|-------|
| HumanEval base | 89.63% | 90.85% | -1.22pp |
| HumanEval+ | 84.76% | 87.20% | -2.44pp |

Both runs had identical give-up rates (0/164), identical problem sets, and identical test suite versions (hash `fe585eb4df8c88d844eeb463ea4d0302`). The 1.22-2.44pp gap is therefore a clean model-quality signal, not an artifact. Heretic's mixed-precision quantization costs ~1-2.5pp on chat coding versus stock Q4_K_M — within acceptable range for the size budget reduction.

---

## 4. Spot-check: 10 heretic chat-run failures (base_status=fail)

10 random failures sampled from base failures (seed=42). Classification method: inspect function body after closing docstring `"""`, check for empty / pass-stub / truncation / cop-out signatures.

| # | Task | Base | Verdict | Evidence |
|---|------|------|---------|----------|
| 1 | HumanEval/135 | fail | REAL_ERROR | Attempted linear scan, off-by-one logic |
| 2 | HumanEval/108 | fail | REAL_ERROR | Attempted digit-sum loop, indentation bug |
| 3 | HumanEval/32 | fail | REAL_ERROR | Comment reasoning + incomplete implementation |
| 4 | HumanEval/148 | fail | REAL_ERROR | Attempted planet-ordering, incorrect planet list |
| 5 | HumanEval/93 | fail | REAL_ERROR | Attempted vowel substitution, wrong cipher |
| 6 | HumanEval/62 | fail | REAL_ERROR | One-liner derivative, enumerate offset wrong |
| 7 | HumanEval/132 | fail | REAL_ERROR | Attempted bracket nesting, logic error |
| 8 | HumanEval/115 | fail | REAL_ERROR | Attempted bucket fill, wrong formula |
| 9 | HumanEval/163 | fail | REAL_ERROR | Attempted even-range filter, boundary error |
| 10 | HumanEval/17 | fail | REAL_ERROR | Attempted music-string parser, missing case |

**10/10 are genuine model errors.** Every failure shows non-trivial attempted logic — multi-line implementations, reasoning comments, real algorithmic attempts. Zero pass-stubs, zero empty bodies, zero truncations. The model is trying and failing, not giving up.

Distribution: 17 base failures total in 164 problems. 8 additional plus-only failures (pass base, fail extended tests). This breakdown is consistent with a model that handles the standard test suite well but struggles on edge cases — the expected profile for a compressed model.

---

## 5. Full score card

| Benchmark | Heretic-27b | Stock-27b | Published baseline |
|-----------|------------|-----------|-------------------|
| ARC-Challenge | **96.93%** | — | — |
| MMLU-Redux | **76.21%** | — | — |
| HumanEval base (chat) | **89.63%** | 90.85% | — |
| HumanEval+ (chat) | **84.76%** | 87.20% | — |

---

## 6. Audit conclusion

No artifacts, no parse failures, no timeout stubs, no label-format bugs, no truncation signatures, no empty responses detected across any benchmark or either chat model run. All reported figures are verified against ground-truth eval data.

The ARC score (96.93%) is the highest in the Cerebellum 27B series and verified clean against all 36 wrong answers. MMLU (76.21%) clears the reported figure with zero artifacts in the 50-entry sample. Chat EvalPlus shows a real but modest -1.22/-2.44pp gap versus stock — no give-up events, all failures are genuine model errors.

**SHIP.**
