# Benchmark Audit: ARC-Challenge + MMLU-Redux
**Model:** heretic-cerebellum-v1  
**Auditor:** adversarial / automated  
**Audit date:** 2026-06-11  
**Audited files:**
- `heretic-cerebellum-v1_arc_detailed.jsonl` (1172 entries)
- `heretic-cerebellum-v1_mmlu_redux_detailed.jsonl` (2400 entries)

---

## Verdict

| Benchmark | Verdict | Reported | Recount | Artifact errors |
|-----------|---------|----------|---------|-----------------|
| ARC-Challenge | **TRUSTWORTHY** | 95.48% | 95.48% | 0 |
| MMLU-Redux | **TRUSTWORTHY** | 75.42% | 75.42% | 0 |

No artifacts, no parse failures, no label-format bugs, no truncation signature detected. All 1172 ARC and 2400 MMLU entries are internally consistent.

---

## 1. Schema Reconnaissance

Both files use the same schema per line:

```json
{
  "question": "...",
  "choices": ["A text", "B text", "C text", "D text"],
  "expected": "C",
  "predicted": "C",
  "raw_response": "C",
  "correct": true,
  "error": null
}
```

MMLU adds a `"subject"` field. The `raw_response` field stores exactly what the model returned — in every entry across both benchmarks this is a single uppercase letter (A/B/C/D). There are no multi-token completions, no reasoning traces, no chain-of-thought artifacts.

---

## 2. Aggregate Verification

Recount performed by independently summing `correct == true` flags:

| Benchmark | Total | Correct | Wrong | Recount acc | Reported acc | Match |
|-----------|-------|---------|-------|-------------|--------------|-------|
| ARC | 1172 | 1119 | 53 | **95.48%** | 95.48% | YES |
| MMLU | 2400 | 1810 | 590 | **75.42%** | 75.42% | YES |

Both match to 2 decimal places. The summary JSONs are not lying.

---

## 3. Wrong-Answer Classification

### ARC-Challenge: all 53 wrong entries

| Class | Count |
|-------|-------|
| REAL_ERROR (model chose wrong letter) | **53** |
| ARTIFACT_EMPTY | 0 |
| ARTIFACT_UNPARSEABLE | 0 |
| ARTIFACT_PARSE_MISMATCH | 0 |
| ARTIFACT_NUMERIC_LABEL | 0 |

**0 artifacts out of 53 wrong answers.**

### MMLU-Redux: 60-entry random sample (seed=42) of 590 wrong entries

| Class | Count |
|-------|-------|
| REAL_ERROR (model chose wrong letter) | **60** |
| ARTIFACT_EMPTY | 0 |
| ARTIFACT_UNPARSEABLE | 0 |
| ARTIFACT_PARSE_MISMATCH | 0 |
| ARTIFACT_NUMERIC_LABEL | 0 |

**0 artifacts out of 60 sampled wrong answers.** At 0/60 artifact rate, the 95% CI for artifact prevalence in the full wrong population is 0–6% (Wilson interval). The most pessimistic reading: ~35 of the 590 wrong answers could be artifacts; even so the corrected score would be 75.42% + (35/2400)*100 = ~76.9%. The floor of the score cannot drop below reported.

---

## 4. Distribution Checks

### 4a. Choice distribution (predicted vs gold)

**ARC:**
| Choice | Predicted | Gold | Delta |
|--------|-----------|------|-------|
| A | 269 | 266 | +3 |
| B | 312 | 311 | +1 |
| C | 304 | 310 | -6 |
| D | 287 | 285 | +2 |

Deltas are ≤6. No evidence of parser defaulting to any single choice.

**MMLU:**
| Choice | Predicted | Gold | Delta |
|--------|-----------|------|-------|
| A | 497 | 537 | -40 |
| B | 614 | 600 | +14 |
| C | 613 | 606 | +7 |
| D | 676 | 657 | +19 |

The model under-picks A and over-picks D relative to gold distribution. This is a model-level tendency, not a parser artifact — A-defaulting (the known parser-fallback bug) would produce the opposite signature (over-picking A).

### 4b. Empty/whitespace raw responses across ALL entries

- ARC: **0 / 1172**
- MMLU: **0 / 2400**

No empty responses anywhere.

### 4c. Parsed choice absent from raw_response (all entries with long responses)

All 3572 entries have single-character raw responses. The parsed `predicted` field equals `raw_response` in 100% of entries (0 mismatches in either benchmark).

### 4d. `correct` flag internal consistency

- ARC entries where `correct=True` but `predicted != expected`: **0**
- MMLU entries where `correct=True` but `predicted != expected`: **0**
- ARC entries where `correct=False` but `predicted == expected`: **0**
- MMLU entries where `correct=False` but `predicted == expected`: **0**

The `correct` flag is computed correctly from `predicted == expected` with no exceptions.

### 4e. First-option bias among wrong answers

- ARC wrong answers predicted as 'A': 12/53 = **22.6%** (expected if random: 25%)
- MMLU wrong answers predicted as 'A': 112/590 = **19.0%** (expected if random: 25%)

If anything, the model slightly under-picks 'A' when wrong — no first-option parser bias.

---

## 5. Truncation Analysis (MMLU)

Prompt length was approximated as `len(question) + sum(len(choice) for choice in choices)`. This is a proxy for the actual tokenized prompt, but faithfully captures long-vs-short relative ordering.

**Wrong rate by prompt-length decile:**

| Decile | Len range (chars) | Wrong rate |
|--------|-------------------|------------|
| 1 (shortest) | 17–101 | 24.2% |
| 2 | 101–129 | 23.8% |
| 3 | 129–154 | 29.2% |
| 4 | 154–184 | 25.4% |
| 5 | 184–219 | 25.4% |
| 6 | 219–259 | 22.1% |
| 7 | 259–315 | 25.0% |
| 8 | 316–382 | 25.0% |
| 9 | 382–489 | 25.0% |
| 10 (longest) | 489–4872 | **20.8%** |

**No truncation signature.** The longest decile (489–4872 chars, including a 4872-char outlier) has the *lowest* wrong rate (20.8%), not the highest. If context truncation were occurring, deciles 9–10 would show elevated error rates. The distribution is flat across deciles, with decile 3 as the minor high point (29.2%) — almost certainly subject-difficulty driven, not length-driven.

**Mean prompt length of correct vs wrong answers:**
- Correct: 277 chars
- Wrong: 259 chars

Wrong answers are marginally *shorter* in prompt length on average, the opposite of what truncation would produce.

**ARC truncation check:**
- Wrong answers mean prompt length (217) < correct answers (249). Same anti-truncation pattern.

No context-per-slot truncation artifacts in either benchmark.

---

## 6. Known Historical Bug Cross-Check

| Bug | Check | Status |
|-----|-------|--------|
| Numeric-vs-letter label mismatch (cost 19 ARC questions) | ARTIFACT_NUMERIC_LABEL count | **0 in both** |
| Empty responses counted as wrong | empty raw_response | **0 in both** |
| Parser fallback picking first option (A-bias) | wrong-answer A% vs expected 25% | **ARC 22.6%, MMLU 19.0% — no bias** |
| API errors counted as wrong | `error` field non-null | **0 in both** |
| Context-per-slot truncation | prompt-length decile wrong rate | **Flat; longest decile lowest error** |

All five known historical bugs: **not present**.

---

## 7. Corrected Scores

No correction needed. Recount matches reported scores exactly. Zero artifact errors detected in all sampled and exhaustively audited wrong answers.

**Final scores:**
- ARC-Challenge: **95.48%** (1119/1172) — confirmed
- MMLU-Redux: **75.42%** (1810/2400) — confirmed
