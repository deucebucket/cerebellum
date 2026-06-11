# HellaSwag Benchmark Audit — heretic-cerebellum-v1

**Date:** 2026-06-11  
**Auditor:** Adversarial automated audit (adversarial-hellaswag-audit)  
**Benchmark run timestamp:** 2026-06-11 16:02  
**File audited:** `heretic-cerebellum-v1_hellaswag_detailed.jsonl`  
**Reported score:** 91.78%

---

## Verdict: TRUSTWORTHY

All six check categories pass. The reported score of 91.78% is correct. No artifacts, no fallbacks, no transport failures, no cache contamination detected.

---

## Section 1: Schema Reconnaissance

**Fields present in every entry (10042/10042):**

| Field | Type | Sample values |
|-------|------|---------------|
| `context` | str | "A man is sitting on a roof. he" |
| `endings` | list[str] | 4 continuation options |
| `expected` | str | "D", "C", "B", "A" |
| `predicted` | str | "D", "C", "B", "A" |
| `raw_response` | str | "D" (exactly 1 char, always) |
| `correct` | bool | true / false |
| `error` | null | always null |

**No timestamp field.** Entries are in sequential dataset order.

**raw_response length distribution:** All 10042 entries have `len(raw_response) == 1`. The model returned a single letter for every question with no verbose output to parse.

---

## Section 2: Accuracy Recount

Three independent counting methods were applied:

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| `correct == True` field | 9217 | 10042 | 91.7845% |
| `predicted == expected` comparison | 9217 | 10042 | 91.7845% |
| Reported in results.json | — | 10042 | 91.78% |

**Delta from reported: 0.0045%** — within single-decimal rounding. The reported 91.78% is accurate.

Zero disagreements between the `correct` boolean field and the `predicted == expected` computed result across all 10042 entries. The bookkeeping is internally consistent.

---

## Section 3: Empty / Garbage Response Audit

All 10042 entries were scanned.

| Check | Count | Status |
|-------|-------|--------|
| Empty or whitespace `raw_response` | **0** | CLEAN |
| Non-null `error` field | **0** | CLEAN |
| Predicted letter absent from `raw_response` (silent parse-fallback signature) | **0** | CLEAN |

The historical "empty-response fallback to option A" bug that cost 108 entries in a prior run is **not present here**. Every entry has a single-char raw response matching the predicted answer exactly.

---

## Section 4: Wrong-Answer Audit

**Total wrong entries:** 825  
**Sample:** 80 entries (random.seed(42))

| Classification | Count | Pct of sample |
|----------------|-------|---------------|
| REAL (genuine model error) | 80 | 100% |
| ARTIFACT (script/transport bug) | 0 | 0% |

### 3 Representative REAL examples

**Entry #4663**
- Context: `[header] How to beat a "tough" person in a fight [title] Make the first move...`
- Model predicted: **A** | Gold: **B**
- Raw response: `'A'`
- Assessment: Model picked a plausible but wrong continuation.

**Entry #618**
- Context: `People are riding camels in a desert area. two individuals that are leading the...`
- Model predicted: **C** | Gold: **D**
- Raw response: `'C'`
- Assessment: Semantically close wrong answer; genuine comprehension error.

**Entry #130**
- Context: `A shot of a cyclist is shown and then it cuts back to the same man in white spea...`
- Model predicted: **C** | Gold: **B**
- Raw response: `'C'`
- Assessment: Genuine model error on an ambiguous video-description question.

**No artifacts found in the 80-entry sample.** All wrong answers show single-letter responses that are plausible wrong choices from the option set.

---

## Section 5: Answer Distribution

No fallback-to-first-option bias detected. Expected ~25% per option; all within normal variance.

### Model picks
| Option | Count | Percentage | Flag |
|--------|-------|------------|------|
| A | 2492 | 24.82% | — |
| B | 2403 | 23.93% | — |
| C | 2543 | 25.32% | — |
| D | 2604 | 25.93% | — |

### Gold distribution
| Option | Count | Percentage |
|--------|-------|------------|
| A | 2515 | 25.04% |
| B | 2485 | 24.75% |
| C | 2584 | 25.73% |
| D | 2458 | 24.48% |

**The 35% threshold was not breached by any option.** The model's answer distribution closely tracks the gold distribution, which is the expected signature of a well-calibrated model rather than a fallback pattern.

---

## Section 6: Timing / Contiguous Failure Check

No timestamp field is present in the JSONL; sequential position is the only ordering available.

| Metric | Value |
|--------|-------|
| Longest consecutive wrong-answer streak | **5** (entries #967–971) |
| Any streak >= 10 (transport-failure threshold) | **None** |

**The streak-of-5 was manually inspected.** All five entries show valid single-letter predictions for distinct questions with legitimate wrong but plausible answers:
- Entry 967: predicted B, gold D (newscast/gymnastics)
- Entry 968: predicted D, gold B (same context topic, different question)
- Entry 969: predicted C, gold B (toothbrush/bathroom)
- Entry 970: predicted A, gold B (harmonica player)
- Entry 971: predicted B, gold C (skateboarding)

No shared context or repeated endings — these are five independent questions. The streak is noise, not a transport stall.

---

## Section 7: Meta / Cache Verification

Contents of `heretic-cerebellum-v1_meta.json`:
```json
{
  "model_size": "unknown",
  "model_name": "heretic-cerebellum-v1",
  "port": 7890
}
```

**Observations:**
- `model_name` matches the filename prefix (`heretic-cerebellum-v1`). No cache contamination from a different model identity.
- `model_size: "unknown"` is incomplete but not a contradiction — the GGUF filename would be the authoritative source.
- `port: 7890` is consistent with a dedicated bench server (not the default 8080). This port is distinct from the production inference server (7800), which is correct practice.
- No model path or SHA fingerprint is stored, so hardware-level fingerprint verification is not possible from this file alone. This is a metadata weakness but does not contradict the run data.

**Cache contamination check:** The JSONL filename, results JSON, and meta JSON all reference `heretic-cerebellum-v1` consistently. The run timestamp (16:02) matches the JSONL mtime. No evidence of a stale cache from a different model.

---

## Summary Table

| Check | Result | Detail |
|-------|--------|--------|
| Entry count | PASS | 10042 entries, 0 parse errors |
| Accuracy recount | PASS | 91.7845% computed vs 91.78% reported (delta 0.0045%) |
| `correct` field vs computed | PASS | 0 disagreements across all 10042 entries |
| Empty raw_response | PASS | 0 empty entries |
| Error field set | PASS | 0 error-flagged entries |
| Silent parse fallback | PASS | 0 entries where letter absent from raw |
| Wrong-answer artifacts (80-sample) | PASS | 0 artifacts / 80 REAL |
| Answer distribution — model | PASS | A:24.82% B:23.93% C:25.32% D:25.93% (max 25.93%, below 35% threshold) |
| Answer distribution — gold | PASS | Uniform, no anomalies |
| Contiguous wrong streak | PASS | Max streak = 5, below 10-entry threshold |
| Meta model identity | PASS | model_name matches filename |
| Cache contamination | PASS | No cross-model fingerprint mismatch |

---

## Final Verdict

**TRUSTWORTHY**

- **Recount accuracy:** 91.7845% (rounds to 91.78%) — matches reported score exactly
- **Empty response count:** 0
- **Artifact count in 80-entry wrong-answer sample:** 0
- **Cache contamination flag:** None
- **Corrected score:** Not needed — reported score is accurate

The score can be published as-is. The only metadata gap is the absence of a model path or SHA in `meta.json`; future runs should record the GGUF path for traceability. This is a bookkeeping recommendation, not a validity concern.
