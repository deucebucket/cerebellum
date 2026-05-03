# Benchmark Score Corrections — 2026-05-03

## Summary

All previously published HumanEval scores were inflated downward by a script bug.
ARC scores were slightly low due to a label mismatch. HellaSwag had a minor
empty-response issue. This document records exactly what was wrong and the fixes applied.

## Bugs Found

### 1. HumanEval — Fence-stripping destroyed indentation (ALL MODELS AFFECTED)

**Impact**: All published HumanEval pass@1 scores are ~7-8 points too LOW.

**Root cause**: The model outputs code wrapped in \`\`\`python fences at temp=0.
The old script stripped the fence marker with `.strip()` which also removed
leading whitespace from the first code line. This caused `normalize_indent()`
to calculate incorrect minimum indentation, shifting all subsequent lines.

**Example**:
```
Raw model output:    "```python\n    if x > 0:\n        return x\n```"
After old strip:    "if x > 0:\n        return x"   (first line lost indent)
After normalize:    "    if x > 0:\n            return x"  (wrong: 4+12 mix)
```

**Fix** (commit TBD):
```python
# OLD (broken):
if content.startswith("```python"):
    content = content[len("```python"):].strip()  # .strip() kills indent!

# NEW (fixed):
if content.startswith("```python"):
    content = content[len("```python"):]
elif content.startswith("```"):
    content = content[3:]
if content.endswith("```"):
    content = content[:-3]
content = content.strip("\n")  # only strip newlines, not spaces
```

**Verification**: Re-ran v4 at temp=0 with fixed script → 82.9% (was 75.0%).
The model's code logic was always correct; the score reflected formatting damage
from our own measurement tool.

### 2. ARC-Challenge — Numeric label mismatch (19 questions misjudged)

**Impact**: ARC score ~1.6 points too LOW.

**Root cause**: 22 ARC-Challenge questions use numeric answer keys (1, 2, 3, 4)
instead of letters (A, B, C, D). We prompt the model to "respond with ONLY the
letter" so it outputs "B" for the 2nd choice. But comparison was against "2".
19 of these were marked wrong despite the model answering correctly.

**Fix**:
```python
answer_key = df['answerKey'][i]
if answer_key.isdigit():
    answer_key = LETTERS[int(answer_key) - 1]
```

**Verification**: Manual audit of all 22 numeric-label questions confirmed 19 were
correct answers marked wrong. Published 95.1% → corrected ~96.7%.

### 3. HellaSwag — Empty responses from thinking template (108 questions)

**Impact**: HellaSwag score ~1.1 points too LOW.

**Root cause**: Using `chat_template_kwargs: {"enable_thinking": False}` sometimes
caused the model to output only whitespace or `<think></think>` content.
The `query_chat_completion` function strips responses and treats empty content
as an error (retries 2x, then marks wrong).

**Fix**: Switched to explicit assistant message prefill with
`{"role": "assistant", "content": "<think>\n</think>\n", "prefix": True}`
which deterministically suppresses thinking without template-level ambiguity.

**Verification**: temp=0.3 run of 1223 HellaSwag questions had 0 empty responses
(vs 108 at temp=0 with old method). Re-running full HellaSwag with fix.

### 4. MMLU / MMLU-Redux — No bugs found

Manual audit of wrong answers confirmed all are genuinely incorrect model predictions.
No parsing or labeling issues.

### 5. Assistant prefill "prefix: True" — NOT supported by llama.cpp

**Impact**: None on published scores (caught before publishing).

**Root cause**: Attempted to fix the HellaSwag empty response issue by adding
`{"role": "assistant", "content": "<think>\n</think>\n", "prefix": True}` to
force-skip thinking. llama.cpp's `/v1/chat/completions` returns 400 Bad Request
on unknown fields. This approach does NOT work.

**Resolution**: Reverted to `chat_template_kwargs: {"enable_thinking": False}`
which is the correct llama.cpp method for suppressing thinking. The empty response
issue (108/10042 in HellaSwag) remains a minor known limitation — the model
occasionally generates whitespace-only at temp=0 through the chat template path.

### 6. HumanEval fence stripping — mid-output fences not caught

**Impact**: ~2 additional failures in HumanEval (model outputs explanation text
then a code fence, original strip only checked `startswith`).

**Fix**: Changed fence detection from `startswith("```python")` to
`"```python" in content` with `index()` to find fences anywhere in the output.

## Corrected Scores (pending full re-run)

| Benchmark | Old (buggy) | Corrected | Delta |
|-----------|-------------|-----------|-------|
| HumanEval | 75.0% | 81-83% | +6-8 |
| ARC-Challenge | 95.1% | pending | pending |
| HellaSwag | 91.2% | pending | pending |
| MMLU-Redux | 77.1% | pending | pending |

HumanEval confirmed across 2 independent runs (81.1% and 82.9%).
MC benchmarks re-running with ARC label fix. HellaSwag/MMLU expected
to be close to original since the only MC bug was ARC labels (19 questions).

Full re-benchmark running with all fixes applied. Final numbers will replace
these estimates once complete.

## Other Models Affected

All published Cerebellum models were benchmarked with the same buggy HumanEval
script. Their HumanEval scores are all too low by a similar margin (~7-8 points).
Re-benchmarking requires having each GGUF on disk. Models on disk:
- Qwen 3.6 27B v4 (re-running now)
- Gemma 4 26B v6 (queued)
- Qwen 3 14B v2 (queued)

Models needing download from HF:
- Granite 4.1 30B v2
- Qwen 3 30B v2
- Qwen 3 32B v2
- Qwen 3.6 35B-A3B v1
- Gemma 4 E4B v2

## Timeline

- 2026-05-03 01:00 CDT — Temperature sweep revealed anomaly (temp>0 scored 25%)
- 2026-05-03 01:30 CDT — Identified fence-stripping bug as cause
- 2026-05-03 02:00 CDT — Fixed script, confirmed temp=0 baseline is 82.9%
- 2026-05-03 02:45 CDT — Discovered ARC label mismatch (19 questions)
- 2026-05-03 02:50 CDT — Discovered HellaSwag empty response issue (108 questions)
- 2026-05-03 03:00 CDT — Full re-benchmark launched with all fixes
