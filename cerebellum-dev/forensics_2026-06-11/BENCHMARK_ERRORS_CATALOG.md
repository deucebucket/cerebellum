# Benchmark / Test / Logging Setup Errors — Merged Catalog
Date: 2026-06-11
Sources: forensics_2026-06-11/claude_sessions.md §3 and §5,
         forensics_2026-06-11/other_clis_sessions.md §I errors list.
Deduplicated: claude_sessions Incidents 1-9 merged with Codex errors 1-13.
All entries are original defects, not reformulations of each other.

Format per entry: date, benchmark, what broke, root cause, how discovered,
corrected number (if applicable), prevention rule.

---

## BE-01: HumanEval fence-stripping destroyed indentation (ALL models)

**Date:** Pre-2026-05-03; discovered 2026-05-03 ~01:30 CDT.
**Benchmark:** HumanEval pass@1, EvalPlus base.
**What broke:** `scripts/benchmark_humaneval.py` stripped code fences using
`.strip()`, which also removed leading whitespace from the first code line.
`normalize_indent()` computed wrong minimum indentation, corrupting all
subsequent lines -> `IndentationError` on execution -> systematic undercount.
```python
# BROKEN:
content = content[len("```python"):].strip()
# FIXED:
content = content[len("```python"):]
content = content.strip("\n")
```
**Root cause:** `.strip()` is not fence-safe; it removes all leading/trailing
whitespace including meaningful indentation.
**How discovered:** Temperature sweep showed anomalous lows (~25%) at temp>0 that
should not exist. Fixed script at temp=0 gave 82.9% immediately vs prior 75.0%.
**Corrected numbers:**
- Qwen 3.6 27B v4: 75.0% -> 81.10% (two independent reruns: 81.1%, 82.9%)
- Affected but not individually re-reported: Gemma 4 26B v4-v6, Qwen 3.5 9B,
  Granite 4.1 30B. All pre-2026-05-03 HumanEval scores are ~7-8 pts too low.
**Prevention rule:** Use EvalPlus upstream runner (`evalplus.codegen`), not
hand-rolled harness. If writing a harness, strip newlines only (`strip("\n")`),
never `.strip()` on code content.

---

## BE-02: ARC-Challenge numeric label mismatch (~22 questions)

**Date:** 2026-05-03 (discovered and fixed same day).
**Benchmark:** ARC-Challenge.
**What broke:** 22 ARC questions use numeric answer keys (1, 2, 3, 4) instead of
letters (A, B, C, D). Script prompted model to answer with a letter, but compared
against the numeric key. 19 of 22 numeric-key questions were marked wrong despite
correct answers.
**Root cause:** Missing normalization step:
```python
if answer_key.isdigit():
    answer_key = LETTERS[int(answer_key) - 1]
```
**How discovered:** Manual inspection of wrong-answer sample showed multiple
questions where model said "A" and key said "1" — clearly the same answer.
**Corrected numbers:** Qwen 3.6 27B v4: ARC ~95.1% -> 96.76%. All models pre-fix
have ARC ~1.6 pts too low.
**Prevention rule:** ARC answer-key normalization is mandatory. Add explicit test
case with a numeric-key question to the benchmark test suite. Audit wrong-answer
JSONL before recording any score (docs/benchmark_protocol.md).

---

## BE-03: HellaSwag empty responses from thinking-template handling

**Date:** 2026-05-03 (Qwen 3.6 27B era).
**Benchmark:** HellaSwag.
**What broke:** `chat_template_kwargs: {"enable_thinking": False}` occasionally
caused model to output only whitespace or `<think></think>`. Script retried twice
then marked wrong. 108/10042 questions affected.
**Root cause:** Template parameter handling for thinking models: `enable_thinking:
False` was not consistently honored by the server at this llama.cpp version.
**How discovered:** Reviewing HellaSwag wrong-answer JSONL showed a cluster of
near-empty response entries.
**Attempted fix that failed:** `prefix: True` (assistant prefill) -> llama.cpp
returned 400 Bad Request.
**Actual fix applied:** Reverted to `enable_thinking: False` accepting 108-question
noise (1.1 pts) as within acceptable variance for HellaSwag.
**Corrected number:** Qwen 3.6 27B v4 HellaSwag: 92.21% (with ~1.1 pt noise).
**Prevention rule:** For thinking models, always set both `enable_thinking: false`
AND `thinking_budget_tokens: 0` in every request. Audit empty/whitespace responses
in wrong-answer JSONL (predicated="?" cluster is the signal).

---

## BE-04: Gemma 4 HumanEval via raw /v1/completions — 3.05% false low

**Date:** 2026-05-18 (Heretic v1 first bad run). Same harness produced v6's
35.97% score on 2026-05-01.
**Benchmark:** HumanEval base, HumanEval+.
**What broke (run 1):** Used `/v1/completions` with `--reasoning off
--reasoning-budget 0`. Gemma 4 is a chat/thinking model; raw completions bypass
chat template -> malformed continuations.
**Score:** Base 3.05%, Plus 3.05% — invalid.
**Root cause:** Wrong API endpoint for chat models.
**How discovered:** 3.05% is implausibly low for a model of this class. Inspection
showed completions were raw text fragments, not code.
**Fix:** Switch to `/v1/chat/completions` with `enable_thinking: false`,
`thinking_budget_tokens: 0`, fence extraction, indentation normalization.
**Prevention rule:** Always confirm endpoint before running coding benchmarks.
Chat models must use `/v1/chat/completions`. Add endpoint as required field in
benchmark provenance metadata.

---

## BE-05: Gemma 4 HumanEval double-indentation bug — 17.07% false low

**Date:** 2026-05-18 (Heretic v1, bad run 2 after endpoint was fixed).
**Benchmark:** HumanEval base, HumanEval+.
**What broke:** Switched to `/v1/chat/completions` but indentation normalization
was wrong. Model outputs: first line indented correctly, subsequent lines one level
too deep (double-indentation pattern). 135 syntax errors in audit.
**Root cause:** Harness extracted raw body without normalizing indentation against
full function signature context.
**How discovered:** EvalPlus audit showed 136 fail/fail, 28 pass/pass, 135 syntax
errors. Pattern was consistent double-indentation.
**Fix:** Patched `scripts/benchmark_evalplus_chat.py`:
1. Use `ast.parse(prompt + body)` validation before writing
2. Normalize indentation with full function signature context
3. Extract `<final_code>...</final_code>` tag if present
4. Handle markdown fences defensively
**Corrected from saved outputs:** Base 92.68% / Plus 90.24%.
**Fresh full rerun (published):** Base 92.07% / Plus 89.63%.
**Prevention rule:** Run `ast.parse()` validation on every extracted code body
before evaluation. Check `scripts/benchmark_evalplus_chat.py` for the reference
implementation.

---

## BE-06: Gemma 4 v6 HumanEval thinking-enabled run — 36% false low

**Date:** 2026-05-01 (v6 session) / 2026-05-08 (rerun attempt).
**Benchmark:** HumanEval base, HumanEval+.
**What broke:** v6 HumanEval run used raw completions endpoint (see BE-04). The
2026-05-08 rerun used `--reasoning auto` without per-request `enable_thinking:
false`, so model output extensive reasoning before code which parser could not
handle. Both runs recorded 35.97% / 36.0%.
**Score in file:** `benchmark_results/cerebellum_v6_humaneval_results.json`:
`pass_at_1: 0.3597`. This file is invalid.
**Corrected estimate:** Heretic v1 on the same tensor map (v6 override file) scored
base 92.07% / plus 89.63% on fresh rerun with fixed harness. That is the best
available proxy for v6's true coding score; v6 itself has never been re-run with the
correct harness.
**Prevention rule:** After every benchmark run, immediately check if `pass_at_1` is
implausibly low vs comparable models. If yes, do not log the result; diagnose first.
Never publish a score from a file whose run provenance is ambiguous.

---

## BE-07: BENCH_WORKERS > 1 cache contamination for Gemma 4 EvalPlus

**Date:** 2026-05-18 (Heretic bench run, established as a rule).
**Benchmark:** EvalPlus (HumanEval+) for Gemma 4.
**What broke:** Multiple workers caused cache contamination or slot-reuse issues
with the llama-server, producing shared-context results across problems.
**Root cause:** llama.cpp's KV cache slot assignment is non-deterministic under
concurrent load with Gemma 4's thinking template.
**How discovered:** Multi-worker run results were inconsistent and showed higher
variance than expected; investigation pointed to shared context.
**Fix:** `BENCH_WORKERS=1` mandatory for all Gemma 4 EvalPlus runs.
**Score impact:** Not precisely quantified, but results were invalid.
**Prevention rule:** `BENCH_WORKERS=1` for any model using thinking/chat template
in EvalPlus. Set explicitly in `scripts/run_benchmarks.sh` so a stale env var
cannot override it. (This rule is already in CLAUDE.md and docs/benchmark_protocol.md.)

---

## BE-08: v7 HellaSwag false 0% (API transport failures counted as wrong)

**Date:** 2026-05-01 (Codex sessions, other_clis_sessions.md item 2).
**Benchmark:** HellaSwag.
**What broke:** Harness counted API transport failures (server not responding or
returning error codes) as wrong answers. HellaSwag reported 0% (or near 0%).
**Root cause:** `benchmark_utils.py` did not distinguish between "model answered
wrong" and "API call failed." All non-200 responses were classified as incorrect.
**How discovered:** Score of 0% on HellaSwag is impossible; investigation found
stale/unreachable server.
**Fix:** `benchmark_utils.py` patched to abort on API errors (not mark wrong). Rerun
gave ~84.05%.
**Prevention rule:** Benchmark harness must abort or raise on any HTTP non-2xx
response. Never silently count transport failures as wrong answers. The current
`benchmark_utils.py` has this fix — do not regress.

---

## BE-09: MMLU context abort at 600/2400 questions

**Date:** 2026-05-05 era (Codex sessions, other_clis_sessions.md item 3).
**Benchmark:** MMLU-Redux (2400 questions).
**What broke:** `--ctx-size 4096 --parallel 4` = 1024 tokens per slot. Some MMLU
prompts reach ~1070 tokens -> HTTP 400 from llama-server -> benchmark aborted at
question 600.
**Root cause:** Insufficient per-slot context for the full MMLU prompt distribution.
**How discovered:** Benchmark run aborted mid-way with HTTP 400 errors in log.
**Fix:** Minimum `--ctx-size 2048` per slot; for full suite use `--ctx-size 24576
--parallel 4` = 6144 tokens/slot. This is the current standard server invocation
per CLAUDE.md.
**Prevention rule:** Server must be launched with `--ctx-size 24576 --parallel 4`
before any benchmark suite run. This is the canonical flag set in CLAUDE.md.
Running a subset with smaller ctx is a known regression risk.

---

## BE-10: enable_thinking produces empty completions (Qwen 3.5 9B)

**Date:** 2026-05 (Codex sessions, other_clis_sessions.md item 7).
**Benchmark:** All (affected at server level).
**What broke:** Server defaulted to thinking mode for Qwen 3.5 9B -> `"content": ""`
for all requests. Every benchmark returned empty, scoring 0% across the board.
**Root cause:** No explicit `enable_thinking: false` + `thinking_budget_tokens: 0`
in benchmark request bodies; server default was thinking-on.
**How discovered:** All benchmark scores were 0% — clearly invalid.
**Fix:** Add `chat_template_kwargs: {"enable_thinking": false}` and
`thinking_budget_tokens: 0` to every request for thinking-capable models.
**Prevention rule:** For any model with thinking capability (Qwen 3.5, 3.6, Gemma 4,
etc.), both flags must be set explicitly in every benchmark request. Do not rely on
server defaults.

---

## BE-11: EvalPlus null pass_at_1_plus (summary parser bug)

**Date:** 2026-05-05 (Codex sessions, other_clis_sessions.md item 5).
**Benchmark:** EvalPlus (HumanEval+).
**What broke:** EvalPlus summary JSON parser did not handle the case where
`pass_at_1_plus` was `null` (no passing solutions). Script crashed or reported 0%
without distinguishing null from actual zero.
**Root cause:** Missing null check in summary parser:
```python
pass_plus = result.get("pass_at_1_plus") or 0.0  # null -> 0
```
**How discovered:** Parser threw exception on first model with 0 passing EvalPlus+
solutions.
**Fix:** Patched 2026-05-05.
**Prevention rule:** Use the upstream EvalPlus scorer (`evalplus.evaluate`), not
hand-written summary parsers. If a custom parser is needed, add a test case for
null/missing fields.

---

## BE-12: ARC TypeError crash (pool.map returned tuple)

**Date:** 2026-05-05 (Codex sessions, other_clis_sessions.md item 4).
**Benchmark:** ARC-Challenge.
**What broke:** `benchmark_arc.py:130` — `query_model()` returned a tuple; `pool.map`
passed it as a string to the next stage -> `TypeError`.
**Root cause:** Function signature mismatch between parallel worker and expected
string return type.
**How discovered:** Benchmark crashed with `TypeError` on line 130.
**Fix:** Patched to unpack tuple correctly.
**Prevention rule:** Always run a short smoke pass (5-10 questions) on any new
benchmark script before a full run. Type errors surface immediately at small scale.

---

## BE-13: Brainloop runs unauditable / generation never ran

**Date:** 2026-06 (conch-poc sessions).
**Benchmark:** HumanEval+ (conch-poc bench_results/).
**What broke:**
- `qwen7b-baseline` and `qwen7b-baseline-real`: 164 literal `"    pass"` completions,
  elapsed=0, pass@1=None. Generation never ran (server likely not started or wrong
  endpoint).
- `brainloop-best-combo` and `brainloop-fix-13k`: byte-identical (MD5 match). One was
  a file copy; no distinct model was benchmarked.
- `brainloop-rag-coding` and `brainloop-sharp-rag`: 97.6% identical; RAG intervention
  was inert.
- `recall_results_deadblock.json` A vs B: 100% identical despite different GGUF paths;
  server ran same model twice.
- All 9 named runs omit checkpoint path, git commit, GGUF path: permanently unauditable.
**Root cause:** Multiple: server not started, file copy passed as second run,
provenance fields not required by result format.
**How discovered:** Forensic audit 2026-06-11 comparing MD5s and elapsed times across
result JSON files.
**Prevention rule:** Result format must include checkpoint path, git commit, GGUF
SHA256, and server flags as required fields. Validate generation ran: `elapsed > 0`,
`pass_at_1 is not None`, completion content is not literal `"    pass"` for all
problems.

---

## BE-14: Published A/B comparison confounded by near-zero gate activations

**Date:** 2026-06 (conch-poc, Brainloop).
**Benchmark:** HumanEval+ A/B comparison (62.2% vs 56.7%).
**What broke:** Published comparison attributed -5.5% difference to the Brainloop
refiner. Actually:
- `fused_refiners.pt` had `tanh(gate) ≈ -0.005` on both injected layers (scaling
  contribution ~0.5%).
- L17 `inj_proj.weight` was exact identity matrix (L2 norm of W-I = 0.000, never
  received gradient updates).
- The -5.5-point difference reflected instruct-wrap prompt format difference, not
  the refiner.
**Root cause:** No gate-activation diagnostic before publishing A/B comparison.
Model was trained but weights showed no update had actually occurred for that layer.
**How discovered:** Forensic audit 2026-06-11 checked `inj_proj.weight` norm and
gate tanh values.
**Correction:** Posted to RESULTS.md and conch-poc/README.md 2026-06-11. The
"~5% logic tax" claim was retracted.
**Prevention rule:** Before publishing any A/B comparison involving trained modules,
verify: (a) gate activations are non-trivial; (b) weight norms differ from identity
or prior checkpoint; (c) prompt format is identical between conditions.

## BE-15 — ReadTimeout silently fabricates "    pass" completions (found 2026-06-11, live audit)
- **Benchmark**: EvalPlus (both `benchmark_evalplus.py:79` and `benchmark_evalplus_chat.py:199`)
- **What broke**: after 3 failed retries (ReadTimeout/ConnectError/HTTP error) the harness returned a literal `"    pass"` body, scored as a model failure. On the dense 27B at WORKERS=1 this hit 11% of heretic answers and 18.9% of stock-v4 answers in the same-night runs.
- **How discovered**: adversarial audit of the heretic 27B's anomalous 65.24 HumanEval base — `audit_evalplus_completions.py` give-up census traced every bare `pass` stub to the timeout handler.
- **Root cause**: silent failure substitution + fixed 2s backoff too short while the server is mid-long-generation.
- **Fix**: both scripts now raise RuntimeError after exhausted retries (runs are checkpoint-resumable) with linear backoff. Artifact-corrected heretic-27B base ≈ 75.6%.
- **Prevention rule**: a harness must never synthesize an answer; abort > fabricate. Audit give-up census after every code bench (already policy via audit_evalplus_completions.py — this is why it worked).
- **Related comparison error**: the May 27B "81.10 HumanEval" was measured with thinking enabled via chat endpoint — not comparable to no-think raw-completions runs. Same-harness same-night pairs are the only valid comparisons.
