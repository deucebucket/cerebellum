# DRAFT A: reply on google/gemma-4-26B-A4B-it discussion #37

Status: DRAFT, not posted. Target: reply to thnamratha in the thread where Jerry
posted the v3 announcement (2026-05-01) and the v6 link (2026-05-09).

---

## POST BODY (everything between the rules is the post)

---

hey, following up since you said you wanted a closer look. a few things have landed since that v6 link.

current 26B builds, both 11.7 GB, measured on an RTX 3090 with llama.cpp:

- [Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF](https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF) is the repo i linked above. the current file is v6.1, same tensor allocation as v6 with updated chat-template metadata. benchmark summary JSONs are in the repo's `benchmark_results` folder.
- [Gemma-4-26B-A4B-it-Heretic-Cerebellum-GGUF](https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Heretic-Cerebellum-GGUF) is the same recipe applied to coder3101's heretic checkpoint, kept as a separate repo on purpose. that one carries per-question EvalPlus samples in its `benchmark_results` folder.

| benchmark | v6/v6.1 | heretic |
|---|---:|---:|
| ARC-Challenge | 95.56 | 95.48 |
| HellaSwag | 84.55 | 83.49 |
| MMLU-Redux | 71.33 | 71.42 |
| HumanEval / HumanEval+ | see note | 92.07 / 89.63 |

note on code scores: the heretic numbers are from the chat-completions harness with thinking off. v6's older HumanEval artifacts are marked for audit on the card (raw completions turned out to be the wrong way to bench gemma 4 code), so the heretic run is the one to trust for code right now.

the rest of the family is covered too: [E4B Cerebellum v2](https://huggingface.co/deucebucket/Gemma-4-E4B-it-Cerebellum-v2-GGUF), an [E4B heretic version](https://huggingface.co/deucebucket/Gemma-4-E4B-it-Heretic-Cerebellum-GGUF), and [E2B v2](https://huggingface.co/deucebucket/Gemma-4-E2B-it-Cerebellum-v2-GGUF).

i also ran the full sensitivity pass on the new 12B (the standard it checkpoint), and the honest result is that mixed precision had nothing to add at that size. the allocation my own data prescribed collapsed back to plain uniform Q4_K_M, byte for byte, so there was nothing of mine worth shipping. whatever went into training that one left no slack for a method like mine to work with. nice work on that model.

i've been keeping notes on a few measured model-level behaviors from all this downstream testing, things like per-architecture quant sensitivity patterns and a long-context reasoning quirk a user reported at 25k+ that i've also seen in my own sessions. the measured findings are written up; happy to share if useful.

---

## REVIEW NOTES (not part of the post, delete before use)

Vetting pass 2026-06-12 (independent re-verification, all sources re-opened):

- Voice sources: Jerry's live posts in discussion #37 (re-pulled 2026-06-12) and
  his replies in the v6 repo discussion 2. Last comment in #37 is Jerry's
  2026-05-09 v6 link with "a couple of changes since then" - the opener and the
  "repo i linked above" phrasing dovetail with that, no redundant bare re-link.
- v6/v6.1 scores 95.56 / 84.55 / 71.33: LIVE card (fetched 2026-06-12) metadata
  values 0.9556 / 0.8455 / 0.7133 AND the published evidence JSON
  `benchmark_results/cerebellum_v6_hellaswag_results.json` in the HF repo (84.55,
  timestamp 2026-05-04, 10042 q). The local
  `benchmarks/gemma4-26b-a4b/cerebellum_v6_hellaswag_results.json` (84.75) is an
  EARLIER run (timestamp 2026-05-01); the published/audited number is 84.55.
  RESOLVED: use 84.55 everywhere.
- Heretic scores 95.48 / 83.49 / 71.42 / 92.07 / 89.63: live heretic card
  metadata + Evaluation table (fetched 2026-06-12). Per-question samples JSONL
  confirmed present in that repo's tree; the v6 repo has summary JSONs only,
  which is why the artifact sentence is split per repo (the old draft's "audit
  files in each repo" claim was wrong and was removed).
- 11.7 GB: HF API file sizes are 11.75 GB (decimal) each; 11.7 GB is the
  figure previously published on the v6 card's At a Glance. Close enough,
  consistent with public record.
- "marked for audit" is the live v6 card's own language ("the retained v6
  HumanEval artifacts are marked for audit"). The old draft's "is being
  re-measured" was not verifiable from the current card and was cut.
- 12B paragraph: rewritten. The campaign quantized the standard
  google/gemma-4-12B-it checkpoint (RUN_PLAN.md inventory, F16 from HF
  safetensors), NOT the QAT release, and
  GEMMA4_12B_ARCH_RESEARCH_2026-06-12.md itself notes QAT alone does not
  explain the flat sensitivity (26B also has QAT releases). The old draft's
  "the QAT training holds up" causal claim was cut; replaced with the
  verified facts: v3 prescription rebuilt uniform Q4_K_M with identical
  sha256 (v3_build_info.json), nothing shipped, "no slack" framing is
  hedged as the campaign conclusion (SUMMARY_FOR_HUMAN.md verdict).
- Long-context sentence: one user report (tima2431, v6 repo discussion 2,
  2026-05-21, literal `enough;` at 25k+) plus Jerry's own confirmed sighting in
  the same thread. "users keep reporting" was cut to match the single-report
  evidence; "i can reproduce" softened to "i've also seen in my own sessions",
  which matches what Jerry already said publicly in that thread.
- All five linked repos re-fetched 2026-06-12, all live.
- No em dashes, no AI-tell constructions, casual lowercase matches his prior
  replies in this thread.
- Disclosure: nothing here goes beyond the live cards plus the degenerate-12B
  outcome (a phenomenon, no thresholds or allocation process named).
