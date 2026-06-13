# Gemma drafts adversarial vetting report — 2026-06-12

Independent verification pass on `DRAFT_A_gemma26b_discussion37_reply.md` and
`GEMMA_TEAM_FINDINGS_DRAFT.md`. Every number re-derived from primary disk
evidence or live fetches; the drafts' own review-notes citations were NOT
trusted, every cited file was opened and every card/thread re-pulled live.

Nothing posted, nothing committed. Both drafts revised in place.

## Verdict summary

- Claims audited: 47
- Verified against primary evidence: 38
- Corrected in revision: 7
- Cut as unverifiable or overclaimed: 5 (QAT attribution x2, "4-bit floor",
  "every repo has JSONL+audit", "is being re-measured")
- Unverifiable claims remaining in either draft: 0

## Flag resolutions (the seven known flags)

(a) **HellaSwag 84.55 vs 84.75 — RESOLVED: 84.55.** The HF repo's published
evidence JSON (`benchmark_results/cerebellum_v6_hellaswag_results.json`, live)
reads 84.55, timestamp 2026-05-04, 10042 questions, and the live card metadata
says 0.8455. The local `benchmarks/gemma4-26b-a4b/` JSON reading 84.7540 is an
earlier run (timestamp 2026-05-01). The published/audited number is the May 4
rerun. Both drafts now use 84.55 (exact decimals, no rounding ambiguity).

(b) **Router curve numbers — VERIFIED, with a status correction.** -2.4%
(Q8_0), ~0% (Q4_0), +15.9% (Q6_K), +17.2% (Q2_K) vs F32 12,356 match the v6
card revision live from 2026-05-01 to 2026-06-12 (verified at HF commit
5fb6cee2). The CURRENT live card (rewritten, fetched today) carries none of it.
Draft B's "full precision curve is on the v6 model card" was false as of today
and now reads "was published on the v6 card ... in the repo's card history."
Disclosure unchanged: previously public.

(c) **E2B KL/EvalPlus — VERIFIED with one framing correction.** KL 0.1651 is
publicly on coder3101's E2B heretic card (fetched live); 0.0058 on the E4B
side (coder3101 + our public E4B heretic card); ratio 28.5x. EvalPlus
53.66/48.78 (stock, same night) vs 22.56/21.95 (heretic build) and ARC 70.39
vs 71.9 match the raw JSONs in `cerebellum-gemma4-e2b-heretic/benchmark_results*`
(2026-06-11 runs) and the no-ship doc. CORRECTION: the EvalPlus collapse was
measured on the QUANT build (same recipe both sides), not at F16; the F16
evidence is perplexity only (150.16 vs stock quant 118.05). Draft B previously
implied the code drop itself was measured at F16; it now separates the two.

(d) **12B numbers — VERIFIED by re-derivation.** From
`ablation_results_multidomain.json`: ffn_down wiki 2591.09 vs baseline 1435.57
= +80.5% (worst domain); mean across wiki/code/math/dialogue deltas
(+80.5/+44.1/+30.1/+55.7) = +52.6%. From `stage1_baselines.tsv`:
2574.4694/1561.6245 = 1.6486 = 1.65x. 7.4 HumanEval+ points: SUMMARY_FOR_HUMAN
(77.4 v2 vs 84.8 same-size uniform Q3_K_M); draft now states the comparison
baseline. attn_v in 40 of 48 layers: counted in `ablation_tensors.txt`.
`gemma4_unified` and `attention_k_eq_v: true`: live config.json. CUT: "MQA 1
KV head global layers" (top-level config shows num_key_value_heads 8; the
per-layer global claim was not independently verifiable) and the QAT causal
attribution (campaign source was the standard it checkpoint per RUN_PLAN.md;
the arch doc itself caveats that QAT alone does not explain flat sensitivity,
since 26B also has QAT releases). K=V explanation now labeled a hypothesis.

(e) **E4B PLE table — VERIFIED, exact.** All six rows (54.58 / 55.74 / 104.74 /
55.82 / 7296.76 / 62.72) match `osmosis-gemma4-e4b/ppl_results.md` to the
digit. The "protected" rows are Q8_0 pins (~0.6 GB over 174 tensors); the
shipped v2 recipe pins Q5_K. Draft B now states both explicitly instead of
staying ambiguous.

(f) **Every benchmark score vs live cards TODAY — VERIFIED.**
- v6: 95.56 / 84.55 / 71.33 (card metadata + tables, fetched 2026-06-12).
- 26B heretic: 95.48 / 83.49 / 71.42 / 92.07 / 89.63, vision smoke 6/6.
- E4B v2: 85.7 / 75.3 / 58.4 / 68.3, 4.2 GB-class file (4,498,725,056 bytes =
  4.19 GiB; stock card says 4.2 GB, heretic card says 4.2 GiB ~4.51 GB).
- E4B heretic: 87.37 / 74.98 / 58.63 / 70.12 / 65.24.
- E2B v2: 71.9 / 50.0 / 47.4 / 46.3 (used only as the stock reference point).
- File sizes: both 26B GGUFs are 11.75 GB decimal via HF API; "11.7 GB"
  matches the previously published At a Glance figure, kept.

(g) **"v6 code marked under audit is public" — VERIFIED.** The live v6 card
states: "the retained v6 HumanEval artifacts are marked for audit in local
notes" and excludes HumanEval from metadata for that reason. Draft A's "is
being re-measured" was NOT verifiable from the current card and was cut;
"marked for audit on the card" stands. Heretic 92.07/89.63 chat-harness
provenance and the audit line (0 echoes / 0 repeats / 0 pass-only / 2 syntax
failures) are on the live heretic card.

## Full claims table

Sources: [L] = live HF fetch 2026-06-12, [D] = disk file opened this pass.

| # | Claim (as now in drafts) | Source | Actual value | Status |
|---|---|---|---|---|
| 1 | v6/v6.1 ARC 95.56 | [L] v6 card metadata + repo JSON | 95.5631 | VERIFIED |
| 2 | v6/v6.1 HellaSwag 84.55 | [L] repo evidence JSON (05-04 run) | 84.55 | VERIFIED (84.75 local = earlier run) |
| 3 | v6/v6.1 MMLU-Redux 71.33 | [L] card + repo JSON | 71.3333 | VERIFIED |
| 4 | v6 code score not quoted, "marked for audit" public | [L] live v6 card | verbatim on card | VERIFIED |
| 5 | heretic ARC 95.48 | [L] heretic card | 95.48 | VERIFIED |
| 6 | heretic HellaSwag 83.49 | [L] heretic card | 83.49 | VERIFIED |
| 7 | heretic MMLU 71.42 | [L] heretic card | 71.42 | VERIFIED |
| 8 | heretic HumanEval 92.07 / + 89.63, chat no-think | [L] heretic card metadata + table | 92.07 / 89.63 | VERIFIED |
| 9 | heretic vision smoke 6/6 | [L] heretic card artifacts table + Jerry's 05-19 comment | 6/6 | VERIFIED |
| 10 | both 26B builds 11.7 GB | [L] HF API tree | 11.75 GB each | VERIFIED (pub. card figure) |
| 11 | v6.1 = v6 allocation, metadata-only template fix | [L] v6 card; memory canon | "zero tensor changes" | VERIFIED |
| 12 | heretic = same recipe on coder3101 checkpoint, separate repo | [L] heretic card provenance; #37 thread | transferred verbatim | VERIFIED |
| 13 | router curve -2.4 / ~0 / +15.9 / +17.2 vs F32 | [L] v6 card rev 5fb6cee2 | exact match | VERIFIED; "is on card" CORRECTED to "was published" |
| 14 | Q8_0 only format that improved router PPL | [L] same | "Q8_0 is the only precision that improves PPL" | VERIFIED (draft's "only safe" CORRECTED, Q4_0 was neutral) |
| 15 | single router demotions improved PPL, top-3 stack worse than baseline | [L] same | 6 layers improve; stack worsens | VERIFIED |
| 16 | layer improved wiki PPL 3.9%, HumanEval -~10 pts | [L] same (layer 10: 71.3 -> 61.6) | -9.7 pts | VERIFIED + harness caveat added |
| 17 | attn_q only degrading non-router group +13.4% | [L] same | +13.4% | VERIFIED (was "most fragile group", CORRECTED: routers +30.7%) |
| 18 | ffn_up improved 18.2% at Q2_K; routers +30.7% as group | [L] same | -18.2% / +30.7% | VERIFIED |
| 19 | super-block scale structure explanation | [L] same (asserted on card) | hypothesis | NOW HEDGED as "best guess" |
| 20 | E4B PLE table, six rows | [D] osmosis-gemma4-e4b/ppl_results.md | exact | VERIFIED |
| 21 | 174 PLE tensors, ~0.6 GB at Q8_0; shipped pin Q5_K | [D] ppl_results.md; [L] both E4B cards | 174 / ~0.6 GB / Q5_K | VERIFIED, now explicit |
| 22 | E4B BF16 wiki PPL ~54, 262k vocab, gemma-3 comparison confusion | [D] ppl_results.md finding 6 | 54.58 / 262K | VERIFIED |
| 23 | E4B v2 85.7/75.3/58.4/68.3, 4.2 GiB | [L] E4B v2 card + heretic card byte count | match; 4.19 GiB | VERIFIED |
| 24 | E4B heretic 87.37/74.98/58.63/70.12/65.24 | [L] E4B heretic card | exact | VERIFIED |
| 25 | E2B stock shipped | [L] E2B v2 repo | live | VERIFIED |
| 26 | E2B heretic KL 0.1651, publicly reported | [L] coder3101 E2B card | 0.1651 | VERIFIED |
| 27 | 28x vs E4B 0.0058 | [L] coder3101/our E4B cards | 28.5x | VERIFIED |
| 28 | stock EvalPlus 53.7/48.8 same night | [D] benchmark_results_stock JSON | 53.66/48.78 | VERIFIED |
| 29 | heretic-source EvalPlus 22.6/22.0, -31 base | [D] heretic-e2b JSONs | 22.56/21.95 | VERIFIED |
| 30 | MC barely moved, ARC 70.4 vs 71.9 | [D] heretic JSON; [L] E2B card | 70.39 / 71.9 | VERIFIED |
| 31 | code collapse "already present at F16" | [D] noship doc: F16 evidence is PPL 150.16 | EvalPlus was on quant | CORRECTED: claims separated |
| 32 | -31 pts figure public | [D] origin/main README (public repo, HTTP 200) | "-31 pts at full precision" | VERIFIED public |
| 33 | 12B is gemma4_unified, attention_k_eq_v true | [L] google/gemma-4-12B-it config.json | exact | VERIFIED |
| 34 | attn_v in 40 of 48 layers | [D] ablation_tensors.txt count | 40 | VERIFIED |
| 35 | 6 of 7 groups need protection | [D] SUMMARY_FOR_HUMAN.md | 6 of 7 | VERIFIED |
| 36 | ffn_down +52.6% mean / +80.5% worst | [D] ablation_results_multidomain.json, re-derived | 52.58% / 80.49% | VERIFIED |
| 37 | attn_v passed PPL screens, cost 7.4 HumanEval+ pts | [D] SUMMARY (77.4 vs 84.8 same-size Q3_K_M) | 7.4 | VERIFIED, baseline now named |
| 38 | K=V sharing explanation for attn_v damage | [D] arch research doc | analysis only | NOW LABELED HYPOTHESIS |
| 39 | Q3_K_M = 1.65x Q4_K_M under screen | [D] stage1_baselines.tsv | 1.6486 | VERIFIED |
| 40 | Q3_K_M beat Q4_K_M on HumanEval+ 84.8 vs 83.5 | [D] SUMMARY scores table | exact | VERIFIED (added as counterpoint) |
| 41 | "quantization floor sits right at 4-bit" | contradicted by #40 | overclaim | CUT |
| 42 | QAT training causal attribution (both drafts) | [D] RUN_PLAN (source = standard it ckpt); arch doc's own caveat | unsupported | CUT |
| 43 | uniform Q4_K_M/Q3_K_M on the frontier, no 12B release | [D] SUMMARY verdict; v3_build_info sha match | NO-SHIP, byte-identical v3 | VERIFIED |
| 44 | 25k+ `enough;` reasoning skip report | [L] v6 repo discussion 2 (tima2431, 05-21) | verbatim incl. `enough;`, 25k+, forced reasoning | VERIFIED |
| 45 | "i can reproduce" | [L] thread + [D] long-context doc (test matrix planned, not run) | own sighting = stall pattern | CORRECTED to "i have seen the matching symptom"; benchmark-in-progress line matches his public 06-12 comment |
| 46 | no NaN failures in gemma 4; SSM contrast | canon (CLAUDE.md gotchas, mamba findings lineage) | consistent | VERIFIED as own-testing statement |
| 47 | "every repo carries per-question JSONL + audit" | [L] HF API trees: v6 = summaries only, E2B v2 = NO benchmark_results dir | false as stated | CORRECTED in intro + data section |

## Disclosure verdicts (fresh read of docs/public_release_scope.md)

- **26B section: CLEAR.** Every magnitude was on the public v6 card from
  2026-05-01 to 2026-06-12 (verified at the archived revision). The scope doc
  itself says to assume previously-public material is cached. No thresholds,
  no allocation process, no override contents.
- **Router curve vs current v6 card: does NOT exceed what was published**, but
  the current card no longer shows it; draft language fixed so it does not
  misdescribe the live card.
- **E2B section: CLEAR, does not leak the screening rule.** The phenomenon,
  the -31 pts magnitude, and even the KL >~0.05 screening note are on the
  public github README (repo returns HTTP 200, paragraph confirmed at
  origin/main). The draft still does not state the ceiling rule. The only
  new-to-public material is per-bench detail (22.6/22.0, ARC 70.4), which is
  benchmark data on a build that never shipped; scope doc treats benchmark
  results as releasable and bans method internals, none of which appear.
- **12B section: the only genuinely never-published numbers in either draft**
  (+52.6%/+80.5%, 7.4 pts, 1.65x, 84.8 vs 83.5). They are measurements and
  phenomena, not selection rules; the four screening domains are not named;
  no PROTECT/DEMOTABLE bands, no budget process. Judged in-scope, but since
  they have appeared nowhere public before, Jerry should consciously approve
  these specific bullets before the doc leaves the kitchen.
- **Long-context section: CLEAR.** Everything traces to the public thread
  plus hedged local observation.

## Thread fit (Draft A)

Discussion #37 re-pulled with auth. Sequence: Jerry's v3 announcement
(2026-05-01), thnamratha's reply asking for a working link "so we can take a
closer look" (2026-05-08), Jerry's bare v6 link with "a couple of changes
since then, but hopefully improvements" (2026-05-09). Nothing after that.
The revised draft opens off "you said you wanted a closer look", refers to the
v6 link as "the repo i linked above" (no redundant bare re-link), contradicts
nothing he said, and stays in the lowercase casual register of his replies in
this and the v6-repo thread. The heretic repo and family links are new to this
thread. The "kept as a separate repo on purpose" phrasing matches his public
05-19 comment in the other thread.

## Residual risk

- Zero unverifiable claims remain by my audit. Two judgment calls to be aware
  of: (1) the 12B never-published numbers above; (2) the layer-10 "~10
  HumanEval points" finding rests on the v6-era harness that is publicly
  marked for audit; the draft discloses that caveat inline, but if Jerry wants
  zero audit-era numbers in front of Google, that bullet can drop without
  weakening the section (finding 1, 2, 4 carry it).
- The E4B stock card's headline "4.2 GB" vs byte-accurate 4.19 GiB (~4.51 GB)
  is a pre-existing card inconsistency, not a draft problem; draft uses GiB.
- The 26B "11.7 GB" is the published card figure; HF API says 11.75 GB
  decimal. Within rounding of the public record.

---

## REGISTER PASS LOG (2026-06-12, voice/register only, zero factual content changed)

Spec applied: no punchy lines, decisions attributed to measurements not
judgment, hypotheses stay labeled, casual lowercase register preserved, no em
dashes / exclamations / rhetorical questions / not-X-but-Y.

DRAFT_A_gemma26b_discussion37_reply.md, 1 line rewritten:

1. "left no slack for a method like mine to harvest" -> "left no slack for a
   method like mine to work with" (metaphor flattened; the hedged "whatever
   went into training" framing and all facts unchanged)

GEMMA_TEAM_FINDINGS_DRAFT.md, 12 lines rewritten:

1. intro: "spending the size budget where the model actually needs it" ->
   "spending the size budget where the measurements say it is needed"
2. finding 2: "compensated for one degraded router, not for multiple at once"
   -> "compensated for one degraded router; it did not compensate for
   multiple at once"
3. finding 4: "the expert mass held the redundancy and the attention and
   routing side held the risk" -> "the expert tensors measured as the
   redundant part, and the attention and routing tensors measured as the
   fragile part"
4. E4B: "Q4_K handled PLE fine, Q3_K destroyed it." -> "PLE measured fine at
   Q4_K; the damage appeared at Q3_K."
5. E4B note 2: "which we attribute to the 262k multimodal vocab" -> "the
   working hypothesis for the high baseline is the 262k multimodal vocab"
   (causal claim relabeled as hypothesis, same content)
6. E4B note 2: "read the high baseline as quant damage. it is not." -> "...as
   quant damage. the table above shows the high baseline at BF16, before any
   quantization." (two-word punch replaced with the measurement it rests on)
7. E2B: "takeaway: ...edits appear to destroy code ability first, and
   ARC/HellaSwag/MMLU style benches will not catch it" -> "in these
   measurements, ...edits appear to hit code ability first, and the
   ARC/HellaSwag/MMLU style benches did not catch it" ("will not" prediction
   narrowed to what was measured; "appear to" hedge preserved)
8. 12B heading: "nothing for me to harvest, with data" -> "the sensitivity
   pass found no headroom, with data"
9. 12B opener: "this one is a compliment. ...there is none." -> opener cut;
   "...the measurements show no such contrast." (the compliment survives in
   the closing line, which was already plain)
10. 12B conclusion: "there is no redundancy pool for mixed precision to
    harvest" -> "the measurements show no redundancy pool for mixed precision
    to draw from"
11. 12B conclusion: "my read is that ...remove the slack a method like mine
    feeds on, and the model behaves as if it arrives pre-compressed" -> "my
    working hypothesis is that ...remove the slack a method like mine depends
    on" (the "arrives pre-compressed" quotable was removed; the
    cannot-separate-architecture-from-training hedge kept verbatim)
12. long-context: "caveat, and it is a real one:" -> "caveat:"; "to measure
    where reasoning falls apart at long context instead of guessing" -> same
    without "instead of guessing"; "a silent reasoning skip rather than an
    error, which is exactly the kind of thing users blame on the model" ->
    "the failure raises no error, it is a silent reasoning skip, and the user
    who reported it read it as model behavior" (claim narrowed to the actual
    reporter)
13. "worth stating because the family is clean here:" -> "worth stating for
    completeness:"
14. "the risk concentrated in routers and attention rather than expert mass"
    -> "the measured risk concentrated in routers and attention" (redundant
    tail dropped; prior sentence already states experts tolerated it)

All benchmark numbers, PPL curves, sizes, dates, links, and hypothesis labels
untouched. Nothing committed, nothing posted.
