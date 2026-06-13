# gemma 4 family quantization findings from downstream testing

**STATUS: DRAFT, not posted or shared. For Jerry's review.**

---

what this is: i build mixed-precision GGUF quants by measuring which parts of a
model tolerate heavy compression and which do not, then spending the size budget
where the measurements say it is needed. doing that across the gemma 4 family turned up
a set of model-level findings. this document records them as measured.
everything below was measured on an RTX 3090 with llama.cpp. published benchmark
numbers trace to summary JSONs in the linked repos; the newer releases also carry
per-question JSONL samples and audit reports.

models covered: 26B-A4B, E4B, E2B, 12B (gemma4_unified). plus one behavior
report that applies family-wide.

## 26B-A4B (MoE)

released builds, both 11.7 GB:

- [Cerebellum v6/v6.1](https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF): ARC-Challenge 95.56, HellaSwag 84.55, MMLU-Redux 71.33 (the v6-era code numbers are marked for audit on the card, so i am not quoting one)
- [Heretic Cerebellum](https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Heretic-Cerebellum-GGUF) (same recipe on coder3101's abliterated checkpoint): ARC 95.48, HellaSwag 83.49, MMLU 71.42, HumanEval 92.07 / HumanEval+ 89.63 (chat harness, thinking off), vision smoke 6/6

findings:

1. **router tensors broke under K-quant formats even at high bit width.** on the
   router layer we shipped demoted: Q8_0 measured -2.4% PPL vs F32, Q4_0 was
   neutral, but Q6_K was +15.9% and Q2_K +17.2%. a 6-bit K-quant was nearly as
   damaging as a 2-bit one, so bit width alone does not explain it; my best
   guess is the 256-element super-block scale structure interfering with expert
   selection, though i have not isolated the mechanism. Q8_0 was the only format
   that improved PPL on routers in this model. the full precision curve was
   published on the v6 card (the current card is a slimmed rewrite; the curve is
   in the repo's card history).
2. **routers tolerated single-point degradation but not stacking.** demoting any
   one of several router layers improved PPL on its own. demoting the top 3
   together was worse than baseline. the model compensated for one degraded
   router; it did not compensate for multiple at once.
3. **sensitivity is inverted vs dense-model intuition.** of the five non-router
   groups tested at Q2_K, attn_q was the only one that degraded (+13.4%); the
   large FFN and expert groups improved wiki PPL outright, ffn_up by 18.2%.
   crushed as a group, the routers were the most sensitive of all (+30.7%). in
   this architecture the expert tensors measured as the redundant part, and the
   attention and routing tensors measured as the fragile part.

## E4B: the PLE cliff

1. **per-layer embedding tensors are a single point of failure, and the failure
   is a cliff between Q4_K and Q3_K.** wiki PPL, 2048 ctx, full wikitext-2 test
   (PLE protection here means the 174 PLE tensors pinned to Q8_0; the shipped
   build pins them at Q5_K):

   | quant | PPL |
   |---|---:|
   | BF16 | 54.58 |
   | Q4_K_M | 55.74 |
   | Q3_K_M, PLE unprotected | 104.74 |
   | Q3_K_M, PLE protected | 55.82 |
   | Q2_K, PLE unprotected | 7296.76 |
   | Q2_K, PLE protected | 62.72 |

   the Q8_0 pins cost about 0.6 GB and removed the cliff in our measurements.
   PLE measured fine at Q4_K; the damage appeared at Q3_K.
2. **baseline calibration note:** E4B's wiki PPL
   measured around 54 at BF16; the working hypothesis for the high baseline is
   the 262k multimodal vocab. some "broken quant" reports compare against
   text-only gemma 3 numbers and read the high baseline as quant damage. the
   table above shows the high baseline at BF16, before any quantization.

released builds: [Cerebellum v2](https://huggingface.co/deucebucket/Gemma-4-E4B-it-Cerebellum-v2-GGUF)
(4.2 GiB: ARC 85.7, HellaSwag 75.3, MMLU 58.4, HumanEval 68.3) and a
[heretic version](https://huggingface.co/deucebucket/Gemma-4-E4B-it-Heretic-Cerebellum-GGUF)
(ARC 87.37, HellaSwag 74.98, MMLU 58.63, HumanEval 70.12 / HumanEval+ 65.24).

## E2B: an evals warning about derivative checkpoints

stock E2B quantizes normally and
[that build shipped](https://huggingface.co/deucebucket/Gemma-4-E2B-it-Cerebellum-v2-GGUF).
what did not ship was our build of a third-party abliterated E2B, and the reason
is relevant to anyone evaluating derivative checkpoints. we applied the same quant recipe to both
checkpoints, same harness, same night. the stock-source build measured 53.7
base / 48.8 plus on EvalPlus (chat, thinking off). the abliterated-source build
measured 22.6 / 22.0, a 31 point drop on HumanEval base, while multiple-choice
barely moved (ARC 70.4 vs 71.9). the source checkpoint's own reported KL from
base is 0.1651 (coder3101's published figure), 28x the 0.0058 of the E4B
heretic we shipped successfully, and its F16 perplexity already sat well above
our stock quant before our quantization touched it.

in these measurements, heavy activation-space edits on the small gemma 4 models
appear to hit code ability first, and the ARC/HellaSwag/MMLU style benches did
not catch it. anyone evaluating derivative checkpoints of the small models
should include a code bench.

## 12B (gemma4_unified): the sensitivity pass found no headroom, with data

i ran the same group-level sensitivity pass on the new 12B (the standard it
checkpoint) that worked across the rest of the family, expecting the usual
contrast between fragile and tolerant groups. the measurements show no such
contrast.

- 6 of 7 tensor groups degraded enough at 2-bit to need protection. ffn_down
  alone was +52.6% mean perplexity across our screening corpora (+80.5% in the
  worst one) when crushed.
- the one group that passed the perplexity screens at 2-bit (attn_v) still cost
  7.4 HumanEval+ points against the same-size uniform baseline when demoted.
  one hypothesis for why: the 12B shares K=V, and attn_v only exists in 40 of
  the 48 layers in the GGUF (the global attention layers carry no separate V),
  so each remaining V weight may carry more unique signal than in a
  fat-attention dense model. i cannot prove the mechanism from the outside. the
  measurement itself is solid, and it is a clean example of damage that shows
  up in code before it shows up in perplexity.
- perplexity moves sharply below 4-bit: Q3_K_M measured 1.65x the Q4_K_M
  wikitext number under our screening setup, though task benchmarks at Q3_K_M
  held up far better than that ratio suggests (it edged out Q4_K_M on
  HumanEval+, 84.8 vs 83.5, in our runs).

the conclusion i recorded: uniform Q4_K_M and Q3_K_M are already on the
frontier for this model. the measurements show no redundancy pool for mixed
precision to draw from. my working hypothesis is that encoder-free unified
weights plus K=V sharing remove the slack a method like mine depends on; i
cannot separate how much of that is architecture and how much is training. i
did not release a 12B build because i could not beat your own work, and that
looks like the correct outcome.

## long-context reasoning skip at 25k+ (behavior report)

what was reported and what i have seen: a user on my 26B thread reported that
gemma 4 based models served through llama.cpp with thinking enabled, at roughly
25k+ tokens of accumulated context, sometimes stop producing the reasoning
phase, emit a short give-up token (literally `enough;` in his report) and jump
straight to the final answer, even with reasoning forced at launch. he saw it
across multiple gemma 4 based models. in my own long agent/coding sessions the
matching symptom is: stall, resume briefly when prompted, stall again.

caveat: llama.cpp's gemma 4 chat-template and reasoning-budget handling have
had several open upstream issues, so i cannot cleanly separate model behavior
from serving behavior yet. i am building a benchmark to measure where reasoning
falls apart at long context. flagging it now anyway because the failure raises
no error, it is a silent reasoning skip, and the user who reported it read it
as model behavior. the first public report is on
[my 26B thread](https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF/discussions/2).

## what we did not find

worth stating for completeness:

- no NaN-class hard failures anywhere in gemma 4 in my testing. for contrast,
  the hybrid SSM architectures i have tested hard-fail below 4-bit with NaNs.
  gemma 4 degraded gradually or with a clean measurable cliff (PLE); it never
  numerically blew up.
- no vision projector fragility. mmproj at F16 over quantized backbones passed
  every vision smoke test i ran.
- no expert pathology on the 26B MoE. experts took heavy quantization
  gracefully. the measured risk concentrated in routers and attention.

## data available

the release repos carry benchmark summary JSONs; the newer releases (the
heretic line and E4B v2) also carry per-question JSONL, EvalPlus samples and
eval JSONs, and audit reports. behind that i have perplexity logs and
sensitivity measurements for everything described above. if any of this is
useful in more depth, ask and i will pull the specific measurements.

---

## REVIEW NOTES (internal, delete before sharing)

Vetting pass 2026-06-12: every number re-derived from primary evidence, all
cards re-fetched live, all cited threads re-pulled. Evidence map:

- 26B v6/v6.1 scores: LIVE card (2026-06-12) metadata 0.9556/0.8455/0.7133 and
  the HF repo's published `cerebellum_v6_hellaswag_results.json` (84.55,
  2026-05-04 run). The local benchmarks/ JSON showing 84.75 is the earlier
  2026-05-01 run; 84.55 is the published/audited number. RESOLVED, draft now
  uses exact card decimals throughout.
- 26B heretic scores + vision smoke 6/6: live heretic card metadata and
  Evaluation table; per-question samples confirmed in the repo tree.
- "marked for audit": live v6 card states it verbatim ("retained v6 HumanEval
  artifacts are marked for audit"). Verified public.
- router curve (-2.4 / ~0 / +15.9 / +17.2), stacking, layer-10 -3.9% PPL with
  HumanEval 71.3->61.6, group deltas (attn_q +13.4, ffn_up -18.2, routers
  +30.7), 18% largest improvement: all verified against the v6 card revision
  that was live 2026-05-01 through 2026-06-12 (fetched at commit 5fb6cee2).
  IMPORTANT: the CURRENT live card no longer carries any of this; the draft now
  says "was published / in the card history" instead of "is on the card".
  Disclosure: previously public, so no new leak. The layer-10 HumanEval drop
  used the v6-era harness now under audit; the draft says so explicitly.
- E4B PLE table: `osmosis-gemma4-e4b/ppl_results.md`, exact match all six rows.
  Protection in that sweep = Q8_0 pins (~0.6 GB, 174 tensors); shipped build
  pins Q5_K. The draft now states both, no format ambiguity. Q5_K pin and the
  ~104 -> ~55 magnitude are public on both E4B cards.
- E4B v2 + heretic scores: live cards (2026-06-12): 85.7/75.3/58.4/68.3 and
  87.37/74.98/58.63/70.12/65.24. Verified.
- E2B no-ship: `cerebellum-dev/forensics_2026-06-11/E2B_HERETIC_NOSHIP_FINDING.md`
  plus the raw JSONs in `cerebellum-gemma4-e2b-heretic/benchmark_results*`:
  heretic build 22.56/21.95 (2026-06-11 run), stock same-night 53.66/48.78,
  ARC 70.39 vs published 71.9, F16 PPL 150.16 vs stock quant 118.05.
  CORRECTED: the EvalPlus collapse was measured on the QUANT build (same recipe
  both sides), not at F16; F16 evidence is perplexity. The draft now separates
  the two claims. KL 0.1651 is on coder3101's public E2B card (verified live);
  0.0058 is on coder3101's E4B card and our public E4B heretic card. The public
  github README already states the E2B failure, the -31 pts figure, and even
  the KL >~0.05 screening note, so this section discloses nothing beyond
  public except per-bench detail (22.6/22.0, ARC 70.4), which is benchmark
  data, not method internals. The screening ceiling RULE is still not stated
  in the draft.
- 12B: SUMMARY_FOR_HUMAN.md (6 of 7 protect, attn_v ~7.4 HumanEval+ cost vs
  same-size uniform Q3_K_M: 77.4 vs 84.8; verdict "uniform Q4_K_M and Q3_K_M
  already on the frontier"); ffn_down re-derived from
  `ablation_results_multidomain.json`: wiki 2591.09/1435.57 = +80.5% worst,
  mean across the four domains = +52.6%. Q3/Q4 ratio re-derived from
  `stage1_baselines.tsv`: 2574.4694/1561.6245 = 1.6486. HumanEval+ 84.8 vs
  83.5 from SUMMARY scores table (audited baseline runs). attn_v in 40 of 48
  layers re-verified by counting `ablation_tensors.txt`; `attention_k_eq_v:
  true` and `model_type: gemma4_unified` re-verified from the live
  google/gemma-4-12B-it config.json. CUT: the "MQA 1 KV head global" detail
  (not independently verifiable from top-level config) and ALL causal QAT
  attribution. The campaign source was the standard it checkpoint
  (RUN_PLAN.md inventory), not the QAT release, and the arch research doc
  itself says QAT alone does not explain the flatness. K=V/V-signal
  explanation is now explicitly labeled a hypothesis.
- "the quantization floor sits right at 4-bit" was CUT as an overclaim: task
  benches at Q3_K_M actually beat Q4_K_M on HumanEval+ (84.8 vs 83.5). The
  draft now states the ratio plus the counterpoint.
- 25k collapse: tima2431's report re-pulled live from v6 repo discussion 2
  (2026-05-21, literal `enough;`, 25k+, forced reasoning, multiple gemma 4
  models) plus Jerry's own reply confirming the stall pattern, plus his
  2026-06-12 comment saying he is building a long-context reasoning benchmark
  (the draft's "i am building a benchmark" line matches what he already said
  publicly). CHANGED: "i can reproduce" softened to match the actual evidence
  (his observed symptom is the stall pattern, not a controlled repro);
  "multiple checkpoints" attributed to the reporting user; the test matrix in
  `docs/gemma4_long_context_thinking_issue_20260521.md` is planned, not run.
- SSM contrast: project canon (CLAUDE.md gotchas, mamba_hybrid_findings.md
  lineage) for Granite 4.0-H-Small / Qwen 3.5 9B; about non-gemma models,
  stated as own-testing experience, public-safe.
- "data available" section REWRITTEN for accuracy: v6 repo has summary JSONs
  only; E2B v2 repo has NO benchmark_results folder; per-question JSONL +
  audit reports confirmed only in the heretic repos and E4B v2. The old
  blanket "every released build carries per-question JSONL plus audit" claim
  was false and is gone.

disclosure check against `docs/public_release_scope.md` (re-read in full):

- shared: phenomena, magnitudes, public card numbers, repro conditions, one
  hypothesis clearly labeled as such.
- NOT shared: allocation/budget process, classification thresholds, override
  file contents, KL screening ceiling rule (already on the public README, but
  still not repeated here), imatrix/pipeline internals, screening corpus
  composition (the four domains are not named).
- 26B magnitudes: previously public on the 2026-05 card revisions (verified at
  commit 5fb6cee2). New-to-public material in this draft: 12B group-level
  magnitudes (+52.6%/+80.5%, 7.4 pts, 1.65x) and E2B per-bench detail. Both
  are measurements/phenomena of the kind the scope doc allows; neither names a
  threshold or selection rule. Jerry should still eyeball the 12B bullets
  before sending since they are the only numbers never published anywhere.
- no em dashes, no "not X but Y" constructions, superlatives removed, causal
  claims hedged or cut.
