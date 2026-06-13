# Vetting Report: docs/how_cerebellum_works.md

Date: 2026-06-12. Private (cerebellum-dev, never push to origin).
Scope: adversarial three-pass vetting of the public explainer draft before it
carries the user's name. Live HF cards re-curled 2026-06-12 ~13:10 CDT
(post-fix: 27B PPL table and 35B "4 of 5"/"7 of 10" corrections are confirmed
live).

Verdict summary: 24 claims VERIFIED as written or after minor rewording,
9 CORRECTED, 3 CUT, 1 residual-risk item retained with scoping (see end).

---

## PASS 1: Fact audit, claim by claim

Sources used (all on disk or curled live):
- PV = cerebellum-dev/PIPELINE_VERIFICATION_2026-06-12.md (claim numbers cited)
- CM = cerebellum-dev/knowledge/CURRENT_METHOD.md
- CS = cerebellum-dev/knowledge/CEREBELLUM_STORY.md
- WM = cerebellum-dev/WINNING_METHOD.md
- OG = cerebellum-dev/OG_CEREBELLUM_RECONSTRUCTION_GUIDES_2026-06-07.md
- FEL = cerebellum-dev/FULL_EXPERIMENT_LOG.md (v-numbering session-local; used
  only where it agrees with OG)
- CARD-27B / CARD-35B / CARD-26B = live HF READMEs curled this session
- DR = community_feedback_data_2026-06-12/discussions_raw.json

| # | Draft sentence (original) | Claim | Source + actual value | Verdict |
|---|---|---|---|---|
| 1 | "for people running 8 to 24 GB cards" | mission range | CS + memory project_mission: "8-12 GB cards" | CORRECTED to 8 to 12 GB |
| 2 | "Standard quantization picks one recipe and applies it to every tensor... Q4_K_M treats [every tensor] exactly the same" | uniform-precision claim | Factually wrong: llama.cpp K-quant recipes vary format by tensor type (Q4_K_M is not single-format), and imatrix is measured from the model. A llama.cpp-literate skeptic kills this sentence on sight. | CORRECTED: rewritten as "fixed rule by tensor kind, written once for all models; imatrix improves rounding inside a format but format assignment is not measured from the specific model" |
| 3 | "crush a part... measure what breaks... damage map" | method description | CM steps 4-5 (group ablation, classify) at public-safe altitude | VERIFIED |
| 4 | "this piece gets better when you crush it... shows up in every model I've done this to" | universal claim | Verified instances: dense (Granite 4.1 30B, CLAUDE.md gotchas), MoE (26B groups -5.5/-12.1/-18.2%, OG + PV#31), hybrid (27B negative tensor deltas PV#6; 35B 7 of 10 groups, PV M4 + CARD-35B). NOT verified for all 17 releases. | CORRECTED: "measured it in dense, mixture-of-experts, and hybrid attention models" |
| 5 | "Lower precision can act like regularization" | causal mechanism | Interpretation, not proven. CARD-35B publicly uses "Q2_K regularization" language, so the framing is already public. | KEPT, explicitly hedged: "The interpretation I lean on... I have not proven that. What I have is the measurements." |
| 6 | "fixed file size budget... spend it where the measurements say" | method | CM step 8 (budget stage) | VERIFIED |
| 7 | "completely normal GGUF built with stock llama.cpp. No fork, no custom runtime, no special loader" | toolchain | CLAUDE.md, CM step 8 (stock llama-quantize), all three cards | VERIFIED (trimmed list of three to two, style) |
| 8 | "Everything is measured and built on a single RTX 3090" | hardware | CARD-27B/35B/26B test-hardware/protocol lines all say RTX 3090. Modal exists since 06-12 but no Modal-built model has shipped (CS ch.10). | CORRECTED scope: "Every release so far was measured and built on one RTX 3090" |
| 9 | "Some layer types fail hard at low precision... just NaN" | SSM hard-fail | docs/mamba_hybrid_findings.md, CLAUDE.md gotchas, CM | VERIFIED |
| 10 | "In MoE, the fragile part is the opposite of what dense intuition says" | MoE inversion | CLAUDE.md gotchas, CM | VERIFIED (scoped "the MoE models I measured") |
| 11 | "In some dense models, lowering attention precision improved perplexity. I didn't believe it either. The benchmarks backed it up." | dense attention demotion + benchmark confirmation | First half: CLAUDE.md (Granite 4.1 30B). "The benchmarks backed it up": no primary source isolates attention demotion in task benchmarks. | First half VERIFIED (scoped to "one dense model"); "benchmarks backed it up" CUT, replaced with a statement about not trying it without ablation data |
| 12 | "Every model gets its own map" | per-model maps | Heretic builds transfer maps verbatim between stock and abliterated variants of the same base (CS ch.5, memory heretic recipe), so "every model" is too strong | CORRECTED: "Every base model gets its own map" |
| 13 | "Early on I built a search that optimized perplexity alone" | chronology | Hillstep era was 2026-06-03 to 06-06 (CS ch.7), AFTER the main releases. "Early on" is false. | CORRECTED: "At one point" |
| 14 | "improved wiki perplexity 35 percent... dropped HumanEval+ by 14 points" | hillstep block-10 numbers | CM (DP-1): -35% wiki PPL, -14.03 HumanEval+ pts, +1.02 GiB | VERIFIED; +1 GiB added ("grew the file by a gigabyte") because it strengthens the honest version |
| 15 | "clear task benchmarks (MMLU, ARC, HellaSwag, HumanEval+)" | gate suite | CM/WM gate list is ARC, HellaSwag, MMLU-Redux, HumanEval+ | CORRECTED: MMLU-Redux named |
| 16 | "A build that can't beat its baseline doesn't ship" | absolute gate claim | CONTRADICTED by the live 26B card: shipped v6 loses HellaSwag (84.55 vs 86.57) and MMLU-Redux (71.33 vs 73.67) to its local Q3_K_M baseline (CARD-26B, PV#35/36). A skeptic holds these two public statements side by side and the doc dies. | CORRECTED: "the comparison goes on the card, wins and losses both. Finished builds have been rejected at this stage" + two real rejection examples |
| 17 | "I also killed a finished build outright when the damage traced back to the source weights... The finding got written up. The model didn't get uploaded." | E2B no-ship | CS ch.5: E2B heretic, source KL 0.1651, heretic F16 PPL already +27% over the stock quant, EvalPlus -31 pts, no-ship commit 0f7ae48, finding doc E2B_HERETIC_NOSHIP_FINDING.md. All true, but the writeup is PRIVATE; "got written up" implies a public document that does not exist. | CORRECTED: "I kept the notes and never uploaded the model" (no model name, no KL numbers: those are method-adjacent and private) |
| 18 | "Early published HumanEval rows lost 6 to 8 points to harness bugs" | correction magnitude | CARD-27B (live): "scores were ~6 points too low" plus "19 questions misjudged" (ARC) and "108 empty responses incorrectly counted as wrong" (HellaSwag). Private catalog (CS ch.4, PV#42) says 7-8 pts systematic; the public card says ~6 for the 27B. Publishing "6 to 8" creates a mismatch with the card. | CORRECTED: anchored to the card's exact public facts: "about 6 points on the 27B build", 19 ARC answers, 108 empty HellaSwag responses |
| 19 | "Benchmark scripts lie more often than models do" | aphorism | Unverifiable as stated (a universal about scripts). The verified fact is stronger anyway. | CUT, replaced with CS corrective truth #3: "every bad number I have published traced back to my harness or my process, not to a model" (scoped "so far") |
| 20 | "Qwen 3.6 27B build is 12 GB and scores 81.1 on HumanEval against 47.0 for a uniform 2-bit quant two gigabytes smaller" | 27B headline | CARD-27B (live): v4 12 GB / 11.98 GB, HumanEval pass@1 81.1%, Q2_K imatrix 47.0%, sizes 11.98 vs 9.98 GB = 2.0 GB. PV#11/13 verify against benchmarks/qwen36-27b JSONs (0.81097, 47.0). | VERIFIED ("pass@1" made explicit) |
| 21 | "the 35B MoE build is 11 GB and beats the 15.6 GB uniform Q3_K_M on HumanEval+ by more than 8 points" | 35B headline | CARD-35B (live, corrected today): v3 11 GB, Q3_K_M 15.6 GB, HumanEval+ 65.2 vs 56.7 (= +8.5). PV#23/24 verify the underlying JSONs (65.24 vs 56.71). Card now reads "4 of the 5 measured benchmarks" and "7 of 10 groups", both of today's fixes confirmed live. | VERIFIED; exact scores 65.2/56.7 now stated instead of "more than 8 points", and the ARC loss (95.8 vs 96.1) added explicitly |
| 22 | "Same weights, same harness, only the bit allocation differs" | comparison conditions | True for the 27B pair (same F16 source, same imatrix). For the 35B baseline, provenance of the Q3_K_M file's imatrix is not pinned by a primary build log (PV UNVERIFIABLE section). | CORRECTED: "measured with the same harness on the same machine at temperature 0" (both cards state temp 0, RTX 3090) |
| 23 | "The numbers are public, per question, in every model card" | evidence location | Per-question detail lives in the public GitHub repo (benchmarks/, 85 files across 9 models, CS ch.10) and in HF repo benchmark_results/ dirs; cards link out. "In every model card" overstates. | CORRECTED: "public in the project repo, and the model cards link to their benchmark artifacts" |
| 24 | "The cards also show where my builds lose to baseline" | honest-losses claim | CARD-26B shows two losses; CARD-35B shows the ARC loss | VERIFIED |
| 25 | "It's not mystique, it's that the measurement discipline is the product" | style/claim | "it's not X, it's Y" is a banned construction; "the discipline is the product" is a slogan | CUT, replaced with the verifiable version: gateless variants produce worse models that look fine on perplexity, "I have the rejected builds to prove that" (hillstep block-10 and 26B v5 are both real rejected builds) |
| 26 | "short-answer benchmarks [vs] code generation" gap pattern | results pattern | CARD-27B: "Short-answer benchmarks (ARC, HellaSwag) are nearly identical... The gap opens on tasks requiring precise code generation (HumanEval: +28%)" | VERIFIED, hedged with "The pattern so far" |
| 27 | "discussions on the model pages are open, and I answer them" | discussion activity | DR: multiple substantive replies by deucebucket | VERIFIED |

### Pass 1 claims for the NEW v1-to-v6 section (Pass 3 content)

| # | New sentence | Source + actual value | Verdict |
|---|---|---|---|
| 28 | "Six candidate precision maps, v1 through v6, each built into a full GGUF and measured" | OG lineage list + PV#29: all six override files on disk with exact counts (120/90/99/91/91/91) | VERIFIED ("measured", not "fully benchmarked": v1-v3 task-bench coverage is not proven, PPL measurement is) |
| 29 | "v2 removed one demotion the measurements had said was helping... came back worse. The demotion went back in." | OG: "v2: 90 overrides, kept ffn_gate_up_exps at default; lost the useful regularization and was worse"; v3 restores it (99 overrides, Q2_K=90) | VERIFIED (group name withheld: sauce) |
| 30 | "v3 and v4 narrowed the map: fewer overrides, more targeted" | OG: v3 layer-level pruning to 9 important layers, v4 task-balanced edits; counts 99 then 91 (PV#29) | VERIFIED at public-safe altitude |
| 31 | "v5 raised precision in a handful of places... Perplexity improved. HumanEval and MMLU-Redux regressed. It was never released." | OG line 454: "v5: 91 overrides, un-demoted 7 attn_k layers to Q3_K; PPL improved but HumanEval/MMLU regressed, so it was not the general release winner". FEL Step 2 agrees in shape (PPL -21.2%, HumanEval -3.7, MMLU-Redux -5.6, DO NOT SHIP). Numbers NOT included in the doc: they exist only in private logs, and FEL v-numbering is flagged session-local (CS graveyard #9). Layer count and group name withheld (sauce). | VERIFIED as written, numbers deliberately omitted |
| 32 | "v6 adjusted that trade based on further measurement, and shipped" | OG: v6 = selected attn_k changes + router work, shipped; specifics withheld (router surgery and group identities = sauce, also absent from the live card) | VERIFIED at public-safe altitude |
| 33 | "v6.1, three weeks later, kept the v6 tensor allocation with zero tensor changes... only updated chat-template and runtime metadata" | CARD-26B (live): "keeps the v6 tensor allocation and updates GGUF/runtime-facing metadata", "zero tensor changes (metadata-only update)"; dates: v6 repo 2026-05-01 (CS, HF API), v6.1 2026-05-22 (card + CS) = 3 weeks | VERIFIED |
| 34 | "11 GB, winning ARC-Challenge (95.56 vs 95.22), losing HellaSwag (84.55 vs 86.57) and MMLU-Redux (71.33 vs 73.67)" | CARD-26B live table, exact; PV#34/35 verify the same numbers against local JSONs (95.5631/95.2218, 84.55/86.5664, 71.3333/73.6667) | VERIFIED |

### Known-claims re-verification requested in the brief

- 35%/14-point hillstep numbers: VERIFIED (CM DP-1: -35% PPL, -14.03 HumanEval+, +1.02 GiB). Chronology in the draft was wrong and was fixed (#13).
- 27B 81.1 vs 47.0: VERIFIED on the live card and PV#11/13.
- 35B 11 GB vs 15.6 GB / 8 points: VERIFIED; live card now carries the corrected "4 of the 5 measured" wording. Doc states the exact pair 65.2 vs 56.7 and names the ARC loss.
- Killed-E2B story: VERIFIED internally; public phrasing scoped so it does not imply a public writeup or name the model (#17).
- "6-8 points harness bugs": the live card says ~6 for the 27B; doc now matches the card exactly (#18).
- M1 (27B card 8.256 PPL error): confirmed FIXED on the live card (size table now reads Q2_K no imatrix 7.649). The doc does not cite that table, no action needed.

## PASS 2: Claims-language audit (every edit and why)

1. "Early on" -> "At one point": chronology fix (also a Pass 1 item).
2. "It's not mystique, it's that..." removed: banned "not X, it's Y" construction.
3. "Benchmark scripts lie more often than models do" removed: aphorism, replaced with the measured statement scoped by "So far".
4. "shows up in every model" -> three named architecture families: superlative-adjacent universal reduced to verified instances.
5. "The benchmarks backed it up" removed: certainty about a confirmation that was never isolated per-change.
6. "A build that can't beat its baseline doesn't ship" removed: contradicted by the shipped 26B v6's own card; replaced with "wins and losses both" plus concrete rejections.
7. "No fork, no custom runtime, no special loader" -> two items: rule-of-three flourish trimmed.
8. "## Does it work" -> "## Results": heading was a rhetorical question.
9. "The part that actually matters: the gates" -> "## The gates": winner-flavored heading flattened.
10. "the measurement discipline is the product" removed: marketing cadence.
11. Em dashes: zero in the original, zero introduced. Checked the final file.
12. Exclamation marks: zero. Superlatives (best/only/first): zero. Remaining interpretation is carried by "The interpretation I lean on", "The pattern so far", "I have not proven that", "So far".
13. Hedges added: "so far" (hardware, bad-number record, results pattern), "the ones I measured" (MoE), "one dense model" (attention finding).
14. Kept on purpose: "Dissect the brain, cut one of the five senses so the other four come back sharper" (his own established tagline, memory project_tagline); "I don't guess"; "Recipes don't transfer on vibes"; "You find the floor by hitting it". These match his public register in DR (plain, self-aware, no marketing cadence) and assert nothing numeric.

Voice check against DR replies and the cards: short declarative sentences, willing admissions of limits ("I'm just one guy testing what I can", "No clue what in the break down also improved accuracy") map to the doc's "I have not proven that" and "I would not have guessed". No exclamation marks, no emoji, no rule-of-three crescendos, no rhetorical questions in the final text.

## PASS 3: The v1-to-v6 section (final text, as shipped in the doc)

> ## The 26B, v1 to v6
>
> The Gemma 4 26B-A4B build took six tries, which makes it a useful record of
> the process. Six candidate precision maps, v1 through v6, each built into a
> full GGUF and measured.
>
> - v1 was the first map drawn from the ablation measurements.
> - v2 removed one demotion the measurements had said was helping, to check
>   whether the effect was real. The build came back worse. The demotion went
>   back in.
> - v3 and v4 narrowed the map: fewer overrides, more targeted.
> - v5 raised precision in a handful of places that looked like easy wins.
>   Perplexity improved. HumanEval and MMLU-Redux regressed. It was never
>   released.
> - v6 adjusted that trade based on further measurement, and shipped.
> - v6.1, three weeks later, kept the v6 tensor allocation with zero tensor
>   changes. It only updated chat-template and runtime metadata.
>
> The live card shows where v6 landed: 11 GB, winning ARC-Challenge against a
> local uniform Q3_K_M baseline (95.56 vs 95.22) and losing HellaSwag (84.55 vs
> 86.57) and MMLU-Redux (71.33 vs 73.67) to it. The losses are printed on the
> card because hiding them would defeat the point of the gates.

Sauce audit of this section: no tensor group names, no layer counts or
indices, no classification thresholds, no router surgery, no imatrix
provenance, no override counts per quant type. The only numbers are the four
score pairs and the size that sit on the live card today.

## Counts

- Claims traced: 34 (27 original-draft claims + 7 new-section claims).
- VERIFIED as written or with scoping reword: 22.
- CORRECTED (number, scope, or chronology fixed against a primary source): 9
  (#1, #2, #8, #12, #13, #15, #17, #18, #22/#23 counted with their rows).
- CUT outright: 3 (#11 second half, #19, #25).

## Residual risk (anything not nailed to a primary source that remains)

1. "Every release so far was measured and built on one RTX 3090." Verified on
   the three cards re-curled this session (27B, 35B, 26B all state RTX 3090)
   and consistent with CS; the other ~14 release cards were not individually
   re-curled this pass. Risk: very low (the launch-args rule mandates
   measured-on-real-hardware cards, and no Modal-built model has shipped).
2. "v2... came back worse" rests on the OG reconstruction guide's lineage
   narrative. The guide itself was triple-verified for every checkable count
   (PV#29 verified all six override files exactly), but the v2 PPL log was not
   re-opened this pass. Risk: low; the claim carries no number.
3. Nothing else. Every number in the doc (35%, 14 points, 1 GiB, ~6 points,
   19, 108, 12 GB, 81.1, 47.0, 2 GB, 11 GB, 15.6 GB, 65.2, 56.7, 95.8, 96.1,
   95.56, 95.22, 84.55, 86.57, 71.33, 73.67, 8 to 12 GB, six tries, three
   weeks) traces to the live cards, PV-verified JSONs, or the knowledge canon
   as tabled above.

Nothing committed, nothing posted. Both files are working-tree only.

---

## REGISTER PASS LOG (2026-06-12, voice/register only, zero factual content changed)

Spec applied: no punchy lines, decisions attributed to measurements not judgment,
no sentence that invites a defend-your-reasoning follow-up, hypotheses stay
labeled, no em dashes / exclamations / rhetorical questions / not-X-but-Y.

docs/how_cerebellum_works.md, 14 lines rewritten:

1. "smaller without being dumber, for people on 8 to 12 GB cards"
   -> "make models smaller while losing as little measured ability as possible,
   for people on 8 to 12 GB cards"
   (NOTE: this flattened the established mission phrase. Restore if it should
   stay as brand; spec said tagline-class phrases stay only on the org card.)
2. "But conservative means big, and big means the people I care about can't
   run it." -> "Conservative rules also produce large files, and large files
   do not fit on the 8 to 12 GB cards this project is for."
3. "The bet behind Cerebellum is that the model itself can tell you where the
   bits matter. You just have to ask it the hard way." -> "Cerebellum replaces
   the fixed rule with measurements taken from the specific model being
   quantized."
4. Heading "What I actually do" -> "What I do"
5. "I don't guess. I take a part..." -> "I take a part..." (punchy opener cut)
6. "this piece is fragile, this piece doesn't care, and this piece, weirdly,
   gets better when you crush it. That last category is real. I have measured
   it in..." -> "some pieces measure as fragile, some measure as indifferent,
   and some measure better after being crushed. That last category has shown
   up in..."
7. "The interpretation I lean on is... I have not proven that. What I have is
   the measurements." -> "One interpretation is... That is a hypothesis, and I
   have not proven it. The measurements are recorded either way." (hypothesis
   label preserved)
8. "I set a fixed file size budget and spend it where the measurements say.
   Fragile parts keep their precision. Parts that proved they don't care get
   cut hard. Dissect the brain, cut one of the five senses so the other four
   come back sharper." -> "a fixed file size budget gets spent where the
   measurements say. Parts that measured as fragile keep their precision.
   Parts that measured as tolerant take the deepest cuts."
   (NOTE: tagline sentence removed from body prose per spec; it remains only
   where it already exists as the org-card tagline.)
9. "No fork, no custom runtime." (fragment) -> folded into the sentence:
   "built with stock llama.cpp, with no fork and no custom runtime."
10. Heading "Why measure instead of trusting intuition" -> "Why each model
    gets its own measurements"; opener "Because intuition from one
    architecture is wrong in the next." -> "In the models measured so far,
    what held in one architecture did not hold in the next."
11. "Not gradual degradation, NaN. You find the floor by hitting it." ->
    "produced NaN output below a precision floor, with no gradual degradation
    leading up to it. The floor only showed up in the measurements once a
    build crossed it."
12. "Every base model gets its own map. Recipes don't transfer on vibes." ->
    "Every base model gets its own map, because the maps measured so far have
    not transferred between architectures."
13. "Perplexity is a damage sensor, not a quality gate." -> "Since then,
    perplexity has been used to detect damage and never to pass a build on
    its own."
14. "traced back to my harness or my process, not to a model." -> "...my
    harness or my process. None traced back to a model."
15. "v5 raised precision in a handful of places that looked like easy wins."
    -> "v5 raised precision in a handful of additional places." (judgment
    basis removed rather than invented; the regression facts are unchanged)
16. "The losses are printed on the card because hiding them would defeat the
    point of the gates." -> "The losses are printed on the card next to the
    wins."
17. "I have the rejected builds to prove that. I'd rather publish..." -> "The
    rejected builds above are the record of that. I would rather publish..."
18. Closer "Ask about the results. The how will follow." -> "The method
    writeup will follow."

Left in place deliberately: "traced back to the source weights I had built on
rather than to the quantization" (factual attribution, not rhetorical
antithesis). All numbers, scores, sizes, version history, and hypothesis
labels untouched. Nothing committed, nothing posted.
