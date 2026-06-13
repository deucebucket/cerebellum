# The Cerebellum Story

Canonical, dated history of the Cerebellum project and its sibling projects.
Written 2026-06-12 from primary sources: git histories (origin, dev, conch-poc,
clanker, clanker-soul), dated devlogs and forensics files, the HF API, and the
community feedback sweep. Every claim cites a checkable source.

**PRIVATE. This lives in cerebellum-dev (dev remote territory). Never push to origin.**

If you are a fresh agent: read this before proposing anything. Most "new ideas"
in this project's space have already been tried, and several of them broke
published numbers. The graveyard section at the bottom is not optional reading.

---

## Origins (before 2026-04-26): the brain in a jar

In the user's own words, from his HF comment (verbatim in
`cerebellum-dev/community_feedback_data_2026-06-12/discussions_raw.json`):

> "it was originally a project to try to make a MoA of Bonsai 1bit 9b models,
> then turned into a way to generate I matrix on cpu with osmosis. Then while
> thinking about quantz, I was just thinking about how much of the brain you'd
> need just to think and have thoughts. Well you could trim away a large amount,
> which is funny because in movies the brain in a jar is always a full brain...
> So I just wondered how deep into models we can go when quanting. So I just ran
> ppl tests on quanted layers to find the perfect zone. And there's actually
> layers that perform better at lower quants, the way I look at it is: if you
> know a lot about cooking, but are bad at math, what happens to your math
> knowledge if you were to silence the noise of cooking in a way that the
> numbers spoke louder"

Same comment, on why nobody else did it: "I honestly feel my lack of knowledge
on the subject as a whole also takes away what's impossible. I had it explained
to me like I was 5 how quantz work, and I explained back like I was 5 how I
think it should work."

The mission crystallized as: **smaller without being dumber, for people with
8-12 GB cards**. The tagline frame: "dissect the brain, cut 1 of 5 senses to
make the other 4 stronger." (memory: project_mission.md, project_tagline.md)

---

## Chapter 1: The Osmosis era (2026-04-26 to 2026-04-29)

First commit: `aa84641` 2026-04-26, "init: Model Osmosis, perceptually lossless
neural network compression." The first three days were a rapid sequence of
ideas tried and shed (git log, origin):

- 04-26: custom GGUF quant types, custom writer/reader. Abandoned: the winning
  insight later was to emit *standard* GGUF that vanilla llama.cpp consumes.
- 04-27: block-wise quantization, "repair LoRA" (distill knowledge back into a
  crushed model), sensitivity-guided targeted LoRA, NIM wikitext training
  pairs. All abandoned. Same day, `2617e40` "v0.2.0: clean repo, imatrix
  generation only": the project shrank to its honest core.
- 04-28: `e6f37e2` streaming imatrix generator (any model size, constant RAM),
  then `20251be` "feat: cerebellum, ablation-informed tensor precision
  allocator." The word "cerebellum" enters the repo as a module name first.
- 04-29: full ablation sweep, memory controller design doc
  (`docs/cerebellum_memory_controller.md`), and `fadd037` "docs: rebrand to
  Cerebellum." The project name died as Osmosis on day four.

Why the name died: "osmosis" described the imatrix-generation phase. The thing
that actually worked, ablation-informed mixed precision, is brain surgery, and
the brain-in-a-jar framing fit. The *code* rename lagged the project rename by
six weeks (see Chapter 10); `osmosis/` paths haunted the repo until 2026-06-12.

Also on 04-29: an in-house refusal-direction ablation pipeline was built and
found ineffective on the Qwen 3.6 27B hybrid architecture (`30340e4` "results:
refusal ablation ineffective"). This dead end matters later: the Heretic line
(Chapter 5) uses *external* abliterated bases instead of in-house abliteration.

## Chapter 2: The first win, Qwen 3.6 27B v1 to v4 (2026-04-27 to 2026-04-29)

Source of record: `cerebellum-dev/QWEN36_27B_V1_TO_V4_PLAYBOOK_2026-06-07.md`
and `WINNING_METHOD.md` Instantiation B.

The lineage that defined the method: BF16 HF source converted to F16 GGUF
(51 GB at `/var/tmp/osmosis-qwen36/`), imatrix, then smash aggressively with a
Q2_K base, measure which tensors survive or improve, and spend precision back
only on tensors that proved they mattered.

| Version | Size | Wiki PPL | What it proved |
|---|---:|---:|---|
| v1 | 14.86 GiB | 7.6713 | Q4-ish mixed; baseline |
| v2 | 10.67 GiB | 7.0868 | Q2_K-base smashing: smaller AND better than v1 |
| v3/v3c | ~10.5 GiB | 7.0870 | budget experiments; found the size-estimation bug (#16) |
| v4 | 11.97 GiB | 7.0344 | 23 sparse PPL probes + budget allocator + 181 overrides; beat Unsloth Q2_K_XL (7.034 vs 7.040 at 12GB, commit `aa793fd` 04-29) |

Key validating number: Q2_K+imatrix (PPL 7.4996) already beat Q3_K_M+imatrix
(7.6413) on this model, which made aggressive demotion the correct base.

27B v4 is the only flagship that got the full *tensor-level* budget-allocator
treatment (181 per-tensor overrides from `osmosis.cerebellum` + `budget`). The
35B (Chapter 3) never did; it shipped on group-level ablation. That asymmetry
is a standing BACKLOG item (35B v4 tensor-level pass, BACKLOG.md NEXT #1).

HF releases: Qwen3.6-27B-Osmosis-Q2_K-GGUF created 2026-04-28 (the only repo
ever published under the old name; renamed 2026-06-12),
Qwen3.6-27B-Cerebellum-GGUF created 2026-04-29 (HF API, createdAt).

Caveat for honesty: the repo's `start.md` claims a "448-tensor ablation" but
`ablation_results.json` holds exactly 23 measured probes; the allocator
extrapolated the rest (PIPELINE_VERIFICATION_2026-06-12.md §M2). The 23-probe
number is the truth.

## Chapter 3: The fleet era (2026-04-30 to 2026-05-05): the releases that built the audience

Nine models shipped in six days (HF API creation dates):

- 04-30: Gemma-4-E4B-it-Cerebellum-v2 (PLE protection discovered: Q4_K to Q3_K
  cliff; PPL 104 without PLE protection, 55 with PLE@Q5_K)
- 05-01: Gemma-4-26B-A4B-it-Cerebellum (v3) and the v6 repo. The 26B work
  produced the router discoveries: llama-quantize silently ignores
  `--tensor-type-file` for `ffn_gate_inp` tensors, so `gguf_tensor_surgery.py`
  was built to recast routers post-quantize
  (DEVLOG_2026-05-01_router_road_mapping.md); K-quants are *broken* on Gemma
  routers (Q6_K +15.9%, Q2_K +17.2% PPL; Q8_0 only safe format), and layer 8
  was the surgical target. v6 shipped with 91 overrides + router surgery.
- 05-01: Qwen 3.6 35B-A3B work began (DEVLOG_2026-05-01_qwen36_35b_start.md):
  hybrid SSM+attention MoE, 256 experts. The SSM hard-fail-below-4-bit law was
  established on this family (docs/mamba_hybrid_findings.md).
- 05-02: Qwen3.6-35B-A3B, Granite-4.0-H-Small, Qwen3-30B-A3B, Granite-4.1-30B,
  Qwen3-32B, Qwen3-14B all created on HF (six repos in one day).
- 05-03: Gemma-4-E2B-it-Cerebellum-v2.
- 05-05: Qwen3.5-122B-A10B (the largest; COMPARISON_122B.md is the private
  competitive analysis). Same day, the user authorized autonomous overnight
  pipeline execution (memory: feedback_autonomous_overnight_authorized).

Method evolution across the fleet, in order: **group ablation** (crush a named
tensor group to Q2_K, ~20-25 probes) -> **reverse ablation** (restore groups
from a fully-demoted v1 and see what genuinely regresses) -> **budget
allocator** (per-tensor promotion under a GB cap). The 35B v3 used
group+reverse only (360-entry override, all-Q2_K expert demotion over Q3_K_M
base, WINNING_METHOD.md Instantiation C). 35B v3 shipped 2026-06-02 (commit
`b36e3bf`: 11 GB, +8.5 HumanEval+ vs Q3_K_M at 29% smaller); the head-to-head
was re-confirmed 2026-06-11 (`780ff06`: 11.96GB beats uniform Q3_K_M 16.87GB).

Also in this era: Qwen 3.5 9B got a 202-tensor ablation (commit `d4650ac`
04-30) and weeks of deep work (the entire `docs/qwen35_9b_*_2026052*.md` series
plus row-block experiments, 05-20/21), but **was never released on HF** (no
repo exists, HF API). v1 failed on a wiki-only imatrix (false failure, DP-5),
v2_code won at 53.0% EvalPlus+, v3_rowblock passed classical benchmarks but
failed a real agent task (DP-6, unresolved anomaly).

Per-architecture laws established in this era (CLAUDE.md "gotchas" section all
trace here): SSM params hard-fail below 4-bit; MoE expert weights are the
fragile part (opposite of dense intuition); PLE needs protection; dense models
sometimes *improve* when attention is demoted. Ablation data, not intuition,
drives every per-model decision.

## Chapter 4: The benchmark-bug era and the audit discipline (pre-2026-05-03, fixed 05-03 onward)

The most consequential infrastructure story. Full catalog:
`cerebellum-dev/forensics_2026-06-11/BENCHMARK_ERRORS_CATALOG.md` (14 entries,
BE-01 through BE-16).

BE-01, the worst: the hand-rolled HumanEval harness stripped code fences with
`.strip()`, destroying leading indentation, causing a systematic 7-8 point
undercount on **every published HumanEval row before 2026-05-03**. Discovered
2026-05-03 ~01:30 CDT via a temperature sweep that showed impossible lows.
Qwen 27B v4 jumped from 75.0% to 81.1% with the fixed script. Corrections
committed 05-03 (`3f47634`, `7be0455`).

Other entries in the same family: ARC numeric-label mismatch (BE-02, fixed
05-03), raw /v1/completions garbage for chat models (DP-3, bit twice: 05-01 and
05-18), EvalPlus fabricating 'pass' on exhausted retries (BE-15, fixed
2026-06-11, commit `d124175`).

The discipline this created, now baked into CLAUDE.md and memory:
1. After every benchmark run, audit wrong answers before recording the score.
2. Run `scripts/audit_evalplus_completions.py` after every code bench.
3. Use upstream runners (evalplus.codegen, lm-eval-harness), never hand-rolled
   wrappers. Four wrapper bugs were edge cases upstream had already solved.
4. HumanEval+ is single-client sequential (WORKERS=1); the server invariants
   live in docs/benchmark_protocol.md.

Every previously-published bad score in this project came from skipping step 1.

## Chapter 5: The Heretic line (2026-05-09 to 2026-06-12): born from a user request

- 2026-05-09: HF user tima2431 asks for an uncensored 26B build (discussion #2
  on Gemma-4-26B-v6, COMMUNITY_FEEDBACK_2026-06-12.md §2). User Koitenshin
  suggests coder3101's heretic base.
- 2026-05-19: Gemma-4-26B-A4B-it-Heretic-Cerebellum-GGUF ships (HF createdAt).
  The transfer protocol is the discovery: **transfer the proven override map
  verbatim, same imatrix, no re-ablation**, because abliteration modifies
  activations, not tensor layout (WINNING_METHOD.md, Heretic Transfer
  Protocol). Result beat the stock build; refusal rate 57.8% -> 2.2%. tima2431
  made it a daily driver ("almost on par with Gemini 2.5 Pro").
- 2026-05-31: TheodoreH requests a Qwen 35B heretic (35B discussion #3, the
  most engaged user on any repo).
- 2026-06-03: first 35B heretic attempt FAILS. Source was the MTP-preserved
  variant with extra block blk.40: load failure, -14 HellaSwag / -32
  HumanEval+ (DP-2). Rule since: always check for blk.40; never use
  MTP-preserved sources (memory: project_heretic_transfer_recipe).
- 2026-06-11: the fleet ships properly. HERETIC_FLEET_PLAN.md (trusted sources:
  llmfan46, coder3101; banned: huihui-ai). 35B heretic audited and shipped
  (`97d6230`), 27B heretic gate PASS (`2a81a27`), E4B heretic gate PASS, beats
  stock 3/4 (`ce2a97d`). HF repos: 35B 06-11, 27B and E4B 06-12.
- 2026-06-11: **E2B heretic NO-SHIP** (`0f7ae48`,
  forensics_2026-06-11/E2B_HERETIC_NOSHIP_FINDING.md). Source KL 0.1651 (28x
  the E4B's 0.0058); heretic F16 PPL was already +27% over the stock *quant*;
  EvalPlus collapsed -31 points while multiple-choice survived. The KL ceiling
  rule was adopted: screen the source's reported KL before transfer; ≤0.006
  transfers clean, ≳0.05 is suspect and needs an F16 PPL + code screen first
  (memory: project_heretic_kl_ceiling).

## Chapter 6: Template fixes, 9B row-blocks, codex experiments (2026-05-11 to 2026-05-22)

- 05-11: Carl runtime notes, 128K context and CUDA build flags documented
  (`b7fed81`; the branch this story is being written on,
  docs/runtime-notes-cuda-arch-and-128k, dates from here).
- 05-20/21: the Qwen 3.5 9B deep-dive sweep, ~40 dated findings docs
  (docs/qwen35_9b_*_20260520/21.md): per-domain sensitivity, row-block ladders
  (the first real sub-tensor work, feeding the memory-controller endgame),
  agent-probe findings.
- 05-21/22: Gemma 4 26B v6.1: a metadata-only re-release (zero tensor changes,
  chat-template fix, commit ded491334; memory: project_v61_is_metadata_only).
  Upstream Gemma template issues researched and documented
  (docs/gemma4_*_20260521.md).
- 05-22: Gemma 26B codex experiments: LoRA merge, codex-aware tensor map, MTP
  dead end (`ae9c909`).

## Chapter 7: The hillstep detour (2026-06-03 to 2026-06-06): the most expensive lesson

Source: DEVLOG_2026-06-03_gemma4-12b.md, DEAD_PATHS.md DP-1.

The Gemma 4 12B work started with genuine wins: the world's first Gemma 4 12B
GGUF (one-line llama.cpp patch) and the discovery of a hard quant floor
(Q3_K_M outputs "a a a a"; Q4_K_M works). Then came hillstep: a resumable,
SQLite-journaled, bidirectional per-tensor hill-climber optimizing wiki PPL.
The 06-03 devlog called it "proven better than forward ablation." It was not.

The block-10 checkpoint (06-04/06): **-35% wiki PPL, +1.02 GiB size, and
-14.03 points HumanEval+** versus the Q4_K_M baseline. Wiki PPL and task
performance diverged catastrophically. Contributing failures: a poisoned
starting map that demoted attn_v/attn_o, K-quant router corruption inside the
demotion chain, no benchmark gate before committing locks, plus five distinct
hillstep.py bugs.

The deprecation was canonized 2026-06-11 (WINNING_METHOD.md, DEAD_PATHS.md,
memory: project_winning_method_og_formula). PPL-only gating is dead.
Targeted hillstep *after* a group-first scan remains a documented optional
step, but exhaustive wiki-PPL hill-climbing is never coming back.

Ironically the detour produced lasting infrastructure: the 06-04/05 burst of
~80 commits built the `cerebellum` CLI, watch TUI, FastAPI dashboard control
plane, job scheduler, benchmark-gate plumbing, artifact inventory, and the
public-release audit tooling that the 06-12 overnight relied on.

## Chapter 8: Brainloop / conch-poc (2026-06-09 to 2026-06-11): the sibling experiment

Repo: `cerebellum-dev/conch-poc/`, public as
github.com/deucebucket/cerebellum-brainloop (renamed from Conch-POC 06-10,
commit `0241017`). HF dataset: deucebucket/cerebellum-brainloop.

What it is: a dual-stage hidden-state interceptor for Qwen2.5-3B. Frozen base
model, identity-initialized "refiner" blocks after layers 17 and 30, subspace
masking (only 25% of hidden dims writable), goal: inject factual knowledge
directly into the residual stream with zero context tokens, exportable as a
standard GGUF via metadata surgery (`unroll_vanilla_gguf.py`).

Timeline (conch-poc git log):
- 06-09: initial commits, RAG-trained refiners, C++ port (GPU allocation fix),
  cartridge KV injection research, Gemini strategy sessions.
- 06-10: public release, weight-baking "breakthrough" claims, vanilla GGUF
  unrolling, 13k Python stdlib mass-production pipeline, MTP hijacker C++.
- 06-11: the honesty pass. Three public retractions in one day:
  1. `d570ab3`: the "5% logic tax" claim retracted; a wiring audit found the
     refiners were *functionally inert* during that measurement (gate
     contribution 0.5%, injection projection never trained). The gap was
     prompt-format difference.
  2. `c61789a`: the c=2048 regression retracted; baseline wiki/python PPL
     labels were crossed in a parallel measurement session.
  3. An earlier recall harness compared a model against itself; pre-fix recall
     numbers are void (RESULTS.md).

The real, verified results (all on stock llama.cpp, 06-11):
- **Dead-block structural parity**: inserting two trained blocks into a
  38-block GGUF costs nothing measurable (161/164 HumanEval completions
  token-identical), but knowledge injection was null
  (conch-poc/DEADBLOCK_STATUS.md, RESULTS.md).
- **LM-trained insertion blocks**: first verified recall through a vanilla
  GGUF. Symbol recall 10.0% -> 25.5%, *including Python 3.14 stdlib symbols
  that postdate the base model's training data*, with zero context tokens. A
  wikitext-trained block improved wiki PPL -10% at its training context.
- **Layer sweep** (`24aba95`): static delta injection cannot write knowledge
  at any depth. LM-loss training is the only path that worked.
- Honest open problem: corpus-trained blocks regress HumanEval (best
  containment so far 55.5% vs 62.8% baseline). Retaining recall while holding
  HumanEval at baseline is unsolved and is the active work.

DP-9 warning stands: the early brainloop bench_results/ JSONs have no
provenance (no checkpoint path, no commit, no GGUF path) and several are
byte-identical copies. Treat pre-06-11 brainloop benchmark files as
untrustworthy.

What brainloop gives cerebellum: GGUF surgery experience (block insertion,
metadata unrolls) that directly feeds the memory-controller endgame
(docs/cerebellum_memory_controller.md), and a second proof that benchmark
wiring must be audited before claims.

## Chapter 9: Memory archaeology and the forensics canon (2026-06-05 to 2026-06-11)

- 06-05: public history audit, artifact inventory (`4405848`, `70cde91`,
  ARTIFACT_INVENTORY.md). The public GitHub repo was taken down for cleanup
  around this time; user ibaldonl noticed 06-06 (community digest §2).
- 06-07: the reconstruction guides. The original winning sessions were
  recovered from raw Claude JSONL transcripts
  (OG_CEREBELLUM_RECONSTRUCTION_GUIDES_2026-06-07.md,
  QWEN36_27B_V1_TO_V4_PLAYBOOK_2026-06-07.md). Key honest finding: the
  successful models were **not one identical workflow**. 27B v4 = tensor-level
  allocator; 35B v3 = group/reverse recipe; 26B v6 = hand-evolved 91-entry map
  plus router surgery. Same day: QWEN3_0_6B_OG_V4_METHOD_SMOKE (method
  verified end-to-end on a tiny model).
- 06-11: forensics day (`2afd423`). Produced the canon this file leans on:
  WINNING_METHOD.md, DEAD_PATHS.md, BENCHMARK_ERRORS_CATALOG.md (14 entries),
  PRUNE_LOG.md (~259 GB reclaimed, small evidence archived first, cmp-verified
  dups only), RECIPE_heretic_qwen36_35b.md, leaderboards/eval-badge research
  (`bb6e2a3`, LEADERBOARDS_AND_EVAL_BADGES.md: model-index YAML,
  .eval_results, UGI/OLL2 status), HERETIC_FLEET_PLAN.md, and the heretic
  fleet ships (Chapter 5).

## Chapter 10: The 2026-06-12 overnight: the factory comes online

All in one overnight (BACKLOG.md DONE section, REBRAND_LOG.md, git log,
claude-mem observations #1195-#1255, 03:00-04:10 CDT):

- **Benchmark evidence published**: 85 files across 9 released models pushed
  to the restored public repo (`7878780`), with docs/benchmark_protocol.md.
- **Public GitHub repo restored** (sanitized data dump; closes the ibaldonl
  complaint and the dead links on model cards). AI-attribution trailers
  scrubbed from public history.
- **The rebrand finally completed** (REBRAND_LOG.md): engine code moved
  `osmosis/` -> `cerebellum/` with a deprecation-shim package so `import
  osmosis.X` and `python -m osmosis.X` still work; 225 tests passing before
  and after; HF repo Qwen3.6-27B-Osmosis-Q2_K-GGUF renamed to
  -Cerebellum-Q2K- with server-side GGUF rename. Commits `174e503`,
  `29e9afe`, `73c86d7`, `f4297ae`, `16d6849`.
- **Org + fleet metadata**: DB-Cerebellum org card, collections cleanup, 17
  model cards got model-index eval panels, 18 GGUFs renamed for the HF
  hardware badge.
- **Modal cloud factory**: harness built and verified for $0.06
  (cerebellum-dev/modal_harness/), watchdog live. This breaks the
  single-RTX-3090 bottleneck.
- **Three campaigns launched**: Gemma 4 12B (local, ablation done,
  continuation running), GLM-4.7-Flash on Modal (converted to BF16 GGUF at
  04:02, RUN_PLAN at 04:00), and North-Mini-Code-1.0 pre-positioning (Cohere's
  first open code model, llama.cpp PR #24260 branch building; play: first
  benchmark-gated sub-4-bit).
- **Community sweep** (COMMUNITY_FEEDBACK_2026-06-12.md, ~03:27): ~18,100
  all-time downloads, 9 real user threads, a Google Gemma org member engaged,
  someone already repackaged the work (LordAce9). Headline insight: users
  react to base model + uncensored availability, not method granularity; the
  audience is exactly the low-VRAM crowd from the mission statement.
- **Pipeline triple-verification** (PIPELINE_VERIFICATION_2026-06-12.md,
  ~03:40): 43 claims verified, 5 mismatches found, including a live public
  card error (27B card publishes 8.256 as Q2_K-no-imatrix PPL; the real
  number is 7.6494; 8.256 is the ablation baseline).
- **This reorganization**: knowledge base consolidation launched ~04:04
  (observation #1246), producing the `cerebellum-dev/knowledge/` directory
  this file lives in.

---

## Sibling projects

### Brainloop (cerebellum-brainloop / conch-poc)
Covered in Chapter 8. Lives inside this repo at `cerebellum-dev/conch-poc/`,
public remote `public` -> github.com/deucebucket/cerebellum-brainloop. Active
06-09 to 06-11. Honest status: structural parity proven, LM-trained recall
through vanilla GGUF proven (including post-cutoff knowledge), knowledge
retention vs HumanEval damage unsolved. Its GGUF surgery skill set is the
bridge to the cerebellum memory-controller endgame.

### Clanker (/var/home/deucebucket/ai-drive/clanker)
A deterministic emotional stance resolver: 7-dimensional VADUGWI coordinates
(Valence, Arousal, Dominance, Urgency, Gravity, self-Worth, Intent) computed
from text via explainable structural pattern recognition, no neural inference.
"You can ask WHY and get a real answer" (clanker/README.md). Git timeline:
V8 VADUGWI engine 2026-04-24 (predates Cerebellum's first commit by two days),
AGPL->MIT relicense 2026-05-25, then the 2026-06-11 "Council-v2 evaluation
era" burst: 373-probe sealed holdout, **honest 41.3% baseline** published in
the commit message itself, V8.4 "every number re-measured", V8.5 slang
register overhaul.

What clanker offers cerebellum: the evaluation discipline in its purest form.
Sealed holdouts, baselines published even when embarrassing, every number
re-measured on doc refresh. Cerebellum's audit-gate culture and clanker's
council evaluation are the same person's same instinct, converged
independently. What cerebellum offers clanker: the local-inference fleet
(clanker's LLM-rated charge calibration can run on Cerebellum quants).

### Clanker-soul (/var/home/deucebucket/ai-drive/clanker-soul)
"An emotional learning tool for AI agents": persistent VADUGWI-based state,
motivation-driven actions, and a consequence feedback loop, extracted from
CARL (the user's agent runtime) on 2026-05-08 (`0492733`). 90 commits,
releases v0.16 (05-09), v0.17 (05-10), v0.18 (05-25, MIT relicense), then
06-11 measurement-calibrated reflexes wired to the real clanker engine.
Notable as the most process-disciplined repo in the family: issue-numbered
PRs, CI, spec docs per knob. Shared truth across all three siblings: the user
works in rapid bursts (04-24, 05-09/11, 05-25, 06-11 appear in multiple repos'
logs), prefers honest regressions in writing over silent fixes, and treats
historical artifacts as evidence to preserve, never delete.

---

## The defunct-methods graveyard

Read this before proposing methodology. Each entry: what, when, and why an
agent suggesting it is wrong. Full detail in `cerebellum-dev/DEAD_PATHS.md`
and `forensics_2026-06-11/BENCHMARK_ERRORS_CATALOG.md`.

1. **Exhaustive wiki-PPL hillstep / PPL-only gating** (active 2026-06-03 to
   06-06, deprecated 06-11). If an agent suggests optimizing or gating on
   perplexity alone, it is wrong because the block-10 checkpoint proved PPL
   and task performance diverge: -35% wiki PPL came with -14.03 HumanEval+
   and +1 GiB. Benchmark gates (ARC/HellaSwag/MMLU/HumanEval+, plus
   BigCodeBench as Gate 3) vs a same-size uniform baseline are mandatory.
   (DP-1; memory: project_winning_method_og_formula)

2. **Raw /v1/completions HumanEval harness for chat models** (used 05-01 and
   05-18). Wrong because it bypasses the chat template and produces garbage:
   the Gemma v6 "35.97%" and Heretic "3.05%" scores are artifacts. Use
   `scripts/benchmark_evalplus_chat.py` against `--jinja` with thinking
   disabled. (DP-3)

3. **Any published score from before 2026-05-03**. Wrong to cite because the
   fence-stripping bug undercounted HumanEval by 7-8 points on every model
   (BE-01), and ARC had a label mismatch (BE-02). Canonical numbers live in
   `benchmarks/<model>/` post-correction files; note the `corrected_*` vs
   `fixed_*` trap (PIPELINE_VERIFICATION M3: published 27B MMLU is 76.58
   from `corrected_*`, not 76.875 from `fixed_*`).

4. **Hand-rolled benchmark wrappers**. Wrong because four separate wrapper
   bugs (fence stripping, label mismatch, fabricated passes, empty-response
   counting) were all edge cases upstream runners had already solved. Default
   to evalplus.codegen / lm-eval-harness / bigcode-evaluation-harness.
   (memory: feedback_use_upstream_runners)

5. **MTP-preserved heretic sources** (failed 2026-06-03). Wrong because the
   extra blk.40 is an architecture mismatch: load failure plus -14 HS / -32
   HE+. Always grep the source for blk.40 first. (DP-2)

6. **Heretic transfer without KL screening** (E2B no-ship, 2026-06-11). Wrong
   because abliteration damage is inherited, not caused by quantization: KL
   0.165 source lost 31 code points at F16 already. Screen reported KL;
   ≳0.05 requires an F16 PPL + code screen before any build. (E2B finding;
   memory: project_heretic_kl_ceiling)

7. **Locally generated imatrix with partial tensor coverage on MoE** (DP-4,
   2026-04-30). Wrong because the 205-entry Gemma imatrix covered zero expert
   or router tensors and made Q4_K_M *worse than no imatrix*. For MoE, verify
   full expert coverage or import bartowski/unsloth imatrices. The harmful
   file still sits at `osmosis-gemma4-26b/imatrix.dat`; the correct one is
   `google_gemma-4-26B-A4B-it-imatrix.gguf` in the same directory.

8. **Wiki-only imatrix for code-capable models** (DP-5, 9B v1). Wrong because
   it produced a false model failure; the code-weighted v2_code imatrix
   fixed it. Calibration must match target domains.

9. **Re-quantizing identified CRITICAL layers in either direction** (Gemma
   26B, 2026-05-01). Wrong because critical attn_k layers calibrated by the
   imatrix at a specific quant are fragile both ways: the v6/v7 experiments
   in FULL_EXPERIMENT_LOG.md show promotion attempts of those layers
   costing +46-51% PPL and -13.4 HumanEval. Note an unresolved log
   discrepancy: FULL_EXPERIMENT_LOG.md (Step 3) records "v6 = Q4_K on
   critical layers, catastrophic" while DEAD_PATHS DP-8 and WINNING_METHOD
   record the *shipped* v6 carrying those layers at Q4_K with the catastrophe
   being Q3_K-or-below. The shipped-v6 override file
   (`cerebellum_v6_overrides.txt`, 91 entries) is the ground truth; the
   FULL_EXPERIMENT_LOG version labels do not line up with shipped naming.
   Trust the override file and WINNING_METHOD, treat FULL_EXPERIMENT_LOG's
   v-numbering as session-local.

10. **In-house refusal-direction ablation** (2026-04-29). Wrong to retry on
    hybrid architectures because it was measured ineffective on Qwen 3.6 27B
    (`30340e4`). The Heretic transfer protocol with trusted external sources
    (llmfan46, coder3101; huihui-ai banned) replaced it.

11. **Benchmarks without provenance logging** (brainloop, 2026-06). Wrong
    because seven runs became permanently unauditable (no checkpoint path, no
    commit, no GGUF path) and two "different" runs were byte-identical
    files. Every result JSON must record model path, commit, and harness.
    (DP-9)

12. **Row-block allocation without agent-task gates** (9B v3_rowblock, DP-6).
    Not confirmed dead, but an open caution: it passed ARC/HellaSwag/MMLU and
    failed a real agent loop. Any sub-tensor allocation work needs
    agent-capable tasks in the gate suite.

13. **Naming anything new "osmosis"** (rename completed 2026-06-12). Wrong
    because the rebrand is done; `osmosis.*` imports only survive as
    deprecation shims. New artifacts are `cerebellum_*`.
    (REBRAND_LOG.md; memory: feedback_no_osmosis_naming)

---

## Appendix: research and decision dates

| Date | Event | Source |
|---|---|---|
| 2026-04-24 | Clanker V8 VADUGWI engine (family's first commit) | clanker git log |
| 2026-04-26 | Model Osmosis init | `aa84641` |
| 2026-04-27 | v0.2.0 strip-down to imatrix-only; repair-LoRA path abandoned | `2617e40` |
| 2026-04-28 | Streaming imatrix; cerebellum allocator born; first HF repo | `e6f37e2`, `20251be`, HF API |
| 2026-04-29 | Memory controller design; v4 beats Unsloth; rebrand to Cerebellum; refusal ablation found ineffective | `9336013`, `aa793fd`, `fadd037`, `30340e4` |
| 2026-04-30 to 05-05 | Fleet era: 11 HF releases | HF API createdAt |
| 2026-05-03 | Benchmark-bug discovery and corrections; audit discipline born | BE-01/BE-02, `3f47634` |
| 2026-05-05 | 122B ships; autonomous overnight authorized | HF API, memory |
| 2026-05-08/09 | clanker-soul extracted from CARL | clanker-soul `0492733` |
| 2026-05-09 | tima2431 uncensored request (Heretic line origin) | community digest §2 |
| 2026-05-19 | First Heretic ships (26B) | HF API |
| 2026-05-20/21 | 9B row-block sweep (first sub-tensor work) | docs/qwen35_9b_*_2026052*.md |
| 2026-05-22 | v6.1 metadata-only release; codex experiments | ded491334, `ae9c909` |
| 2026-05-31 | TheodoreH 35B thread (most engaged user) | community digest §2 |
| 2026-06-02 | 35B v3 ships | `b36e3bf` |
| 2026-06-03 | MTP heretic failure; hillstep era opens (Gemma 12B) | DP-2, DEVLOG 06-03 |
| 2026-06-04/05 | CLI + dashboard control-plane buildout (~80 commits) | git log |
| 2026-06-05 | Artifact inventory; public history audit; public repo down for cleanup | `70cde91`, `4405848` |
| 2026-06-07 | OG reconstruction guides (session-transcript archaeology); 0.6B method smoke | dated filenames |
| 2026-06-09 to 06-11 | Brainloop sprint, then its three retractions | conch-poc git log |
| 2026-06-11 | Forensics day: WINNING_METHOD + DEAD_PATHS canon, hillstep deprecated, 259GB prune, leaderboard/eval-badge research, heretic fleet (35B/E4B audited), E2B no-ship + KL ceiling, EvalPlus fabricated-pass fix | `2afd423`, `bb6e2a3`, `97d6230`, `0f7ae48`, `d124175` |
| 2026-06-12 (overnight) | Community sweep (~03:27); rebrand complete (~03:36); pipeline triple-verification (~03:40); benchmark evidence publish (85 files); org card + 17 model-index cards + 18 GGUF renames; public repo restored; Modal harness verified $0.06; 12B + GLM-4.7-Flash campaigns launched (~04:00); North-Mini-Code pre-positioning; GLM-5.1 feasibility demoted after landscape sweep; knowledge-base reorganization (~04:04) | BACKLOG.md DONE, REBRAND_LOG.md, observations #1195-#1255 |

## The five corrective truths (if you read nothing else)

1. **PPL is a screening tool, not a gate.** The hillstep block-10 checkpoint
   improved wiki PPL 35% and lost 14 HumanEval+ points. Every build clears
   four task benchmarks plus audits against a same-size uniform baseline
   before it ships.
2. **The winning method is the OG group-first, benchmark-gated formula**
   (WINNING_METHOD.md), and it was never one identical workflow: 27B v4 is
   tensor-level allocator, 35B v3 is group-level recipe, 26B v6 is a
   hand-evolved map plus router surgery. The 35B has never had the
   tensor-level treatment; that is queued work, not done work.
3. **Audit before you publish.** Every bad published number in this project's
   history (pre-05-03 HumanEval, v6 35.97%, brainloop logic tax, the live 27B
   card PPL error found 06-12) came from skipping wrong-answer audits or
   provenance logging.
4. **Heretic builds transfer override maps verbatim** (no re-ablation, same
   imatrix), but only from trusted non-MTP sources with screened KL: blk.40
   present = abort; KL ≳0.05 = screen at F16 first; KL 0.165 broke E2B
   beyond saving.
5. **The project is Cerebellum, the rename is finished** (2026-06-12), the
   public/dev split is absolute (cerebellum-dev content never goes to
   origin), and historical artifacts are evidence: preserve, never delete
   without a verified backup and a plan.
