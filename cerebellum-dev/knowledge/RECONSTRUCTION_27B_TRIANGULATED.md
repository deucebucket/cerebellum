# Qwen 3.6 27B Cerebellum — Triangulated Version-by-Version Reconstruction

**Scope:** v1 -> v4, assembled from the per-version multi-agent consensus reports (3 agents per
version, every load-bearing claim re-verified against the `osmosis-qwen36-27b/` artifact tree).
This is the **evidence-based** reconstruction: each step is tagged CONFIRMED / DISPUTED / GAP by
whether a primary artifact proves it. Where agents disagreed, the disagreement is surfaced, not
smoothed.

**Model:** Qwen3.6-27B, arch `qwen35`, 64 blocks, 26.90 B params, 851 tensors. Hybrid SSM
(`ssm_alpha/beta/out`, `in_proj_a/b/z`, `attn_gate`) + full attention every 4 layers
(`attn_qkv/q/k/v/output`) + dense MLP (`ffn_gate/up/down`).

**Shared infrastructure (all versions):**
- **imatrix:** one file, `cerebellum_imatrix.dat` (13,582,641 bytes, mtime Apr 27 12:40), reused
  v1->v4. It is a **sensitivity proxy**, NOT a standard wikitext-activation imatrix: dataset label
  `osmosis-sensitivity`, **496 entries, 8 chunks**, embedded in every shipped GGUF (kv41-44).
- **F16 source:** `/var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf`.
- **Ablation baseline:** Q4_K_M + imatrix base GGUF, **wikitext PPL 8.2556** (145 chunks, n_ctx
  2048). Every single-tensor crush delta is measured against this 8.2556.
- **Ablation shape:** 23 individually-named tensors, each crushed one-at-a-time Q4_K_M -> Q2_K,
  full-model PPL re-measured. **Per-tensor, not per-tensor-type group.** Selection KL-screened
  (`ablation_plan.json`: `kl2/kl3/kl4 + damage + reason`) — but see the v3 KL-provenance dispute.

---

## v1 — first cut, superseded same night

**Window:** Apr 27 (download / imatrix / f16) -> Apr 28 (ablate / build / PPL). First Cerebellum
cut for this model; superseded the same night by v2.

### CONFIRMED (in order)
1. **Download + F16 convert.** HF safetensors (15 shards) -> `qwen3.6-27b-f16.gguf`, 851 tensors,
   arch `qwen35`, 64 blocks. (`overnight.log` STEP 1/3; v1 KV metadata.)
2. **imatrix = sensitivity proxy** (496 entries / 8 chunks), embedded in the v1 GGUF. Not standard
   wikitext calibration.
3. **Q4_K_M ablation baseline, PPL = 8.2556 ±0.06480.** The method quantizes *down from Q4_K_M*,
   not up from Q2_K. (`baseline_q4km.log`; `ablation_results.json` `baseline_ppl`.)
4. **23-tensor selection via KL proxy.** `ablation_plan.json` reasons `highest_kl2 / lowest_kl2 /
   highest_damage`. SSM + attention + MLP across ~17 layers. Individual named tensors. (Commit
   `20251be`.)
5. **Single-tensor PPL ablation (NOT group, NOT coding).** Each of 23 tensors crushed Q4_K_M->Q2_K
   alone, PPL delta recorded. 23 logs Apr 28 20:24 -> Apr 29 00:53. ~7 tensors *improve* when
   crushed (e.g. `blk.2.ffn_gate` 8.1089, `blk.0.ffn_up` 8.1988); most sensitive `blk.63.q_proj`
   8.4178, `blk.63.ffn_down` 8.3933; SSM tensors are noise (|delta| < 0.02). No group/category PPL
   ablation existed at v1 — all three agents assert this.
6. **Ablation-aware budget allocator.** `cerebellum.py@20251be`: `classify_tensors`
   (noise_threshold=0.02 -> demote / sacred / neutral), `extrapolate_layer_sensitivity` over 64
   layers, then `ablation_aware_allocate` from base Q4_K_M.
7. **Build via stock `llama-quantize`** + imatrix + tensor-type override, base Q4_K_M. Output
   `qwen3.6-27b-cerebellum-v1.gguf`. **Size = 14.86 GiB, 4.74 BPW, file_type=15 (Q4_K_M)**, 238
   q4_K tensors. (`cerebellum_v1.log` — `file size = 14.86 GiB (4.74 BPW)`.)
8. **Evaluation = PPL ONLY. v1 PPL = 7.6713 ±0.05718.** No ARC / HellaSwag / MMLU / HumanEval.
   Superseded same night by v2.

### DISPUTED
- **D1 — was v1's on-disk size directly proven?** Agent 1 said unconfirmed (fell back to the
  14531 MiB runtime CUDA buffer). Agents 2/3 cited `cerebellum_v1.log` `file size = 14.86 GiB
  (4.74 BPW)` directly. **Resolved: Agents 2/3 correct; Agent 1 missed the line. v1 = 14.86 GiB /
  4.74 BPW CONFIRMED.**
- **D2 — how heavy was the v1 override?** Agent 1 described a heavy iq2_xs floor — but read that
  from `quantize_mixed_8.5gb.log`, which is a **separate ~8.5 GB Q2_K/iq2_xs sibling build**
  (`qwen3.6-27b-osmosis-mixed-8.5gb.gguf`), NOT v1. **Resolved: Agents 2/3 light-touch
  characterization is correct** (4.74 BPW, near-Q4_K_M, a handful of demotions). Agent 1's
  override map does not apply to v1.

### GAPS
- Exact v1 tensor-type override `.txt` **does not survive** (only `tensor_types_v3*/v4*.txt`, all
  dated May 3). Composition inferred from allocator logic + 4.74 BPW + 238-q4_K-tensor count.
- Literal `--budget-gb` for v1 not journaled (allocator default 12.0 GB, but output is 14.86 GiB).
- No captured v1 `llama-quantize` command line; v1 GGUF deleted from disk.
- imatrix calibration corpus content only weakly evidenced (tag + 8 chunks).
- wiki PPL corpus filename not named in surviving logs (145 chunks / n_ctx 2048 recorded).

---

## v2 — first allocator product, first benchmark suite

**Build benched Apr 28 23:34** (`cerebellum_v2.log`).

### CONFIRMED (in order)
1. **Same `osmosis-sensitivity` imatrix** (496/8), embedded in v2 GGUF.
2. **Baseline quants:** Q4_K_M baseline PPL 8.2556 (ablation harness reference); uniform Q2_K
   7.4996, Q3_K_M 7.6413.
3. **Per-tensor PPL ablation** (single-tensor, not group, not coding). 17/23 tensors tested at
   build time (commit `20251be` body, verbatim "Key findings (17/23 tensors tested)"); logs span
   Apr 28 20:24 -> Apr 29 00:53. Sacred: `blk.63.ffn_down` (+0.138), `blk.63.attn_q`.
4. **Build output** `qwen3.6-27b-cerebellum-v2.gguf`: 851 tensors, **file_type=10 (Q2_K)**, type
   counts f32:353 / q2_K:241 / q3_K:187 / q4_K:49 / q5_K:20 / q6_K:1; CUDA0 buffer ~10.5 GB.
5. **PPL gate: v2 PPL = 7.0868 ±0.04737.** Beats v1 (7.6713) and uniform Q2_K (7.4996); approaches
   Unsloth Q2_K_XL (~7.04 at 12 GB) while smaller.
6. **Benchmark suite (Apr 30, thinking off, temp=0):** HumanEval pass@1 63.4% (104/164); ARC-C
   84.81%; HellaSwag 75.02%; MMLU-Redux 57.29%. (`benchmark_results/cerebellum_v2_*`.) v2 is the
   first version to receive a downstream suite.

### DISPUTED
- **D1 — did the 23-tensor sweep actually drive v2, or did v2 ship mid-sweep?** Agents 1/2: it
  drove v2 (commit `20251be` narrates v2 as the allocator's first product). Agent 3: the sweep was
  **still running** (logs to Apr 29 00:53) when v2 was benched (Apr 28 23:34), so it fed v3, not
  v2. **Adjudication (not collapsed):** mtimes confirm Agent 3's premise — v2 was built before the
  sweep finished. Agents 1/2 have a commit *asserting* the link; Agent 3 has a timestamp *proving*
  the sweep was incomplete. Genuinely in tension; the override that fed v2 does not survive to
  settle it.
- **D2 — base type passed to `llama-quantize`?** Final GGUF file_type is unambiguously 10 (Q2_K).
  But `general.file_type` records the dominant result, not the CLI positional arg. Agent 1's
  two-hypothesis framing (Q2_K base + promotions, OR Q4_K_M base + mass demotion) is the honest
  state; no v2 quantize command survives.
- **D3 — `cerebellum_v2b` variant** appears in `benchmark_results` (Apr 30, Agent 2 only), no
  committed override/build log. Provenance is a GAP.

### GAPS
- **v2 override `.txt` does not exist** anywhere (repo, git history via
  `git log --all --diff-filter=A`, disk). Composition reconstructable only from final type-counts
  + allocator code.
- v2 `llama-quantize` build command / log not preserved.
- Exact `--budget-gb` unrecorded (only the ~10.7 GB / 3.41 BPW result survives).
- On-disk `ablation_results.json` / `ablation_plan.json` carry **May 3** mtimes (post-v4); the
  Apr-28 git blob at `20251be` is the v2-era authoritative copy.

---

## v3 (+ v3c) — full 23-tensor ablation, undersizing bug, tie with v2

**Shipped Apr 29** (commit `a8ac0a9`, 02:02:47). v3c is a byte-identical-map rebuild.

### CONFIRMED (in order)
1. **Same imatrix** (496/8), embedded in shipped GGUF.
2. **Ablation baseline PPL 8.2556** (Q4_K_M + imatrix, wikitext, 145 chunks).
3. **Per-tensor (NOT group) PPL ablation, 23 tensors.** `ablation_results.json` has exactly 23
   `gguf_tensor` entries, flat `layer_N.<tensor>` keys, no group aggregates. Helps (e.g.
   `layer_2.mlp.gate_proj` 8.1089) to hurts (`layer_63.self_attn.q_proj` 8.4178, most sacred).
   *(Note: agents 2/3 labeled this step "group-ablation" in their `phase` field but their prose
   describes per-tensor work — labeling artifact, not substance.)*
4. **Budget allocation, 4 budget points** (8.0 / 10.7 / 12.0 / 14.0 GB). 12.0 and 14.0 GB files
   byte-identical (allocator caps ~11.17 GB). 12 GB map = 181 lines, quant histogram q2_K×22 /
   q3_K×119 / q4_K×22 / q5_K×18.
5. **v3 build:** stock `llama-quantize`, Q2_K base floor (file_type=10) + imatrix + 181-tensor
   override. Came in **undersized ~10.15 GiB / 3.24 BPW** — the dry-run size estimator
   overestimated the imatrix-compressed base by ~720 MB.
6. **v3 PPL = 7.3156 ±0.04950.** Underperformed v2 (7.0868) purely from undersizing. PPL-only.
7. **v3c rebuild = identical map, inflated budget.** `tensor_types_v3c_11.4gb.txt` is
   **byte-identical** to `tensor_types_v3_12.0gb.txt` (matching md5 `d36b8871...`); only the budget
   target was inflated to compensate the ~720 MB bug. Result **10.58-10.59 GiB / 3.38 BPW**.
8. **v3c PPL = 7.0870 ±0.04723 — ties v2's 7.0868 exactly.** Recorded conclusion: 23 tensors of
   ablation data did not beat the 18-tensor (v2) extrapolation; v3 line abandoned, effort moved to
   v4. PPL-only, no task suite.

### DISPUTED
- **D1 — the 23-tensor selection mechanism (KL pre-screen).** Agent 1 attributes selection to
  nothing (does not cite `ablation_plan.json` for v3). Agents 2/3 assert a KL screen
  (`damage = kl2 × param_count`, reasons `highest_kl2 / lowest_kl2 / ...`) selected the 23,
  citing `ablation_plan.json`. **The dispute is real and NOT smoothed:** `ablation_plan.json` does
  contain those KL fields — but its **mtime is May 3 06:44, four days AFTER v3 shipped (Apr 29)**
  and contemporaneous with the v4-era coding-ablation dirs. So the artifact agents 2/3 lean on to
  prove v3's selection logic is itself dated to the v4 era. **Treat "a KL pre-screen selected v3's
  23 tensors" as plausible but NOT v3-artifact-proven; Agent 1's silence is arguably the more
  defensible position.**

### GAPS
- Exact `llama-quantize` invocation for v3/v3c not preserved (base ftype + imatrix + override
  inferred from GGUF metadata).
- Allocator command/flags + the v3c "inflated budget" value not saved.
- Raw imatrix calibration corpus / token count not pinned by any v3 artifact.
- On-disk byte sizes (10.15 / 10.59 GB) come from commit prose, not `ls` (GGUFs on the games
  partition, possibly deleted).
- Whether the 8.0 / 10.7 GB override variants were ever built + measured: only the 12.0 GB map
  (=v3=v3c) has a PPL log; `v3b_10.7gb.txt` is byte-identical to `v3_10.7gb.txt`.

---

## v4 — multi-pass promotion, the canonical build

**Allocator commit `aa793fd`, frozen Apr 29; benchmarks corrected May 3.** Canonical GGUF
`qwen3.6-27b-cerebellum-v4.gguf`, wiki PPL 7.0344.

### CONFIRMED (in order)
1. **imatrix** — same `osmosis-sensitivity` file (496/8), reused project-wide.
2. **Source prep** — HF bf16 -> F16 GGUF, arch `qwen35`, 64 blocks, 26.90 B params.
3. **23-tensor SAMPLED single-tensor PPL ablation** (NOT exhaustive over 851 tensors). baseline
   8.2556; exactly 23 `gguf_tensor` entries. KL-screened via `ablation_plan.json`. Key signals:
   `blk.63.attn_q` +0.162 (sacred), `blk.63.ffn_down` +0.138; `blk.2.ffn_gate` -0.147 and
   `blk.32.attn_qkv` -0.133 (demoting HELPS). This is the only ablation JSON present and is the
   sensitivity input to v4.
4. **Budget allocation, `cerebellum.py@aa793fd`, `--budget-gb 12.0`, base Q2_K** ->
   `tensor_types_v4_12gb.txt`. **The v4-distinguishing change is MULTI-PASS PROMOTION:** tensors
   climb q2_K -> q3_K -> q4_K -> q5_K -> q6_K -> q8_0 across repeated passes until budget is
   exhausted, replacing single-step promotion. Override = 181 explicit tensor lines. **Override-line
   histogram (re-counted, unanimous): q2_K 22, q3_K 19, q4_K 22, q5_K 70, q6_K 41, q8_0 7.**
5. **Build:** stock `llama-quantize`, base Q2_K (file_type=10), `--imatrix` the osmosis-sensitivity
   imatrix, `--tensor-type-file tensor_types_v4_12gb.txt`. Output `qwen3.6-27b-cerebellum-v4.gguf`;
   non-override tensors fall through to Q2_K.
6. **Wiki PPL = 7.0344 ±0.04625.** Ladder: uniform Q2_K 7.4996, Q2_K no-imatrix 7.6494, v3 7.3156,
   v3c 7.0870, **v4 7.0344**. Headline: "beats Unsloth Q2_K_XL 7.034 vs 7.040 at 12 GB".
7. **Benchmark suite (corrected May 3** after HumanEval fence/indent, ARC numeric-label, HellaSwag
   empty-response fixes): ARC-C 96.76% (1134/1172), HellaSwag 92.21% (9260/10042), MMLU-Redux
   76.58% (1838/2400). An Apr-30 llama-cpp-python HumanEval run scored 0.0% (runner artifact,
   superseded).
8. **POST-v4 group coding ablation** (diagnostic, NOT a build input) — see below.
9. **POST-v4 layer coding drill** (diagnostic, NOT a build input, Phase 2 truncated) — see below.

### changed_from_prev (CONFIRMED)
vs v3c (7.0870) -> v4 (7.0344): the single mechanical change is **multi-pass promotion** in
`cerebellum.py@aa793fd`. Same imatrix, same Q2_K base, same 23-tensor ablation data, same 12 GB
budget; only the allocator promotion loop changed.

### DISPUTED
- **D1 — ablation naming.** Agents 1/3 call it "group/sampled ablation"; Agent 2 says "per-TENSOR,
  NOT per-group". **Artifact resolves:** per-individual-tensor measurements on a KL-sampled subset
  of 23 tensors — both "per-tensor" and "sampled/not-exhaustive" are correct; it is NOT a
  per-tensor-TYPE group ablation. Terminology, not substance.
- **D2 — ablation baseline context.** Agent 3 alone characterizes the 8.2556 baseline as a
  "Q4_K_M-crush baseline" distinct from the uniform-Q2_K 7.4996. The 8.2556 value is
  artifact-proven; the "Q4_K_M-crush" characterization is Agent 3's assertion, **not independently
  confirmed** — treat as unverified.
- **D3 — ablation file mtime.** `ablation_results.json` / `ablation_plan.json` mtime May 3 06:44 is
  a later re-save; content is the Apr 29 sweep (commit `9336013`). mtime != run time.
- **D4 — final whole-GGUF tensor distribution (post-fall-through).** Agent 1: q2_K 253 / q3_K 77 /
  q4_K 49 / q5_K 70 / q6_K 42 / q8_0 7. Agent 2: f32:353 / q8_0:7 / q2_K:253 / q3_K:77 / q4_K:49
  (omits q5_K/q6_K, adds f32). **These loader-histogram readings are inconsistent and not
  reconciled.** The override-level counts (step 4) are solid; the whole-GGUF post-fall-through
  histogram is DISPUTED.
- **D5 — HellaSwag 92.21%.** Agents 1/2 cite a fixed result file; Agent 3 could not open a fixed
  JSON and notes the on-disk `cerebellum_v4_hellaswag_results.json` still shows the PRE-fix 91.18%,
  so the +1.0 correction rests partly on the commit message. Treat 92.21% as likely-correct with
  caveat.
- **D6 — HumanEval pass@1: 81.1% vs 82.9%, a GENUINE unresolved discrepancy.** BOTH are temp=0
  fixed-script runs (NOT a temperature artifact): `cerebellum_v4_fixed_humaneval_results.json` =
  0.8109756 (81.1%, temperature 0.0); `coding_promotion/v4_baseline_fixed_humaneval_results.json`
  = 0.8292683 (82.9%, temperature 0.0). `BENCHMARK_CORRECTIONS.md` itself says "confirmed across 2
  independent runs (81.1% and 82.9%)". 81.1% is the value in the saved `benchmarks/` file; 82.9% is
  the equally-temp=0 second run the corrections doc treats as the verified baseline. **The artifacts
  do NOT reconcile which is canonical.** (Separate v5-directed temperature sweep also exists:
  temp0.1=79.9%, 0.2=81.1%, 0.3=82.9%, 0.5=80.5% — not the published number.) The buggy **75.0%**
  is the OLD pre-fix score — which is why the coding-ablation deltas below are computed off 75.0%
  rather than 81/83.

### GAPS
- Exact `llama-quantize` command line for the v4 build is in NO captured log (inputs reconstructed
  from GGUF metadata).
- v4 GGUF true on-disk byte size not directly stated (~11.86 GiB / 11862 MiB CUDA0 per Agent 3;
  "12 GB" per benchmark.log; two possibly-different filenames `qwen3.6-27b-cerebellum-v4.gguf` vs
  `qwen3.6-27b-osmosis-budget-12gb-v4.gguf` never reconciled).
- imatrix calibration corpus contents unconfirmed (only tag + 8 chunks / 496 entries).
- The upstream KL-screening harness that PRODUCED `ablation_plan.json` was not located by any agent.
- PPL-worker count for the 23-tensor ablation not logged (project default N=2, not artifact-confirmed
  here).
- Whether v4 (PPL-driven) or a later coding-ablation-informed build became the public/HF release is
  out of scope of these artifacts.

---

## Coding ablation across all four versions — the decisive timeline

| Version | Build/freeze date | Group coding ablation (HumanEval per group)? | Layer coding drill? |
|---------|-------------------|----------------------------------------------|---------------------|
| v1 | Apr 28 | **NO** | **NO** |
| v2 | Apr 28 23:34 | **NO** | **NO** |
| v3 / v3c | Apr 29 02:02 | **NO** | **NO** |
| v4 | Apr 29 (frozen) | **YES — but POST-HOC, NOT a build input** | **YES (Phase 1 only) — POST-HOC, NOT a build input** |

**v1/v2/v3 — unanimous, artifact-proven NO.** Every file in `coding_ablation/` and
`coding_ablation_layers/` is mtime **May 2-3**, days after these builds. The log headers read
verbatim `Base: v4 (12 GB, 75.0% HumanEval baseline)`, `Started: 2026-05-02 22:42:26` (group) and
`2026-05-02 23:24:36` (layer). This is a **v4-base** campaign. v1/v2/v3 were gated on **PPL only**.

**v4 — YES, but the ablations ran AGAINST the already-frozen v4 GGUF, 3 days after the v4 override
(Apr 29 -> May 2).** The override `tensor_types_v4_12gb.txt` predates both ablations. They informed
FUTURE (v5 / code-optimized) variants only — **they did NOT feed shipped v4.** All three agents
agree, date-proven.

- **Group coding ablation (v4 base, May 2):** 7 tensor groups each demoted to Q2_K, HumanEval delta
  measured (temp=0, max_tokens=512, parallel=4, 164 problems). Results table (deltas off the buggy
  75.0% baseline): ssm_alpha 22.6% (-52.4%) [CRITICAL], ssm_beta 25.6% (-49.4%), attn_qkv 28.7%
  (-46.3%), output 29.9% (-45.1%), attn_v 30.5% (-44.5%), attn_output 42.1% (-32.9%), ffn_down
  70.7% (-4.3%, disposable).
- **Layer coding drill (v4 base, May 2-3):** Phase 1 (thirds) COMPLETED 12/12 and is verified (e.g.
  attn_qkv middle 40.2% best / late 27.4% worst; ssm_alpha late 25.0% worst). **Phase 2 (84
  individual layers) STARTED but TRUNCATED** — log ends after only ssm_alpha l43/l44/l45 builds;
  only 3 override files, NO Phase-2 result JSONs. (Agent 2 correct; Agents 1/3 overstated Phase 2
  as having run.)

---

## What the multi-domain-PPL reconstruction got WRONG vs this evidence-based one

The competing "multi-domain-PPL" reconstruction held that **coding ablation (HumanEval per group
and per layer) was part of the real method that produced the 27B builds** — i.e. that HumanEval-gated
group/layer ablation was a *build input* shaping the shipped tensor allocations.

**From the consensus evidence, that is WRONG.** Stated plainly from the artifacts:

1. **Coding ablation was NOT part of the method for v1, v2, or v3 — at all.** Every coding-ablation
   artifact is dated May 2-3 and headed `Base: v4`. v1 (Apr 28), v2 (Apr 28), and v3/v3c (Apr 29)
   shipped before any coding ablation existed. They were gated on **PPL only** — single-tensor
   Q4_K_M->Q2_K wikitext PPL deltas against the 8.2556 baseline, fed to the demote/sacred budget
   allocator. Unanimous across all 9 agents (3 per version), artifact-proven by mtime and log
   headers.

2. **Coding ablation was NOT a build input even for v4.** It ran 3 days AFTER the v4 override was
   frozen (`tensor_types_v4_12gb.txt` predates `coding_ablation.log`). v4 was selected and shipped
   on the **PPL ladder** (7.0344, beating v3c 7.0870 via multi-pass promotion) — not on any
   HumanEval-per-group signal. The coding ablation was a **post-hoc diagnostic** pointed at the
   already-built v4 GGUF, intended to steer FUTURE (v5 / code-optimized) variants.

3. **The HumanEval-per-layer drill never even completed.** Its Phase 2 (the 84-individual-layer
   sweep) was truncated after 3 builds with no recorded results — so it could not have driven any
   shipped allocation regardless of date.

4. **The genuine v1->v4 method is PPL-gated throughout.** Sensitivity-proxy imatrix -> 23-tensor
   single-tensor PPL ablation (baseline 8.2556) -> demote/promote budget allocation from a Q4_K_M
   (v1) or Q2_K (v2-v4) base -> stock `llama-quantize`. The only methodological evolution across
   versions is allocator mechanics (v1 Q4_K_M base; v2 Q2_K base; v3 4 budget points + undersizing
   fix; v4 multi-pass promotion). Coding/HumanEval signal enters the project's history **only after
   v4 was already frozen**, and only as forward-looking diagnostics.

**Bottom line:** the multi-domain-PPL account overstates coding ablation's role by treating a
post-v4 diagnostic campaign as a build-time method input. The evidence shows the real 27B v1->v4
pipeline was **PPL-gated end to end**; HumanEval-per-group and HumanEval-per-layer ablation were
never an input to any of the four shipped builds.

---

### Key artifact paths (absolute)
- `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/ablation_logs/cerebellum_v1.log`
  (v1: file_type 15, 14.86 GiB / 4.74 BPW, PPL 7.6713)
- `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/ablation_logs/cerebellum_v2.log`
  (v2: file_type 10, PPL 7.0868)
- `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/ablation_logs/cerebellum_v3_ppl.log`
  / `cerebellum_v3c_ppl.log` (v3 7.3156 / v3c 7.0870)
- `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/ablation_logs/cerebellum_v4_ppl.log`
  (v4 7.0344)
- `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/ablation_logs/baseline_q4km.log`
  (baseline 8.2556)
- `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/ablation_results.json` (23
  single-tensor PPL deltas) / `ablation_plan.json` (KL screen; mtime May 3 06:44)
- `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/coding_ablation/coding_ablation.log`
  (`Base: v4`, May 2 — POST-v4, NOT a build input for any version)
- `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/coding_ablation_layers/layer_ablation.log`
  (`Base: v4`, May 2-3; Phase 2 truncated)
- `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/quantize_mixed_8.5gb.log`
  (separate Q2_K/iq2_xs 8.5 GB sibling build — NOT v1)
- `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/benchmark_results/` (v2 / v2b / v4
  only; zero v1 or v3 entries)
- Commits: `20251be` (allocator + v2), `a8ac0a9` / `0d57e7d` (v3/v3c), `aa793fd` / `9336013` (v4)
