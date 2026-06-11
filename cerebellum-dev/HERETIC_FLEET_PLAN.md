# Heretic Fleet Plan — Cerebellum Transfer Across All Gemma 4 and Qwen Releases
# Date: 2026-06-11
# Protocol: WINNING_METHOD.md §Heretic Transfer Protocol
# Prior validated runs: Gemma4 26B Heretic v1 (coder3101), Qwen3.6 35B Heretic v2 (IN PROGRESS)

Private. Do not push to `origin`.

---

## Source Quality Rules

- **TRUSTED**: llmfan46, coder3101
- **BANNED**: huihui-ai (tried, was a mess)
- **Preference**: plain heretic over ultra; flag ultra-only as separate option
- **Hard exclude**: MTP-preserved variants (blk.40+ = architecture mismatch, load failure,
  benchmark regression proven on Qwen3.6-35B June 3 attempt)

---

## Artifact Inventory (verified local)

| Model | Override file | Status | Imatrix | Status |
|---|---|---|---|---|
| Gemma4 26B-A4B v6 | `osmosis-gemma4-26b/cerebellum_v6_overrides.txt` (91 entries, 2636 B) | VERIFIED | `osmosis-gemma4-26b/google_gemma-4-26B-A4B-it-imatrix.gguf` (56.9 MB, bartowski, 590 tensors) | VERIFIED |
| Qwen3.6 27B v4 | `osmosis-qwen36-27b/tensor_types_v4_12gb.txt` (181 overrides, 4982 B) | VERIFIED | `osmosis-qwen36-27b/cerebellum_imatrix.dat` (13.6 MB, 496 entries) | VERIFIED |
| Qwen3.6 35B-A3B v3 | `/var/home/deucebucket/games/qwen36-35b-v2/cerebellum_v3_overrides.txt` (360 entries, 11230 B) | VERIFIED | `/var/home/deucebucket/games/qwen36-35b-v2/imatrix_unsloth.gguf_file` (192 MB, 1020 tensors) | VERIFIED |
| Qwen3.5 9B v1 | `osmosis-qwen35-9b/tensor_types_v1_4.5gb.txt` (1788 B) | VERIFIED | `osmosis-qwen35-9b/cerebellum_imatrix.dat` (5.0 MB) | VERIFIED |
| Qwen3.5 122B v4 | `osmosis-qwen35-122b/ablation/override_v1_full.txt` (422 entries, 12722 B) | VERIFIED | `osmosis-qwen35-122b/imatrix_unsloth.gguf` (343 MB, 612 entries, unsloth calib) | VERIFIED |
| Qwen3 14B v2 | `osmosis-qwen3-14b/ablation/cerebellum_v2.txt` (3290 B) | VERIFIED | `/var/home/deucebucket/games/models/imatrix_unsloth.dat` (7.7 MB) | VERIFIED |
| Qwen3 30B-A3B v3 | `osmosis-qwen3-30b/ablation/cerebellum_v2.txt` (v2 only; v3 override NOT locally present) | PARTIAL | `/var/home/deucebucket/games/models/Qwen_Qwen3-30B-A3B.imatrix` | NOT DOWNLOADED (lock only) |
| Qwen3 32B v2 | `osmosis-qwen3-32b/ablation/cerebellum_v2.txt` (5282 B) | VERIFIED | `/var/home/deucebucket/games/models/Qwen_Qwen3-32B.imatrix` | NOT DOWNLOADED (lock only) |
| Gemma4 E4B v2 | `osmosis-gemma4-e4b/ple_overrides.txt` (174 entries) | VERIFIED | `osmosis-gemma4-e4b/imatrix.dat` (4.6 MB, weight-stat) | VERIFIED |
| Gemma4 E2B v2 | `osmosis-gemma4-e2b/cerebellum_v2_overrides.txt` (3 entries) | VERIFIED | embedded in `osmosis-gemma4-e2b/google_gemma-4-E2B-it-Q3_K_M.gguf` (3.1 GB) | VERIFIED |

Notes on 30B v3: The shipped version is v3 (coder imatrix, attn_q sacred). v2 override is local but is slightly different allocation. v3 override must be reconstructed or re-derived before heretic build.

---

## Heretic Source Matching

| Our Model | Base Model | Trusted Heretic Source | Variant | BF16 Source Format | Approx BF16 Size |
|---|---|---|---|---|---|
| Gemma4 26B-A4B | google/gemma-4-26B-A4B-it | **coder3101/gemma-4-26B-A4B-it-heretic** | plain ✓ | safetensors (2-part) | 51.6 GB |
| Gemma4 26B-A4B | google/gemma-4-26B-A4B-it | llmfan46/gemma-4-26B-A4B-it-uncensored-heretic-GGUF | plain ✓ | BF16 GGUF (single) | ~52 GB |
| Qwen3.6 27B | Qwen/Qwen3.6-27B | **llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF** | plain ✓ | BF16 GGUF | ~55 GB |
| Qwen3.6 35B-A3B | Qwen/Qwen3.6-35B-A3B | **llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF** | plain ✓ | BF16 GGUF | ~70 GB |
| Qwen3.5 9B | Qwen/Qwen3.5-9B | **coder3101/Qwen3.5-9B-heretic** | plain ✓ | safetensors (single) | 18.8 GB |
| Qwen3.5 122B | Qwen/Qwen3.5-122B-A10B | NO TRUSTED SOURCE | — | — | — |
| Qwen3 14B | Qwen/Qwen3-14B | NO TRUSTED SOURCE (llmfan46/coder3101 have no Qwen3-14B) | — | — | — |
| Qwen3 30B-A3B | Qwen/Qwen3-30B-A3B | NO TRUSTED SOURCE | — | — | — |
| Qwen3 32B | Qwen/Qwen3-32B | NO TRUSTED SOURCE | — | — | — |
| Gemma4 E4B | google/gemma-4-E4B-it | **coder3101/gemma-4-E4B-it-heretic** | plain ✓ | safetensors (single) | 16.0 GB |
| Gemma4 E2B | google/gemma-4-E2B-it | **coder3101/gemma-4-E2B-it-heretic** | plain ✓ | safetensors (single) | 10.2 GB |

Ultra-only flags:
- llmfan46/Qwen3.5-9B: only ultra variants exist. coder3101 plain is preferred.
- llmfan46/gemma-4-E2B/E4B: only ultra variants. coder3101 plain is preferred for both.
- Qwen3.5-122B: trohrbaugh/Qwen3.5-122B-A10B-heretic exists (safetensors) but is NOT from a
  trusted author. Sabomako has a 6-part BF16 GGUF but also not trusted. BLOCKED pending
  finding a trusted heretic for this base.

NO TRUSTED HERETIC: Qwen3-14B, Qwen3-30B-A3B, Qwen3-32B — neither llmfan46 nor coder3101
carry these. Qwen3.5-122B also blocked on trusted source.

---

## Priority-Ordered Fleet Plan

Heretic transfer is already proven twice (Gemma4 26B Heretic v1, Qwen3.6 35B in-progress).
Priority = impact (popular model + recipe quality) × effort (recipe completeness).

### Priority 1 — Gemma 4 26B-A4B Heretic v2 (PROVEN RECIPE, router surgery required)

| Field | Value |
|---|---|
| Our model / HF release | deucebucket/Gemma-4-26B-A4B-it-Heretic-Cerebellum-GGUF (existing, update to v2) |
| Target repo name | deucebucket/Gemma-4-26B-A4B-it-Heretic-Cerebellum-GGUF |
| Heretic source | coder3101/gemma-4-26B-A4B-it-heretic (preferred: plain, trusted author) |
| Source file | model-00001-of-00002.safetensors + model-00002-of-00002.safetensors |
| Source size | ~51.6 GB (safetensors) → convert to F16 GGUF ~52 GB |
| Override | osmosis-gemma4-26b/cerebellum_v6_overrides.txt (91 entries, VERIFIED) |
| Imatrix | osmosis-gemma4-26b/google_gemma-4-26B-A4B-it-imatrix.gguf (VERIFIED) |
| Base quant | Q3_K_M |
| Router surgery | YES: `blk.8.ffn_gate_inp.weight → Q8_0` via gguf_tensor_surgery.py post-quantize |
| Output size | ~17 GB (Q3_K_M) |
| Disk needed | ~52 GB (source) + ~17 GB (output) = ~69 GB peak on /games (351 GB free) |
| Est. pipeline time | 4-6 hrs (convert) + 3-4 hrs (quantize) + 2 hrs (bench) ≈ 10-12 hrs |
| Alt source | llmfan46/gemma-4-26B-A4B-it-uncensored-heretic-GGUF (BF16 GGUF, single file, skip convert step) |
| Blocker | None. Recipe proven. Router surgery step documented. |

Architecture gotchas:
- K-quants broken on Gemma4 26B routers (Q6_K = +15.9% PPL, Q2_K = +17.2%); Q8_0 is only safe option
- Layer 8 is the critical router layer (identified by per-layer ablation of all 30 router layers)
- Router surgery must happen AFTER llama-quantize, not before

Per-model checklist:
- [ ] Download coder3101/gemma-4-26B-A4B-it-heretic safetensors
- [ ] Convert safetensors → F16 GGUF (or use llmfan46 BF16 GGUF directly)
- [ ] Verify no blk.40 (MTP check): `llama-gguf --list src.gguf | grep "blk\.40"` → expect nothing
- [ ] Verify override file 91-entry match: `wc -l osmosis-gemma4-26b/cerebellum_v6_overrides.txt` → 91
- [ ] Run llama-quantize with imatrix + override + Q3_K_M base
- [ ] Apply router surgery: `blk.8.ffn_gate_inp.weight → Q8_0`
- [ ] PPL sanity (expect ~12,050 ± 200)
- [ ] Benchmark gauntlet: ARC, HellaSwag, MMLU-Redux, HumanEval+ (gate: ≥ Heretic v1 on all four)
- [ ] Audit EvalPlus completions (audit_evalplus_completions.py)
- [ ] Release to deucebucket/Gemma-4-26B-A4B-it-Heretic-Cerebellum-GGUF

---

### Priority 2 — Qwen 3.6 35B-A3B Heretic v2 (IN PROGRESS — exclude from new planning)

Already executing. See RECIPE_heretic_qwen36_35b.md.
Target: deucebucket/Qwen3.6-35B-A3B-Heretic-Cerebellum-GGUF (new repo, version-agnostic)

---

### Priority 3 — Qwen 3.6 27B Heretic v1

| Field | Value |
|---|---|
| Target repo | deucebucket/Qwen3.6-27B-Heretic-Cerebellum-GGUF |
| Heretic source | llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF |
| Source file | Qwen3.6-27B-uncensored-heretic-v2-BF16.gguf |
| Source size | ~55 GB |
| Override | osmosis-qwen36-27b/tensor_types_v4_12gb.txt (181 overrides, VERIFIED) |
| Imatrix | osmosis-qwen36-27b/cerebellum_imatrix.dat (13.6 MB, VERIFIED) |
| Base quant | Q2_K |
| Router surgery | None (Qwen3.6 27B: no router curve study; not MoE) |
| Output size | ~12 GB |
| Disk needed | ~55 GB (source) + ~12 GB (output) = ~67 GB peak on /games |
| Est. pipeline time | 2 hrs (download) + 3 hrs (quantize Q2_K from BF16) + 2 hrs (bench) ≈ 7-8 hrs |
| Blocker | None. Verify blk.* naming matches before quantize (no MTP check needed — single-BF16). |

Architecture gotchas:
- Dense model; no MoE, no SSM. Straightforward transfer.
- Q2_K base + imatrix beats Q3_K_M without (validated in stock v4)
- Verify tensor count matches stock (expect ~28 blk.* layers × dense tensor names)

Per-model checklist:
- [ ] Download llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF BF16
- [ ] MTP check: `llama-gguf --list src.gguf | grep "blk\."` → expect max blk.27
- [ ] Verify tensor_types_v4_12gb.txt (181 entries match model naming)
- [ ] Run llama-quantize: Q2_K base + imatrix + override file
- [ ] PPL sanity (stock v4 was 7.0344; expect heretic ≤ 7.4)
- [ ] Benchmark gauntlet (gate: ≥ stock v4 or same-size heretic Q2_K uniform)
- [ ] Audit EvalPlus completions
- [ ] Release to deucebucket/Qwen3.6-27B-Heretic-Cerebellum-GGUF

---

### Priority 4 — Gemma 4 E4B Heretic v1

| Field | Value |
|---|---|
| Target repo | deucebucket/Gemma-4-E4B-it-Heretic-Cerebellum-GGUF |
| Heretic source | coder3101/gemma-4-E4B-it-heretic |
| Source file | model.safetensors (single, ~16 GB) |
| Source size | 16 GB safetensors → ~16 GB F16 GGUF |
| Override | osmosis-gemma4-e4b/ple_overrides.txt (174 entries, VERIFIED) |
| Imatrix | osmosis-gemma4-e4b/imatrix.dat (4.6 MB, VERIFIED) |
| Base quant | Q3_K_M |
| Router surgery | None |
| Output size | ~4.2 GB |
| Disk needed | ~16 GB (source) + ~4 GB (output) = ~20 GB peak on /games |
| Est. pipeline time | 1 hr (download) + 1.5 hrs (convert + quantize) + 1.5 hrs (bench) ≈ 4 hrs |
| Alt source (flag) | llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic-GGUF — ULTRA only, prefer coder3101 |
| Blocker | None. |

Architecture gotchas (PLE model — critical):
- Gemma4 E4B is a Per-Layer Embedding model. Q4_K → Q3_K cliff: without PLE protection,
  PPL = 104; with PLE@Q5_K = PPL 55. The ple_overrides.txt handles this.
- The PLE tensor names (per_layer_token_embd, inp_gate, proj) must be present in heretic.
  Verify: `llama-gguf --list src_f16.gguf | grep "per_layer"` → expect entries
- coder3101 heretic only modifies o_proj/out_proj (attention output projections); PLE tensors
  are untouched. Transfer should be safe.

Per-model checklist:
- [ ] Download coder3101/gemma-4-E4B-it-heretic safetensors
- [ ] Convert to F16 GGUF (llama-convert-hf-to-gguf)
- [ ] MTP check (E4B has 42 layers, expect max blk.41)
- [ ] Verify ple_overrides.txt PLE tensor names exist in heretic F16
- [ ] Run llama-quantize: Q3_K_M + imatrix + ple_overrides.txt
- [ ] PPL sanity (stock v2 = 52.20; expect heretic ≤ 58)
- [ ] Benchmark gauntlet
- [ ] Audit EvalPlus completions
- [ ] Release to deucebucket/Gemma-4-E4B-it-Heretic-Cerebellum-GGUF

---

### Priority 5 — Qwen 3.5 9B Heretic v1

| Field | Value |
|---|---|
| Target repo | deucebucket/Qwen3.5-9B-Heretic-Cerebellum-GGUF |
| Heretic source | coder3101/Qwen3.5-9B-heretic |
| Source file | model.safetensors (single, ~18.8 GB) |
| Override | osmosis-qwen35-9b/tensor_types_v1_4.5gb.txt (VERIFIED) |
| Imatrix | osmosis-qwen35-9b/cerebellum_imatrix.dat (5.0 MB, VERIFIED) |
| Base quant | (check: stock v1 was 4.5 GB target — likely Q4_K_M or IQ3_XXS) |
| Router surgery | None |
| Output size | ~4.5 GB |
| Disk needed | ~19 GB (source) + ~5 GB (output) = ~24 GB peak |
| Est. pipeline time | 1 hr (download) + 1.5 hrs (convert + quantize) + 1.5 hrs (bench) ≈ 4 hrs |
| Alt source (flag) | llmfan46 only has ultra variants for Qwen3.5-9B — AVOID unless coder3101 blocked |
| Blocker | Verify base quant from tensor_types_v1_4.5gb.txt before build. |

Architecture gotchas:
- Qwen3.5-9B is a hybrid SSM model. SSM params (in_proj_a/b, A_log, dt_bias, conv1d, in_proj_z)
  hard-fail below 4-bit: NaN, no gradual degradation. Verify tensor_types_v1 preserves
  SSM floor (expect Q4_K minimum on all SSM tensors).
- Confirm with: `grep "ssm\|conv1d\|A_log\|dt_bias" osmosis-qwen35-9b/tensor_types_v1_4.5gb.txt`

Per-model checklist:
- [ ] Check base quant from tensor_types_v1_4.5gb.txt
- [ ] Verify SSM tensor floors in override file
- [ ] Download coder3101/Qwen3.5-9B-heretic safetensors
- [ ] Convert to F16 GGUF
- [ ] Run llama-quantize with imatrix + override
- [ ] PPL sanity
- [ ] Benchmark gauntlet
- [ ] Audit EvalPlus completions
- [ ] Release to deucebucket/Qwen3.5-9B-Heretic-Cerebellum-GGUF

---

### Priority 6 — Gemma 4 E2B Heretic v1

| Field | Value |
|---|---|
| Target repo | deucebucket/Gemma-4-E2B-it-Heretic-Cerebellum-GGUF |
| Heretic source | coder3101/gemma-4-E2B-it-heretic |
| Source file | model.safetensors (single, ~10.2 GB) |
| Override | osmosis-gemma4-e2b/cerebellum_v2_overrides.txt (3 entries: blk.11/13/14 ffn_gate → Q2_K) |
| Imatrix | Embedded in `osmosis-gemma4-e2b/google_gemma-4-E2B-it-Q3_K_M.gguf` (3.1 GB, VERIFIED) |
| Base quant | Q3_K_M + --allow-requantize |
| Router surgery | None |
| Output size | ~3.0 GB |
| Disk needed | ~10 GB (source) + ~3 GB (output) = ~13 GB peak |
| Est. pipeline time | 0.5 hr (download) + 1 hr (convert + quantize) + 1 hr (bench) ≈ 2.5 hrs |
| Alt source (flag) | llmfan46/gemma-4-E2B-it-ultra-uncensored-heretic-GGUF — ULTRA only; coder3101 preferred |
| Blocker | imatrix path: the bartowski imatrix is embedded in google_gemma-4-E2B-it-Q3_K_M.gguf. For heretic build, use the same GGUF as imatrix source OR re-derive from E2B heretic F16. |

IMPORTANT imatrix note for E2B: The stock E2B recipe used --allow-requantize from the
Q3_K_M.gguf which embeds a bartowski imatrix. For the heretic build, there are two options:
1. Convert heretic safetensors → F16 GGUF, then quantize directly to Q3_K_M (no allow-requantize
   needed). Use stock E2B imatrix.dat if bartowski imatrix is needed separately, or generate
   fresh from E2B heretic F16 with `python -m osmosis.imatrix_stream` (45 sec on CPU).
2. OR: use the existing google_gemma-4-E2B-it-Q3_K_M.gguf as imatrix source (--imatrix flag
   accepting a GGUF with embedded imatrix — check if llama-quantize version supports this).

Preferred path: convert heretic safetensors → F16, run imatrix_stream (~45 sec), then quantize.

Architecture gotchas (PLE model):
- E2B is also PLE. The v2 override only demotes 3 ffn_gate tensors — no PLE protection in override.
  Stock E2B v2 did NOT add PLE protection (unlike E4B); benchmark results showed no regression.
  Keep the same 3-entry override.
- E2B has 35 layers (blk.0–blk.34). Verify heretic has same depth.

Per-model checklist:
- [ ] Download coder3101/gemma-4-E2B-it-heretic safetensors
- [ ] Convert to F16 GGUF
- [ ] Generate imatrix from heretic F16 (osmosis.imatrix_stream, ~45 sec)
- [ ] Verify 3 override tensors exist (blk.11/13/14 ffn_gate)
- [ ] Run llama-quantize: Q3_K_M + imatrix + 3-entry override (no allow-requantize needed)
- [ ] PPL sanity (stock v2 = 139.69; heretic should be similar range)
- [ ] Benchmark gauntlet (gate: match stock E2B v2 on HumanEval, ARC, HellaSwag, MMLU)
- [ ] Audit EvalPlus completions
- [ ] Release to deucebucket/Gemma-4-E2B-it-Heretic-Cerebellum-GGUF

---

## Models with NO Trusted Heretic Source (BLOCKED)

| Model | Why blocked |
|---|---|
| Qwen3-14B | No heretic from llmfan46 or coder3101. Untrusted sources exist (opensynthesis, mradermacher). |
| Qwen3-30B-A3B | No heretic from llmfan46 or coder3101. Untrusted sources only. |
| Qwen3-32B | No heretic from llmfan46 or coder3101. Untrusted sources only. |
| Qwen3.5-122B | No heretic from llmfan46 or coder3101. trohrbaugh/Sabomako exist but are untrusted. |

Additional blockers for 30B:
- v3 override file NOT locally present (only v2 is local). v3 differs in what's sacred
  (attn_q protected in v3, not v2) and uses a coder imatrix (not wiki imatrix). Would need
  to rebuild or reconstruct v3 override from recipe documentation before any heretic build.
- 30B imatrix (Qwen_Qwen3-30B-A3B.imatrix) only has download lock/metadata — not downloaded.

Additional blockers for 32B:
- Qwen_Qwen3-32B.imatrix only has download lock/metadata — not downloaded.

---

## Aggregate Disk and Time Estimates

| Priority | Model | Download GB | Output GB | Peak GB | Est. Time |
|---|---|---|---|---|---|
| 1 | Gemma4 26B Heretic v2 | 52 | 17 | 69 | 10-12 hrs |
| 2 | Qwen3.6 35B Heretic v2 | 70 (in progress) | 11 | 81 | IN PROGRESS |
| 3 | Qwen3.6 27B Heretic v1 | 55 | 12 | 67 | 7-8 hrs |
| 4 | Gemma4 E4B Heretic v1 | 16 | 4 | 20 | 4 hrs |
| 5 | Qwen3.5 9B Heretic v1 | 19 | 5 | 24 | 4 hrs |
| 6 | Gemma4 E2B Heretic v1 | 10 | 3 | 13 | 2.5 hrs |
| TOTAL (sequential, delete sources after each) | | ~222 GB max peak | ~52 GB final | 69 GB max at once | ~30 hrs |

Available on /games: **351 GB free** — sufficient for sequential execution (delete source after output verified).
Available on /var/home: 157 GB (92% used) — keep artifacts on /games.

---

## Execution Sequence (recommended)

1. Gemma4 26B Heretic v2 (priority 1 — highest impact, recipe proven)
2. Qwen3.6 27B Heretic v1 (priority 3 — lightweight recipe, fast)
3. Gemma4 E4B Heretic v1 (priority 4 — small model, quick)
4. Qwen3.5 9B Heretic v1 (priority 5 — small model, quick)
5. Gemma4 E2B Heretic v1 (priority 6 — smallest, fastest)
6. BLOCKED: Qwen3-14B, Qwen3-30B, Qwen3-32B, Qwen3.5-122B — pending trusted source discovery

---

## Global Build Notes

- All quantize runs: `distrobox enter ai` (CUDA available)
- llama-quantize bin: `/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize`
- PPL corpus: `/var/home/deucebucket/games/osmosis-quants/wiki.test.raw`
- Bench server: `-ngl 99 --ctx-size 24576 --parallel 4` (mandatory minimum)
- HumanEval+: BENCH_WORKERS=1 (sequential)
- ARC/HellaSwag/MMLU: BENCH_WORKERS=4
- Audit EvalPlus completions with `audit_evalplus_completions.py` after EVERY HumanEval run
- No ship without passing all 4 benchmark gates vs same-size heretic uniform-quant baseline
- Target repo naming: deucebucket/<Model>-Heretic-Cerebellum-GGUF (version-agnostic)
- Do NOT push to `origin`: no devlogs, no private pipeline code, no intermediate results

---

## Compact Priority Table (summary)

| Pri | Model | Heretic Source | Source (GB) | Output | Disk Peak | Time | Router Surgery | Blockers |
|---|---|---|---|---|---|---|---|---|
| 1 | Gemma4 26B-A4B | coder3101/gemma-4-26B-A4B-it-heretic | 52 GB (ST) | ~17 GB | 69 GB | 10-12 hrs | YES: blk.8 → Q8_0 | None |
| 2 | Qwen3.6 35B-A3B | llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF | 70 GB (BF16) | ~11 GB | 81 GB | IN PROGRESS | None | None |
| 3 | Qwen3.6 27B | llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF | 55 GB (BF16) | ~12 GB | 67 GB | 7-8 hrs | None | None |
| 4 | Gemma4 E4B | coder3101/gemma-4-E4B-it-heretic | 16 GB (ST) | ~4 GB | 20 GB | 4 hrs | None | Verify PLE tensors |
| 5 | Qwen3.5 9B | coder3101/Qwen3.5-9B-heretic | 19 GB (ST) | ~5 GB | 24 GB | 4 hrs | None | Verify SSM floor |
| 6 | Gemma4 E2B | coder3101/gemma-4-E2B-it-heretic | 10 GB (ST) | ~3 GB | 13 GB | 2.5 hrs | None | Re-generate imatrix |
| BLOCK | Qwen3-14B/30B/32B | None (trusted) | — | — | — | — | — | No trusted heretic |
| BLOCK | Qwen3.5-122B | None (trusted) | — | — | — | — | — | No trusted heretic |

ST = safetensors (requires convert step). BF16 = BF16 GGUF (quantize directly).
