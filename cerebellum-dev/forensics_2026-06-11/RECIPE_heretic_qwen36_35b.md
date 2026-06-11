# Heretic Qwen 3.6 35B-A3B — Cerebellum Recipe
# Date: 2026-06-11
# Target: llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF (BF16 GGUF, 69.4 GB)
# Source for this recipe: local archaeology of stock 35B Cerebellum v3 + existing heretic v1 attempt

Private. Do not push to `origin`.

## TL;DR

The proven recipe for Qwen 3.6 35B-A3B Cerebellum is a Q3_K_M-base quantization
with a 360-entry tensor override file (40 layers × 9 tensor types at Q2_K).
Ablation was run on the stock model; the heretic simply REUSES the same override
file verbatim without re-running ablation. The prior heretic attempt (June 3) proved
this works mechanically — the regression there was architectural (MTP-preserved source
from a different repo). The new target (llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF)
is non-MTP and should transfer cleanly.

---

## Q1: Gemma heretic — did we re-run ablation?

NO. The Gemma 4 26B Heretic (coder3101 source) reused the stock v6 override map verbatim:
- Same override file: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/cerebellum_v6_overrides.txt`
- Same imatrix: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/google_gemma-4-26B-A4B-it-imatrix.gguf`
- Only the source BF16 GGUF changed.
The heretic had the same tensor architecture as the stock model. This is the general pattern:
re-ablation is only needed if the architecture changes. Heretic ablations only modify activations
(orthogonal projection on o_proj/out_proj), not weight tensor layout — so tensor sensitivity
transfers exactly.

The Qwen 3.6 35B heretic first attempt (2026-06-03) confirmed the same: v3_overrides.txt was
reused, the build ran, 360 overrides all matched. The regression was from a MTP-preserved GGUF
(extra block blk.40 caused llama.cpp load failure later).

## Q2: Exact winning pipeline for Qwen 3.6 35B-A3B

Architecture: 40 layers, hybrid SSM+Attention MoE, 256 experts, 3B active params.

### Original stock ablation findings (from osmosis-qwen36-35b/ablation/):

Forward ablation vs Q3_K_M baseline PPL 7.1758 — all groups at Q2_K:

| Group | PPL delta | Verdict |
|---|---:|---|
| ssm_alpha | -0.01% | FREE |
| attn_gate | +0.1% | FREE |
| ssm_beta | +0.06% | FREE |
| ffn_gate_shexp | +0.6% | DEMOTABLE |
| ffn_down_shexp | +0.7% | DEMOTABLE |
| ffn_up_exps | +1.2% | DEMOTABLE |
| ffn_up_shexp | +1.2% | DEMOTABLE |
| attn_qkv | +1.4% | DEMOTABLE |
| ffn_gate_exps | +1.4% | DEMOTABLE |
| ffn_down_exps | +1.9% | DEMOTABLE |
| ssm_out | +3.3% | PROTECT |

Reverse ablation from fully-demoted v1: only attn_qkv, ffn_down_exps, ffn_up_exps
genuinely improved when restored — but per-layer ablation showed the effect was
spread across all 40 layers equally (no surgical layers). Benchmark result on v3
showed that demoting all three anyway (while keeping ssm_out and attn_qkv at Q3_K_M
default) beat v2's benchmark scores at smaller size.

### v3 override file (VERIFIED LOCAL):
- Path: `/var/home/deucebucket/games/qwen36-35b-v2/cerebellum_v3_overrides.txt`
- Size: 11230 bytes, 360 lines
- Pattern: blk.0–39, each with 9 entries: ffn_gate_exps, ffn_up_exps, ffn_down_exps,
  ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp, attn_gate, ssm_alpha, ssm_beta — all Q2_K
- Protected (stays at Q3_K_M): attn_qkv, ssm_out, all F32 norms/SSM state params, router tensors

### Imatrix used for v3 (VERIFIED LOCAL):
- Primary/stronger: `/var/home/deucebucket/games/qwen36-35b-v2/imatrix_unsloth.gguf_file`
  - 192 MB, 1020 tensors, 76 chunks, dataset: unsloth_calibration_Qwen3.6-35B-A3B.txt
  - This is the imported Unsloth coder calibration imatrix used in the shipped v3
- Backup local: `/var/home/deucebucket/games/qwen36-35b-v2/cerebellum_imatrix.dat`
  - 108 MB, 470 entries, ncall=1 (weight-stat only, less calibrated)

### Base quant: Q3_K_M
### Output size: ~11 GB (2.76 BPW)

---

## Exact Step-by-Step Plan for llmfan46 Heretic Build

### Working dirs:

```bash
WORKDIR="/var/home/deucebucket/games/cerebellum-heretic-qwen36-35b-v2"   # new run dir
OVERRIDES="/var/home/deucebucket/games/qwen36-35b-v2/cerebellum_v3_overrides.txt"
IMATRIX="/var/home/deucebucket/games/qwen36-35b-v2/imatrix_unsloth.gguf_file"
LLAMA_BIN="/var/home/deucebucket/ai-drive/llama.cpp/build/bin"
SCRIPTS="/var/home/deucebucket/ai-drive/cerebellum/scripts"
mkdir -p "$WORKDIR"
```

---

### STEP 0: Acquire BF16 GGUF source

Target repo: `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF`
File: `Qwen3.6-35B-A3B-uncensored-heretic-BF16.gguf` (69.4 GB)
mmproj: `Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF16.gguf` (~861 MB)

```bash
# Disk needed: ~71 GB free on target partition
# Current free on /var/home/deucebucket/games: 192 GB — sufficient

# Download BF16 GGUF (resume-capable)
huggingface-cli download \
  llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF \
  Qwen3.6-35B-A3B-uncensored-heretic-BF16.gguf \
  --local-dir "$WORKDIR/source" \
  --local-dir-use-symlinks False

# Download mmproj
huggingface-cli download \
  llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF \
  Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF16.gguf \
  --local-dir "$WORKDIR/source" \
  --local-dir-use-symlinks False
```

NOTE: Check HF cache first — partial download may exist:
`/var/home/deucebucket/.cache/huggingface/hub/models--llmfan46--Qwen3.6-35B-A3B-uncensored-heretic-GGUF/`
Lock files were seen there (`.lock`) suggesting an interrupted prior download.

Source vars:
```bash
SRC_BF16="$WORKDIR/source/Qwen3.6-35B-A3B-uncensored-heretic-BF16.gguf"
SRC_MMPROJ="$WORKDIR/source/Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF16.gguf"
```

**Disk at this point: ~71 GB used for source**

---

### STEP 1: Verify override file matches architecture

```bash
# Count override entries
wc -l "$OVERRIDES"  # expect 360

# Spot-check first/last few lines
head -9 "$OVERRIDES"
tail -9 "$OVERRIDES"
# Expect: blk.0.ffn_gate_exps.weight=Q2_K through blk.39.ssm_beta.weight=Q2_K
```

VERIFIED: The override file at the path above is exactly the stock v3 map, identical
to the one used in the prior heretic attempt (confirmed via diff). No changes needed
— the heretic model has the same architecture as the stock Qwen 3.6 35B-A3B
(same 40 layers, same tensor naming, same MoE structure).

**CRITICAL CAUTION**: This target is the NON-MTP BF16 (69.4 GB), not the
MTP-preserved variant (67 GB with extra block blk.40). The prior June 2026 attempt
used the MTP-preserved source and hit a llama.cpp load failure. This target should
avoid that. Still verify the file is non-MTP before quantizing:

```bash
# Quick check: MTP-preserved builds have blk.40.* tensors
distrobox enter ai -- "$LLAMA_BIN/llama-gguf" --list "$SRC_BF16" 2>/dev/null | grep "blk\.40" || echo "NO BLK.40 — clean non-MTP"
```

---

### STEP 2: Quantize

```bash
OUT_GGUF="$WORKDIR/Qwen3.6-35B-A3B-Heretic-Cerebellum-v1.gguf"

# CUDA required — use distrobox
distrobox enter ai -- "$LLAMA_BIN/llama-quantize" \
  --imatrix "$IMATRIX" \
  --tensor-type-file "$OVERRIDES" \
  "$SRC_BF16" \
  "$OUT_GGUF" \
  Q3_K_M \
  2>&1 | tee "$WORKDIR/quantize.log"
```

Notes:
- No `--allow-requantize` needed: source is BF16.
- Tensor map aliases: The override file uses `Q2_K` format (uppercase). The current
  llama-quantize binary at this path accepts this format for override files (verified
  in prior 35B builds). If it rejects, normalize with:
  `sed -i 's/=Q2_K$/=q2_K/' "$OVERRIDES"` (but make a backup first, or use a copy).
- Expected runtime: 2–4 hours on CPU (no GPU needed for quantize).
- Expected output size: ~11 GB.

**Disk at this point: ~71 GB source + ~11 GB output = ~82 GB total**

---

### STEP 3: PPL sanity check

```bash
WIKI_CORPUS="/var/home/deucebucket/games/osmosis-quants/wiki.test.raw"

distrobox enter ai -- "$LLAMA_BIN/llama-perplexity" \
  --model "$OUT_GGUF" \
  --ctx-size 2048 \
  -f "$WIKI_CORPUS" \
  -ngl 99 \
  --chunks 32 \
  2>&1 | tee "$WORKDIR/ppl.log"
```

Expected PPL: 7.4–7.9 (stock v3 was 7.4307 with Q3_K_M baseline 7.1758; heretic
model has same architecture but slightly modified weights, expect similar range).

---

### STEP 4: Launch server for benchmarks

```bash
distrobox enter ai -- "$LLAMA_BIN/llama-server" \
  --model "$OUT_GGUF" \
  --mmproj "$SRC_MMPROJ" \
  -ngl 99 \
  --ctx-size 24576 \
  --parallel 4 \
  --reasoning off \
  --reasoning-budget 0 \
  --port 8095 \
  2>&1 | tee "$WORKDIR/server.log" &
```

Wait for: `curl -s http://localhost:8095/health | grep ok`

---

### STEP 5: Run benchmarks

```bash
BENCH_DIR="$WORKDIR/benchmark_results"
mkdir -p "$BENCH_DIR"

# Run all four (sequential for evalplus, parallel 4 for others — run_benchmarks.sh handles this)
cd /var/home/deucebucket/ai-drive/cerebellum
bash "$SCRIPTS/run_benchmarks.sh" 8095 qwen36-35b-heretic-cerebellum-v1 "$BENCH_DIR"
```

For HumanEval+, note that this is Qwen3 (not Gemma4) — use raw completions not chat:
```bash
# Standard evalplus (non-chat, fine for Qwen):
BENCH_PORT=8095 BENCH_MODEL=qwen36-35b-heretic-cerebellum-v1 RESULTS_DIR="$BENCH_DIR" \
BENCH_WORKERS=1 python "$SCRIPTS/benchmark_evalplus.py"
```

After benchmarks, audit EvalPlus completions:
```bash
python "$SCRIPTS/audit_evalplus_completions.py" \
  --results "$BENCH_DIR/qwen36-35b-heretic-cerebellum-v1_evalplus_samples.jsonl"
```

---

### STEP 6: Gate comparison

Compare against stock v3 (PROVEN targets) and assess:

| Benchmark | Stock v3 | Prior heretic v1 | Target (this build) |
|---|---:|---:|---|
| ARC-Challenge | 95.82% | 91.60% | >= 92% recommended |
| HellaSwag | 92.28% | 77.40% | >= 85% recommended |
| MMLU-Redux | 75.00% | 63.10% | >= 68% recommended |
| HumanEval base | 70.73% | 43.30% | >= 55% recommended |
| HumanEval+ | 65.24% | 40.85% | >= 50% recommended |

The prior heretic attempt (June 3) used `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF`
and showed severe regression. The current target (`llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF`)
is the simpler BF16 without MTP and SHOULD score closer to stock v3. The MTP-preserved
variant adds extra layer blk.40 that changes routing; the plain heretic should not.

If results are near stock v3, flag success. If there's still significant regression vs v3,
check if heretic ablation method causes architecture-level damage to Qwen's SSM or routing
(the heretic ablates o_proj/out_proj weights).

---

### STEP 7: mmproj handling

The mmproj from the heretic source is 861 MB BF16:
```bash
SRC_MMPROJ="$WORKDIR/source/Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF16.gguf"
```

This mmproj is architecture-shared with the stock Qwen 3.6 35B — the heretic ablation
targets attention output projections in the LM backbone, not the vision encoder. The
stock mmproj from the GGUF repo should work interchangeably if the heretic mmproj is
missing. However: use the heretic mmproj if available, as the prior build at
`/var/home/deucebucket/games/cerebellum-heretic-qwen36-35b/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf`
(861 MB) appears to be from the heretic repo and is locally present (reusable if it matches).

---

## Q3: v6 vs v6.1 — what changed

For Gemma 4 26B (not Qwen 35B, but for reference):

- v6: The winning tensor allocation. 91 overrides. Override file: `cerebellum_v6_overrides.txt`.
  Router layer 8 promoted to Q8_0 via tensor surgery.
- v6.1: KEPT the exact same tensor allocation as v6. Only change was template/runtime metadata
  fix (GGUF metadata for Gemma 4 chat-template compatibility). No tensors changed.

The v6 override file: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/cerebellum_v6_overrides.txt`
is verified to exist (2636 bytes, 91 lines).

The v6.1 GGUF: `/var/home/deucebucket/games/models/Gemma-4-26B-A4B-Cerebellum-v6/gemma-4-26B-A4B-it-cerebellum-v6.1-templatefix.gguf`
(path from OG_RECONSTRUCTION guide — not verified to still exist locally).

---

## Q4: Published benchmark targets

### Qwen 3.6 35B-A3B Cerebellum v3 (stock — our proven release):

| Benchmark | v3 Cerebellum | Q3_K_M baseline |
|---|---:|---:|
| ARC-Challenge | 95.82% | 96.10% |
| HellaSwag | 92.28% | 91.50% |
| MMLU-Redux | 75.00% | 74.12% |
| HumanEval base | 70.73% | 64.02% |
| HumanEval+ | 65.24% | 56.71% |
| Vision smoke (36 images) | 100% | — |
| Wiki PPL | 7.43 (vs baseline 7.18) | 7.18 |
| Size | 11 GB / 2.76 BPW | 15.6 GB |

Source: verified locally from `/var/home/deucebucket/games/qwen36-35b-v2/benchmark_results_v3/*.json`

### Gemma 4 26B-A4B Heretic Cerebellum v1 (published HF):

| Benchmark | Heretic v1 | Regular v6 | Regular Q3_K_M |
|---|---:|---:|---:|
| ARC-Challenge | 95.48% | 95.56% | 95.22% |
| HellaSwag | 83.49% | 84.55% | 86.57% |
| MMLU-Redux | 71.42% | 71.33% | 73.67% |
| HumanEval base (chat) | 92.07% | — | 62.20% |
| HumanEval+ (chat) | 89.63% | — | — |

Source: HF model cards fetched from public READMEs.

---

## Verified Reusable Artifact Paths

| Artifact | Path | Status |
|---|---|---|
| v3 override file (360 entries) | `/var/home/deucebucket/games/qwen36-35b-v2/cerebellum_v3_overrides.txt` | VERIFIED (11230 B, 360 lines) |
| Unsloth imatrix (primary) | `/var/home/deucebucket/games/qwen36-35b-v2/imatrix_unsloth.gguf_file` | VERIFIED (192 MB, 1020 tensors) |
| Local imatrix (backup) | `/var/home/deucebucket/games/qwen36-35b-v2/cerebellum_imatrix.dat` | VERIFIED (108 MB, 470 entries) |
| Gemma v6 overrides (reference) | `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/cerebellum_v6_overrides.txt` | VERIFIED (2636 B, 91 lines) |
| Gemma bartowski imatrix | `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/google_gemma-4-26B-A4B-it-imatrix.gguf` | VERIFIED (56.9 MB) |
| llama-quantize bin | `/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize` | in distrobox ai |
| llama-perplexity bin | `/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-perplexity` | in distrobox ai |
| llama-server bin | `/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-server` | in distrobox ai |
| Wiki PPL corpus | `/var/home/deucebucket/games/osmosis-quants/wiki.test.raw` | from prior runs |
| Prior heretic mmproj | `/var/home/deucebucket/games/cerebellum-heretic-qwen36-35b/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf` | VERIFIED (861 MB) — may be reusable |
| Stock v3 Qwen GGUF (done) | `/var/home/deucebucket/games/qwen36-35b-v2/Qwen3.6-35B-A3B-Cerebellum-v3.gguf` | VERIFIED (11.1 GB) |
| Benchmark scripts | `/var/home/deucebucket/ai-drive/cerebellum/scripts/run_benchmarks.sh` | VERIFIED |

---

## Gaps / Blockers

1. **BF16 GGUF not yet downloaded.** The target `Qwen3.6-35B-A3B-uncensored-heretic-BF16.gguf`
   (69.4 GB) is in the HF cache lock directory but not confirmed complete. Check:
   ```bash
   ls /var/home/deucebucket/.cache/huggingface/hub/models--llmfan46--Qwen3.6-35B-A3B-uncensored-heretic-GGUF/
   ```
   If only `.lock` files, the download needs to be initiated with `huggingface-cli download`.

2. **Disk space: 192 GB free.** Source (69.4 GB) + output (~11 GB) = ~80.4 GB. Fine.
   But note the prior heretic GGUF still on disk at 13 GB:
   `/var/home/deucebucket/games/cerebellum-heretic-qwen36-35b/Qwen3.6-35B-A3B-Heretic-Cerebellum.gguf`
   That was from a DIFFERENT heretic source (MTP-preserved). May be safe to delete after
   confirming the new run succeeds.

3. **llama-quantize tensor-type-file alias check.** Prior smoke test (QWEN3_0_6B doc)
   found the binary rejects `Q4_K_M` per-tensor but accepts `Q2_K`. The v3 override
   file uses `Q2_K` — this is correct and should work without modification.

4. **MTP check.** Before quantizing, confirm the source BF16 has no blk.40:
   the non-MTP HF listing shows this is a single-BF16 file (69.4 GB) vs the MTP-preserved
   (which the prior attempt used). The new target repo shows BF16 at 69.4 GB and
   MTP-preserved is a different file not in this repo.

5. **Prior heretic attempt (June 3) showed regression.** That build used a different
   source (Native-MTP-Preserved), not the plain heretic. Expect this build to score
   closer to stock v3, but cannot guarantee until benchmarked.

---

## Disk Budget Summary

| Phase | Item | Size |
|---|---|---|
| Step 0 | BF16 source GGUF | 69.4 GB |
| Step 0 | mmproj BF16 | 0.9 GB |
| Step 2 | Output Cerebellum GGUF | ~11 GB |
| — | Available on /games | 192 GB |
| — | Peak total usage | ~81 GB |
| — | After deleting source | ~12 GB |

Source can be deleted after output is verified and benchmarked.

---

## Server Invocation for Benchmarks

```bash
# Note: 24576 ctx is required minimum (6144 per slot × 4 = 24576)
distrobox enter ai -- /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-server \
  --model /var/home/deucebucket/games/cerebellum-heretic-qwen36-35b-v2/Qwen3.6-35B-A3B-Heretic-Cerebellum-v1.gguf \
  --mmproj /var/home/deucebucket/games/cerebellum-heretic-qwen36-35b-v2/source/Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF16.gguf \
  -ngl 99 \
  --ctx-size 24576 \
  --parallel 4 \
  --reasoning off \
  --reasoning-budget 0 \
  --port 8095
```

Qwen 3.6 35B uses raw completion benchmarks (NOT Gemma4 chat harness):
- EvalPlus: `BENCH_WORKERS=1` (sequential, script handles this)
- ARC/HellaSwag/MMLU: `BENCH_WORKERS=4`
