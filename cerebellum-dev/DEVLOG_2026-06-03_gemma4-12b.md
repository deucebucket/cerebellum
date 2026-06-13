# Devlog: June 3-4, 2026 — Gemma 4 12B Cerebellum

## Breakthroughs

### World's First Gemma 4 12B GGUF
- One-line patch to llama.cpp: register `Gemma4UnifiedForConditionalGeneration` under `Gemma4Model`
- The unified model wraps text backbone at `model.language_model.*` — Gemma4Model.modify_tensors already strips this
- F16 GGUF: 23.8GB, 667 tensors, 13 tok/s, coherent ("Paris")
- Q4_K_M: 8.0GB, 54 tok/s, coherent ("It appears...")

### Quant Level Floor Discovered
- **Q3_K_M BROKEN** — outputs "a a a a a" (garbage repetition)
- **Q4_K_M WORKS** — coherent text, 2583 PPL
- This is a hard floor for the 12B dense model
- F16→Q8→Q6→Q5→Q4 all coherent; Q3 and below collapse

### Cerebellum Hill-Stepper Proven Better Than Forward Ablation

**Forward ablation:**
- Only tests Q4→Q2 (one direction, one data point)
- Cannot find sweet spots (can't tell if Q4 is optimal or Q3/Q5 is better)
- 18 hours for 200 weight tensors

**Hill-Stepper (bidirectional):**
- Start at Q4 baseline, test q3→q2 down AND q5→q6→f16 up
- Finds TRUE per-tensor sweet spot, not just floor
- Proved layer-dependence: blk.0 vs blk.1 differ for same tensor type
- ABS_BEST tracking with tiebreaker to lower precision
- Timestamps for real timing data

### Bugs Found & Fixed (in hillclimber)
1. **Break on rejection** — q3 rejection exited loop, skipping q2/q5/q6/f16 tests. Fixed: don't break, continue all levels
2. **Δ=0 tiebreaker** — equal PPL should pick LOWER precision. Fixed: precision rank comparison
3. **Drift** — 2% window drifted BEST_PPL upward. Fixed: ABS_BEST tracked separately
4. **Missing post-f16 fallback** — all levels accepted → locked f16 without checking ABS_BEST. Fixed: post-loop fallback
5. **Q3 target in quantize** — hillclimb was quantizing to Q3_K_M target instead of Q4_K_M baseline

### Infrastructure Built
- **Cerebellum Knowledge DB**: 177 benchmarks, 261 ablation results, 11 models, 8 arch gotchas
- **Router**: 30 models, watchdog fixed (no more auto-unload), default=Qwen 3.6 35B v3
- **Tray app**: model switching, VRAM display, router control
- **llama-ctl**: model scanner with family-based auto-config, DB query

### Dead Files Cleared
- 65GB qwen35-35b-a3b-heretic-f16.gguf
- 13GB gemma4-26b-codex-cerebellum-v6-transfer-requant.gguf

## Mistakes / Lessons

### Conversion Hell
- Old broken F16 GGUF was cached → Q3_K tests used wrong source for hours
- Gemma4Unified registration wasn't sticky (git checkout reverted it)
- modify_tensors skip logic was wrong for text-only extraction
- GGUFWriter crashes on Python 3.14 (gguf-py library incompatible)

### Container Corruption
- `distrobox enter ai -- kill -9 -1` corrupted the container
- Had to fully recreate: `distrobox rm ai && distrobox create --nvidia`
- Lost CUDA passthrough on recreate, needed `--nvidia` flag

### Process Management
- Stale llama-quantize/perplexity processes eating GPU (zombies)
- pkill from host can't reach processes inside distrobox
- Must use `distrobox enter ai -- pkill` for container processes
- Always verify nvidia-smi shows only expected processes

### Timing Gaps
- No timestamps in original hillclimb → couldn't tell if quantize was stuck (10+ min) or just slow (2 min)
- Fixed: $(date) on every major echo

## To Remember
1. Q4_K_M is the minimum viable quant for Gemma 4 12B dense
2. Hill-Stepper > Forward Ablation for per-tensor optimization
3. Always verify GPU is clean before starting any llama workload
4. Timestamps on EVERY log line
5. ABS_BEST ≠ last accepted level
6. Two 8GB models fit on 24GB card with KV cache room
