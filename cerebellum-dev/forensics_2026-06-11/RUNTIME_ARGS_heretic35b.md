# Runtime Args — Qwen3.6 35B-A3B Heretic Cerebellum Benchmark Run

Compiled: 2026-06-11
Sources: official Qwen3.6-35B-A3B README, llmfan46 heretic GGUF README,
deucebucket/Qwen3.6-35B-A3B-Cerebellum-GGUF README (v1 card),
local v3 benchmark meta + results (port 7890), gate2 launch scripts,
run_benchmarks.sh, benchmark_vision_smoke.py, benchmark_realworldqa.py,
docs/benchmark_protocol.md.

---

## 1. llama-server command — TEXT-ONLY benchmark suite

```bash
llama-server \
  -m /path/to/Qwen3.6-35B-A3B-Cerebellum-heretic-vN.gguf \
  -ngl 99 \
  --parallel 4 \
  -c 24576 \
  --port 7890 \
  --jinja
```

Notes:
- `--jinja` is REQUIRED for Qwen3.6. The `enable_thinking` chat_template_kwargs
  flag (used by all bench scripts and benchmark_vision_smoke.py) only works when
  the Jinja template path is active. Without --jinja the flag is silently ignored
  and the model thinks by default on every call.
- `-c 24576` is the project-wide minimum (6144 per slot × 4 slots). Covers
  HumanEval+ 4096 max_tokens with headroom and is enough for ARC/HellaSwag/MMLU.
- `--parallel 4` supports the 4-worker ARC/HellaSwag/MMLU runs. HumanEval+ runs
  at BENCH_WORKERS=1 (sequential) and just uses slot 0 in order.
- No `--reasoning-budget` or `--reasoning` flag needed on the server side;
  thinking is toggled per-request via `chat_template_kwargs`.
- Port 7890 matches the v3 meta (`benchmark_results_v3/qwen36-35b-cerebellum-v3_meta.json`).
  Port 8095 is used by the granite gate2 script; pick whichever is free.
- No `--mmproj` for the text-only suite — adding it wastes ~858 MB VRAM on
  every launch; keep vision server separate.

Per-bench BENCH_WORKERS (from run_benchmarks.sh):
  evalplus/humaneval  → BENCH_WORKERS=1
  arc                 → BENCH_WORKERS=4
  hellaswag           → BENCH_WORKERS=4
  mmlu                → BENCH_WORKERS=4

---

## 2. llama-server command — VISION smoke + RealWorldQA

```bash
llama-server \
  -m /path/to/Qwen3.6-35B-A3B-Cerebellum-heretic-vN.gguf \
  --mmproj /var/home/deucebucket/games/qwen36-35b-v2/vision/mmproj-F16.gguf \
  -ngl 99 \
  --parallel 4 \
  -c 24576 \
  --port 7821 \
  --jinja
```

mmproj path:
  `/var/home/deucebucket/games/qwen36-35b-v2/vision/mmproj-F16.gguf` (858 MB, F16)
  This is the same projector used for all prior v2/v3 vision runs and is also
  listed in the deucebucket HF card as `mmproj-F16.gguf`. The heretic GGUF card
  ships `Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF1.gguf` (BF16); either is
  fine but the local F16 copy is already on disk — use it.

### Vision smoke script

```bash
BENCH_PORT=7821 \
BENCH_MODEL=qwen36-heretic-cerebellum-vN \
RESULTS_DIR=/path/to/benchmark_results_heretic_vN/vision_test \
python scripts/benchmark_vision_smoke.py
```

- Generates 24 deterministic images (seed 36635): 4 per category × 6 categories
  (ocr, counting, spatial, chart, table, ui).
- Temperature 0, max_tokens 48, `chat_template_kwargs: {"enable_thinking": False}`.
- Pass bar: **24/24 = 100%**. All prior Cerebellum builds (v2, v3, heretic-v3)
  hit this. Anything below 24 is a regression and must be investigated.

### RealWorldQA script

```bash
BENCH_PORT=7821 \
BENCH_MODEL=qwen36-heretic-cerebellum-vN \
RESULTS_DIR=/path/to/benchmark_results_heretic_vN/vision_test \
python scripts/benchmark_realworldqa.py 50
```

- Draws n=50 samples (seed 36635) from xai-org/RealworldQA test split.
- Temperature 0, max_tokens 48. No `enable_thinking` override in
  benchmark_realworldqa.py (sends plain completions request).
- Expected pass bar: **~78%** (39/50). This is the established prior-run
  baseline from heretic v3 and v3 runs. Note: no explicit v3 RealWorldQA
  result artifact was found locally — the 78% bar comes from the deucebucket
  HF card prose. Treat it as a soft lower bound; flag anything below ~72%.

---

## 3. Recommended user-facing sampling params (model card table)

Source: official Qwen3.6-35B-A3B README + generation_config.json.
The heretic GGUF card contains no llama.cpp invocation or sampling guidance;
it defers entirely to the base model documentation.

| Mode | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty | notes |
|------|-------------|-------|-------|-------|------------------|--------------------|-------|
| Thinking — general | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 1.0 | default thinking mode |
| Thinking — precise coding (WebDev) | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 | lower temp for determinism |
| Non-thinking (instruct) | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 | set enable_thinking: false |
| Benchmark (deterministic) | 0 | — | — | — | — | — | all bench scripts use temp=0 |

generation_config.json (canonical file on HF):
  `do_sample: true, temperature: 1.0, top_k: 20, top_p: 0.95`
  (thinking mode defaults; non-thinking overrides not in config file, only in README prose)

Key Qwen3.6 differences vs Qwen3.5:
- `/think` and `/nothink` soft-switch tokens are NOT supported. Must use
  `chat_template_kwargs: {"enable_thinking": false}` at the API level.
- Thinking mode is ON by default; non-thinking requires explicit flag.
- presence_penalty=1.5 is recommended to suppress repetition loops in
  thinking mode; can be reduced to 0 for precise coding where determinism
  is more important than loop suppression.

---

## 4. Contradictions and flags

1. **`--jinja` not mentioned in prior gate2 launch scripts (run_gate2_granite30b.sh,
   run_gate2_9b.sh).** Those models (Granite 4.1 30B, Qwen 3.5 9B) are not
   Qwen3.6 and do not use `enable_thinking`. The v3 cerebellum server was
   launched manually (port 7890, no recorded shell script) so no explicit
   `--jinja` flag is archived — but the benchmark_vision_smoke.py script
   sends `enable_thinking: False` and the vision results are 24/24, which
   implies `--jinja` was active (the flag would have been silently ignored
   without Jinja, causing the model to emit `<think>` tokens that overflow
   max_tokens=48 and produce wrong answers). Treat `--jinja` as REQUIRED.

2. **RealWorldQA ~78% expected bar is from HF card prose, not a local artifact.**
   No `*realworldqa*` result file was found under `/var/home/deucebucket/games/qwen36-35b-v2/`.
   The 78% figure in the deucebucket card is unverified against a local JSONL.
   The heretic-v3 vision smoke hit 100% (24/24) which is confirmed by a local
   artifact, but RealWorldQA was not run for heretic-v3 locally. Use the 78%
   bar as a soft target; a fresh run on the heretic build will establish the
   true baseline.

3. **benchmark_realworldqa.py does NOT send `enable_thinking: False`.**
   (benchmark_vision_smoke.py does; realworldqa.py does not.) At temperature=0
   in llama.cpp, thinking mode will engage unless suppressed. If the model
   generates `<think>...</think>` preamble, the 48-token max_tokens will be
   exhausted on the thinking block, returning an empty content field. This is
   a latent bug. Either: (a) add `chat_template_kwargs: {"enable_thinking": False}`
   to the realworldqa.py query call, or (b) increase max_tokens to accommodate
   the thinking block. Check the first few completions manually before recording
   a score — if `raw_response` fields are empty or contain only `<think>` fragments,
   the bench is broken.

4. **Cerebellum v1 HF card shows `--ctx-size 8192`** (not 24576). The card was
   written for casual user deployment, not benchmark mode. The benchmark
   protocol and run_benchmarks.sh both require -c 24576. Do not use 8192 for
   benchmark runs — HumanEval+ needs 4096 max_tokens plus prompt headroom and
   will silently clip or fail at 8192.
