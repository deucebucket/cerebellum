# Carl runtime notes — Gemma 4 26B-A4B Cerebellum v6

Source of truth for how this GGUF should be served when Carl uses it as a
local fallback / idle-tier model. The benchmark protocol in
`cerebellum/CLAUDE.md` is for offline measurement; this file is for the
live server Carl talks to.

When in doubt, reach for this file before re-litigating flags in chat.

## TL;DR — copy-paste invocation

```bash
LD_LIBRARY_PATH=/tmp/cuda-toolkit/lib64:/tmp/cuda-toolkit/lib \
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-server \
  --model /var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/gemma-4-26B-A4B-it-cerebellum-v6.gguf \
  --host 127.0.0.1 --port 7801 \
  --alias gemma4-26b-cb \
  --n-gpu-layers 999 \
  --ctx-size 131072 \
  --parallel 2 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --mlock \
  --jinja \
  --reasoning off \
  --reasoning-budget 0 \
  --prio 2 \
  --no-warmup
```

## Flag rationale (don't drop these without reason)

### Required for correctness

| Flag | Why |
|------|-----|
| `--jinja` | Default in newer builds, but pin it. Forces use of the GGUF's embedded jinja chat template instead of llama-server's hardcoded fallback. The fallback was wrong for Gemma 4 in builds before ~2026-05 and would cause **looping output** because the model emitted stop tokens the parser didn't recognize. |
| `--reasoning off` | Gemma 4 is **not** a reasoning model. Leaving reasoning on `auto` lets the parser try to extract `<think>` blocks that don't exist, which corrupts the output stream. |
| `--reasoning-budget 0` | Belt-and-suspenders: even with `--reasoning off`, set the budget to 0 explicitly so no reasoning logit bias gets applied. There was a buggy +inf logit bias commit in llama.cpp around early May 2026 (reverted in `f9cd456ea`); rebuilding past that commit + budget=0 closes the loophole. |

### Speed features (the "turbo cache" stack)

| Flag | What it does |
|------|--------------|
| `--flash-attn on` | Flash attention kernels. Faster + lower VRAM. |
| `--cache-type-k q8_0 --cache-type-v q8_0` | Quantize the KV cache to 8-bit. Big VRAM win at long context, no measurable quality loss for chat. |
| `--mlock` | Pin the model in RAM, no swap. Removes paging stalls during long sessions. |
| `--prio 2` | High process priority (medium-high). Keeps the server responsive when other workloads compete for CPU. |
| `--no-warmup` | Skip the warmup pass. Faster cold start, no quality impact. |
| `--parallel 2` | Two server slots. One for Carl's main reply path, one for background_inference (rumination, distiller drafts) so they don't queue behind each other. |

### Memory / context budget

| Flag | Reason |
|------|--------|
| `--n-gpu-layers 999` | Offload everything to the 3090. Model is 11.7 GB; KV cache @ q8 + 128K context fits in ~16 GB on a 24 GB card. |
| `--ctx-size 131072` | 128K total = 64K per slot with `--parallel 2`. Gemma 4 SWA caps most layers at 1024 tokens, so KV cost grows much slower than ctx-size implies — quadrupling ctx from 32K → 128K only added ~700 MiB. The user complained about cloud fallback on a 16K-overflow at 32K total; 128K total kills that for any plausible chat. |

### Flags **not** to add (intentional omissions)

- `--swa-full` — sliding-window attention default is "compact cache" which is faster per token. Full-cache mode trades speed for tiny accuracy. Skip for serving.
- `--cache-reuse N` — Gemma 4 uses sliding-window attention; KV cache reuse via shifting requires portable positional encodings, which SWA breaks. The server logs `cache_reuse is not supported by this context, it will be disabled` at startup if you set it. No-op for this model — leave unset to keep the log clean.
- `--no-mmap` — slower load. mmap is default and right.
- `--ctx-checkpoints / -ctxcp` — for swappable session contexts. Carl uses one rolling user session post-#354, so checkpoints don't help.

## llama.cpp version requirement

Build must be at or after commit `f9cd456ea` (`common : revert reasoning
budget +inf logit bias (#22740)`). On older builds, the +inf logit bias
breaks sampling and produces visible looping in the output even with
`--reasoning off` and `--reasoning-budget 0`.

Quick check:
```bash
cd /var/home/deucebucket/ai-drive/llama.cpp && git log --oneline | grep -m1 f9cd456e
```
If no output: pull + rebuild before serving.

Build:
```bash
# Pull from the host; build inside the distrobox where the CUDA dev
# toolkit lives. The host shell only has CUDA runtime libs at
# /tmp/cuda-toolkit — no working nvcc — so a host-side rebuild fails
# with "CUDA Toolkit not found". The CMakeCache also hardcodes
# /usr/bin/cmake which only resolves inside the container.
cd /var/home/deucebucket/ai-drive/llama.cpp
git pull --ff-only origin master
distrobox enter ai -- bash -c '
  cd /var/home/deucebucket/ai-drive/llama.cpp &&
  /usr/bin/cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=86 &&
  /usr/bin/cmake --build build --target ggml-cuda llama-server -j$(nproc)
'
```

**Critical: `-DCMAKE_CUDA_ARCHITECTURES=86`** — Carl's GPU is an RTX 3090
(Ampere, sm_86). If you build for sm_89 (Ada / 4090) you get
`CUDA error: no kernel image is available for execution on the device`
when the model loads. If the host GPU changes, look it up:
- 3090 / 3080 / A100 → `86` (Ampere consumer / `80` for A100)
- 4090 / 4080 → `89`
- 5090 → `120`
You can pass a list (`"86;89"`) to build a fat binary, but the build
takes proportionally longer.

**`-DGGML_CUDA=ON` matters** — `cmake -B build` without it (or with the
old CMakeCache holdover) leaves CUDA off and only rebuilds the CPU
backend. Symptom: `libggml-base.so.0` and `libggml-cpu.so.0` get bumped
to a new version but `libggml-cuda.so.0` stays at the old one. Loading
the model then logs `warning: no usable GPU found` and runs on CPU. Fix
by re-running the cmake configure line above.

The `ai` distrobox image is `nvidia/cuda:12.6.3-devel-ubuntu22.04` —
nvcc 12.6 + cmake 3.22 already installed. If that container is gone:

```bash
distrobox create --name ai --image nvidia/cuda:12.6.3-devel-ubuntu22.04
distrobox enter ai -- sudo apt update && sudo apt install -y cmake build-essential
```

## Where this is wired into Carl

`/var/home/deucebucket/ai-drive/carl/config/local_models.yml` —
`models[].extra_args` for the `gemma4-26b-cb` entry. Carl's model
orchestrator spawns the server with these args when it picks the local
tier. Update both the YAML and this file when changing flags so they
don't drift.

## Symptoms → diagnosis

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Output loops on the same line / phrase | Stale llama.cpp build (pre-`f9cd456ea`) OR reasoning still on | Rebuild llama.cpp + confirm `--reasoning off --reasoning-budget 0` in the running cmd |
| Looping with garbled tokens (`<end_of_turn>` leaks, etc.) | Wrong chat template — `--jinja` not active | Add `--jinja`, confirm via `--help` that it's enabled by default in your build |
| First token of every turn slow | Full prefill on every turn (Gemma 4 SWA can't reuse KV via shifting) | This is architectural — accept it. Larger `--ubatch-size` helps prefill throughput but won't avoid the prefill itself. |
| OOM on long context | KV cache not quantized, or `--swa-full` enabled | Confirm `--cache-type-k/v q8_0`, drop `--swa-full` if present |
| Huge cold-start latency | mmap disabled OR mlock missing on large model | Default mmap + add `--mlock` |
