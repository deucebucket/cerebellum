# DRAFT — for Jerry's review, do not post.
#
# Target: github.com/noonghunna/club-3090 issues (scoping issue, per CONTRIBUTING
# "New models: open an issue first to scope" + "one model per PR").
# Evidence dir: cerebellum-club3090/results/ (run of 2026-06-12 on the real 3090).
# Open items before the eventual PR are listed at the bottom of the issue body itself.

---

**Title:** Scoping: Qwen3.6-35B-A3B on mainline llama.cpp, single 3090 (the llama.cpp ❌ cell)

**Body:**

The 35B-A3B row currently reads `llama.cpp ❌`. I have a single-card mainline-llama.cpp config for it that I'd like to land as the third engine path for that model, next to the ik_llama fit-mtp recipe from #243. Opening this to scope before any PR. One model only.

The model artifact is an 11.96 GB mixed-precision GGUF of Qwen3.6-35B-A3B I build and publish on HF (per-tensor quant levels chosen from ablation measurements, runs on stock llama.cpp, no fork). Whole model + KV fits one 3090 with room to spare, and it reaches into 16 GB cards too.

Numbers from today on a single RTX 3090, host llama.cpp build (no Docker on this box, podman only, so everything ran via your `URL=... CONTAINER=none` host flow from discussion #88):

Serve: `llama-server -ngl 99 --parallel 4 -c 24576 --jinja`

- `bench.sh` (3 warm + 5 measured): narrative wall 147.9 / decode 150.5 TPS (CV 0.7%), code wall 146.2 / decode 149.9 TPS, TTFT ~112 ms, 14151 MiB VRAM under load.
- `verify-full.sh`: PASS (rc=0). Genesis and MTP checks skipped as vLLM-only, engine auto-detect worked as documented.
- `verify-stress.sh`: rc=1, but the one real failure was a sizing mismatch, not a model failure: the ~25K-token tool-prefill probe got a clean HTTP 400 from a 6144-token slot (24576 ctx / 4 slots). Probes 3 through 6 all passed; the long-needle rungs skipped themselves as above deployed ctx. Needs a rerun against the long-context launch below.
- Context ceiling, `--parallel 1` with KV q8_0: `-c 131072` peaked at 14981 MiB (79K-token fill completed clean), `-c 196608` peaked at 15869 MiB (119K-token fill completed clean). So 196K ctx fits with ~8 GB VRAM headroom on a 24 GB card.
- `benchlocal-cli --medium` (0.9.6): 63/75 overall. toolcall 13/15, instructfollow 15/15, structoutput 14/15, dataextract 9/15, reasonmath 12/15. The sandboxed packs (bugfind / hermes / cli-40) need Docker and were skipped on this host, so no direct comparison yet against the agentic-pack numbers in the ik_llama row.

Not done yet, will be in the PR: `verify-stress.sh` rerun at the long-ctx config, `SOAK_MODE=continuous` (I know it's required for single-card variants), `PP=1 bench.sh` for prompt-processing throughput, the Docker-sandboxed benchlocal packs, and `report.sh --full`. Today's run was on a local llama.cpp build; the PR will pin a tagged upstream release and re-verify against it.

Two scoping questions before I spend the afternoon on it:

1. Does a mainline-llama.cpp single-card path belong in that cell, or do you consider ik_llama to already own "llama.cpp-family single card" for this model?
2. Is an externally published mixed-precision GGUF acceptable as the model artifact, same way the 27B path builds on the Lorbus AutoRound quant? I'd document the recipe and link the build, with the full gate evidence attached to the PR.

If yes to both, I'll run the complete gate list and open one PR for this one model.
