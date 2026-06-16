# Running jobs / handoff — 2026-06-16 ~05:50 CDT (Jerry exiting)

## ⚠️ KEY DECISION PENDING — v6 qkvq4 is NOT a clean v5 replacement
Full gate on `Qwen3.6-27B-Cerebellum-v6-qkvq4` (QKV@q4, 12.07 GiB):
- 8-pack: **105/150** (≈ v5; deterministic 92 vs v5 90; math best-of-all)
- serving: verify PASS, **31.6/32.1 TPS**, needle clean to **120K**, soak skipped (bare-server autodetect; v4 finding carries)
- games: snake/tetris/chess **hand-verified playable** (think-off), evidence staged
- **HumanEval+: 84.8% (AUDITED CLEAN — 0/164 give-ups, real number)** ← **but v5 = 89.0**, so q4 QKV costs ~4 coding pts vs q5 (matches 27B arch: attn_qkv is the coding-fragile pathway)
- PPL: NOT run (`llama.cpp/build` llama-perplexity has a stale libggml-cuda symbol error — needs a rebuilt perplexity bin)
- ARC/HS/MMLU: **NOT run** — owed via Colab upstream lm-eval (local harness understates; do NOT publish local benchmark_arc/hellaswag/mmlu_redux numbers)

**DECIDED (Jerry, 2026-06-16): NO v6 SHIP.** qkvq4's 84.8 HumanEval+ doesn't clear v5's 89.0 — not a coding-worthy replacement. **v5 stays the 27B.** qkvq4 held in PRIVATE staging (keep, don't delete — it's the q4-QKV coding datapoint, and a possible smaller-tier candidate later). Lesson logged: on 27B, **QKV must stay q5 for coding** (q4 costs ~4 HumanEval+ pts; q3 worse). qkvq3-vs-v4 (8-pack win) is moot for shipping since the coding axis is what matters here.

## v6 staging (HF) — PRIVATE, do NOT flip public yet
- `deucebucket/qwen36-27b-cerebellum-v6-staging` (PRIVATE, verified). Has `Qwen3.6-27B-Cerebellum-v6.gguf` (=qkvq4) + `benchmark_results/games/` (snake/tetris/chess html+txt+README).
- Flip public ONLY after ARC/HS/MMLU + Jerry green-lights the tradeoff. If dropped, delete the staging repo (nothing public saw it).

## cerebellum-brain (PRIVATE repo) — Piece 1 DONE + pushed
- Over-ingestion fixed: model table **168→4** (mmproj/vocab no longer stored as models), blank-arch=0, **7/7 tests**, junk→`ingest_report` (628 deferred, NOT dumped).
- Hybrid harvester: deterministic numbers + LLM prose-extraction. **LLM extraction not yet run — needs `localhost:7800` (DOWN) or `ds` fallback.** Then `brain ingest --all` classifies the 628.
- Wiki LIVE: `github.com/deucebucket/cerebellum-brain/wiki` (6 pages).
- NOT started: Piece 2 (auto-insight engine) · Piece 3 (dashboard/API + public github.io via fo76-data template, allow-list export ONLY — never dump the brain).

## ARCH_RESEARCH backfilled (NEW standing gate per memory)
- `data/{glm/glm-4.7-flash, qwen/qwen3.6-27b, qwen/qwen3.6-35b-a3b, north/north-mini-code-1.0}/ARCH_RESEARCH.md` + `club3090_findings.md`.
- Cross-model headline: **`ffn_down_exps` = sacred coding-anchor across MoE** (North −36.6 HE, GLM anchor); SSM/GDN hard-fail <4bit. North PR #24260 MERGED 06-13 (golden window; community quants pre-merge; cohere2moe→tiny_aya retag).

## Deferred / not done
- **GLM midbudget gate** — never ran (midbudget 13.4GB ungated). Gate on local 3090 or Colab A100; use GLM F32-accum serving flag (`-fa off`/`GGML_CUDA_FORCE_MMQ=1`); verify GLM GGUF built post-Jan2026 router scoring_func fix before trusting numbers.
- club-3090 #415: posted v4/v5 (96/105) are CORRECT (the 90/83 were crashed-hermes runs). No edit needed.

## Background procs surviving SSH exit (setsid)
- Games HTTP server `100.99.191.100:8088` (tailscale) still serving. Kill: `pkill -f "http.server 8088"`.

## NEXT (next session)
1. ARC/HS/MMLU on Colab Pro (A100) via upstream lm-eval, pull qkvq4 from staging → completes v6 gate.
2. Jerry's call on qkvq4-vs-v5 tradeoff → flip public or drop.
3. Bring up localhost:7800 → brain LLM extraction (classify the 628) → Piece 2/3.
4. Rebuild clean llama-perplexity → PPL sanity number.
