# Cerebellum Knowledge Index — START HERE

Single entrypoint for any agent or session. Read this file first, every session.
If a doc here conflicts with an older doc anywhere else in the tree, THIS canon wins.

## The canon (read in this order)

| File | What it answers |
|---|---|
| [CURRENT_METHOD.md](CURRENT_METHOD.md) | "How do we build a Cerebellum quant?" The OG bench-gated formula. What is deprecated. |
| [REAL_PIPELINE_RECONSTRUCTED.md](REAL_PIPELINE_RECONSTRUCTED.md) | The exact end-to-end method from primary artifacts, incl. the **coding ablation** phase the guides lost. Runnable: `scripts/coding_ablation.py`. Read alongside CURRENT_METHOD step 6. |
| [CRITICAL_LOST_STEP_2026-06-13.md](CRITICAL_LOST_STEP_2026-06-13.md) | The mandate: no "no-ship" verdict without the real coding ablation (HumanEval per group/layer, not multi-domain PPL). |
| [PROJECT_HIERARCHY.md](PROJECT_HIERARCHY.md) | "What projects exist, where do they live, what data do they share?" |
| [ARCHITECTURE_MAP.md](ARCHITECTURE_MAP.md) | "Where is everything on disk?" Full folder map, classifications, reorg plan, delete candidates. |

## Deeper truth (one hop away, in `cerebellum-dev/`)

- `../WINNING_METHOD.md` — full canonical formula with per-model instantiations (26B v6, 27B v4, 35B v3) and exact build commands.
- `../DEAD_PATHS.md` — every abandoned approach and why. Check before proposing "new" ideas.
- `../BACKLOG.md` — what's queued.
- `../README.md` — factory index for current experiments.
- `../ARTIFACT_INVENTORY.md` — preservation-first inventory of evidence artifacts.
- `../RESEARCH_IDEAS.md` — already-specced future work (read before pitching research).
- STORY doc — *(slot reserved; being written by another agent — will live at `knowledge/STORY.md` when delivered)*.

## Hard rules (non-negotiable)

1. **PPL never gates alone.** Benchmark gates (ARC, HellaSwag, MMLU-Redux, HumanEval+, BigCodeBench) vs same-size uniform baseline, every build. See CURRENT_METHOD.md.
2. **hillstep exhaustive mode is deprecated.** Do not resume hillstep campaigns; do not cite hillstep results as method.
3. **Evidence dirs are preservation-locked.** Never delete/move `osmosis-*/`, `cerebellum-*/` per-model dirs, or `benchmarks/` without a verified-backup plan.
4. **Two remotes:** `origin` = public, `dev` = private. `cerebellum-dev/` never goes to origin. When unsure, default to dev.
5. **No AI attribution in anything public.** No Co-Authored-By trailers on public-bound commits, no AI-tell prose in cards/READMEs.
6. **Audit wrong answers before recording any score.**
