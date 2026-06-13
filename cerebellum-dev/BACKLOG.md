# Cerebellum Backlog

Canonical work queue. Agents and sessions: this file is the source of truth for
"what's next". Move items between sections, never delete history (strike or move
to Done with date). Community-sourced items link their thread.

Last consolidated: 2026-06-12 (from community sweep, campaign launches, research notes).

## NOW (running tonight)

- **North-Mini-Code-1.0 golden window (days-urgent)**: Cohere's first open code model, 30B-A3B MoE (our proven lane), trending top-5, llama.cpp PR #24260 approved 06-12 and near merge. ALL existing community quants are pre-merge and unvalidated below Q4. Play: first benchmark-gated sub-4-bit. Pre-positioning started: Q8 source downloading, PR branch building (llama.cpp-pr24260). On merge: re-verify conversion, ablate, gate, ship with evidence.

- Gemma 4 12B campaign: VERDICT 2026-06-12 — group-level door CLOSED (Jerry), model stays OPEN. v3 was degenerate (attn_v was the only demotable group; restoring it = uniform Q4_K_M byte-for-byte). Arch research explains why: `gemma4_unified` is encoder-free, K=V-shared, MQA-global, QAT-trained — no group-level redundancy pools (cerebellum-dev/GEMMA4_12B_ARCH_RESEARCH_2026-06-12.md). Sub-group paths parked in LATER.
- GLM-4.7-Flash campaign on Modal: convert -> baselines -> 10-group ablation -> verdicts + RUN_PLAN stop (~$9-12 credits)
- Pipeline triple-verification + public how-it-works explainer draft (agent, in flight)
- Two community reply drafts (TheodoreH, tima2431) awaiting Jerry's go

## NEXT (committed, ordered)

1. **club-3090 wedge (the no-selling community entry)**: fill their literal ❌ cell — mainline llama.cpp single-card Qwen3.6-35B-A3B recipe using our 11.96GB build (+16GB-card reach). Sequence: run THEIR measurement checklist on our 3090 against their pinned image (recon report: cerebellum-dev/knowledge/ + agent report 2026-06-12 — ctx sweeps, soak, bench.sh, benchlocal-cli 8-pack), ISSUE POSTED 2026-06-12: github.com/noonghunna/club-3090/issues/390 (numbers-from-your-rig template, soak PASS, 131k@15.1GB recipe config, compose offer). Awaiting maintainer response; expect intake-with-credit. CADENCE (Jerry 06-12): interest-driven — feed #390 with ONE consolidated improvement comment (agentic curve + NIAH ladder + pin-check + dataextract re-probe) when measurements land; the 122B issue (their catalog tops out at 40B — we'd add a weight class) fires after maintainer engagement or 2-3 quiet days, fleet data arms it overnight. One model per PR; 27B + Gemma follow after trust (fleet measurements running tonight feed them). Their benchlocal-cli is also the BYO-endpoint quality bench for our own use.
2. Qwen 3.6 35B v4: tensor-level budget pass (flagship never got it; drill per-tensor inside attn_qkv/ffn_down_exps/ffn_up_exps, allocator at 11.96GB cap; bench-gate; transfer winner to Heretic). Modal GPU work.
3. Size ladders: intermediate Q3/Q4-class budgets per released model (TheodoreH ask: fits 14GB RAM; the budget tool exists for exactly this) — https://huggingface.co/deucebucket/Qwen3.6-35B-A3B-Cerebellum-GGUF/discussions/3
4. Extreme-small line (XXS): IQ2-class budget candidates from existing maps; "smallest working quant of X" claim; Modal-cheap. STATUS 2026-06-12: Flash XXS STAGED — 3 candidates (IQ2_M ~8.8GB / IQ2_XS ~8.3GB / IQ2_XXS ~7.9GB floor probe), ~$1.15 est, $5 cap; gated launcher armed, fires when north v1 map lands (cerebellum-dev/modal_harness/cerebellum_flash_xxs.py). 12B XXS dead (no demotable groups). Coder XXS candidate rides along after its v1 map.
5. ~~GLM-5.1~~ RESOLVED BY DATA (2026-06-12): 744B/1.65TB is beyond means; requester's hardware class already served by ubergarm ik_llama + unsloth dynamic quants; landscape sweep confirms no window. GLM line continues via 4.7-Flash (campaign running) and optionally 4.5-Air. Dev issues #48/#52 updated with the finding.
6. MTP-weights-baked 27B (arbv, promised): llama.cpp MTP support landed; needs MTP-preserved source + recipe compat check — https://huggingface.co/deucebucket/Qwen3.6-27B-Cerebellum-GGUF/discussions/1
7. .eval_results YAML fleet-wide (free, self-hosted — the ecosystem-native eval surface; no Jobs spend involved).

## LATER (real, not yet scheduled)

- **Evidence backfill (found by Gemma vetting 06-12)**: the ship-with-evidence rule is violated by older repos — v6 26B has summary JSONs only, E2B v2 has NO benchmark_results dir. Backfill per-question JSONL + audit files from local evidence to every released repo, audit the other 12 repos for the same gap. Cheap, mechanical, and it makes every future "evidence is in the repo" claim true everywhere.

- **Gemma 4 12B below group granularity** (Jerry 2026-06-12: "leave Gemma 12 parked for now. I want the coder" — PARKED, coder takes priority): group-level is exhausted, but sub-group paths are untried — per-block attn_v (demote only blocks whose code-domain delta was negative), attn_v at Q3_K instead of Q2_K, and row-block/shard analysis once that tooling matures (the memory-controller endgame). The multi-domain ablation map + arch research are the inputs; any iteration needs a coding gate, PPL already lied once on attn_v. Note: XXS-from-existing-maps (NEXT #4) is likely dead for 12B specifically — the map says nothing is demotable.

- Launch-args retrofit: add measured best-invocation sections (fits-VRAM + CPU-offload `-ot` split, footprints) to the popular existing cards (35B, 27B, 26B v6, heretics) — 122B card is the pattern; the low-VRAM crowd from the sweep is the audience. New releases get this as a standard post-gate step (see memory: feedback-ship-tuned-launch-args).

- Gemma 4 31B quant (arbv ask)
- EQ-Bench submission (maintainers re-run themselves; pitch drafted in spirit, needs send)
- LocalBench pitch (oobabooga KLD comparisons; one comment could get cerebellum into next roundup)
- ~~Post v6/Heretic results on Google's #37~~ DONE 2026-06-12: vetted reply posted (results table + method explainer link, explainer live at docs/how_cerebellum_works.md on public repo). GEMMA_TEAM_FINDINGS_DRAFT.md vetted and held in cerebellum-dev/drafts/ — shares as published data only if they engage.
- 25k+ context reasoning collapse: reasoning-drift benchmark (RESEARCH_IDEAS), then check if tensor map can help (tima2431 thread)
- Heretic-vs-stock explainer section on cards (igottempmail question)
- Llama 4 family map (research value; saturated quant space, demand-gate it)
- Planka one-way mirror of this file (if Jerry wants the visual board)
- Strip imatrix local-path metadata leak in future builds (privacy hygiene)
- Mozilla Builders application (Local AI theme bullseye); Prime Intellect compute grant email
- Distrobox CUDA stack repair (driver/glibc mismatch; campaigns use host CUDA path meanwhile)

## RESEARCH FINDINGS (validated)

- **Cheap metrics CANNOT replace HumanEval for coding ablation — VALIDATED NEGATIVE 2026-06-13** (KLD_VALIDATION_2026-06-13.md). KLD/PPL/top-token are ANTI-correlated with coding damage on the 27B oracle (Spearman -0.14 to -0.43); PPL ranks the safest group (ffn_down) as most damaging. Real code execution stays mandatory for coding-precision allocation. The achievable speedup is a SMALLER code-execution set (e.g. 20-problem HumanEval smoke, ~8x faster than full 164), NOT a proxy metric. This confirms the project thesis and the hillstep deprecation with a controlled experiment.

## ICEBOX (ideas, unvetted)

- HF Jobs neutral-infra bench receipts: PARKED by Jerry 2026-06-12 (off-brand: nobody asks for it, verified badge doesn't exist yet for anyone, and the one-guy-with-a-3090 story is the brand). Pilot script stays ready; revisit only if HF ships verifyToken issuance and it becomes a level playing field.

- Sub-tensor row-block ablation (the memory-controller endgame; see docs/cerebellum_memory_controller.md)
- CPU-offload-optimized split recipes as a product line (VRAM-pin sacred set, crush experts for RAM bandwidth)
- TurboQuant fork pairing (user reported 12x speedup; investigate compat)
- Quant leaderboard of our own (the gap is real: nobody benches GGUFs by API)

## DONE (recent)

- 2026-06-12: org card + collections cleanup; 17 cards got model-index eval panels (0-shot labels fixed); 18 GGUFs renamed for HF hardware badge; public GitHub repo restored (sanitized data dump, 85 evidence files); AI-attribution trailers scrubbed from public history; osmosis->cerebellum rebrand (code + HF repo); Modal harness built + verified ($0.06); watchdog live; community reception record written; 12B + Flash campaigns launched
- 2026-06-11: Heretic fleet shipped (35B/27B/E4B gate PASS), E2B no-ship finding published
