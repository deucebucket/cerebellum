# Cerebellum Community Feedback Digest, 2026-06-12

Read-only sweep of Hugging Face (all repos, all discussions, verbatim) plus a best-effort
external web sweep. Raw data behind this report:
`cerebellum-dev/community_feedback_data_2026-06-12/` (repos_models.json, discussions_raw.json).

No replies were posted anywhere as part of this sweep.

## 1. Scoreboard

Downloads = HF rolling 30-day count. All-time = downloadsAllTime. Disc = discussions (open/closed),
excluding bot threads. Method column: tensor = per-tensor override builds (27B v4 era and later),
group = group-level ablation builds (35B v1/v3 era), per repo card and release notes.

| Repo | Method | DL 30d | DL all-time | Likes | Disc open | Disc closed |
|---|---|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF | tensor (v6) | 2,732 | 4,535 | 10 | 2 | 0 |
| Gemma-4-26B-A4B-it-Heretic-Cerebellum-GGUF | tensor (transfer recipe) | 2,339 | 2,339 | 5 | 1 | 0 |
| Qwen3.6-35B-A3B-Cerebellum-GGUF | group (v1, 400 overrides from group ablation) | 1,760 | 2,494 | 10 | 3 | 0 |
| Qwen3-30B-A3B-Cerebellum-GGUF | group | 649 | 1,075 | 1 | 0 | 0 |
| Gemma-4-E4B-it-Cerebellum-v2-GGUF | group/early | 437 | 942 | 3 | 0 | 1 (junk PR) |
| Qwen3.6-27B-Cerebellum-GGUF | tensor (v4, 181 overrides) | 330 | 2,030 | 9 | 1 | 0 |
| Qwen3.6-27B-Osmosis-Q2_K-GGUF | legacy name | 221 | 990 | 2 | 0 | 0 |
| Gemma-4-E2B-it-Cerebellum-v2-GGUF | group/early | 191 | 606 | 3 | 0 | 0 |
| Qwen3.5-122B-A10B-Cerebellum-GGUF | group | 175 | 723 | 5 | 0 | 0 |
| Gemma-4-26B-A4B-it-Cerebellum-GGUF (v3) | group/early | 131 | 797 | 4 | 1 | 0 |
| Granite-4.1-30B-Cerebellum-GGUF | group | 68 | 383 | 1 | 0 | 0 |
| Qwen3-32B-Cerebellum-GGUF | group | 66 | 427 | 1 | 0 | 0 |
| Granite-4.0-H-Small-Cerebellum-GGUF | group | 61 | 353 | 1 | 0 | 0 |
| Qwen3-14B-Cerebellum-GGUF | group | 36 | 423 | 1 | 0 | 0 |
| Qwen3.6-27B-Heretic-Cerebellum-GGUF (new 06-12) | tensor (transfer) | 0 | 0 | 2 | 0 | 0 |
| Qwen3.6-35B-A3B-Heretic-Cerebellum-GGUF (new 06-11) | tensor (transfer) | 0 | 0 | 1 | 0 | 0 |
| Gemma-4-E4B-it-Heretic-Cerebellum-GGUF (new 06-12) | tensor (transfer) | 0 | 0 | 1 | 0 | 0 |
| dataset: cerebellum-eval-logs (new 06-12) | n/a | 0 | n/a | 1 | 0 | 0 |
| dataset: cerebellum-brainloop | n/a | 4 | n/a | 0 | 1 (bot) | 0 |
| space: Qwen36-27B-Cerebellum-v4-Demo | n/a | n/a | n/a | 0 | 1 (own grant request) | 0 |

Totals across model repos: roughly 9,200 downloads 30d, 18,100 all-time, 60 likes, 9 real user discussions.
HF profile: 17 followers. Org DB-Cerebellum: 1 follower, collections only, no model repos yet.

Tensor vs group reception: the three repos with real conversation (26B v6, 26B Heretic, 35B) split
across both methods, so the method itself is not what users react to. They react to which base model
it is (Gemma 4 26B and Qwen 3.6 35B are the popular bases) and to whether an uncensored variant exists.
Nobody in any thread asked about tensor-level vs group-level. The pitch that lands is size-for-quality,
not the ablation granularity.

## 2. Every user interaction, one by one

### Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF

- #2 "What about uncensored/abliterated version?" (tima2431, 2026-05-09, open)
  https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF/discussions/2
  Praise ("very cool model, works so well") plus a request for an uncensored build. This thread
  directly caused the Heretic line: Koitenshin suggested coder3101's heretic base, deucebucket shipped
  Heretic-Cerebellum on 05-19, tima2431 made it their daily driver ("almost on par with Gemini 2.5 Pro,
  35-45 t/s on an RTX 5060"). Open bug inside the thread: reasoning collapses at 25k+ context (model
  prints "enough;" and skips the think phase). Acknowledged, attributed to chat template plus an
  unmerged llama.cpp fix, not resolved.
- #1 "Gemma-4-E4B-it-Cerebellum-v1" (dont-remember-it, 2026-05-02, open)
  https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF/discussions/1
  Started as "the E4B repo vanished" (accidentally privated, fixed same day). Then turned into the
  strongest praise on record: on an 8GB RTX 4070 laptop with the AtomicBot TurboQuant fork plus MTP
  speculative decoding, 2.4 to 30+ tok/s vs Q4_K_M, "one of the most impressive quantization efforts
  I've seen". Asked for vision support; root cause was the missing mmproj file, shipped, confirmed
  working ("seamlessly"). Resolved.

### Gemma-4-26B-A4B-it-Heretic-Cerebellum-GGUF

- #1 "which version is better heretic or nomal" (igottempmail, 2026-06-09, open)
  https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Heretic-Cerebellum-GGUF/discussions/1
  Confusion: user cannot tell from the cards whether heretic or normal is smarter, worried heretic
  models "go dumber". Answered (heretic tested better here), but the cards do not state this comparison
  anywhere. A side question (should I use Gemma-4-12B-it instead?) was never answered.

### Qwen3.6-35B-A3B-Cerebellum-GGUF

- #3 "Amazing quality for the size!" (TheodoreH, 2026-05-31, open, long thread)
  https://huggingface.co/deucebucket/Qwen3.6-35B-A3B-Cerebellum-GGUF/discussions/3
  The most engaged user on any repo. Praise: best among specialty quants (names APEX, REAP, Magic
  quants as worse), runs almost CPU-only, low hallucination, other languages survive the quant, beats
  HauHau IQ4-XS that is 2 GB larger, and vision shipping in-repo is a differentiator. Requests:
  Qwen 35B heretic version (delivered 06-11 after a reminder nudge on 06-10), and intermediate Q3/Q4
  size points to fit 14 GB RAM (not yet addressed). Reports 8-10 t/s on 4 GB VRAM + 16 GB RAM.
- #2 "first adjusted model that actually seems equal-or-better" (Tribbler, 2026-05-13, open)
  https://huggingface.co/deucebucket/Qwen3.6-35B-A3B-Cerebellum-GGUF/discussions/2
  Praise (equal or better than IQ3_XXS while smaller) plus the best bug report received: vision worse
  than IQ3_XXS, looping, cut-off output. Worked through publicly; final answer was the missing
  mmproj-F16.gguf, not quant damage. Retested 78.0% RealWorldQA vs 77.5% stock. Resolved with data.
  The looping-in-general note was acknowledged but never separately closed out.
- #1 release announcement thread (deucebucket, 2026-05-02, open)
  https://huggingface.co/deucebucket/Qwen3.6-35B-A3B-Cerebellum-GGUF/discussions/1
  One thank-you from cai2023. No issues raised.

### Qwen3.6-27B-Cerebellum-GGUF

- #1 "Please consider producing an MTP version" (arbv, 2026-05-20, open)
  https://huggingface.co/deucebucket/Qwen3.6-27B-Cerebellum-GGUF/discussions/1
  Request: MTP weights baked in at native precision now that llama.cpp supports it, plus a Gemma 4 31B
  mixed-precision quant. Both acknowledged as planned; neither shipped yet.

### Gemma-4-26B-A4B-it-Cerebellum-GGUF (v3)

- #1 "Github repo doesn't exist anymore" (ibaldonl, 2026-06-06, open)
  https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-GGUF/discussions/1
  Confusion: the GitHub repo the card points to is down. Answered (taken down for cleanup, "it'll be
  back soon"), but the dead link is presumably still on the cards until the public repo returns.

### Gemma-4-E4B-it-Cerebellum-v2-GGUF

- #1 "chanti" (chantidas, 2026-05-18, closed)
  Empty junk PR, closed. No action needed.

### Upstream: google/gemma-4-26B-A4B-it discussion #37

- https://huggingface.co/google/gemma-4-26B-A4B-it/discussions/37
  deucebucket's announcement of Cerebellum v3 on the official Gemma repo got a reply from thnamratha
  (Google org): thanked for sharing, expressed interest in the approach, flagged a 404 link, link fixed.
  Friendly official attention; no follow-up after the fixed link.

### Space: Qwen36-27B-Cerebellum-v4-Demo

- GPU community grant application (deucebucket, 2026-05-01, open)
  No response from HF after six weeks. Either nudge it or treat it as declined.

Zero-discussion repos (silent downloads only): Granite both, Qwen3 14B/32B/30B, E2B, 122B, Osmosis-Q2_K,
and all three Heretic repos launched 06-11/06-12 (too new). cerebellum-eval-logs: no discussions.

## 3. External mentions

- Google org engagement: gemma-4-26B-A4B-it discussion #37 (above). The only "institutional" notice found.
- Third-party derivative, now gone: LordAce9/Qwen3.6-27B-Cerebellum-GGUF-RTX5080-TurboQuant-KV-Runtime
  is in the Google index (title suggests someone repackaged the 27B Cerebellum with TurboQuant KV for an
  RTX 5080) but the repo now returns 401/not-found, so it was deleted or made private. Evidence someone
  built on the work; content unrecoverable.
  https://huggingface.co/LordAce9/Qwen3.6-27B-Cerebellum-GGUF-RTX5080-TurboQuant-KV-Runtime
- Ecosystem adjacency: users run Cerebellum builds on the AtomicBot TurboQuant llama.cpp fork
  (https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant); pairing Cerebellum with MTP/speculative
  decoding is where the 12x speed story came from.
- Name collision, not a mention: dmitchelljackson/cerebellum-e4b-lora and
  cerebellum-qwen35-history-actions-lora are an unrelated Android-UI project also named Cerebellum.
  No reference to this work. Worth knowing the name is not unique on the Hub.
- Reddit r/LocalLLaMA: none found. Hacker News: none found. Blogs/newsletters (kaitchup etc. cover
  mixed-precision GGUF generally): no Cerebellum coverage found. Searches tried: "deucebucket" alone
  and with quant/GGUF/cerebellum, "cerebellum GGUF", "cerebellum quant", model repo names, site:reddit.com,
  HN. Absence of buzz is real: all traction so far is organic HF discovery, with zero social amplification.

## 4. Themes

1. The uncensored demand drove the strongest engagement. The single most-requested thing was an
   abliterated version, and the Heretic line (born from a user thread) hit 2,339 downloads in under a
   month and produced the "daily driver" and "on par with Gemini 2.5 Pro" quotes. Users also report the
   heretic builds test better than stock, which matches internal findings.
2. The audience is the low-VRAM/CPU-offload crowd, exactly the project mission. Reported rigs: 8 GB
   4070 laptop, RTX 5060, 4 GB VRAM + 16 GB RAM CPU-offload. They quote tok/s numbers unprompted and
   compare against IQ3_XXS / IQ4_XS / APEX / REAP. Cerebellum is winning those user-run comparisons.
3. Vision packaging was the recurring early bug, not quant quality. Both real bug reports (35B looping
   "vision damage", 26B "no vision") were the missing mmproj file. Now fixed, and shipping mmproj is
   cited as an advantage over competitors. Lesson encoded: every multimodal release must include mmproj
   and a smoke test before upload.
4. People build on top and ask for speed features: MTP-baked weights, the TurboQuant fork pairing, the
   deleted LordAce9 RTX 5080 repack. Speed-adjacent integration (MTP, speculative decoding) is the most
   concrete shipping request on the table.
5. Reliability of project surfaces lags the models: the GitHub repo is down and users notice, a repo got
   accidentally privated, and one user could not tell heretic vs normal apart from the cards. The models
   earn trust; the surrounding links and explanations leak it.

## 5. Open items / to-do list

- [ ] 25k+ context reasoning collapse on Gemma 4 builds (26B v6 #2). Acknowledged, never resolved or
      tracked. Retest on current llama.cpp; post findings to the thread.
- [ ] TheodoreH's request for Q3/Q4 intermediate sizes to fit 14 GB RAM (35B #3). Unanswered.
- [ ] arbv's MTP-baked-weights request and Gemma 4 31B quant request (27B #1). Promised "on the list",
      nothing shipped; the Heretic transfer recipe note even says avoid MTP-preserved sources, so if MTP
      builds are out of scope, say so in the thread.
- [ ] igottempmail's side question (Gemma-4-12B-it vs 26B uncensored) never answered, and the cards do
      not explain heretic-vs-normal quality. Add a one-paragraph comparison to both 26B cards.
- [ ] GitHub repo still down; ibaldonl thread promised "back soon" on 06-07. Restore it or update the
      model cards to stop pointing at a dead link.
- [ ] Tribbler's general looping observation (35B #2) was folded into the mmproj fix but never separately
      confirmed fixed. One reply closing the loop would finish the best bug thread on the account.
- [ ] HF GPU community grant application open since 05-01 with no response. Nudge or abandon.
- [ ] Tell TheodoreH and tima2431 that the Qwen 3.6 35B Heretic and 27B/E4B Heretic repos are live
      (35B was pointed at on 06-11; the other two launched after). These two users are the de facto QA
      team; they asked to be pinged.
- [ ] cerebellum-brainloop parquet-converter bot thread can be closed, cosmetic.

## 6. Surprises worth knowing

- Google replied. An official Gemma org member engaged with the v3 announcement. That thread is a
  standing invitation to post v6/Heretic results upstream.
- Someone repackaged the work (LordAce9) and then pulled it. The method is being treated as a base
  layer by strangers already.
- The two Gemma 26B repos alone are about 55% of all 30-day downloads. Gemma 4 26B is the franchise.
- Not one user asked about methodology rigor, benchmarks provenance, or tensor vs group ablation.
  The audience trusts vibes and their own A/B tests; the published benchmark JSONs are for credibility,
  not because users read them.
- There is another "Cerebellum" on the Hub (Android UI agent). No conflict yet, but the name is shared.
