# Network Map, 2026-06-12

Read-only recon across Hugging Face and GitHub for deucebucket. Nothing was posted,
followed, liked, or starred during this sweep. All suggestions below are suggestions only.

Sources: HF followers API, HF model likers API, HF user overview API, GitHub followers,
stargazers of cerebellum / clanker / clanker-soul / cerebellum-brainloop, GitHub issue search,
and the local `COMMUNITY_FEEDBACK_2026-06-12.md` digest plus `community_feedback_data_2026-06-12/` raw JSON.

## 1. Counts

| Metric | Number | Notes |
|---|---:|---|
| HF followers (deucebucket) | 17 | full list, one page |
| HF followers (org DB-Cerebellum) | 1 | just deucebucket himself |
| GitHub followers | 4 | tpowellcio, sydches, swstegall, shimo4228 |
| GitHub stargazers (4 target repos, deduped, excl. self) | 4 | shimo4228, dandenkijin, swstegall, ivanbaldo |
| Multi-repo stargazers | 1 | shimo4228 starred 3 of 4 (cerebellum, clanker, cerebellum-brainloop) |
| Cross-platform people (HF + GH confirmed same human) | 2 | shimo4228/Shimo4228, ivanbaldo/ibaldonl |
| HF model likers harvested (top 5 repos) | 22 unique | likers API works; overlap with followers is high |

Reality check: the raw follower numbers are small. The signal is not in volume, it is in
who the specific humans are. Several of them turn out to be exactly the right people.

## 2. Ranked shortlist

### 1. noonghunna, club-3090 maintainer
- Platforms: GitHub (36 followers). No HF account found under that handle.
- Why they matter: maintains club-3090 (1,321 stars), THE community hub for serving LLMs on
  3090/4090/5090, shipping configs for the exact models Cerebellum quantizes (Qwen3.6 27B/35B,
  Gemma 4 26B/31B). Also authors beellama.cpp (DFlash + TurboQuant in llama.cpp) and
  genesis-vllm-patches (45+ vLLM patches, Qwen3.6-35B on Ampere). This person sits at the center
  of the target audience.
- Evidence: https://github.com/noonghunna/club-3090 ; deucebucket already filed
  https://github.com/noonghunna/club-3090/issues/390 (bench: qwen3.6-35b-a3b, 131k ctx at 15.1GB,
  opened 2026-06-12, 0 comments so far).
- Suggested lean-in: issue #390 is the live thread, when the maintainer replies, follow through
  fast. Natural next step is contributing a Cerebellum recipe config to club-3090 (a Cerebellum
  GGUF + launch args entry alongside the existing model configs). That puts the models in front
  of 1,321 stargazers of exactly the right demographic.

### 2. coder3101 (Ashar), heretic author, DigitalOcean
- Platforms: HF (191 followers, 61 models, heretic-org member) + GitHub (67 followers,
  company: DigitalOcean, blog coder3101.com, twitter @coder3101).
- Why they matter: author of the heretic base models the entire Cerebellum Heretic line is
  built on (gemma-4-26B heretic: 90 likes, 18k downloads). Infra-company employee with a real
  following. Upstream dependency and big fish in one person.
- Evidence: https://huggingface.co/coder3101 ; https://github.com/coder3101 ;
  Heretic transfer recipe in this repo consumes his sources.
- Suggested lean-in: share the measured result that his heretic bases benchmark equal-or-better
  after the Cerebellum pass (the KL-screen data is genuinely interesting to an abliteration
  author). A discussion post on one of his heretic repos with benchmark evidence is a
  zero-ask, high-credibility introduction.

### 3. thnamratha, Google org member on HF
- Platforms: HF (google org, 3 followers, 22 discussions, account created 2026-04).
- Why they matter: the only institutional contact on record. Replied warmly to the Cerebellum v3
  announcement on the official google/gemma-4-26B-A4B-it repo, thanked for sharing, expressed
  interest in the approach, flagged a broken link (fixed).
- Evidence: https://huggingface.co/google/gemma-4-26B-A4B-it/discussions/37 ;
  https://huggingface.co/thnamratha
- Suggested lean-in: that thread is a standing invitation. Posting v6 + Heretic results
  (downloads, user quotes, benchmark deltas) as a follow-up in the same thread re-engages the
  Google contact without cold outreach. Low follower count suggests a DevRel/community role,
  which is exactly who amplifies community projects.

### 4. llmfan46, dominant uncensored-quant publisher
- Platforms: HF only (1,148 followers, 175 models, Pro, Patreon + Ko-fi).
- Why they matter: by far the biggest fish in the orbit. Their heretic GGUFs of the same bases
  pull 66k to 126k downloads each. One repo of theirs outdraws the entire Cerebellum catalog
  by 5x. Same audience, same bases, complementary method (they abliterate, Cerebellum compresses).
- Evidence: https://huggingface.co/llmfan46 ; their Qwen3.6-35B heretic was cited inside the
  TheodoreH thread on the 35B Cerebellum repo as an alternative base.
- Honest caveat: no direct contact yet. They have not engaged deucebucket; their model was only
  linked by a third party in a discussion. This is the coldest lead on the list but the largest.
- Suggested lean-in: a Cerebellum mixed-precision pass on one of llmfan46's most-downloaded
  heretic bases, shipped with credit and benchmark evidence, then a discussion post on their repo.
  Their bio says they take requests via Patreon votes, meaning they are collab-receptive.

### 5. shimo4228 (Tatsuya Shimomoto), the cross-platform superfan
- Platforms: GitHub (56 followers, Japan, Substack substack.com/@shimo4228, twitter @shimo4228)
  + HF (Shimo4228, 8 datasets). The only person who follows on BOTH platforms.
- Why they matter: strongest multi-touch signal in the dataset: HF follower, GH follower, and
  starred 3 of 4 target repos (cerebellum, clanker, cerebellum-brainloop). Builds thoughtful
  local-agent tooling (contemplative-agent runs entirely on a local 9B model, agent-knowledge-cycle,
  claude-harness). Writes publicly on Substack. Local-small-model philosophy aligned with the
  project mission.
- Evidence: https://github.com/shimo4228 ; https://huggingface.co/Shimo4228 ; stargazer lists
  of deucebucket/cerebellum, deucebucket/clanker, deucebucket/cerebellum-brainloop.
- Suggested lean-in: he runs a local 9B for his agent. The Qwen3.5-9B Cerebellum work is directly
  useful to him. A reply or issue engagement when he next touches one of the starred repos,
  or simply making the brainloop README better, serves the one person already watching everything.
  Possible long-game: he is a writer, and Cerebellum has zero external coverage so far.

### 6. ibaldonl / ivanbaldo (Ivan Baldo), MLOps at Netlabs Uruguay
- Platforms: HF (8 followers, 10 discussions, bio: MLOps, Scalability, Performance, OnPremises)
  + GitHub (ivanbaldo, 26 followers, company @netlabsuy, Montevideo).
- Why they matter: professional on-prem MLOps person, cross-platform engaged: HF follower,
  liked the 35B and 27B models, starred cerebellum-brainloop, and filed the
  "Github repo doesn't exist anymore" discussion. On-prem deployment people are the
  commercial-adjacent audience for VRAM-budget quantization.
- Evidence: https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-GGUF/discussions/1 ;
  https://github.com/ivanbaldo
- Suggested lean-in: there is an open promise to him ("repo will be back soon", 06-07). Restoring
  the public GitHub repo and replying in his thread closes a loop with the most professionally
  relevant warm lead. Costs nothing, repairs a trust leak the feedback digest already flagged.

### 7. TheodoreH (Theo), the de facto QA lead
- Platforms: HF only (new account 2026-05-31, 0 followers). GH handle exists but is empty.
- Why they matter: the single most engaged user on any repo. Long structured thread on the 35B:
  comparative testing against APEX/REAP/Magic/HauHau quants, multilingual checks, CPU-offload
  perf numbers (8-10 t/s on 4GB VRAM + 16GB RAM), drove the 35B Heretic release with a reminder
  nudge. Asked to be pinged on releases.
- Evidence: https://huggingface.co/deucebucket/Qwen3.6-35B-A3B-Cerebellum-GGUF/discussions/3
- Honest caveat: zero reach, brand-new account. Value is QA and testimonial, not network.
- Suggested lean-in: already on the backlog: ping him that 35B Heretic shipped and answer the
  open Q3/Q4 intermediate-size request. He produces the best public testimonial quotes
  on the account, keep him fed.

### 8. arbv (Artem Boldariev), veteran OSS dev with a blog
- Platforms: HF (5 discussions, new account) + GitHub (50 followers, Kharkiv Ukraine,
  blog chaoticlab.io, long OSS record: daemonize 42 stars, emacs-msi-installer 52 stars,
  AVR coroutines, Lisp/Genera patches).
- Why they matter: a real systems engineer, HF follower, liked the 27B, and filed the most
  technical feature request on record (MTP weights baked at native precision, plus Gemma 4 31B).
  Has an active technical blog, another potential coverage vector.
- Evidence: https://huggingface.co/deucebucket/Qwen3.6-27B-Cerebellum-GGUF/discussions/1 ;
  https://github.com/arbv ; https://chaoticlab.io/
- Suggested lean-in: his MTP request was answered "on the list" but internal findings say avoid
  MTP-preserved sources for the transfer recipe. An honest technical close-out reply (here is why
  MTP-baked is deprioritized, here is what we found) respects the kind of engineer he is and
  keeps the relationship.

### 9. VykosX, club-3090 tooling author
- Platforms: GitHub (21 followers; ControlFlowUtils 147 stars, ModernVB 121 stars,
  club-3090-server 12 stars, AgenticStudio) + HF (bio: LLMs, ComfyUI, LM Studio; 1 discussion,
  known from club-3090 orbit).
- Why they matter: builds the serving/orchestration layer around club-3090, and AgenticStudio's
  README literally says "Recommended to use with Qwen 3.6 27B", the exact model the 27B
  Cerebellum quant serves. Toolmaker whose users need small fast 27B GGUFs.
- Evidence: https://github.com/VykosX/AgenticStudio ; https://github.com/VykosX/club-3090-server
- Suggested lean-in: when the club-3090 bench issue thread is active, a pointer that the 27B
  Cerebellum build fits AgenticStudio's recommended-model slot at a smaller footprint is a
  natural, non-spammy fit. Suggestion only.

### 10. pegasus912 (Thomas N), peer quanter
- Platforms: HF (10 followers, 27 models). GH handle exists but empty.
- Why they matter: an active quant publisher working the same bases: gemma-4-26b-a4b heretic
  UD-Q4-K-XL repack at 2,505 downloads, plus 31B and 12B variants. HF follower and liked the
  Heretic 26B. A peer, possibly a future comparison point or collaborator, possibly a competitor
  studying the work.
- Evidence: https://huggingface.co/pegasus912
- Suggested lean-in: low priority. His UD repacks of the same heretic bases are the obvious
  head-to-head benchmark targets; publishing a fair comparison would either win publicly or
  surface something to learn.

## 3. Warm-but-low-reach bench (not shortlisted, do not lose them)

- tima2431: caused the entire Heretic line with discussion #2 on 26B v6, made it their daily
  driver, quote "almost on par with Gemini 2.5 Pro". Zero followers, zero models. Pure QA and
  testimonial value. Backlog already says ping them about the new Heretic repos.
- Tribbler: best bug report on record (35B vision/mmproj). 2 models, 0 followers. The looping
  observation was never formally closed out, one reply finishes the account's best bug thread.
- Hyphonical: HF follower, 18 models, 11 followers, publishes abliterated/APEX-style quants of
  the same bases (Qwen3.6-35B-abliterated-MAX-APEX-i-nano). Another peer quanter, smaller scale.
- Koitenshin: not a follower, but the person who pointed the 26B thread at coder3101's heretic
  base, indirectly creating the Heretic line. 8 models, 48 discussions, an active connector
  type in these comment sections.
- igottempmail, dont-remember-it: engaged discussions (heretic vs normal confusion; the E4B
  vanished thread that became the TurboQuant 12x speed story). Throwaway-style accounts, value
  is the threads, not the people.
- LordAce9 (Muhla): made and then deleted a TurboQuant RTX 5080 repack of the 27B Cerebellum.
  2 models, 0 followers. Evidence the method is treated as a base layer; the person themself is
  currently low-signal.

## 4. Honest nulls

- GitHub followers are mostly pre-AI acquaintances, not scene contacts: tpowellcio (DevOps,
  Collective Medical, account from 2011), sydches (1 follower, inactive), swstegall (JPMorganChase
  Android dev, 106 followers, starred brainloop, probably a friend; not an AI-scene node).
- Roughly a third of the HF followers are empty accounts with zero publications and zero
  discussions (AXFSSD, cunzai97, Kazao53, Jonipentti, andyoneal-on-HF, yybl-adjacent likers like
  user3762, valer4iksheva, TATERHATER, ithilelda). They downloaded, liked, followed, and that is all.
- No HF staff, no Mistral/Cohere/unsloth/Nous members anywhere in the follower or liker lists.
  The only lab-adjacent contact is thnamratha (Google org).
- Org DB-Cerebellum has exactly one follower: deucebucket. The org has no gravity yet.
- dandenkijin starred clanker, account has 67 repos but no bio/identity signal, unclassifiable.
- Rhonstin (Bohdan Kikot): present on both platforms, 59 GH repos, but no visible interaction
  trail beyond the club-3090 orbit. Thin.
- noonghunna club-3090 contributors (easel, danbedford, JohnTheNerd, Whamp, hlo-world) were not
  individually profiled; they are second-degree orbit, reachable through the club itself.
- External coverage remains zero (matches the 06-12 feedback sweep): no Reddit, no HN, no blogs.
  Every relationship above came from organic HF discovery.

## 5. One-line read

The network is small but unusually well-aimed: one community hub with a live thread (club-3090),
one upstream author at an infra company (coder3101), one Google contact with a standing thread
(thnamratha), one giant adjacent publisher not yet contacted (llmfan46), and a tiny circle of
cross-platform true believers (shimo4228, ivanbaldo) plus a volunteer QA team (TheodoreH, tima2431,
Tribbler). Almost every suggested move is "finish a conversation already started" rather than
cold outreach.
