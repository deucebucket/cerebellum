# Project Hierarchy — The Umbrella

Cerebellum is the **head project**. Everything else is a member project that either
proves the head's thesis or consumes/produces data for it.

```
CEREBELLUM (head)  — ablation-informed mixed-precision GGUF quantization
├── cerebellum quants   — the public proof (HF releases + benchmarks/ evidence)
├── brainloop           — bolt-on refiner research (member)
├── clanker             — deterministic emotion resolver (member)
└── clanker-soul        — emotional learning library (member)
```

## Member directory

| Project | What it is (one line) | Where it lives | What data it offers the others |
|---|---|---|---|
| **Cerebellum** (head) | Per-tensor → per-shard precision allocation; "smaller without being dumber" for 8-12 GB cards | `/var/home/deucebucket/ai-drive/cerebellum` (remotes: `origin`=public cerebellum, `dev`=private cerebellum-dev) | Ablation maps, override files, imatrices, benchmark evidence (`benchmarks/`, per-model dirs), the proven build formula |
| **Cerebellum quants** | The shipped GGUFs — public proof the method works | HF repos (deucebucket/*) + per-model dirs + `/var/home/deucebucket/games/cerebellum-*` build areas | Bench-gated quant recipes, measured launch args, model cards |
| **Brainloop** (conch-poc) | Bolt-on trainable refiner block for frozen small LLMs; knowledge injection via delta vectors, zero custom C++ | `/var/home/deucebucket/ai-drive/cerebellum/cerebellum-dev/conch-poc` — **nested git repo**, own remote `origin` → `cerebellum-brainloop.git` (public!), own CLAUDE.md | Residual-stream/refiner findings, frozen-model knowledge-recall data, HumanEval harnesses for GGUF |
| **Clanker** | Deterministic 7-dim (VADUGWI) emotional state resolver — rules, not ML | `/var/home/deucebucket/ai-drive/clanker` (own repo, CLAUDE.md at root) | Deterministic emotion coordinates other agents/projects can condition on; benchmarks + datasets |
| **Clanker-soul** | Emotional learning library for AI agents (extracted from CARL); host-integrated, no CLI | `/var/home/deucebucket/ai-drive/clanker-soul` (own repo, CLAUDE.md + AGENTS.md at root) | Persistence/learning layer over clanker coordinates; integration examples |
| **Clanker-drift** | Integration shim between clanker and drift | `/var/home/deucebucket/ai-drive/clanker-drift` (AGENTS.md only, no CLAUDE.md) | Integration patterns (`CLANKER_DRIFT_INTEGRATION.md`) |

## Where each project's knowledge surface lives

| Project | CLAUDE.md | Memory | Logs / evidence |
|---|---|---|---|
| Cerebellum | repo root (gitignored, local) + `AGENTS.md` (keep consistent) | `~/.claude/projects/-var-home-deucebucket-ai-drive-cerebellum/memory/` (MEMORY.md + ~29 topic files) | `cerebellum-dev/` devlogs, `docs/` dated findings, `benchmarks/` |
| Brainloop | `cerebellum-dev/conch-poc/CLAUDE.md` | "Logging Is Law" append-only ops log inside conch-poc; RESEARCH_LOG.md, RESULTS.md, DEADBLOCK_STATUS.md | `conch-poc/bench_results/`, eval logs, checkpoints-* dirs |
| Clanker | `/var/home/deucebucket/ai-drive/clanker/CLAUDE.md` (+ `.bak`) | clanker project memory (incl. `user-working-style.md`) | `clanker/benchmarks/`, `clanker/docs/`, CHANGELOG.md |
| Clanker-soul | `/var/home/deucebucket/ai-drive/clanker-soul/CLAUDE.md` + AGENTS.md | (own project memory dir if used) | `clanker-soul/logs/`, docs/, CHANGELOG.md |

## Rules of engagement between projects

- Repos stay separate. Never merge trees, never move a member into another's repo.
- Brainloop's nested-repo status is intentional but sharp-edged: `git` commands run
  inside `conch-poc/` hit the **public** brainloop remote, not cerebellum's dev remote.
  The parent repo also carries a `public` remote pointing at cerebellum-brainloop.git — do not push to it casually.
- Cross-project truth flows through each repo's CLAUDE.md pointer block → this knowledge dir (see INDEX.md).
