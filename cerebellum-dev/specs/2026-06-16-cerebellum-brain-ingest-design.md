# Cerebellum-Brain — Piece 1: Core DB + Umbrella Harvester (Design)

Date: 2026-06-16. Status: design, pending user review. Location: private (cerebellum-dev).

## Purpose

`cerebellum-brain` (private GitHub repo) is the literal brain of the **entire
cerebellum-projects umbrella** — not just the quant models. It folds in viable
testing/experimental data from every sibling project so we never miss a datapoint
that could help. Shared, not gated.

Piece 1 is the foundation: a family-organized SQLite database + an idempotent,
pluggable harvester that ingests data from all umbrella sources. Pieces 2 (auto-insight
engine) and 3 (dashboard/API + public github.io face) read this DB and are out of scope here.

## Umbrella sources (the harvester's registry)

Pluggable per-source adapters. Each adapter knows where that project's viable data lives
and how to normalize it. Registry is a YAML/TOML config so new projects are one entry.

| Source | What to harvest |
|--------|-----------------|
| `cerebellum` / `cerebellum-dev` (local repo) | `data/<family>/<model>/`: ablation results, cost_ledger.json, ARCH_RESEARCH.md, club3090_findings.md, 8-pack results, override/recipe maps, imatrix meta. `sensitivity_db/master_catalog.tsv` (seed). |
| `cerebellum-brainloop` | RESULTS.md, bench_humaneval*/recall_bench results, layer_divergence/injection-sweep data — refiner-block experiments that may inform tensor sensitivity. |
| clanker family (`clanker`, `-soul`, `-drift`, `evil-clanker`, …) | Any eval/test/benchmark data (VADUGWI emotional-engine evals) — different domain, harvested as generic "findings" so nothing is lost. |
| `cerebellum-cua`, `llama-cpp-turboquant`, llama.cpp forks | Test/perf data, quant experiments, runtime quirks. |

Adapters degrade gracefully: a project with no recognizable data yields zero rows, never errors.

## Schema (family-organized; centers the capability-sensitivity ledger)

Core tables (SQLite, one file `brain.db`, committed to the private repo since data is ~1GB):

- **`source`** — registry of harvested projects (name, repo_url, local_path, last_ingest_commit, last_ingest_ts).
- **`model`** — id, family (`moe`/`hybrid_ssm`/`ple`/`dense`), name, params_total, params_active, base_arch (GGUF `general.architecture`), license, notes. (e.g. GLM-4.7-Flash → family=moe, base_arch=deepseek2.)
- **`tensor`** — model_id, tensor_name, tensor_group (attn_qkv / ffn_down_exps / ssm_* / router …), layer_idx.
- **`sensitivity`** — the spine. tensor_id, crush_format (Q2_K…), capability (humaneval / ppl / arc / dataextract / needle …), delta (vs baseline), source_run, provenance. Enables FAT/MUSCLE: crushable only if low-delta on EVERY capability.
- **`build`** — model_id, label (v4/v5/qkvq3…), bpw, size_bytes, override_map_ref, imatrix_ref.
- **`benchmark`** — build_id, suite (8pack/evalplus/arc…), pack, score, n, harness, run_ts, evidence_path. (Must capture per-pack so anomalies like a crashed `hermesagent 0/20` are visible, not averaged away.)
- **`finding`** — generic cross-project knowledge: source_id, model_id (nullable), topic, text, tags, url. Catches brainloop/clanker/club-3090 notes that don't fit the structured tables.
- **`provenance`** on every row — source + file + commit/ts, so any datum traces back and re-ingest is safe.

## Extraction architecture (HYBRID — model reads prose, parser reads numbers)

Do NOT build a pure-regex harvester (brittle → junk drawer) and do NOT let a model invent
numbers (hallucination). Split by data type:

- **Deterministic parsers** for exact/structured values: sensitivity deltas, 8-pack per-pack
  scores, sizes/bpw, cost_ledger.json, master_catalog.tsv, run.log score lines. Numbers are
  parsed exactly, never LLM-guessed.
- **LLM extraction** for prose/heterogeneous docs (ARCH_RESEARCH.md, club3090_findings.md,
  brainloop RESULTS.md, devlogs): a model reads the doc and returns **controlled-vocab
  structured JSON** — family, base_arch, tensor-group classifications, and narrative
  `finding` rows (topic/tags/model). This "gets the gist" instead of pattern-matching.
- **Extraction is extract-only, grounded:** the model must return a supporting quote/span for
  each `finding` and may NOT emit numeric benchmark/sensitivity values (those come only from
  the deterministic parsers). Each LLM row carries a `confidence`.
- **Configurable model endpoint:** default `localhost:7800` (local llama.cpp, free; currently
  DOWN — bring up before a full ingest), fallback to `ds`/deepseek. Endpoint + model in config.
  Build/unit-test the harvester with a mockable extractor interface so it doesn't require a
  live model to compile or run schema/idempotency tests; the LLM-extraction tests run when an
  endpoint is configured.
- **Low confidence / unparseable → ingest report**, never silent storage.

## Classification discipline (NO JUNK DRAWER — hard requirement)

Prior data systems failed by dumping most content into an "other/misc" bucket until they
were useless. This harvester must classify aggressively into the structured tables; the
`finding` table is a *last resort*, never a default.

- **Controlled vocabularies** for `family`, `tensor_group`, `capability`, and `finding.topic`.
  Adapters map to these explicitly — no free-text sprawl, no inventing buckets.
- **Structured-first**: a sensitivity/benchmark/build number ALWAYS goes to its typed table.
  Only genuinely narrative knowledge (a community insight, a caveat) may go to `finding` —
  and even then it MUST carry `topic` (from the vocab), `tags`, and `model_id` (or an explicit
  reason it's model-agnostic). No untyped blobs, ever.
- **Junk-drawer guard (enforced by test):** after a full ingest, the fraction of rows landing
  in `finding` (and the fraction of `finding` rows with `topic='other'`) must stay under a
  threshold. Exceed it → the harvest is treated as FAILED and names the under-classifying
  adapter. A growing "other" pile is a bug, not an acceptable state.
- **Unparseable input is reported, not swallowed**: if an adapter can't classify a file, it
  logs it to an ingest report for adapter improvement — it does NOT silently file it under "other".

## Harvester behavior

- **Idempotent**: re-runnable; upsert keyed on (source, natural-key); a re-ingest of unchanged data is a no-op.
- **Provenance-locked**: never overwrites evidence; records where each datum came from.
- **Incremental**: per-source `last_ingest_commit` so we only re-scan changed projects.
- **Read-only at source**: harvester never mutates source repos.
- **CLI**: `brain ingest [--source X] [--all]`, `brain stats`, `brain query "<SQL>"` — usable by any agent.

## Explicitly out of scope (later pieces)

- Piece 2: auto-insight engine (universal-MUSCLE detection, ablation-map similarity, anomaly flags).
- Piece 3: dashboard/API (extend `cerebellum/dashboard/`) + public github.io face (template: `fo76-data`/`deucebucket.github.io`), curated published-only, AI-attribution scrubbed.

## Build plan (delegated)

Per the delegate-to-agents mandate (save Claude tokens): the implementation (schema DDL,
adapters, CLI) is handed to `opencode` (deepseek) / `codex` (gpt) agents against this spec.
Claude orchestrates, reviews output, and gates correctness (does a re-ingest stay idempotent?
does provenance trace? does the GLM row show base_arch=deepseek2?). Tests with fixed seeds,
real fixtures from `data/glm/` and `data/north/`.

## Publication boundary (HARD RULE — never leak the sauce)

The brain is the sauce. The public face must NEVER be a filtered dump of the brain —
that's how leaks happen. Instead:

- **Default deny.** A datum reaches the public face ONLY if explicitly allow-listed.
  Schema enforces this: every publishable row carries a `publish` flag (default `false`)
  and/or a `public_ref` (proof it's *already* public — e.g. a shipped HF benchmark or a
  posted club-3090 number). No flag → never exported. Piece 3 reads only `publish=true`.
- **The sauce is, by definition, not flagged:** raw per-tensor sensitivity deltas,
  ablation maps, override recipes, cross-model correlation insights, unshipped candidates,
  failure data. These stay private permanently.
- **Already-public is safe to surface:** shipped-model benchmark rows, published model-card
  numbers, club-3090 posts we authored.
- **Piece 3 export is a separate, auditable step** (its own diff/review), never automatic,
  AI-attribution scrubbed. A human (Jerry) approves the allow-list before any public push.
- `cerebellum-brain` repo stays private; nothing in Piece 1 produces any public artifact.

## No public exposure (Piece 1)

Piece 1 builds the private brain only. It emits zero public artifacts. The curated export
lives entirely in Piece 3, gated by the publication boundary above.
