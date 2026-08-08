# Cerebellum

Cerebellum is the public home for model releases, benchmark summaries, model
cards, and high-level reproducibility notes for Cerebellum quantized GGUFs.

The public repository is intentionally scoped to safe release material:

- model cards and release notes
- benchmark summaries and audited result metadata
- high-level recipes for using released GGUFs
- public CLI surface documentation
- release artifact hashes, runtime flags, and links

## Released Models

Public Cerebellum GGUF releases and their reproducibility notes:

| Model | Base | Size | Recipe / Benchmarks |
|---|---|---|---|
| [KAT-Coder-V2.5-Dev-Cerebellum-GGUF](https://huggingface.co/deucebucket/KAT-Coder-V2.5-Dev-Cerebellum-GGUF) | Kwaipilot/KAT-Coder-V2.5-Dev (Qwen3.6-35B-A3B) | 12.1 GB | [docs/kat_coder_v25_cerebellum.md](docs/kat_coder_v25_cerebellum.md) |

The private development repository contains the factory: pipeline automation,
tensor-selection internals, raw ablation traces, experimental scripts, and
working notes.

## Releases — Heretic Cerebellum series (2026-06-11)

Cerebellum mixed-precision quants of community Heretic (abliterated) variants.
Each repo includes the model card, full audited per-question benchmark
artifacts, and adversarial audit reports. All wrong answers were individually
verified as genuine model errors before publication.

| Release | Size | Highlights |
|---|---|---|
| [Qwen3.6-35B-A3B-Heretic-Cerebellum-GGUF](https://huggingface.co/deucebucket/Qwen3.6-35B-A3B-Heretic-Cerebellum-GGUF) | 11.96 GB | ARC 95.48, HellaSwag 91.78, MMLU-Redux 75.42, HumanEval+ 64.63, vision 24/24. Head-to-head: beats the 16.87 GB uniform Q3_K_M of the same weights on PPL, MMLU and HumanEval+ at 29% smaller. |
| [Qwen3.6-27B-Heretic-Cerebellum-GGUF](https://huggingface.co/deucebucket/Qwen3.6-27B-Heretic-Cerebellum-GGUF) | 12.87 GB | ARC 96.93, MMLU-Redux 76.21, HumanEval+ 84.76 (chat no-think), vision 24/24, RealWorldQA 78%. |
| [Gemma-4-E4B-it-Heretic-Cerebellum-GGUF](https://huggingface.co/deucebucket/Gemma-4-E4B-it-Heretic-Cerebellum-GGUF) | 4.51 GB | Beats stock Cerebellum v2 on ARC (87.37), MMLU-Redux (58.63) and HumanEval (70.12) at identical size. |

A fourth candidate (Gemma 4 E2B heretic) failed release gates — the source
abliteration measurably damaged the base model (code benchmarks -31 pts at
full precision) — and was not published. Methodology note: heretic transfers
are screened by the source's reported KL divergence; heavy ablations
(KL >~ 0.05) require full-precision screening before quantization.

Org collections: [DB-Cerebellum](https://huggingface.co/DB-Cerebellum).

How the method works, at the public level: [docs/how_cerebellum_works.md](docs/how_cerebellum_works.md).

## Benchmark Evidence

The `benchmarks/` directory holds the published per-model evidence: summary
JSONs and per-question detailed answer files for every benchmark number in the
model cards. `docs/benchmark_protocol.md` documents the exact harness rules,
runtime flags, and audit requirements a result must clear before it is
published. Anyone with the model file and these documents can reproduce or
dispute any published score. That is the point.
