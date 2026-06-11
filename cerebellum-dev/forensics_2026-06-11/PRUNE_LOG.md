# Prune Log — 2026-06-11

Executed by orchestrator after agent inventory + review. ai-drive: 33G → 118G free. games: 195G → 369G free (~259GB total). Small text evidence from deleted dirs archived first in `scratch_evidence.tar.zst` (81K). Server logs compressed (zstd), not deleted.

## Deleted — high confidence
| Path | Size | Reason |
|---|---|---|
| `.cerebellum-scratch/` (4 trial dirs) | 55.7G | Dead June 4–6 hill-climber gemma4-12b trial runs; method deprecated; small files archived |
| `osmosis-gemma4-26b/.cache/.../*.incomplete` | 10.4G | Abandoned interrupted HF download (27 days stale) |
| `osmosis-gemma4-26b/gemma-4-26B-A4B-it-cerebellum-v6.gguf` | 10.9G | Byte-identical dup of games/models copy (verified with cmp before rm) |
| `ai-drive/models/Gemma-4-26B-A4B-it-Heretic/...heretic-cerebellum-v1.gguf` | 10.9G | Byte-identical dup of games/models copy (cmp-verified); mmproj kept |
| `games/cerebellum-pipeline-tmp/qwen3-0.6b-...-smoke-abort.../` | 0.7G | Aborted smoke run |
| `games/cerebellum-runs/.../gemma4-12b-targeted-attnv.../current_baseline.gguf` | 7.8G | Stale hill-climber checkpoint |
| `games/cerebellum-runs/.../01-q2_K.gguf.tmp` | 6.3G | Orphaned .tmp |
| `unsloth_compiled_cache/`, `__pycache__`, empty `cerebellum-osmosis-pipeline-tmp/` | <2M | Regenerable |

## Deleted — judged (re-downloadable sources of completed experiments)
| Path | Size | Reason |
|---|---|---|
| `games/qwen3-32b/` (safetensors) | 62G | osmosis-qwen3-32b complete; re-downloadable |
| `games/qwen3.6-27b/` (safetensors) | 52G | osmosis-qwen36-27b complete; re-downloadable |
| `games/qwen36-35b-v2/source_heretic/` | 65G | Source of failed June 3 MTP build; re-downloadable |
| `games/cerebellum-pipeline-tmp/gemma4-12b/text-model/` | 22.2G | Derived duplicate of source safetensors |
| `games/qwen36-35b-v2/Qwen3.5-35B-A3B-Heretic-Cerebellum-v3.gguf` | 11.1G | Failed June 3 heretic build (regressed −14 HS / −32 HE+) |
| `games/cerebellum-heretic-qwen36-35b/Qwen3.6-35B-A3B-Heretic-Cerebellum.gguf` | 12.1G | Failed June 3 MTP-preserved heretic build |

## Compressed (kept)
~4.5G of server stdout logs across osmosis-*/cerebellum-* dirs → *.log.zst (results JSONLs untouched).

## Explicitly NOT touched
games/models/ + pms-models/ (model collections), ai-suite/ (284G, separate hobby), conch-poc (active experiment), both venvs, gemma4-12b source/F16/v1 GGUF, MoE v0/v1 experiments, ~91G image-gen models in ~/.cache/huggingface (possible ComfyUI duplicates — flagged for user, not acted on).
