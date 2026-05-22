# Devlog 2026-05-22 — Gemma 4 26B Codex Experiments

## Summary

Three concurrent experiments on Gemma 4 26B: coding LoRA merge + Cerebellum quantization, codex-aware tensor map, and MTP speculative decoding. Two successes, one dead end.

## Experiment 1: Python LoRA Merge + Cerebellum Quant (v7)

Source: `google/gemma-4-26B-A4B-it` BF16 + `hotdogs/gemma4-26b-python-18k-alpaca-lora` (227 MB LoRA)

- Built streaming per-tensor LoRA merge (~300 MB RAM, never loads full 52 GB model)
- 394/394 LoRA tensors merged in 1.7 min
- Converted to F16 GGUF (50.5 GB, 3 min)
- Cerebellum v6 tensor map applied → 10.1 GB, 3.37 BPW

**Result: 40.24% EvalPlus base / 36.59% plus** — catastrophic. The Q2_K quantization crushed the LoRA deltas.

## Experiment 2: Codex-Aware Tensor Map (v7b)

Key insight: the LoRA targets EXACTLY the tensors v6 demotes to Q2_K. The LoRA is a free ablation — it tells us which tensors are code-critical.

- Built codex-aware override map: 192 LoRA-target tensors promoted from Q2_K to Q4_0 minimum
- Re-quantized same merged F16 → 10.4 GB, 3.45 BPW

**Result: 74.39% EvalPlus base / 69.51% plus** — 34 percentage point recovery.
ARC: 81.66%, HellaSwag: 44.09%, MMLU Redux: 56.54%

| Model | Size | EvalPlus Base | Plus |
|-------|------|---------------|------|
| v7 (Q2_K all) | 10.1 GB | 40.24% | 36.59% |
| v7b (codex-aware) | 10.4 GB | 74.39% | 69.51% |
| dwojcik codex Q4_K_M | 16 GB | 78.66% | 75.00% |

v7b is 35% smaller than dwojcik codex, within 4 points. The methodology is proven: LoRA targets map directly to code-critical tensors for Cerebellum precision allocation.

## Experiment 3: MTP Speculative Decoding (DEAD END)

- Google ships official MTP drafter: `google/gemma-4-26B-A4B-it-assistant` (420M params, 800 MB F16)
- Built ik_llama.cpp with PR #1744 (Gemma 4 MTP support, merged 2026-05-10)
- Loaded v6 Cerebellum GGUF + assistant on RTX 3090

**Result: 141 t/s with MTP vs 160 t/s baseline — MTP is 12% SLOWER.**

The assistant was trained against BF16 distribution. Our Q2_K-quantized v6 diverges too far — 55% acceptance rate means the assistant costs more GPU cycles to verify/reject than just generating tokens directly. Published benchmarks show 85%+ acceptance with Q8_0 target + Q8_0 drafter, but that's 25 GB — defeats the purpose.

Distillation (training assistant against Q2_K distribution) is blocked: the assistant architecture takes backbone hidden states as input, not raw tokens. Requires C++ integration with the GGUF backend.

## MTP Files (ik_llama.cpp + Patches)

- Cloned: `/var/home/deucebucket/games/llamacpp-gemma4-mtp/build/ik_llama.cpp`
- Patches: `patches/0001-PR-1744-gemma4-mtp.patch`
- Built binaries in distrobox `ai`
- Assistant GGUF: `/var/home/deucebucket/games/models/gemma4-26b-assistant/gemma4-26b-assistant.gguf` (839 MB F16)
- Assist GPT safetensors: `/var/home/deucebucket/games/models/gemma4-26b-assistant/`
- Distillation scripts: `/var/home/deucebucket/games/models/distill_v2.py`, `gen_teacher.py`, `train_student.py`

## v6 Transfer Experiment (Earlier, Reflected)

Prior transfer requant test (Q4_K_M codex → Cerebellum v6 tensor map, `--allow-requantize`) was a probe:
- 60/658 tensors used fallback — double-quantization damage
- EvalPlus 62.20% / 58.54% (vs Q4_K_M baseline 78.66% / 75.00%)
- Confirmed: `--allow-requantize` is always destructive. Clean F16 source is mandatory.

## Key Artifacts on Disk

```
/var/home/deucebucket/games/models/gemma4-26b-codex-cerebellum/
  gemma4-26b-codex-cerebellum-v7.gguf              10.1 GB (v7, broken Q2_K all)
  gemma4-26b-codex-cerebellum-codexaware-v7b.gguf  10.4 GB (v7b, codex-aware)
  codex_aware_v7_overrides.txt                      235 overrides (192 promoted)

/var/home/deucebucket/games/models/gemma4-26b-codex-merged/    (merged F16 HF model)
/var/home/deucebucket/games/models/gemma4-26b-python-lora/     (hotdogs LoRA adapter)
/var/home/deucebucket/games/models/gemma4-26b-base/            (base BF16 instruct model)
/var/home/deucebucket/games/models/gemma4-26b-assistant/       (MTP assistant)
```

## Next Steps (If Revisited)

1. Run v6 through EvalPlus chat harness for proper baseline comparison
2. Pivot to better coding finetunes (dwojcik adapter if obtainable)
3. MTP: revisit if someone ships a Q2_K-trained drafter, or build C++ distillation
4. Heretic v1 at 92% EvalPlus is the strongest coder we have — submit agentic coding tasks to validate real-world performance
