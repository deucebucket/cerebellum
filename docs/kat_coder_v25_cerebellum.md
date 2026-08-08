# KAT-Coder-V2.5-Dev Cerebellum

Sensitivity-guided mixed-precision quantization of
[Kwaipilot/KAT-Coder-V2.5-Dev](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev),
a fine-tune of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).

- **HF model repo:** [deucebucket/KAT-Coder-V2.5-Dev-Cerebellum-GGUF](https://huggingface.co/deucebucket/KAT-Coder-V2.5-Dev-Cerebellum-GGUF)
- **GGUF file:** `KAT-Coder-V2.5-Dev-Cerebellum-14GB-v2.gguf`
- **Size:** 12.1 GB (2.93 BPW)
- **Architecture:** Qwen3.6 35B-A3B MoE (35B total / ~3B active)

## Why this model

User feedback on earlier Qwen3.6 Cerebellum quants pointed to two issues:
high thinking-token output and weaker coding/agentic performance relative to
fine-tuned coding variants. KAT-Coder-V2.5-Dev addresses both, so it was a
natural Cerebellum target.

## Quant recipe

1. **Source:** Kwaipilot/KAT-Coder-V2.5-Dev BF16 merged GGUF.
2. **Imatrix:** KAT-specific lite coder imatrix built from HumanEval+ / MBPP+
   samples.
3. **Base quant:** Q3_K_M.
4. **Key override:** promote late-layer expert `ffn_down` weights from Q2_K to
   Q3_K. The override file is in
   [`benchmarks/kat-coder-v25-dev/tensor_types_v2.txt`](../benchmarks/kat-coder-v25-dev/tensor_types_v2.txt).
5. **Build command:**

```bash
llama-quantize \
  --imatrix kat_coder_coder_imatrix_lite.dat \
  --tensor-type-file tensor_types_v2.txt \
  KAT-Coder-V2.5-Dev-bf16.gguf \
  KAT-Coder-V2.5-Dev-Cerebellum-14GB-v2.gguf \
  Q3_K_M
```

## Benchmarks

Measured on an RTX 3090 with `llama-server -ngl 999 --flash-attn on --reasoning off --reasoning-budget 0`.

| Benchmark | Score |
|---|---:|
| ARC-Challenge | 95.56% |
| HellaSwag | 90.34% |
| MMLU-Redux | 74.92% |
| HumanEval+ chat base | 92.07% |
| HumanEval+ chat plus | 89.02% |
| BigCodeBench hard | 28.05% |

Full summary and detailed JSONL artifacts are in
[`benchmarks/kat-coder-v25-dev/`](../benchmarks/kat-coder-v25-dev/).

## Runtime stats

| Scenario | TPS |
|---|---:|
| Single-request 1K context | ~78 tok/s |
| Single-request 8K context | ~83 tok/s |
| Batched 6K decode | ~158 tok/s |

## Notes

- Text-only. Vision was not tested.
- Native context is 262,144 tokens; practical daily-driver range on a 24 GB card
  is 24K–65K.
- The 14 GB v2 outperformed both the 16 GB maxx-v4 experiment and the 24 GB
  pure-Q5_K_M baseline on coding benchmarks, confirming that the coding-specific
  imatrix + targeted late-layer promotion extracts more performance per gigabyte
  than simply raising the base quant.

## License

Apache-2.0, matching the base model.
