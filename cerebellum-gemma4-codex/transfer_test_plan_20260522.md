# Gemma 4 26B Codex Transfer Test

Date: 2026-05-22

## Candidate

Repository: `dwojcik/gemma4-26b-a4b-it-codex-gguf-4bit`

Reason selected:

- Same base architecture claimed on the card: `google/gemma-4-26B-A4B-it`.
- Coding-focused finetune, card says Evol-Instruct-Code-80k-v1 via Unsloth QLoRA.
- GGUF file is available, so it can be tested locally without first rebuilding a
  merged HF checkpoint.

Caveat:

- The card claims coding optimization, but does not publish hard benchmark
  numbers. Treat this as a candidate, not a proven high-result model, until our
  local before/after benchmarks run.
- Available source appears to be Q4_K_M GGUF, not a clean F16/BF16 full text
  model. Applying the v6 tensor map to this file is a transfer/requantization
  experiment and may include double-quantization damage. A later clean test
  should use a merged F16/BF16 source if one is found.

## Transfer Test

1. Download Q4_K_M GGUF and mmproj to game drive.
2. Validate tensor-name compatibility with existing Gemma 4 26B v6 tensor map:
   `osmosis-gemma4-26b/cerebellum_v6_overrides.txt`.
3. Build a transferred Cerebellum GGUF with the same tensor map.
4. Benchmark against the downloaded Q4_K_M source, not against unrelated base
   models:
   - WikiText-2 PPL
   - EvalPlus/HumanEval chat harness if Gemma 4 raw completions are bad
   - optional BigCodeBench/MBPP if runtime allows
   - ARC/HellaSwag/MMLU-Redux only as general-regression checks

## Build Artifact

Build command:

```bash
distrobox enter ai -- /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --allow-requantize \
  --tensor-type-file /var/home/deucebucket/games/cerebellum-gemma4-codex/cerebellum_v6_transfer_overrides.txt \
  /var/home/deucebucket/games/models/gemma4-26b-codex-dwojcik/gemma-4-26B-A4B-it.Q4_K_M.gguf \
  /var/home/deucebucket/games/cerebellum-gemma4-codex/gemma4-26b-codex-cerebellum-v6-transfer-requant.gguf \
  Q4_K_M
```

Files:

- Baseline GGUF: `/var/home/deucebucket/games/models/gemma4-26b-codex-dwojcik/gemma-4-26B-A4B-it.Q4_K_M.gguf`
  - Size: 16G
  - SHA256: `db82e6e25e327238c7be8034365f29e12711d8324b3825867c7ad763964bafca`
- Vision projector: `/var/home/deucebucket/games/models/gemma4-26b-codex-dwojcik/gemma-4-26B-A4B-it.BF16-mmproj.gguf`
  - Size: 1.2G
  - SHA256: `5d6e62072f7890a0b6572da3acb9096a8a2903b3bf1caebdc484b4d6f8923a1b`
- Transferred Cerebellum requant: `/var/home/deucebucket/games/cerebellum-gemma4-codex/gemma4-26b-codex-cerebellum-v6-transfer-requant.gguf`
  - Size: 13G
  - SHA256: `4df7d2a9de0dc6c4359ae807685011b95894f53d77d374ff54b007ee7990b0cf`
  - Quantizer estimate: 12553.37 MiB, 4.17 BPW
  - Warning: 60 of 658 tensors used fallback quantization.

Important interpretation note: `--allow-requantize` was required because the
available coder source is already Q4_K_M. This artifact is useful for transfer
testing and stack testing, but it is not a clean final quant. A clean final test
needs a merged F16/BF16 source checkpoint.

HF source check:

- The repo advertises a BF16 variant in generated HF usage snippets, but the
  actual file listing only includes
  `gemma-4-26B-A4B-it.BF16-00002-of-00002.gguf` at about 582 MB plus the
  mmproj. Shard `00001-of-00002` is missing, so this is not a usable full BF16
  text source.
- Correct next experiment: find the original LoRA adapter or a merged
  F16/BF16/safetensors checkpoint for the coder finetune, then quantize from
  that clean source with the v6 tensor map. The current Q4_K_M -> Cerebellum
  artifact is a transfer/requant probe only.

Same-harness PPL panel:

| Domain | Coder Q4_K_M | Coder v6 transfer requant | Regular Gemma 4 v6.1 |
|--------|--------------|---------------------------|-----------------------|
| wiki | 9.8655 | 15.2815 | 9038.9341 |
| code | 3.2257 | 3.9142 | 79.4824 |
| math | 2.6571 | 2.9774 | 10.0182 |
| dialogue | 3.5457 | 4.2967 | 154.1396 |
| agent | 1.9791 | 2.2471 | 11.1748 |

The regular v6.1 PPL control is not a clean rank against the coder GGUFs; use
its published task benchmark artifacts as the stronger comparison for regular
v6.1. The coder-vs-coder rows are the meaningful same-source PPL comparison.

## Expected Interpretation

This tests whether the regular Gemma 4 26B Cerebellum tensor map transfers to a
matching coding finetune. It does not prove the tensor map is optimal for code;
for that, run code-domain ablations later.

## Full Stack Benchmark Matrix

Run both models through the same llama.cpp stack and harness settings.

### Models

Baseline:

- Source file: downloaded `dwojcik/gemma4-26b-a4b-it-codex-gguf-4bit/gemma-4-26B-A4B-it.Q4_K_M.gguf`
- Role: upstream coder Q4_K_M baseline.

Candidate:

- Source: same downloaded coder GGUF if no merged F16/BF16 source is available.
- Tensor map: `osmosis-gemma4-26b/cerebellum_v6_overrides.txt`.
- Role: transferred Gemma 4 26B Cerebellum tensor map applied to coder finetune.

### Server Stack

Use the patched Gemma 4 llama.cpp server:

```bash
llama-server \
  --model <gguf> \
  --mmproj <mmproj> \
  --n-gpu-layers 99 \
  --ctx-size 24576 \
  --parallel 4 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja \
  --reasoning auto \
  --media-path /tmp/ \
  --alias <alias>
```

Carl runtime note: Carl's idle Heretic Gemma 4 server is launched as a child of
`carl.service` on port 7809 and occupies about 15G VRAM. Stop `carl.service`
only while running this benchmark stack, then restart it after the benchmark.

For EvalPlus/HumanEval chat harness, use request-level no-thinking controls:

```json
{
  "chat_template_kwargs": {"enable_thinking": false},
  "thinking_budget_tokens": 0
}
```

### Benchmarks

Minimum before/after:

1. WikiText-2 PPL with `llama-perplexity`.
2. EvalPlus/HumanEval chat harness for Gemma 4.
3. BigCodeBench or MBPP if available locally.
4. ARC/HellaSwag/MMLU-Redux as general regression checks.
5. Optional opencode/agentic coding smoke only after direct benchmarks finish.

### Publishable Result

Record:

- source repo and file hashes,
- baseline Q4_K_M benchmark artifacts,
- transferred Cerebellum GGUF hash,
- transferred benchmark artifacts,
- tensor map used,
- exact server flags.
