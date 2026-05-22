# Gemma 4 26B-A4B Heretic Cerebellum v1.1 Templatefix

Release date: 2026-05-22

## Files

```text
gemma-4-26B-A4B-it-heretic-cerebellum-v1.1-templatefix.gguf
sha256: 103f973317a0daa2d59f94559c64ae7925257606b8c105c9dbdc8996a86310b1

gemma-4-26B-A4B-it-heretic.mmproj-f16.gguf
sha256: ed0e132c0ec1c97437e7eef555f654fd17ee0c090fbd1ffeea54e30402680476
```

The v1.1 file keeps the existing Heretic Cerebellum tensor allocation and
updates GGUF/runtime-facing metadata for Gemma 4 chat-template use.

## Tensor Map

The release tensor map is:

```text
model-data/gemma4-26b-heretic/cerebellum_v1_overrides.txt
```

## Runtime Used For Templatefix Checks

```text
llama.cpp fork: https://github.com/deucebucket/llama.cpp
branch: cerebellum/gemma4-runtime-fixes
fork commit: ded491334 fix: harden Gemma 4 server budgets
base build: b8930-59fa0b455
```

Server shape used locally:

```bash
llama-server \
  --model gemma-4-26B-A4B-it-heretic-cerebellum-v1.1-templatefix.gguf \
  --mmproj gemma-4-26B-A4B-it-heretic.mmproj-f16.gguf \
  --n-gpu-layers 99 \
  --ctx-size 65536 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja \
  --reasoning auto \
  --media-path /tmp/
```

No-thinking request controls used:

```json
{
  "chat_template_kwargs": {"enable_thinking": false},
  "thinking_budget_tokens": 0
}
```

Bounded-thinking smoke requests used `thinking_budget_tokens: 128`.

## Public Artifacts

Historical v1 benchmark artifacts:

```text
benchmarks/gemma4-26b-heretic-v1/
```

Creative-writing smoke artifacts:

```text
benchmarks/gemma4-26b-heretic-v1/templatefix_20260522/creative_eval/
```

Non-coding tool-use artifacts:

```text
benchmarks/gemma4-26b-heretic-v1/templatefix_20260522/agentic_eval/
```

The v1 benchmark artifacts are retained as historical same-line measurements.
The v1.1 templatefix artifacts are runtime/template/tool/creative checks, not a
fresh full benchmark suite.
