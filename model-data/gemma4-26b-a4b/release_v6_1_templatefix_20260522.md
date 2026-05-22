# Gemma 4 26B-A4B Cerebellum v6.1 Templatefix

Release date: 2026-05-22

## Files

```text
gemma-4-26B-A4B-it-cerebellum-v6.1-templatefix.gguf
sha256: d24229facdef8360a7ffa8b37a50e1de636b9139a5eba0efe899828e45ae7989

gemma-4-26b-a4b-it.mmproj.gguf
sha256: b762c43119ebdc3e3c36d929d958e827fac35b03278dda9203f87131aee1f185
```

The v6.1 file keeps the v6 tensor allocation and updates GGUF/runtime-facing
metadata for Gemma 4 chat-template use.

## Tensor Map

The release tensor map is:

```text
model-data/gemma4-26b-a4b/cerebellum_v6_overrides.txt
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
  --model gemma-4-26B-A4B-it-cerebellum-v6.1-templatefix.gguf \
  --mmproj gemma-4-26b-a4b-it.mmproj.gguf \
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

Creative-writing smoke artifacts:

```text
benchmarks/gemma4-26b-a4b/templatefix_20260522/creative_eval/
```

Non-coding tool-use artifacts:

```text
benchmarks/gemma4-26b-a4b/templatefix_20260522/agentic_eval/
```

Historical v6 benchmark artifacts remain in:

```text
benchmarks/gemma4-26b-a4b/
```

The v6 benchmark artifacts are retained as historical same-line measurements.
The v6.1 templatefix artifacts are runtime/template/tool/creative checks, not a
fresh full benchmark suite.
