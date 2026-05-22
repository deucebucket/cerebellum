# Gemma 4 Reasoning and Vision Hard Notes - 2026-05-21

Purpose: preserve the concrete runtime fix for Gemma 4 thinking loops/token burn,
plus the verified path for image input through GGUF + mmproj.

## Bottom Line

Do not treat the fix as "disable reasoning." Reasoning works in llama.cpp when
the server is launched so per-request budgets can actually take effect.

Correct app/agent server shape:

```bash
distrobox enter ai -- /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-server \
  --model <gemma4-text.gguf> \
  --mmproj <gemma4-mmproj.gguf> \
  --host 127.0.0.1 --port <port> \
  --n-gpu-layers 99 \
  --ctx-size 131072 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --reasoning auto \
  --media-path /tmp/ \
  --alias <alias> \
  --no-warmup
```

Important: omit `--reasoning-budget` for app/agent servers that send
`thinking_budget_tokens` per request. In this llama.cpp build, request-level
`thinking_budget_tokens` is only honored when the server-level reasoning budget
is the default/unrestricted value. A fixed server budget such as
`--reasoning-budget 4096` makes the request budget ineffective.

## Verified Runtime

llama.cpp checkout:

```text
/var/home/deucebucket/ai-drive/llama.cpp
branch: pr/22340-gg
HEAD: 59fa0b455
build: b8930-59fa0b455
commit message: common : do not pass prompt tokens to reasoning budget sampler (#22488)
```

This is the PR-specific fix we kept re-discovering. Keep this pinned in release
notes and test logs when comparing older Gemma 4 behavior.

Relevant upstream/runtime issue trail:

- llama.cpp PR #21697: Gemma 4 reasoning-budget sampler support.
- llama.cpp issue #21902: Gemma 4 reasoning-budget issue.
- llama.cpp PR #22488: do not pass prompt tokens to reasoning budget sampler.
- llama.cpp issue #21338: reports around inability to disable thinking.
- llama.cpp issue #21375: PEG parser/tool-call repetition failure mode.
- llama.cpp issue #21825: outdated Gemma 4 chat template warning with mmproj.
- Google/HF Gemma 4 template PR #36: SI and tool-call handling update.
- Google/HF Gemma 4 template PR #38: multimodal placeholders in tool response
  content parts.

## Reasoning Budget Finding

Bad launch for per-request budget tests:

```bash
--jinja --reasoning auto --reasoning-budget 4096
```

Observed result with request `thinking_budget_tokens=128`, `max_tokens=700`:

```text
finish_reason: length
content_len: 0
reasoning_len: 3097
completion_tokens: 700
```

Same bad server with `max_tokens=4600` eventually produced final content, but
only after consuming the large fixed server budget. That behavior looks like a
closed thinking loop when the app caps output too low.

Good launch for app/agent work:

```bash
--jinja --reasoning auto
```

Observed result with request `thinking_budget_tokens=128`, `max_tokens=700`:

```text
finish_reason: stop
content_len: 961
reasoning_len: 476
completion_tokens: 327
server log: reasoning-budget activated, budget=128 tokens
server log: budget exhausted, forcing end sequence
```

Clean visible-answer prompt also passed:

```text
finish_reason: stop
content_len: 768
reasoning_len: 498
completion_tokens: 287
```

Rule: `max_tokens` must have headroom for both hidden reasoning and the visible
answer. If `max_tokens <= reasoning budget + expected final answer`, a correct
runtime can still return empty or truncated visible content.

## Request Pattern

Thinking enabled with controlled budget:

```json
{
  "chat_template_kwargs": {"enable_thinking": true},
  "thinking_budget_tokens": 128,
  "max_tokens": 700,
  "temperature": 0.2
}
```

No-thinking/writing mode:

```json
{
  "chat_template_kwargs": {"enable_thinking": false},
  "thinking_budget_tokens": 0,
  "max_tokens": 700
}
```

Do not rely on model metadata alone for app behavior. OpenAI-compatible clients
must pass the runtime thinking controls when the server supports them.

## Vision / Image Input

Verified Heretic v1.1 templatefix with mmproj:

```text
text model:
/var/home/deucebucket/games/models/Gemma-4-26B-A4B-it-Heretic/gemma-4-26B-A4B-it-heretic-cerebellum-v1.1-templatefix.gguf

mmproj:
/var/home/deucebucket/games/models/Gemma-4-26B-A4B-it-Heretic/gemma-4-26B-A4B-it-heretic.mmproj-f16.gguf
```

Server startup confirmed:

```text
loaded multimodal model
has vision encoder
projector: gemma4v
/v1/models capabilities: completion, multimodal
```

The local-image route requires `--media-path`. With `--media-path /tmp/`, this
OpenAI-compatible request passed:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What shapes and colors are in this image?"},
        {"type": "image_url", "image_url": {"url": "file://gemma4_vision_smoke.png"}}
      ]
    }
  ],
  "chat_template_kwargs": {"enable_thinking": true},
  "thinking_budget_tokens": 128,
  "max_tokens": 450,
  "temperature": 0.2
}
```

Observed result:

```text
content: The image contains a black square and a red circle.
reasoning_content present: 413 chars
finish_reason: stop
prompt_tokens: 323
completion_tokens: 143
```

This verifies image input and parsed reasoning together.

The same mmproj/no-fixed-budget server also passed the text context smoke:

```bash
python3 scripts/gemma4_context_smoke.py \
  --base-url http://127.0.0.1:7830 \
  --model gemma4-26b-cerebellum-v6.1-templatefix-vision \
  --words 1200 \
  --max-tokens 512 \
  --thinking-budget 128
```

Result:

```text
no-thinking: finish_reason=stop, content=LONG_CONTEXT_OK, reasoning_len=0
thinking: finish_reason=stop, content=LONG_CONTEXT_OK, reasoning_len=431
bad_hits: none
```

## Vision Traps

The current llama.cpp build cannot fetch HTTPS image URLs:

```text
HTTPS is not supported. Rebuild with one of:
-DLLAMA_BUILD_BORINGSSL=ON
-DLLAMA_BUILD_LIBRESSL=ON
-DLLAMA_OPENSSL=ON
```

Do not mistake this for a model/mmproj failure.

Update from 2026-05-22: this was fixed in the local Cerebellum llama.cpp fork
branch by installing OpenSSL development files in the `ai` distrobox and
rebuilding `llama-server` with `-DLLAMA_OPENSSL=ON`.

```text
fork repo: https://github.com/deucebucket/llama.cpp
branch: cerebellum/gemma4-runtime-fixes
base commit: 59fa0b455
build: b8930-59fa0b455
runtime doc: docs/gemma4-cerebellum-server.md
```

The fork also changes llama-server request parsing so an explicit
`thinking_budget_tokens` request overrides a stale fixed server
`--reasoning-budget`. This was tested by launching with the intentionally bad
`--reasoning-budget 4096` and sending request budgets of `0` and `128`; server
logs showed `reasoning-budget: activated, budget=0 tokens` and
`reasoning-budget: activated, budget=128 tokens`.

Verified on the fork build:

```text
no-thinking request: finish=stop, content_len=3, reasoning_len=0
thinking request: finish=stop, content_len=13, reasoning_len=257
local image request: finish=stop, content_len=63, reasoning_len=423
HTTPS image request: finish=stop, content_len=54, reasoning_len=481
HTTPS server log: downloaded 91814 bytes
```

The earlier data URI test failed because `/tmp/gemma4_vision_example.jpg` was
actually an HTML document, not an image. Reuse a known-good generated PNG/JPEG
for smoke tests.

Local `file://` media is intentionally blocked unless the server is launched
with `--media-path`. The URL path is relative to that directory. Example:

```text
--media-path /tmp/
image_url.url = file://gemma4_vision_smoke.png
```

## Release Gates

Before claiming fixed Gemma 4 GGUFs on Hugging Face:

1. Template probe passes against embedded GGUF chat template.
2. No-thinking direct chat returns visible `content`.
3. Thinking-enabled direct chat returns bounded `reasoning_content` plus final
   visible `content`.
4. Long-context smoke passes at 24k-32k prompt tokens with no marker leakage,
   no `enough;`, and no length-only hidden-reasoning burn.
5. Tool-call smoke returns valid parsed tool calls.
6. Opencode run uses default format + `--thinking` when human-visible reasoning
   proof is needed.
7. Opencode app/agent server omits fixed `--reasoning-budget` unless the whole
   test is explicitly about server-fixed budgets.
8. Vision claim requires mmproj load plus one passed image request.

## HF/User Packaging Rule

Do not bake Scrithub, Carl, opencode, or any local app policy into public GGUFs.
Ship upstream-compatible Gemma 4 template metadata and document runtime flags
separately:

- `--jinja --reasoning auto` for capable servers.
- omit fixed `--reasoning-budget` when apps send per-request budgets.
- use `chat_template_kwargs.enable_thinking` and `thinking_budget_tokens` per
  request.
- use `--mmproj` for vision/audio-capable Gemma 4 variants.
- use `--media-path` for local file image tests, or rebuild llama.cpp with TLS
  support for HTTPS images.
