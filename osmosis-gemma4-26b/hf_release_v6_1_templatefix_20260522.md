# Gemma 4 26B v6.1 Templatefix HF Release Checkpoint - 2026-05-22

## Verdict

Regular v6.1 templatefix is ready to package for Hugging Face as a fixed-template
GGUF release with documented runtime requirements and tested non-coding agentic
behavior. The supported agentic claim is assistant-style tool automation, not
coding-agent performance.

Do not claim that opencode coding-agent behavior is clean. The current proof is
for direct API runtime, creative writing, vision, reasoning controls, and
non-coding tool automation.

Heretic v1.1 templatefix also passed the same creative and strict non-coding
agentic gates. Treat it as a separate release/card with its own positioning.

## Files

Regular:

```text
/var/home/deucebucket/games/models/Gemma-4-26B-A4B-Cerebellum-v6/gemma-4-26B-A4B-it-cerebellum-v6.1-templatefix.gguf
sha256: d24229facdef8360a7ffa8b37a50e1de636b9139a5eba0efe899828e45ae7989

/var/home/deucebucket/games/models/Gemma-4-26B-A4B-Cerebellum-v6/gemma-4-26b-a4b-it.mmproj.gguf
sha256: b762c43119ebdc3e3c36d929d958e827fac35b03278dda9203f87131aee1f185
```

Heretic:

```text
/var/home/deucebucket/games/models/Gemma-4-26B-A4B-it-Heretic/gemma-4-26B-A4B-it-heretic-cerebellum-v1.1-templatefix.gguf
sha256: 103f973317a0daa2d59f94559c64ae7925257606b8c105c9dbdc8996a86310b1

/var/home/deucebucket/games/models/Gemma-4-26B-A4B-it-Heretic/gemma-4-26B-A4B-it-heretic.mmproj-f16.gguf
sha256: ed0e132c0ec1c97437e7eef555f654fd17ee0c090fbd1ffeea54e30402680476
```

## Runtime

Use the patched llama.cpp fork for best Gemma 4 behavior:

```text
repo: https://github.com/deucebucket/llama.cpp
branch: cerebellum/gemma4-runtime-fixes
commit: ded491334 fix: harden Gemma 4 server budgets
base build: b8930-59fa0b455
```

Server shape:

```bash
distrobox enter ai -- /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-server \
  --model <text-gguf> \
  --mmproj <mmproj-gguf> \
  --host 127.0.0.1 --port 7830 \
  --n-gpu-layers 99 \
  --ctx-size 65536 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --reasoning auto \
  --media-path /tmp/ \
  --alias <model-alias> \
  --no-warmup
```

For app/agent servers, prefer request-level thinking controls:

```json
{
  "chat_template_kwargs": {"enable_thinking": false},
  "thinking_budget_tokens": 0
}
```

For bounded thinking:

```json
{
  "thinking_budget_tokens": 128
}
```

Avoid publishing examples that require a fixed server-level
`--reasoning-budget`. The fork allows request budgets to override stale fixed
server defaults, but the cleaner release guidance is still request-level
control.

## Test Artifacts

Runtime/reasoning/vision notes:

```text
docs/gemma4_reasoning_vision_hard_notes_20260521.md
docs/gemma4_template_refresh_release_checkpoint_20260521.md
```

Creative writing:

```text
osmosis-gemma4-26b/creative_eval_20260522/regular_v6_1_templatefix_creative.jsonl
osmosis-gemma4-26b/creative_eval_20260522/regular_v6_1_templatefix_creative_summary.json
osmosis-gemma4-26b/creative_eval_20260522/regular_v6_1_templatefix_creative_rerun_longcaps.jsonl
osmosis-gemma4-26b/creative_eval_20260522/regular_v6_1_templatefix_creative_rerun_longcaps_summary.json
osmosis-gemma4-26b/creative_eval_20260522/heretic_v1_1_templatefix_creative.jsonl
osmosis-gemma4-26b/creative_eval_20260522/heretic_v1_1_templatefix_creative_summary.json
```

Non-coding agentic tool use:

```text
osmosis-gemma4-26b/agentic_eval_20260522/README.md
osmosis-gemma4-26b/agentic_eval_20260522/regular_v6_1_noncoding_agentic_tools_strict.jsonl
osmosis-gemma4-26b/agentic_eval_20260522/regular_v6_1_noncoding_agentic_tools_strict_summary.json
osmosis-gemma4-26b/agentic_eval_20260522/heretic_v1_1_noncoding_agentic_tools.jsonl
osmosis-gemma4-26b/agentic_eval_20260522/heretic_v1_1_noncoding_agentic_tools_adjudicated_summary.json
osmosis-gemma4-26b/agentic_eval_20260522/heretic_v1_1_noncoding_agentic_tools_strict_retry.jsonl
osmosis-gemma4-26b/agentic_eval_20260522/heretic_v1_1_noncoding_agentic_tools_strict_retry_summary.json
```

## Creative Results

Regular v6.1:

- Six prompt suite: microfiction, dialogue subtext, noir style control,
  worldbuilding, long continuation, non-explicit adult romance.
- Initial run had four `finish_reason=length` cases due low output caps.
- Long-cap rerun: 4/4 previously truncated cases stopped cleanly.
- No hidden reasoning in no-thinking mode.
- No bad markers: no `<think>`, no template leakage, no waiting-loop marker.
- Style-control prompt passed sentence-start constraint.

Heretic v1.1:

- Same six prompt suite.
- 6/6 stopped cleanly.
- No hidden reasoning in no-thinking mode.
- No bad markers.
- Low repetition by mechanical checks.

## Non-Coding Agentic Results

Harness: OpenAI-compatible tool calling through llama-server with mock tools:

- `list_calendar`
- `create_calendar_hold`
- `search_notes`
- `save_note`
- `add_task`

Regular v6.1 strict run:

- 3/3 clean pass.
- Tasks: scheduling assistant, release-note workflow, creative-brief workflow.
- Required tools called in each task.
- No invalid tool JSON.
- No repeated search/tool loop.
- Calendar task preserved literal `day: Tuesday`.

Heretic v1.1:

- First permissive run showed tool capability but quality warnings:
  - scheduling invented an ISO date instead of preserving `Tuesday`
  - note tasks over-called `search_notes`
- Strict retry: 3/3 clean pass with the same task categories.
- No invalid tool JSON.
- No bad markers.

## Known Caveats

- Opencode coding-agent test on White and Black is not a release claim. The
  model can connect to MCP and run tests, but one smoke run damaged an untracked
  notes file and repeated invalid edit-tool calls. That belongs in internal
  agentic-coding notes, not the public quality headline.
- Creative quality was inspected by prompt/result artifacts plus mechanical
  checks. It was not judged by a human preference panel.
- Adult-romance testing was non-explicit only.
- If publishing vision claims, upload/document the matching mmproj and mention
  that HTTPS image URLs require the SSL-enabled llama.cpp fork/build.

## Suggested HF Model Card Claims

Safe claims:

- Fixed Gemma 4 chat template metadata for llama.cpp/GGUF use.
- Tested with llama-server `--jinja --reasoning auto`.
- Supports request-level no-thinking and bounded-thinking control.
- Vision works with matching mmproj on the tested llama.cpp build.
- Creative writing smoke suite passed without template leakage or reasoning
  leakage.
- Non-coding OpenAI-style tool automation passed a strict three-task harness
  covering scheduling, note drafting, and task creation.

Avoid for this upload:

- "Proven coding agent."
- "Fully autonomous agent."
- "No human review required."
- Claims based on opencode coding behavior.

## Upload Checklist

1. Upload regular v6.1 templatefix GGUF and regular mmproj.
2. Add `benchmark_results/` and `creative_eval_20260522/` summaries.
3. Add `agentic_eval_20260522/` summaries.
4. Add runtime section pointing at the patched llama.cpp fork and exact flags.
5. Add caveat that opencode coding-agent behavior is still under test.
6. Publish Heretic as a separate release/card or separate repo entry with its
   own creative and agentic results.
