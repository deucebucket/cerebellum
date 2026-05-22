# Gemma 4 26B Non-Coding Agentic Tool Evaluation - 2026-05-22

This directory records non-coding agentic tool-use tests for the Gemma 4 26B
Cerebellum templatefix releases.

The goal is narrow: prove OpenAI-style tool automation for assistant workflows,
not coding-agent performance. Coding-agent behavior is not treated as a release
claim in this directory.

## Harness

Runtime:

- `llama-server`
- `--jinja`
- `--reasoning auto`
- request payload included `chat_template_kwargs: {"enable_thinking": false}`
- request payload included `thinking_budget_tokens: 0`

Mock tools:

- `list_calendar`
- `create_calendar_hold`
- `search_notes`
- `save_note`
- `add_task`

Tasks:

- Scheduling assistant: inspect calendar, find a Tuesday slot, create a hold.
- Release-note workflow: search internal notes, save a draft, create a follow-up
  task.
- Creative-brief workflow: search style notes and save a production brief.

Pass criteria:

- Required tools are called.
- Tool arguments are valid JSON.
- No repeated search/tool loop.
- No template or thinking leakage in no-thinking mode.
- The scheduling task preserves the user-provided literal day instead of
  inventing an ISO date.

## Results

Regular v6.1 templatefix:

- File: `regular_v6_1_noncoding_agentic_tools_strict_summary.json`
- Result: 3/3 clean pass.
- Passed cases: `schedule_strict`, `release_notes_strict`,
  `creative_brief_strict`.
- Warnings: none.
- Failures: none.

Heretic v1.1 templatefix:

- File: `heretic_v1_1_noncoding_agentic_tools_strict_retry_summary.json`
- Result: 3/3 clean pass on strict retry.
- Passed cases: `schedule_strict`, `release_notes_strict`,
  `creative_brief_strict`.
- Warnings: none on strict retry.
- Failures: none on strict retry.

The first Heretic permissive run demonstrated tool capability but had quality
warnings: it invented an ISO date for a Tuesday scheduling task and over-called
`search_notes`. The strict retry corrected those issues.

## Release Claim Supported

Supported:

- Non-coding OpenAI-compatible tool automation passed a strict three-task
  harness.
- The model can plan over simple external state, call tools with valid JSON,
  preserve user constraints, write notes, and create tasks.

Not supported by this directory:

- Proven coding-agent behavior.
- Fully autonomous agent behavior without human review.
- General external-tool reliability beyond the listed mock tools.
