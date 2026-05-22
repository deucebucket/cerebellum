# Benchmark Protocol

Benchmark results are release artifacts. They should be reproducible from the
model file, runtime flags, summary JSON, and detailed answer files.

## Public Artifact Requirements

For each published model, keep:

- model filename and SHA256,
- source model and source file hash when available,
- release configuration: ablation or sensitivity JSON, tensor type file,
  allocator command, budget, base quant type, and imatrix provenance,
- runtime name, version or commit, and server flags,
- benchmark summary JSON,
- detailed JSONL outputs for multiple-choice tasks,
- EvalPlus samples JSONL and EvalPlus eval JSON for code tasks,
- notes describing whether reasoning/thinking was enabled, disabled, or not
  applicable.

## Run Rules

- Use the same runtime configuration for the model and any same-size baseline.
- Record temperature, max tokens, context size, parallelism, and API route.
- Disable reasoning/thinking for no-thinking benchmark rows when the runtime
  supports it.
- Do not mix cached samples across different model files.
- Do not publish a corrected score by editing only the summary. If a harness bug
  changes the score materially, rerun generation and keep the new detailed
  artifacts.

## Audit Gate

Before publishing a score, inspect failures.

For multiple-choice tasks:

```bash
jq 'select(.correct == false)' detailed.jsonl | head -30
```

Check for:

- parser misses where the answer is visible in the raw response,
- clusters of `predicted="?"`,
- empty or near-empty responses,
- numeric/letter label mismatches,
- many identical wrong predictions.

For EvalPlus or other code tasks:

- run syntax parsing over every generated solution,
- count prompt echoes,
- count repeated target function definitions,
- count pass-only or empty bodies,
- inspect several passing and failing samples manually.

The score is not ready if failures show a harness or parser pattern.

## Gemma 4 Chat-Template Note

Gemma 4 code benchmarks should use the chat-template path rather than raw
completions unless a specific run proves raw completions are valid for that
runtime and template.

The local Cerebellum rule for no-thinking Gemma 4 runs is:

```text
llama-server --jinja --reasoning auto
chat_template_kwargs: {"enable_thinking": false}
thinking_budget_tokens: 0
BENCH_WORKERS=1 for EvalPlus-style generation
```

## Release Note Language

Use factual wording:

- what model file was tested,
- which benchmark was run,
- which runtime flags were used,
- where the detailed artifacts are stored,
- whether the result is fresh or historical.

Avoid unsupported comparisons, broad quality claims, or claims that are not
backed by artifacts in the repo or model repository.
