# Public Release Scope

This document defines what belongs in the public Cerebellum repository.

## Public Repository

Public `origin` should contain:

- the Cerebellum engine files,
- public docs that explain the manual process,
- release configurations that let others rebuild the same GGUFs,
- Cerebellum ablation outputs used to choose release tensor types,
- benchmark summaries and detailed benchmark artifacts intended for release,
- model-card support files that are safe to publish.

Public `origin` should not contain:

- private pipeline automation,
- local dashboards or web services,
- credentials or machine-specific paths,
- unfinished devlogs,
- one-off experiment scripts,
- large source GGUFs, generated GGUFs, or imatrix binaries.

Release configuration is public when it is sanitized and reproducible. Examples:

- `ablation_results.json` or `sensitivity_multi.json`,
- `tensor_types.txt`,
- allocator arguments,
- source model ID and source file hashes,
- imatrix source or hash,
- final GGUF hash,
- benchmark runtime flags,
- result summaries and detailed answer artifacts.

Raw local logs and temporary GGUF paths can stay private. The measured ablation
JSON and final tensor map should be public for released models.

## Private Development

Private `dev` is the place for:

- exploratory scripts,
- local automation,
- working notes,
- dashboard experiments,
- unfinished ablation plans,
- temporary logs,
- repo-management scratch work.

## Public Engine Files

The current public engine set is:

```text
osmosis/cerebellum.py
osmosis/budget.py
osmosis/imatrix_stream.py
osmosis/imatrix_gen.py
osmosis/micro_quantizer.py
osmosis/sensitivity_stream.py
```

The package path remains `osmosis` until the rename is completed. New docs and
artifacts should use the Cerebellum name unless they are referring to the
legacy Python package path.

## Release Checklist

Before pushing public docs or artifacts to `origin`:

- Check for absolute local paths.
- Check for credentials or tokens.
- Check that commands match files actually present in public.
- Check that benchmark numbers link to detailed artifacts.
- Mark proxy-based results as proxy-based.
- Keep dashboard and automation notes out of public.
