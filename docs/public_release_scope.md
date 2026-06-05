# Public Release Scope

This repository has two remotes with different jobs.

## Public `origin`

Current containment status:

- `deucebucket/cerebellum` was made private on 2026-06-05 after audit showed
  public `origin/main` contained pipeline internals and raw selection artifacts.
- Do not make `origin` public again from existing history. Re-release from a
  sanitized branch or rewritten history after review.
- Assume anything previously public may have been cached or cloned.

Public `origin` should contain:

- model cards and release notes,
- benchmark summaries and detailed benchmark artifacts intended for release,
- reproducible high-level recipes,
- public CLI surface docs,
- safe data/results that do not reveal tensor-selection internals,
- release artifact metadata such as model hashes, runtime flags, and links.

Public `origin` should not contain:

- full automation pipeline code,
- tensor-selection heuristics,
- streaming quant internals,
- raw ablation data when it reveals the method,
- private pipeline automation,
- local dashboards, APIs, or web services,
- credentials or machine-specific paths,
- unfinished devlogs,
- one-off experiment scripts,
- large source GGUFs, generated GGUFs, or imatrix binaries.

Safe release metadata can include:

- source model ID and source file hashes,
- final GGUF hash,
- benchmark runtime flags,
- result summaries and selected detailed answer artifacts,
- high-level quant family/size target,
- public-safe `cerebellum watch --public` screenshots,
- model-card provenance statements.

Raw local logs, tensor maps, ablation JSON, temporary GGUF paths, and method
notes stay private unless they are explicitly reviewed and sanitized for a
specific release.

## Private `dev`

Private `dev` is the place for:

- exploratory scripts,
- local automation,
- working notes,
- dashboard experiments,
- unfinished ablation plans,
- temporary logs,
- repo-management scratch work.

## Public Engine Files

Public releases should not ship the engine implementation by default. If a
minimal public CLI is released later, it must expose only the safe user-facing
surface and avoid private allocator, streaming, ablation, and automation
internals.

## Release Checklist

Before making `origin` public again:

- Rebuild the public branch from a sanitized tree, or rewrite history with
  `git filter-repo` and force-push only after review.
- Remove engine internals, private scripts, dashboards, automation, raw
  ablation data, tensor maps, and devlogs.
- Check for absolute local paths, credentials, tokens, account details, and
  machine-specific paths.
- Check that public commands match files actually present in public.
- Check that benchmark numbers link to safe detailed artifacts.
- Mark proxy-based results as proxy-based.
- Add GitHub Sponsors, Ko-fi, and commission/priority-run links to README and
  model cards.
- Use `cerebellum watch --public --once --plain` or crop/redact raw watch
  screenshots before publishing.
- Use default `cerebellum package` / `upload` for public-safe sidecars; pass
  `--private` only for private dev uploads.
- Keep releasing GGUFs and benchmark artifacts even while the factory remains
  private.
