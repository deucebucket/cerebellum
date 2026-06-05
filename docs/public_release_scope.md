# Public Release Scope

Public Cerebellum is the brand, artifacts, benchmarks, and high-level
reproducibility layer. Private Cerebellum development is the factory.

## Public Repository

Public material may include:

- model cards and release notes
- benchmark summaries and selected safe detailed artifacts
- high-level reproducible recipes
- public CLI surface documentation
- safe release metadata such as model hashes, runtime flags, and links
- sponsor, commission, and priority-run links

Public material must not include:

- full automation pipeline code
- tensor-selection heuristics
- streaming quant internals
- raw ablation data or tensor maps that reveal the method
- devlogs explaining private implementation details
- private scripts, dashboards, unfinished experiments, credentials, local paths,
  account details, source GGUFs, generated GGUFs, or imatrix binaries

## History Status

The public repository history was replaced with an unrelated sanitized history
after sensitive factory files were found in earlier public refs. Assume any
previously public material may have been cached or cloned elsewhere.

Future public releases should be built from a reviewed sanitized tree.
