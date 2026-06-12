"""Deprecated shim: ``osmosis.budget`` moved to ``cerebellum.budget``.

The project is Cerebellum; the osmosis import path is kept only so
in-flight runs and old scripts keep working. Import ``cerebellum.budget`` instead.
"""

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "osmosis.budget is deprecated; use cerebellum.budget instead",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    import runpy as _runpy

    _runpy.run_module("cerebellum.budget", run_name="__main__", alter_sys=True)
else:
    import cerebellum.budget as _target

    _sys.modules[__name__] = _target
