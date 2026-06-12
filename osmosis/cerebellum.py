"""Deprecated shim: ``osmosis.cerebellum`` moved to ``cerebellum.cerebellum``.

The project is Cerebellum; the osmosis import path is kept only so
in-flight runs and old scripts keep working. Import ``cerebellum.cerebellum`` instead.
"""

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "osmosis.cerebellum is deprecated; use cerebellum.cerebellum instead",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    import runpy as _runpy

    _runpy.run_module("cerebellum.cerebellum", run_name="__main__", alter_sys=True)
else:
    import cerebellum.cerebellum as _target

    _sys.modules[__name__] = _target
