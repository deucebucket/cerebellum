"""Deprecated shim: ``osmosis.hillstep`` moved to ``cerebellum.hillstep``.

The project is Cerebellum; the osmosis import path is kept only so
in-flight runs and old scripts keep working. Import ``cerebellum.hillstep`` instead.
"""

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "osmosis.hillstep is deprecated; use cerebellum.hillstep instead",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    import runpy as _runpy

    _runpy.run_module("cerebellum.hillstep", run_name="__main__", alter_sys=True)
else:
    import cerebellum.hillstep as _target

    _sys.modules[__name__] = _target
