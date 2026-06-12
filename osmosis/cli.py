"""Deprecated shim: ``osmosis.cli`` moved to ``cerebellum._legacy.cli``.

The project is Cerebellum; the osmosis import path is kept only so
in-flight runs and old scripts keep working. Import ``cerebellum._legacy.cli`` instead.
"""

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "osmosis.cli is deprecated; use cerebellum._legacy.cli instead",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    import runpy as _runpy

    _runpy.run_module("cerebellum._legacy.cli", run_name="__main__", alter_sys=True)
else:
    import cerebellum._legacy.cli as _target

    _sys.modules[__name__] = _target
