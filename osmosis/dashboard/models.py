"""Deprecated shim: ``osmosis.dashboard.models`` moved to ``cerebellum.dashboard.models``.

The project is Cerebellum; the osmosis import path is kept only so
in-flight runs and old scripts keep working. Import ``cerebellum.dashboard.models`` instead.
"""

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "osmosis.dashboard.models is deprecated; use cerebellum.dashboard.models instead",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    import runpy as _runpy

    _runpy.run_module("cerebellum.dashboard.models", run_name="__main__", alter_sys=True)
else:
    import cerebellum.dashboard.models as _target

    _sys.modules[__name__] = _target
