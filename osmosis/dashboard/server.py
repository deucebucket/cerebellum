"""Deprecated shim: ``osmosis.dashboard.server`` moved to ``cerebellum.dashboard.server``.

The project is Cerebellum; the osmosis import path is kept only so
in-flight runs and old scripts keep working. Import ``cerebellum.dashboard.server`` instead.
"""

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "osmosis.dashboard.server is deprecated; use cerebellum.dashboard.server instead",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    import runpy as _runpy

    _runpy.run_module("cerebellum.dashboard.server", run_name="__main__", alter_sys=True)
else:
    import cerebellum.dashboard.server as _target

    _sys.modules[__name__] = _target
