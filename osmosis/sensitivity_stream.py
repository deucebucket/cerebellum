"""Deprecated shim: ``osmosis.sensitivity_stream`` moved to ``cerebellum.sensitivity_stream``.

The project is Cerebellum; the osmosis import path is kept only so
in-flight runs and old scripts keep working. Import ``cerebellum.sensitivity_stream`` instead.
"""

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "osmosis.sensitivity_stream is deprecated; use cerebellum.sensitivity_stream instead",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    import runpy as _runpy

    _runpy.run_module("cerebellum.sensitivity_stream", run_name="__main__", alter_sys=True)
else:
    import cerebellum.sensitivity_stream as _target

    _sys.modules[__name__] = _target
