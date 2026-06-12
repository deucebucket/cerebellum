"""Deprecated shim: ``osmosis.micro_quantizer`` moved to ``cerebellum.micro_quantizer``.

The project is Cerebellum; the osmosis import path is kept only so
in-flight runs and old scripts keep working. Import ``cerebellum.micro_quantizer`` instead.
"""

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "osmosis.micro_quantizer is deprecated; use cerebellum.micro_quantizer instead",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    import runpy as _runpy

    _runpy.run_module("cerebellum.micro_quantizer", run_name="__main__", alter_sys=True)
else:
    import cerebellum.micro_quantizer as _target

    _sys.modules[__name__] = _target
