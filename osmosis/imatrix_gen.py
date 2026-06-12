"""Deprecated shim: ``osmosis.imatrix_gen`` moved to ``cerebellum.imatrix_gen``.

The project is Cerebellum; the osmosis import path is kept only so
in-flight runs and old scripts keep working. Import ``cerebellum.imatrix_gen`` instead.
"""

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "osmosis.imatrix_gen is deprecated; use cerebellum.imatrix_gen instead",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    import runpy as _runpy

    _runpy.run_module("cerebellum.imatrix_gen", run_name="__main__", alter_sys=True)
else:
    import cerebellum.imatrix_gen as _target

    _sys.modules[__name__] = _target
