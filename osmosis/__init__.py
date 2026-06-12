"""Deprecated compatibility shim for the old ``osmosis`` package name.

The project is Cerebellum and all code now lives in the ``cerebellum``
package. Every ``osmosis.X`` submodule re-exports its ``cerebellum``
counterpart (legacy experimental modules live under ``cerebellum._legacy``).
This shim exists so in-flight runs and old scripts keep working; do not add
new code here.
"""

import warnings as _warnings

__version__ = "0.2.0"

_warnings.warn(
    "the 'osmosis' package is deprecated; import 'cerebellum' instead",
    DeprecationWarning,
    stacklevel=2,
)
