"""Handstand form analysis: reference Python pipeline.

The package is a flat layout at ``pipeline/handstand``. Runtime paths are
resolved by :mod:`handstand.paths` rather than hard-coded at import time so
tests and the Mac/CI runners can point the pipeline at their own data.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
