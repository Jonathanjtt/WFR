"""WALNUTS-R2P sampling baseline.

The sampler is vendored from bob-carpenter/walnuts (MIT); see :mod:`._vendor`.  The
GMMVI-facing entry point is :func:`gmmvi.algorithms.walnuts.sampler.fit`.
"""

from .sampler import fit, lp_fun

__all__ = ["fit", "lp_fun"]
