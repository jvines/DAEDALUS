"""Experimental / prototype code — NOT part of the stable daedalus API.

Currently: `vectorized_ns`, a Phase-1 spike of vectorized nested sampling
(batched-likelihood NS) for expensive GEMM-shaped likelihoods. Validated
unbiased on Gaussians; fixed-dim / box-prior / DE-only. See its docstring.
"""
from .vectorized_ns import vectorized_ns  # noqa: F401
