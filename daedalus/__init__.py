"""daedalus — trans-dimensional and standard nested sampling via MoMS.

Public API:
    NestedSampler   — main entry point (handles both static-gamma and trans-dim)
    Group           — declares a toggleable model component
    Results         — return value of NestedSampler.run_nested
    State           — joint (beta, gamma) point on the MoMS state space
    NSProgress      — per-iteration state passed to run_nested's on_progress
                      callback, for callers that need programmatic progress
                      rather than a terminal bar

Submodules:
    bounds          — bounding regions (NoBound, SingleEllipsoid, MultiEllipsoid)
    samplers        — within-model constrained-MCMC kernels (unif, rwalk, rslice)
    birth_proposals — pluggable trans-dim birth proposals (GaussianRW, GLS, ...)
    benchmarks      — built-in test problems (Gaussian, eggbox, diabetes, ...)
    plotting        — post-processing plots (cornerplot, traceplot, runplot,
                      model/inclusion probability plots) -- optional, requires
                      matplotlib + corner via ``pip install daedalus[plotting]``
"""

from __future__ import annotations

from . import benchmarks, birth_proposals, bounds, plotting, samplers
from .birth_proposals import (
    BirthProposal,
    BirthProposalResult,
    BLSPeriodBirth,
    GaussianRWBirth,
    GLSPeriodBirth,
    PriorDrawBirth,
)
from .diagnostics import (
    InsertionTestResult,
    insertion_index_test,
    multirun_logZ_error,
)
from .groups import Group
from .results import Results, load_results
from .sampler import NestedSampler, NSProgress
from .state import State
from .validation import (
    BirthConsistencyWarning,
    validate_birth_consistency,
)

__all__ = [
    "NestedSampler",
    "NSProgress",
    "Group",
    "Results",
    "State",
    "BirthProposal",
    "BirthProposalResult",
    "BLSPeriodBirth",
    "GaussianRWBirth",
    "GLSPeriodBirth",
    "PriorDrawBirth",
    "BirthConsistencyWarning",
    "validate_birth_consistency",
    "InsertionTestResult",
    "insertion_index_test",
    "multirun_logZ_error",
    "load_results",
    "bounds",
    "samplers",
    "birth_proposals",
    "benchmarks",
    "plotting",
]

__version__ = "0.0.1.dev0"
