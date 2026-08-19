"""End-to-end NS validation against the 2D Rosenbrock-banana benchmark.

The Rosenbrock posterior is unimodal but heavily degenerate: a thin
non-linear ridge along $y = x^2$. Tests the within-model RW sampler's
ability to mix along curvature; bound type matters less here than
on the eggbox or shells benchmarks.

Pin: log Z within 0.5 of the quadrature reference, recovered with
both single- and multi-ellipsoidal bounds.
"""

from __future__ import annotations

import pytest

import daedalus
from daedalus.benchmarks import rosenbrock


@pytest.mark.benchmark
def test_rosenbrock_logz_with_single_ellipsoid_rwalk() -> None:
    problem = rosenbrock.make_problem()
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="single",
        sample="rwalk",
        n_live=400,
        seed=42,
    )
    results = sampler.run_nested(dlogz=0.5, n_mcmc=50, show_progress=False)
    err = abs(results.log_Z - problem.log_Z_true)
    assert err < 0.5, (
        f"log Z = {results.log_Z:.3f} vs quadrature {problem.log_Z_true:.3f} "
        f"(diff {err:.3f}); NS error estimate {results.log_Z_err:.3f}"
    )


@pytest.mark.benchmark
def test_rosenbrock_logz_with_multi_ellipsoid_slice() -> None:
    """Slice sampler is the natural choice on degenerate-ridge posteriors;
    confirm it also recovers log Z under the multi-ellipsoid bound."""
    problem = rosenbrock.make_problem()
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="multi",
        sample="rslice",
        n_live=400,
        seed=42,
    )
    # 100 slice sweeps. Written as n_mcmc=20 when n_mcmc counted step_once
    # calls (each doing n_slices=5 sweeps); n_mcmc is now a sweep budget, so
    # the same configuration is spelled 100.
    results = sampler.run_nested(dlogz=0.5, n_mcmc=100, show_progress=False)
    err = abs(results.log_Z - problem.log_Z_true)
    assert err < 0.5, (
        f"log Z = {results.log_Z:.3f} vs quadrature {problem.log_Z_true:.3f} "
        f"(diff {err:.3f}); NS error estimate {results.log_Z_err:.3f}"
    )
