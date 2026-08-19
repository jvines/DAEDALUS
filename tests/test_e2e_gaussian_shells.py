"""End-to-end NS validation against the Mukherjee+ 2006 / Feroz+ 2009
two-Gaussian-shells benchmark in 2D.

The shells have disjoint support modulo negligible tails -- exactly
the test that distinguishes a multi-ellipsoidal bound from a
single-ellipsoid one. Single-ellipsoid runs cover both shells with one
inflated ellipsoid and waste samples in the central gap; multi-
ellipsoid splits into two per-shell ellipsoids and recovers the
analytic evidence efficiently.

Pin: log Z within 0.5 of the analytic value -1.75 (Feroz+ 2009 Table 6),
recovered with the multi-ellipsoidal bound.
"""

from __future__ import annotations

import pytest

import daedalus
from daedalus.benchmarks import gaussian_shells


@pytest.mark.benchmark
def test_gaussian_shells_logz_with_multi_ellipsoid_rw() -> None:
    problem = gaussian_shells.make_problem(ndim=2)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="multi",
        sample="rwalk",
        n_live=600,
        seed=42,
    )
    results = sampler.run_nested(dlogz=0.5, n_mcmc=50, show_progress=False)
    err = abs(results.log_Z - problem.log_Z_true)
    assert err < 0.5, (
        f"log Z = {results.log_Z:.3f} vs analytic {problem.log_Z_true:.3f} "
        f"(diff {err:.3f}); NS error estimate {results.log_Z_err:.3f}"
    )


@pytest.mark.benchmark
def test_gaussian_shells_logz_with_single_ellipsoid_rwalk() -> None:
    """Same problem, single-ellipsoid bound. Less efficient (the bound
    inflates to cover both shells with a central low-mass void) but
    still recovers the right log Z for a 2D shell pair, given enough
    n_live.
    """
    problem = gaussian_shells.make_problem(ndim=2)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="single",
        sample="rwalk",
        n_live=600,
        seed=42,
    )
    results = sampler.run_nested(dlogz=0.5, n_mcmc=50, show_progress=False)
    err = abs(results.log_Z - problem.log_Z_true)
    assert err < 0.5, (
        f"log Z = {results.log_Z:.3f} vs analytic {problem.log_Z_true:.3f} "
        f"(diff {err:.3f}); NS error estimate {results.log_Z_err:.3f}"
    )
