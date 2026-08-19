"""End-to-end NS validation against the multimodal eggbox benchmark.

The eggbox has 25 isolated likelihood peaks; recovering log Z requires the
constrained-MCMC kernel to mix between modes. RW + bound is sufficient at
modest n_live; this test pins that both Single and Multi ellipsoid bounds
recover the analytical evidence.
"""

from __future__ import annotations

import pytest

import daedalus
from daedalus.benchmarks import eggbox


@pytest.mark.benchmark
def test_eggbox_logz_with_single_ellipsoid_rw() -> None:
    problem = eggbox.make_problem()
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="single",
        sample="rwalk",
        n_live=500,
        seed=42,
    )
    results = sampler.run_nested(dlogz=0.5, n_mcmc=50)
    err = abs(results.log_Z - problem.log_Z_true)
    assert err < 0.5, (
        f"log Z = {results.log_Z:.3f} vs analytical {problem.log_Z_true:.3f} "
        f"(diff {err:.3f}); NS error estimate {results.log_Z_err:.3f}"
    )


@pytest.mark.benchmark
def test_eggbox_logz_with_multi_ellipsoid_rw() -> None:
    problem = eggbox.make_problem()
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="multi",
        sample="rwalk",
        n_live=500,
        seed=42,
    )
    results = sampler.run_nested(dlogz=0.5, n_mcmc=50)
    err = abs(results.log_Z - problem.log_Z_true)
    assert err < 0.5, (
        f"log Z = {results.log_Z:.3f} vs analytical {problem.log_Z_true:.3f} "
        f"(diff {err:.3f}); NS error estimate {results.log_Z_err:.3f}"
    )
