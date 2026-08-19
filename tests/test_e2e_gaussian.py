"""End-to-end NS validation against the multivariate Gaussian benchmark.

This is the first integration test that exercises the full NS loop:
initialisation, dead-point accounting, bound fitting (no-op for NoBound),
constrained-MCMC replacement, termination, residual live accounting, and
importance resampling. Pass criterion follows the spec: |log Z_estimated -
log Z_true| < 0.5 at modest n_live.

Uses the simplest sampler combo (NoBound + UniformSampler) to keep the
test independent of the not-yet-implemented ellipsoidal bounds and Markov
samplers. As prior volume shrinks the rejection rate degrades; we cap the
problem at low ndim and stop early to keep this test under a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

import daedalus
from daedalus.benchmarks import gaussians


@pytest.mark.benchmark
def test_unif_nobound_gaussian_2d_logz_within_half() -> None:
    problem = gaussians.make_problem(ndim=2, prior_half_width=10.0)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="none",
        sample="unif",
        n_live=200,
        seed=42,
    )
    results = sampler.run_nested(dlogz=0.5)
    err = abs(results.log_Z - problem.log_Z_true)
    assert err < 0.5, (
        f"log Z = {results.log_Z:.3f} vs analytical {problem.log_Z_true:.3f} "
        f"(diff {err:.3f}); NS error estimate {results.log_Z_err:.3f}"
    )


@pytest.mark.benchmark
def test_unif_nobound_gaussian_2d_returns_well_formed_results() -> None:
    problem = gaussians.make_problem(ndim=2, prior_half_width=10.0)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="none",
        sample="unif",
        n_live=100,
        seed=7,
    )
    results = sampler.run_nested(dlogz=1.0)
    assert results.samples.shape[1] == problem.ndim
    assert results.samples.shape[0] == results.log_likelihoods.size
    assert results.gamma.shape[1] == 0  # no toggleable groups
    assert np.isfinite(results.log_Z)
    assert results.n_iter > 0
    assert results.n_calls >= results.n_iter
    assert 0.0 < results.eff <= 1.0


@pytest.mark.benchmark
def test_rwalk_single_ellipsoid_gaussian_5d_logz_within_half() -> None:
    """RandomWalkSampler + SingleEllipsoid on a 5D Gaussian. RW is the
    practical default sampler at any prior shrinkage; this test pins that
    it composes correctly with the ellipsoidal bound.
    """
    problem = gaussians.make_problem(ndim=5, prior_half_width=10.0)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="single",
        sample="rwalk",
        n_live=400,
        seed=11,
    )
    results = sampler.run_nested(dlogz=0.5, n_mcmc=25)
    err = abs(results.log_Z - problem.log_Z_true)
    assert err < 0.5, (
        f"log Z = {results.log_Z:.3f} vs analytical {problem.log_Z_true:.3f} "
        f"(diff {err:.3f}); NS error estimate {results.log_Z_err:.3f}"
    )


@pytest.mark.benchmark
def test_rslice_single_ellipsoid_gaussian_5d_logz_within_half() -> None:
    """RandomSliceSampler + SingleEllipsoid on a 5D Gaussian. Slice sampling
    is tuning-free and should hit the analytical truth without any
    parameter tweaking.
    """
    problem = gaussians.make_problem(ndim=5, prior_half_width=10.0)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="single",
        sample="rslice",
        n_live=400,
        seed=42,
    )
    # 25 slice sweeps -- the kernel's own recommendation at ndim=5. Written as
    # n_mcmc=5 when n_mcmc counted step_once calls (each doing n_slices=5
    # sweeps); n_mcmc is now a sweep budget, so the same configuration is
    # spelled 25.
    results = sampler.run_nested(dlogz=0.5, n_mcmc=25)
    err = abs(results.log_Z - problem.log_Z_true)
    assert err < 0.5, (
        f"log Z = {results.log_Z:.3f} vs analytical {problem.log_Z_true:.3f} "
        f"(diff {err:.3f}); NS error estimate {results.log_Z_err:.3f}"
    )


@pytest.mark.benchmark
def test_rwalk_nobound_gaussian_2d_logz_within_half() -> None:
    """RandomWalkSampler with NoBound -- the worst-bound configuration. RW
    should still recover log Z because it's a Markov sampler that doesn't
    rely on the bound for sampling efficiency, only for proposal scaling.
    """
    problem = gaussians.make_problem(ndim=2, prior_half_width=10.0)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="none",
        sample="rwalk",
        n_live=200,
        seed=23,
    )
    results = sampler.run_nested(dlogz=0.5, n_mcmc=50)
    err = abs(results.log_Z - problem.log_Z_true)
    assert err < 0.5, (
        f"log Z = {results.log_Z:.3f} vs analytical {problem.log_Z_true:.3f} "
        f"(diff {err:.3f}); NS error estimate {results.log_Z_err:.3f}"
    )


@pytest.mark.benchmark
def test_unif_single_ellipsoid_gaussian_5d_logz_within_half() -> None:
    """SingleEllipsoid + UniformSampler on a 5D Gaussian. Must beat the
    NoBound combo by an order of magnitude in likelihood evaluations and
    still recover log Z to within 0.5 of the analytic truth.
    """
    problem = gaussians.make_problem(ndim=5, prior_half_width=10.0)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="single",
        sample="unif",
        n_live=400,
        seed=11,
    )
    results = sampler.run_nested(dlogz=0.5)
    err = abs(results.log_Z - problem.log_Z_true)
    assert err < 0.5, (
        f"log Z = {results.log_Z:.3f} vs analytical {problem.log_Z_true:.3f} "
        f"(diff {err:.3f}); NS error estimate {results.log_Z_err:.3f}"
    )
    # NS efficiency should be much better than 1/d_prior_volume for the bound
    # to be doing real work; require at least 10% efficiency.
    assert results.eff > 0.10, f"NS efficiency only {results.eff:.2%}"


@pytest.mark.benchmark
def test_trans_dim_smoke_returns_well_formed_results() -> None:
    """Trans-dim path runs end-to-end and returns gamma-aware results."""
    problem = gaussians.make_problem(ndim=2, prior_half_width=10.0)
    g0 = daedalus.Group(name="g0", params=[0], off_values=np.array([0.0]))
    g1 = daedalus.Group(name="g1", params=[1], off_values=np.array([0.0]))
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=[g0, g1],
        bound="none",
        sample="rwalk",
        n_live=100,
        seed=0,
    )
    results = sampler.run_nested(dlogz=1.0, n_mcmc=20, transdim_fraction=0.3)
    assert results.gamma.shape[1] == 2
    assert results.gamma.dtype == bool
    assert np.isfinite(results.log_Z)
    inc = results.inclusion_probabilities()
    assert set(inc.keys()) == {"g0", "g1"}
    for name, p in inc.items():
        assert 0.0 <= p <= 1.0, f"P(gamma_{name}=1) = {p} not in [0,1]"
