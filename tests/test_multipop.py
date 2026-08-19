"""Regression tests for ``MultiPopNestedSampler``.

The per-γ-population sampler runs each γ-configuration as an
independent static NS chain (migrations off by default; see
see :mod:`daedalus.multipop_sampler`). These tests verify:

1. The sampler runs end-to-end on the SBC toy without errors.
2. log Z agrees with single-cloud baseline within MC noise.
3. The flat-L γ-population mass is correctly represented in the
   importance-resampled joint cloud (otherwise inclusion marginals
   are biased upward toward γ-configs with non-flat continuous L).
"""

from __future__ import annotations

import numpy as np
import pytest

import daedalus
from daedalus.benchmarks import sbc_toy

# MultiPopNestedSampler is experimental and intentionally not exported
# from the daedalus namespace (see daedalus/multipop_sampler.py). It
# remains importable from its submodule, which is what these regression
# tests exercise.
from daedalus.multipop_sampler import MultiPopNestedSampler


@pytest.fixture(scope="module")
def sbc_problem():
    return sbc_toy.make_sbc_problem(
        ndim=2, N=12, sigma2=1.0, tau2=4.0, inclusion_prior=0.5
    )


def _make_chain(sbc_problem, seed):
    rng = np.random.default_rng(seed)
    gamma_true, beta_true = sbc_problem.sample_truth(rng)
    y = sbc_problem.sample_data(beta_true, rng)
    return sbc_problem.make_chain_problem(y), gamma_true, beta_true


def test_multipop_runs_endtoend(sbc_problem):
    chain, _, _ = _make_chain(sbc_problem, seed=10042)
    groups = [daedalus.Group(**kw) for kw in chain.groups_kwargs]
    sampler = MultiPopNestedSampler(
        loglike=chain.loglike,
        prior_transform=chain.prior_transform,
        ndim=chain.ndim,
        groups=groups,
        n_live=60,
        seed=0,
    )
    results = sampler.run_nested(dlogz=0.5, n_mcmc=20, show_progress=False)
    assert np.isfinite(results.log_Z)
    assert results.samples.shape[1] == chain.ndim
    assert results.gamma.shape[1] == len(groups)


def test_multipop_matches_singlecloud_logz(sbc_problem):
    """Per-pop migrations-off Z should agree with single-cloud baseline.

    Both samplers integrate the same joint posterior, so the log Z
    estimates should be equal up to NS Skilling noise. Tolerance set
    generously (0.5 nats) — the SBC toy has small absolute log Z and
    n_live=80 per pop gives modest error budgets.
    """
    chain, _, _ = _make_chain(sbc_problem, seed=10042)
    groups = [daedalus.Group(**kw) for kw in chain.groups_kwargs]

    single = daedalus.NestedSampler(
        loglike=chain.loglike,
        prior_transform=chain.prior_transform,
        ndim=chain.ndim,
        groups=groups,
        n_live=200,
        seed=0,
    )
    res_single = single.run_nested(
        dlogz=0.5, n_mcmc=20, transdim_fraction=0.5, show_progress=False
    )

    multi = MultiPopNestedSampler(
        loglike=chain.loglike,
        prior_transform=chain.prior_transform,
        ndim=chain.ndim,
        groups=groups,
        n_live=80,
        seed=0,
    )
    res_multi = multi.run_nested(dlogz=0.5, n_mcmc=20, show_progress=False)

    assert abs(res_multi.log_Z - res_single.log_Z) < 0.5, (
        f"multipop log Z = {res_multi.log_Z:.3f} vs "
        f"single log Z = {res_single.log_Z:.3f}"
    )


def test_multipop_migrations_off_warns_when_enabled(sbc_problem):
    """transdim_per_iter > 0 must emit a bias warning."""
    chain, _, _ = _make_chain(sbc_problem, seed=10042)
    groups = [daedalus.Group(**kw) for kw in chain.groups_kwargs]
    sampler = MultiPopNestedSampler(
        loglike=chain.loglike,
        prior_transform=chain.prior_transform,
        ndim=chain.ndim,
        groups=groups,
        n_live=60,
        seed=0,
    )
    with pytest.warns(UserWarning, match="bias"):
        sampler.run_nested(
            dlogz=2.0, n_mcmc=10, transdim_per_iter=1,
            max_iter_per_population=5, show_progress=False,
        )


def test_multipop_multi_bound_runs(sbc_problem):
    """bound='multi' (GammaScopedMultiEllipsoid) end-to-end smoke."""
    chain, _, _ = _make_chain(sbc_problem, seed=10042)
    groups = [daedalus.Group(**kw) for kw in chain.groups_kwargs]
    sampler = MultiPopNestedSampler(
        loglike=chain.loglike,
        prior_transform=chain.prior_transform,
        ndim=chain.ndim,
        groups=groups,
        bound="multi",
        n_live=80,
        seed=0,
    )
    results = sampler.run_nested(dlogz=0.5, n_mcmc=25, show_progress=False)
    assert np.isfinite(results.log_Z)
    assert results.samples.shape[1] == chain.ndim


def test_gamma_scoped_multi_ellipsoid_finds_modes():
    """The bound should split a bimodal active subspace into >1 ellipsoid."""
    from daedalus.bounds import GammaScopedMultiEllipsoid
    rng = np.random.default_rng(0)
    ndim, n_active = 8, 4
    n_pts = 300
    pts = np.empty((n_pts, ndim))
    labels = rng.choice([0, 1], size=n_pts)
    for i in range(n_pts):
        if labels[i] == 0:
            pts[i, :n_active] = rng.normal(loc=0.2, scale=0.03, size=n_active)
        else:
            pts[i, :n_active] = rng.normal(loc=0.8, scale=0.03, size=n_active)
        pts[i, n_active:] = rng.uniform(0, 1, size=ndim - n_active)
    bound = GammaScopedMultiEllipsoid(
        ndim=ndim, active_dims=list(range(n_active)),
    )
    bound.fit(pts)
    assert len(bound._ellipsoids) >= 2, (
        f"expected >=2 ellipsoids on bimodal active subspace, got "
        f"{len(bound._ellipsoids)}"
    )
    s = bound.sample(rng)
    assert s.shape == (ndim,)
    assert bound.contains(s)


def test_multipop_flat_L_pop_in_joint_cloud(sbc_problem):
    """The all-off γ-config has flat continuous L and must be
    represented in the importance-resampled joint cloud, or
    inclusion marginals are biased upward."""
    chain, _, _ = _make_chain(sbc_problem, seed=10042)
    groups = [daedalus.Group(**kw) for kw in chain.groups_kwargs]
    sampler = MultiPopNestedSampler(
        loglike=chain.loglike,
        prior_transform=chain.prior_transform,
        ndim=chain.ndim,
        groups=groups,
        n_live=80,
        seed=0,
    )
    results = sampler.run_nested(dlogz=0.5, n_mcmc=20, show_progress=False)
    # All-off γ-config (γ = (False, False)) should appear in the joint
    # importance-resampled cloud whenever sbc_toy's flat-L marginal Z
    # is a non-trivial fraction of the joint Z. For inclusion_prior=0.5
    # this fraction is ~exp(log P(γ=(0,0)) + log L_const - log_Z) and
    # is typically > 1%, so at least one resampled point should land
    # in γ=(0,0).
    all_off = ~results.gamma.any(axis=1)
    assert all_off.sum() > 0, (
        "No resampled posterior points landed in the all-off γ-config; "
        "flat-L pop is not being injected into the joint cloud."
    )
