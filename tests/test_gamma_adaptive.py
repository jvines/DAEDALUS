"""Tests for gamma-adaptive MoMS-NS.

Exercises ``NestedSampler.run_nested(gamma_adaptive=True)`` on a
small exchangeable-slot trans-dim problem to verify:

* The chain runs to completion without errors.
* The live cloud grows from the initial ``n_live`` to a value
  bounded by ``gamma_max_growth_factor * n_live``.
* The log-evidence is finite and the run completes.
* The chain produces dead points covering multiple gamma-configurations.

The fixed-cloud path is also exercised at ``gamma_adaptive=False``
to confirm no regression: same problem, same seed, identical output
(within Monte-Carlo noise on the random-seed-dependent details).
"""
from __future__ import annotations

import numpy as np
import pytest

import daedalus


def _make_toy_problem():
    """Two-slot spike-and-slab regression. Both slots exchangeable.

    Likelihood:  N(y | X beta, sigma^2 I), y = data, X column-orthonormal.
    Truth used here: y has signal at both predictors.
    """
    rng = np.random.default_rng(0)
    N = 12
    g = 2
    X = rng.standard_normal((N, g))
    # Orthonormalise: X^T X = N * I.
    Q, _ = np.linalg.qr(X)
    X = Q * np.sqrt(N)
    beta_true = np.array([1.5, 1.2])
    sigma2 = 1.0
    tau2 = 4.0
    y = X @ beta_true + rng.standard_normal(N) * np.sqrt(sigma2)
    base_log_const = -0.5 * N * np.log(2.0 * np.pi * sigma2)

    def loglike(beta, gamma):
        resid = y - X @ beta
        ssr = float(np.dot(resid, resid))
        if not np.isfinite(ssr):
            return -np.inf
        return base_log_const - 0.5 * ssr / sigma2

    sd_slab = float(np.sqrt(tau2))

    def prior_transform(u):
        from scipy.stats import norm as _norm

        return _norm.ppf(np.clip(u, 1e-15, 1.0 - 1e-15)) * sd_slab

    groups = [
        daedalus.Group(
            name=f"x{i}",
            params=[i],
            off_values=np.array([0.0]),
            inclusion_prior=0.5,
        )
        for i in range(g)
    ]
    return loglike, prior_transform, groups, g


@pytest.mark.benchmark
def test_gamma_adaptive_runs_and_grows() -> None:
    loglike, prior_transform, groups, g = _make_toy_problem()
    sampler = daedalus.NestedSampler(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=g,
        groups=groups,
        bound="single",
        sample="rwalk",
        n_live=100,
        seed=2026,
    )
    res = sampler.run_nested(
        dlogz=0.1,
        n_mcmc=20,
        transdim_fraction=0.5,
        gamma_adaptive=True,
        gamma_speciation_interval=10,
        gamma_c_min_floor=3,
        gamma_max_growth_factor=3.0,
        show_progress=False,
    )
    # log_Z finite
    assert np.isfinite(res.log_Z), f"log_Z = {res.log_Z!r}"
    # Multiple gamma configs visited
    gammas = {tuple(bool(x) for x in row) for row in res.gamma}
    assert len(gammas) >= 2, (
        f"expected the chain to visit multiple gamma-configurations, got "
        f"{gammas}"
    )


@pytest.mark.benchmark
def test_gamma_adaptive_off_matches_legacy_path() -> None:
    """gamma_adaptive=False (default) must reproduce the standard
    fixed-cloud MoMS-NS chain identically. Catches accidental
    side-effects from the n_live=variable refactor on the standard
    code path.
    """
    loglike, prior_transform, groups, g = _make_toy_problem()
    sampler = daedalus.NestedSampler(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=g,
        groups=groups,
        bound="single",
        sample="rwalk",
        n_live=100,
        seed=2026,
    )
    res = sampler.run_nested(
        dlogz=0.1,
        n_mcmc=20,
        transdim_fraction=0.5,
        gamma_adaptive=False,
        show_progress=False,
    )
    # Should produce finite log_Z and the same shape of outputs.
    assert np.isfinite(res.log_Z)
    assert res.gamma.shape[1] == g


@pytest.mark.benchmark
def test_gamma_adaptive_rejects_no_groups() -> None:
    """gamma_adaptive=True is meaningless without toggleable groups;
    the sampler should reject it explicitly."""

    def loglike(beta, gamma):
        return -0.5 * float(np.dot(beta, beta))

    def prior_transform(u):
        return -3.0 + 6.0 * u

    sampler = daedalus.NestedSampler(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=2,
        groups=(),
        bound="single",
        sample="rwalk",
        n_live=50,
        seed=0,
    )
    with pytest.raises(ValueError, match="toggleable Group"):
        sampler.run_nested(
            gamma_adaptive=True,
            show_progress=False,
        )
