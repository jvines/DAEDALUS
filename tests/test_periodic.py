"""Periodic-parameter handling end-to-end tests.

Exercises the ``periodic`` argument of ``NestedSampler`` on problems
where the posterior straddles a u-space boundary. The non-periodic
chain produces an artefactually bimodal posterior in u (mass piles up
at both u=0 and u=1 because the chain cannot cross the boundary
without rejection); the periodic chain wraps the boundary and produces
the correct unimodal posterior on the circle.

Both problems use a von Mises likelihood with mean at the periodic-
boundary point, so the diagnostic is whether the chain places posterior
mass on BOTH halves of the unit cube ([0, 0.05] and [0.95, 1]) at
comparable density. A periodic chain does; a non-periodic chain ends
up locked in whichever half it landed in first, producing posterior
samples concentrated on one side only.
"""
from __future__ import annotations

import numpy as np
import pytest

import daedalus


def _von_mises_problem(kappa: float = 8.0):
    """1D unit-cube problem: u in [0, 1) maps to theta in [0, 2 pi) via
    theta = 2 pi u; likelihood is von Mises with mean 0 and concentration
    kappa, i.e. log L = kappa * cos(theta). The posterior mode is at
    theta = 0 (equivalently theta = 2 pi), which lives exactly at the
    u-space boundary u = 0 / u = 1.
    """

    def loglike(beta, gamma):
        theta = float(beta[0])
        return kappa * np.cos(theta)

    def prior_transform(u):
        return 2.0 * np.pi * u

    return loglike, prior_transform


@pytest.mark.benchmark
def test_periodic_wraps_boundary_mode() -> None:
    """With periodic=[0] the posterior in u is concentrated near
    u = 0 / u = 1 with comparable density on both ends. Without
    periodic, the chain locks to one side only.
    """
    loglike, prior_transform = _von_mises_problem(kappa=8.0)

    sampler_periodic = daedalus.NestedSampler(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=1,
        groups=(),
        bound="single",
        sample="rwalk",
        n_live=500,
        periodic=[0],
        seed=7,
    )
    res_p = sampler_periodic.run_nested(
        dlogz=0.05, n_mcmc=40, show_progress=False
    )
    u_samples = res_p.samples[:, 0] / (2.0 * np.pi)
    # Fold u into the wrapped distance from 0 (or equivalently 1).
    wrapped = np.minimum(u_samples, 1.0 - u_samples)
    # Both ends should contribute: fraction of samples within
    # 0.1 of either boundary.
    near_zero = float(np.mean(u_samples < 0.10))
    near_one = float(np.mean(u_samples > 0.90))
    assert near_zero + near_one > 0.7, (
        f"periodic chain did not concentrate near the boundary: "
        f"near_zero={near_zero:.3f}, near_one={near_one:.3f}"
    )
    # Both halves should have comparable mass under wrap symmetry
    # (kappa = 8 is concentrated enough that the boundary lobe has
    # most mass within +/- 0.1 of u=0 or u=1, split roughly evenly).
    assert min(near_zero, near_one) > 0.3 * max(near_zero, near_one), (
        f"periodic chain locked one-sidedly: "
        f"near_zero={near_zero:.3f}, near_one={near_one:.3f}"
    )
    # And the posterior should be tight around the boundary mode.
    assert float(np.median(wrapped)) < 0.10, (
        f"posterior is not concentrated at the boundary: "
        f"median wrap-distance = {float(np.median(wrapped)):.3f}"
    )


@pytest.mark.benchmark
def test_periodic_evidence_matches_analytic() -> None:
    """The von Mises evidence on theta in [0, 2 pi) with uniform prior
    is Z = (1 / (2 pi)) * integral_0^{2 pi} exp(kappa cos theta) dtheta
        = I_0(kappa),
    where I_0 is the modified Bessel function of the first kind. The
    chain log-evidence should match log I_0(kappa) within a few times
    the Skilling error budget when periodic handling is on.
    """
    from scipy.special import i0

    kappa = 4.0
    loglike, prior_transform = _von_mises_problem(kappa=kappa)
    log_Z_true = float(np.log(i0(kappa)))

    sampler = daedalus.NestedSampler(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=1,
        groups=(),
        bound="single",
        sample="rwalk",
        n_live=500,
        periodic=[0],
        seed=11,
    )
    res = sampler.run_nested(dlogz=0.05, n_mcmc=40, show_progress=False)
    err = res.log_Z_err
    assert abs(res.log_Z - log_Z_true) < 5.0 * err, (
        f"log Z = {res.log_Z:.4f} +- {err:.4f} vs log I_0(kappa) = "
        f"{log_Z_true:.4f}"
    )


@pytest.mark.benchmark
def test_periodic_inside_transdim_group_wraps_boundary() -> None:
    """The use-case that motivates the HD 10180 deployment: a periodic
    coordinate (orbital phase) lives inside a toggleable group. When the
    chain has the group active (gamma=1), the chain's posterior on the
    periodic coord must wrap correctly across the u=0 / u=1 boundary.

    Setup: 1D periodic coord inside a single trans-dim group, with
    likelihood concentrated near u=0/u=1 when gamma=1 (von Mises at the
    boundary) and a slightly lower constant likelihood when gamma=0.
    Inclusion prior favours active so the chain spends time on the slab.
    The off-value is at the slab midpoint (u=0.5) and is therefore not
    a special low-likelihood point.

    Diagnostic: among the gamma=1 samples, fraction near each boundary
    end must be comparable (within a factor of 3) and the joint mass at
    the boundary must exceed 0.5 (von Mises at kappa=8 is concentrated
    enough to satisfy this).
    """
    kappa = 8.0

    def loglike(beta, gamma):
        if gamma[0]:
            theta = float(beta[0])
            return kappa * np.cos(theta)
        return -2.0  # active state strictly preferred when boundary hit

    def prior_transform(u):
        return 2.0 * np.pi * u

    group = daedalus.Group(
        name="phase",
        params=[0],
        off_values=np.array([np.pi]),  # off at slab midpoint
        inclusion_prior=0.7,
    )

    sampler = daedalus.NestedSampler(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=1,
        groups=[group],
        bound="single",
        sample="rwalk",
        n_live=500,
        periodic=[0],
        seed=23,
    )
    res = sampler.run_nested(dlogz=0.1, n_mcmc=40, show_progress=False)

    active_mask = res.gamma[:, 0]
    assert active_mask.sum() > 100, (
        f"chain produced too few active samples: {active_mask.sum()}"
    )
    u_active = res.samples[active_mask, 0] / (2.0 * np.pi)
    near_zero = float(np.mean(u_active < 0.10))
    near_one = float(np.mean(u_active > 0.90))
    assert near_zero + near_one > 0.5, (
        f"trans-dim chain did not concentrate active phase near the "
        f"boundary: near_zero={near_zero:.3f}, near_one={near_one:.3f}"
    )
    assert min(near_zero, near_one) > 0.25 * max(near_zero, near_one), (
        f"trans-dim chain locked one-sidedly on the periodic boundary: "
        f"near_zero={near_zero:.3f}, near_one={near_one:.3f}"
    )


@pytest.mark.benchmark
def test_periodic_does_not_perturb_nonperiodic_axis() -> None:
    """A 2D problem with one periodic and one non-periodic axis must
    leave the non-periodic posterior unchanged. The periodic axis is a
    von Mises on [0, 2 pi) and the non-periodic axis is a unit-cube
    Gaussian on [-3, 3] centred at 0; the marginal posteriors are
    independent and the periodic flag should not affect the latter.
    """
    kappa = 4.0
    sigma = 1.0

    def loglike(beta, gamma):
        theta = float(beta[0])
        x = float(beta[1])
        return kappa * np.cos(theta) - 0.5 * (x / sigma) ** 2

    def prior_transform(u):
        v = np.empty_like(u)
        v[0] = 2.0 * np.pi * u[0]
        v[1] = -3.0 + 6.0 * u[1]
        return v

    sampler = daedalus.NestedSampler(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=2,
        groups=(),
        bound="single",
        sample="rwalk",
        n_live=500,
        periodic=[0],
        seed=19,
    )
    res = sampler.run_nested(dlogz=0.1, n_mcmc=40, show_progress=False)
    x_samples = res.samples[:, 1]
    # Marginal in x should be ~N(0, sigma^2) truncated to [-3, 3]; for
    # sigma = 1 the truncation tails are negligible.
    assert abs(float(np.mean(x_samples))) < 0.2, (
        f"non-periodic axis mean drifted: {float(np.mean(x_samples)):.3f}"
    )
    assert abs(float(np.std(x_samples)) - sigma) < 0.15, (
        f"non-periodic axis std drifted: {float(np.std(x_samples)):.3f}"
    )
