"""Synthetic spike-and-slab benchmark with closed-form inclusion probabilities.

A small hand-pickable problem to validate the MoMS flip kernel before
graduating to the diabetes dataset. Two parameters:

    beta_0  -- always-on, uniform prior on [-W, +W].
    beta_1  -- toggleable group; off-value 0; uniform slab on [-W, +W] when
               active; inclusion prior `inclusion_prior`.

Likelihood is an isotropic unit Gaussian centered at the origin:

    log L(beta) = -(1/2) (beta_0^2 + beta_1^2) - log(2 pi)

Closed-form posterior inclusion probability for beta_1:

    Z_off = L(beta_0, 0) integrated over beta_0 prior
          = (1/(2W)) * (1/(2 pi)) * sqrt(2 pi) * erf(W/sqrt(2))
          = erf(W/sqrt(2)) / (2W * sqrt(2 pi))

    Z_on  = ∫∫ L(beta_0, beta_1) / (2W)^2 dbeta_0 dbeta_1
          = (1/(4 W^2 * 2 pi)) * 2 pi * erf(W/sqrt(2))^2
          = erf(W/sqrt(2))^2 / (4 W^2)

    P(gamma_1 = 1 | y) = pi_inc * Z_on / (pi_off * Z_off + pi_inc * Z_on)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import erf


@dataclass
class SpikeSlabProblem:
    loglike: Callable[[np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    prior_half_width: float
    inclusion_prior: float
    log_Z_true: float
    inclusion_prob_true: float


def make_problem(
    prior_half_width: float = 10.0, inclusion_prior: float = 0.5
) -> SpikeSlabProblem:
    if prior_half_width <= 0.0:
        raise ValueError("prior_half_width must be positive")
    if not 0.0 < inclusion_prior < 1.0:
        raise ValueError("inclusion_prior must be in (0, 1)")

    W = float(prior_half_width)
    norm = -float(np.log(2.0 * np.pi))

    def loglike(theta: np.ndarray) -> float:
        return norm - 0.5 * float(np.dot(theta, theta))

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return 2.0 * W * u - W

    erf_W = float(erf(W / np.sqrt(2.0)))
    sqrt_2pi = float(np.sqrt(2.0 * np.pi))

    # Z_off: integrate L(beta_0, 0) over beta_0 in [-W, W] under uniform prior
    Z_off = erf_W / (2.0 * W * sqrt_2pi)
    # Z_on: integrate L(beta_0, beta_1) over both in [-W, W] under uniform priors
    Z_on = erf_W * erf_W / (4.0 * W * W)
    pi_off = 1.0 - inclusion_prior
    pi_on = inclusion_prior
    log_Z_true = float(np.log(pi_off * Z_off + pi_on * Z_on))
    inclusion_prob_true = pi_on * Z_on / (pi_off * Z_off + pi_on * Z_on)

    return SpikeSlabProblem(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=2,
        prior_half_width=W,
        inclusion_prior=inclusion_prior,
        log_Z_true=log_Z_true,
        inclusion_prob_true=inclusion_prob_true,
    )
