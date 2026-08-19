"""Multivariate Gaussian benchmark with closed-form log Z.

A continuous-only NS test (no toggleable groups). The first validation gate
for the NS skeleton: log Z_estimated must agree with log Z_true to within
0.5 at n_live = 500.

The problem is an isotropic unit-variance Gaussian likelihood centered at
the origin, with a uniform prior on the box [-prior_half_width,
+prior_half_width]^ndim:

    log L(theta) = -(1/2) sum_i theta_i^2 - (ndim/2) log(2 pi)
    Z = (1 / (2 W)^ndim) * (2 pi)^(ndim/2) * erf(W / sqrt(2))^ndim
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import erf


@dataclass
class GaussianProblem:
    loglike: Callable[[np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    prior_half_width: float
    log_Z_true: float


def make_problem(ndim: int = 5, prior_half_width: float = 10.0) -> GaussianProblem:
    """Construct the isotropic unit Gaussian over a uniform box prior.

    Args:
        ndim: dimensionality of the parameter space.
        prior_half_width: prior is Uniform([-prior_half_width, +prior_half_width]^ndim).

    Returns:
        A `GaussianProblem` carrying loglike, prior_transform, and the
        analytical log_Z.
    """
    if ndim < 1:
        raise ValueError(f"ndim must be >= 1, got {ndim}")
    if prior_half_width <= 0.0:
        raise ValueError(f"prior_half_width must be positive, got {prior_half_width}")

    norm_const = -0.5 * ndim * float(np.log(2.0 * np.pi))
    prior_width = 2.0 * prior_half_width
    log_prior_density = -ndim * float(np.log(prior_width))  # uniform PDF height

    def loglike(theta: np.ndarray) -> float:
        return norm_const - 0.5 * float(np.dot(theta, theta))

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return prior_width * u - prior_half_width

    # L is the unit-Gaussian PDF (the (2 pi)^(-d/2) normalisation is folded
    # into `norm_const`). The evidence integrates the PDF against the box
    # prior, giving:
    #   Z = (1 / (2W)^ndim) * erf(W / sqrt(2))^ndim
    # In log:
    #   log Z = -ndim * log(2W) + ndim * log(erf(W / sqrt(2)))
    log_Z_true = log_prior_density + ndim * float(
        np.log(erf(prior_half_width / np.sqrt(2.0)))
    )

    return GaussianProblem(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=ndim,
        prior_half_width=prior_half_width,
        log_Z_true=log_Z_true,
    )
