"""Rosenbrock-banana benchmark for fixed-dim NS.

The classical 2D Rosenbrock function

    f(x, y) = (1 - x)^2 + 100 (y - x^2)^2

defines a thin, curved ridge of high posterior density along $y = x^2$
near $x = 1$. Used here as

    log L(x, y) = -f(x, y) / 2

over a uniform prior on $[-5, 5]^2$. The likelihood is well-defined
but extremely degenerate: posterior mass concentrates on a narrow,
non-linear ridge. The benchmark probes the within-model sampler's
ability to mix along curvature, not the bound's ability to handle
multimodality. Single- and multi-ellipsoidal bounds both inflate
significantly to enclose the ridge, so the test is harder for the
sampler than for the bound.

There is no clean analytic log evidence for the Rosenbrock posterior
on a finite box, so we provide the reference value by 2D adaptive
quadrature on the (densely-sampled) likelihood. The reference is
computed once at module-import time and cached on
:attr:`RosenbrockProblem.log_Z_true`. Tolerance is set generously in
the e2e test because the reference itself carries quadrature error
of order $10^{-3}$.

Reference: Rosenbrock 1960, *The Computer Journal* 3, 175. Used as a
benchmark in optimisation and Bayesian inference literature; see
e.g. \\citet{Feroz2009} Section 6.4 and the dynesty examples
(\\citealt{Speagle2020}).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class RosenbrockProblem:
    loglike: Callable[[np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    box_half_width: float
    log_Z_true: float


def _log_evidence_quadrature(box_half_width: float) -> float:
    """Compute log Z by 2D adaptive quadrature.

    The Rosenbrock posterior on a finite uniform box has

        Z = (1 / (2W)^2) * int_{-W}^{W} int_{-W}^{W} exp(log L(x, y)) dx dy

    where W is ``box_half_width``. Quadrature is run on log L directly
    via ``scipy.integrate.dblquad`` with adaptive subdivision; the
    integrand is smooth so machine-precision tolerances suffice.
    """
    from scipy.integrate import dblquad

    W = float(box_half_width)
    log_prior_norm = -2.0 * float(np.log(2.0 * W))

    def integrand(y: float, x: float) -> float:
        return float(np.exp(-0.5 * ((1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2)))

    val, _err = dblquad(integrand, -W, W, -W, W, epsabs=1e-10, epsrel=1e-10)
    return float(np.log(val)) + log_prior_norm


def make_problem(box_half_width: float = 5.0) -> RosenbrockProblem:
    W = float(box_half_width)

    def loglike(theta: np.ndarray) -> float:
        x, y = float(theta[0]), float(theta[1])
        return -0.5 * ((1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2)

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return 2.0 * W * u - W

    log_Z_true = _log_evidence_quadrature(W)

    return RosenbrockProblem(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=2,
        box_half_width=W,
        log_Z_true=log_Z_true,
    )
