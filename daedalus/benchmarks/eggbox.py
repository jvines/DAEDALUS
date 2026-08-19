"""Eggbox benchmark (Feroz, Hobson & Bridges 2008, MNRAS 398, 1601).

A heavily multimodal 2D test problem: the likelihood is

    L(x, y) = exp((2 + cos(x/2) cos(y/2))^5)

over a uniform prior on [0, 10 pi]^2. The likelihood has 25 isolated peaks
arranged in a 5 x 5 grid. The analytical log evidence is approximately
235.88 (Feroz+ 2008 Sec. 6).

This is the canonical test problem for multimodal NS. SingleEllipsoid will
collapse all peaks into one bound and waste ~95% of samples; MultiEllipsoid
must split into per-peak ellipsoids to maintain efficiency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class EggboxProblem:
    loglike: Callable[[np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    log_Z_true: float


# Analytical evidence reported in Feroz+ 2008. Exact value depends on the
# numerical integration of the eggbox over the prior; we use the value from
# the original paper.
LOG_Z_TRUE = 235.88


def make_problem() -> EggboxProblem:
    box = 10.0 * float(np.pi)

    def loglike(theta: np.ndarray) -> float:
        x, y = theta[0], theta[1]
        return float((2.0 + np.cos(0.5 * x) * np.cos(0.5 * y)) ** 5)

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return box * u

    return EggboxProblem(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=2,
        log_Z_true=LOG_Z_TRUE,
    )
