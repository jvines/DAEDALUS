"""Gaussian-shells benchmark (Mukherjee, Parkinson & Liddle 2006,
ApJ 638, L51; Feroz, Hobson & Bridges 2009 Section 6.2).

Two well-separated thin Gaussian shells in $n$-dimensional space, the
canonical nested-sampling multimodal-on-disjoint-support benchmark. The
likelihood is

    L(theta) = circ(theta - c1; r, w) + circ(theta - c2; r, w)

with

    circ(theta; r, w) = (1 / sqrt(2 pi w^2)) *
                        exp(-(|theta| - r)^2 / (2 w^2)),

where ``c1`` and ``c2`` are the shell centres, ``r`` is the shell
radius, and ``w`` is the radial width. With ``w << r``, mass is
concentrated on two thin (n-1)-dimensional spherical shells and the
two modes have *disjoint support* up to negligible exponential tails
--- the test that distinguishes a multi-ellipsoidal NS bound from a
single-ellipsoidal one.

Default geometry follows Feroz+ 2009 Section 6.2: 2D, ``r = 2``,
``w = 0.1``, centres at ``(-3.5, 0)`` and ``(+3.5, 0)``, prior uniform
on ``[-6, 6]^n``. The analytic log-evidence under this geometry is
``-1.75`` (their Table 6); we use it as the validation target.

Generalisation to higher ``ndim`` keeps the same ``r``, ``w``, prior
box, and centre offsets along the first coordinate; the analytic
log-evidence then has to be recomputed numerically, so the bundled
``log_Z_true`` is only correct for ``ndim = 2``. Set ``log_Z_true=None``
on higher-dim instantiations and validate via internal consistency
(seed-to-seed std vs reported err).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


# Analytic log-evidence in 2D (Feroz, Hobson & Bridges 2009, Table 6).
LOG_Z_TRUE_2D = -1.75


@dataclass
class GaussianShellsProblem:
    loglike: Callable[[np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    radius: float
    width: float
    centres: np.ndarray  # (2, ndim)
    box_half_width: float
    log_Z_true: float | None  # None for ndim != 2


def make_problem(
    ndim: int = 2,
    radius: float = 2.0,
    width: float = 0.1,
    centre_offset: float = 3.5,
    box_half_width: float = 6.0,
) -> GaussianShellsProblem:
    """Build the Gaussian-shells problem.

    Parameters
    ----------
    ndim
        Dimensionality of the parameter space. Default 2 matches the
        Mukherjee+ 2006 / Feroz+ 2009 reference geometry; higher ``ndim``
        retains the same ``r``, ``w``, and centre placement along the
        first coordinate.
    radius
        Radial location of the shells (same for both shells).
    width
        Radial Gaussian width. ``width << radius`` gives well-resolved
        thin shells; the disjoint-support property holds for ``width``
        up to ~ ``centre_offset / 2``.
    centre_offset
        Half-distance between the two shell centres along the first
        coordinate.
    box_half_width
        Uniform-prior half-width: ``[-box_half_width, +box_half_width]^ndim``.
    """
    if ndim < 2:
        raise ValueError(f"ndim must be >= 2, got {ndim}")

    c1 = np.zeros(ndim, dtype=np.float64)
    c2 = np.zeros(ndim, dtype=np.float64)
    c1[0] = -centre_offset
    c2[0] = +centre_offset
    centres = np.stack([c1, c2], axis=0)

    log_norm = -0.5 * float(np.log(2.0 * np.pi * width * width))
    inv_2w2 = 0.5 / (width * width)

    def _shell_log_density(theta: np.ndarray, centre: np.ndarray) -> float:
        d = float(np.linalg.norm(theta - centre))
        return log_norm - inv_2w2 * (d - radius) ** 2

    def loglike(theta: np.ndarray) -> float:
        l1 = _shell_log_density(theta, c1)
        l2 = _shell_log_density(theta, c2)
        # Sum of two shells; numerically safe via logaddexp.
        return float(np.logaddexp(l1, l2))

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return 2.0 * box_half_width * u - box_half_width

    log_Z_true = LOG_Z_TRUE_2D if ndim == 2 else None

    return GaussianShellsProblem(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=ndim,
        radius=radius,
        width=width,
        centres=centres,
        box_half_width=box_half_width,
        log_Z_true=log_Z_true,
    )
