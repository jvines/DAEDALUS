"""A tiny synthetic spike-and-slab regression for simulation-based
calibration (SBC; Talts, Betancourt, Simpson, Vehtari 2020,
arXiv:1804.06788) of daedalus.

The setup is the smallest non-trivial trans-dim problem: ``g`` toggleable
predictors, each with one continuous coefficient under an independent
Zellner slab. Likelihood is Gaussian with fixed nuisance variance. The
problem is small enough that M = 100+ trials are feasible (each chain
runs in seconds) and the SBC rank statistics actually have power to
detect bias.

Because this benchmark generates fresh ``(gamma_true, beta_true, y)``
triples per trial, it does *not* expose precomputed ground truth like
:mod:`diabetes`; instead it provides:

    * :func:`make_sbc_problem`         — define the prior, design matrix,
                                         nuisance scales.
    * :meth:`SBCProblem.sample_truth`  — draw a ``(gamma, beta)`` pair from
                                         the spike-and-slab prior.
    * :meth:`SBCProblem.sample_data`   — generate ``y`` given a ``beta``.
    * :meth:`SBCProblem.make_chain_problem` — assemble the daedalus-ready
                                         loglike + prior_transform + groups
                                         for a given dataset ``y``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class ChainProblem:
    """daedalus-ready specification of a single SBC trial's posterior."""
    loglike: Callable[[np.ndarray, np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    groups_kwargs: list[dict]


@dataclass
class SBCProblem:
    """Tiny synthetic spike-and-slab regression for SBC."""

    X: np.ndarray              # design matrix, shape (N, g)
    sigma2: float              # observation noise variance (fixed)
    tau2: float                # slab variance (fixed)
    inclusion_prior: float     # P(gamma_k = 1) per group, iid

    @property
    def N(self) -> int:
        return int(self.X.shape[0])

    @property
    def ndim(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_groups(self) -> int:
        return self.ndim  # one group per predictor

    def sample_truth(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``(gamma_true, beta_true)`` from the spike-and-slab prior.

        ``gamma_true`` is a length-``g`` boolean vector;
        ``beta_true`` is a length-``g`` float vector with zeros where
        ``gamma_true`` is False, slab draws elsewhere.
        """
        gamma = rng.uniform(size=self.n_groups) < self.inclusion_prior
        beta = np.zeros(self.ndim, dtype=np.float64)
        sd_slab = float(np.sqrt(self.tau2))
        active = np.where(gamma)[0]
        if active.size > 0:
            beta[active] = rng.standard_normal(active.size) * sd_slab
        return gamma, beta

    def sample_data(
        self, beta: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample ``y ~ N(X beta, sigma2 I)``."""
        return self.X @ beta + rng.standard_normal(self.N) * float(
            np.sqrt(self.sigma2)
        )

    def make_chain_problem(self, y: np.ndarray) -> ChainProblem:
        """Build the daedalus-ready loglike + prior_transform + groups for
        a fixed dataset ``y``. Reused across SBC trials by passing in the
        per-trial ``y``."""
        from scipy.stats import norm as _norm

        sigma2 = self.sigma2
        tau2 = self.tau2
        N = self.N
        g = self.n_groups
        X = self.X
        inv_2sigma2 = 0.5 / sigma2
        log_2pi_sigma2 = float(np.log(2.0 * np.pi * sigma2))
        base_log_const = -0.5 * N * log_2pi_sigma2
        sd_slab = float(np.sqrt(tau2))

        def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                resid = y - X @ beta
                ssr = float(np.dot(resid, resid))
            if not np.isfinite(ssr):
                return -np.inf
            return base_log_const - inv_2sigma2 * ssr

        def prior_transform(u: np.ndarray) -> np.ndarray:
            return _norm.ppf(np.clip(u, 1e-15, 1.0 - 1e-15)) * sd_slab

        groups_kwargs = [
            dict(
                name=f"x{i}",
                params=[i],
                off_values=np.array([0.0]),
                inclusion_prior=self.inclusion_prior,
            )
            for i in range(g)
        ]
        return ChainProblem(
            loglike=loglike,
            prior_transform=prior_transform,
            ndim=g,
            groups_kwargs=groups_kwargs,
        )


def make_sbc_problem(
    ndim: int = 2,
    N: int = 12,
    sigma2: float = 1.0,
    tau2: float = 4.0,
    inclusion_prior: float = 0.5,
    design_seed: int = 2026,
) -> SBCProblem:
    """Construct a small SBC problem with a fixed orthonormal design matrix.

    Parameters
    ----------
    ndim
        Number of predictors / toggleable groups.
    N
        Number of observations. Choose ``N`` near the noise level so the
        posterior is in the interesting "ambiguous" regime where both
        included and excluded models are plausible. ``N=12`` with the
        defaults below sits in that regime.
    sigma2
        Observation noise variance.
    tau2
        Slab variance.
    inclusion_prior
        ``P(gamma_k = 1)`` per group, iid across groups.
    design_seed
        RNG seed for the design matrix. Held fixed across SBC trials so
        that only the truth and noise vary.

    Notes
    -----
    The design matrix is constructed by drawing ``X ~ N(0, I)`` of shape
    ``(N, ndim)``, then QR-orthonormalising and rescaling so
    ``X^T X = N I``. This makes the per-coefficient posteriors
    independent, which sharpens the SBC diagnostic on each coordinate
    individually.
    """
    if ndim < 1:
        raise ValueError(f"ndim must be >= 1, got {ndim}")
    if N <= ndim:
        raise ValueError(f"N must exceed ndim for orthonormal X, got N={N}, ndim={ndim}")
    rng = np.random.default_rng(design_seed)
    X_raw = rng.standard_normal((N, ndim))
    Q, _ = np.linalg.qr(X_raw)
    X = Q * float(np.sqrt(N))
    return SBCProblem(
        X=X,
        sigma2=sigma2,
        tau2=tau2,
        inclusion_prior=inclusion_prior,
    )
