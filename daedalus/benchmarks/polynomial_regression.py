"""Polynomial regression with unknown polynomial degree (trans-dim demo).

Generates synthetic-but-realistic data ``y_i = sum_k beta_k * x_i^k +
epsilon_i`` on a fixed ``x``-grid with epsilon ~ N(0, sigma^2), then asks
the chain to recover the active subset of polynomial degrees.

Setup
-----
* Degrees ``k = 0..K_max`` form ``K_max + 1`` toggleable groups, each
  with a single continuous parameter (the coefficient ``beta_k``).
* Spike: when ``gamma_k = 0``, ``beta_k`` is pinned to 0.
* Slab: ``beta_k | gamma_k = 1 ~ N(0, tau2)``, independent across k.
* Likelihood: ``y | beta, sigma2 ~ N(X beta, sigma2 I)`` with the
  Vandermonde design ``X[i, k] = x_i^k``. ``sigma`` is fixed (not
  sampled).
* Closed-form ground truth comes from enumerating all
  ``2 ** (K_max + 1)`` inclusion patterns -- structurally identical to
  the conjugate Gaussian / independent-slab calculation in
  :func:`daedalus.benchmarks.diabetes.enumerate_full_posterior`.

Default true coefficients: ``beta_0 = 1.0``, ``beta_1 = 2.0``,
``beta_3 = -3.0``, all others zero. With ``K_max = 6`` the chain should
return high inclusion probability for degrees ``{0, 1, 3}`` and low
inclusion probability for ``{2, 4, 5, 6}``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def _design_matrix(x: np.ndarray, K_max: int) -> np.ndarray:
    """Vandermonde design matrix with columns ``x**0, x**1, ..., x**K_max``."""
    return np.vander(x, N=K_max + 1, increasing=True)


def _per_gamma_moments(
    X: np.ndarray, y: np.ndarray, sigma2: float, tau2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-gamma posterior log-marginal likelihood and beta moments.

    Mirrors :func:`daedalus.benchmarks.diabetes._per_gamma_moments`. For
    each of the ``2**p`` inclusion patterns gamma, returns:

    * ``log_marginal[m]`` = ``log p(y | gamma_m)`` under the conjugate
      Gaussian / independent-Zellner-slab pair.
    * ``mu[m]`` of shape ``(p,)`` -- ``E[beta_i | y, gamma_m]``; zero
      where ``gamma_{m,i} = 0``.
    * ``var[m]`` of shape ``(p,)`` -- ``Var[beta_i | y, gamma_m]``; zero
      where ``gamma_{m,i} = 0``.
    """
    N, p = X.shape
    n_models = 1 << p
    inv_tau2 = 1.0 / tau2
    inv_sigma2 = 1.0 / sigma2
    yty = float(np.dot(y, y))
    log_2pi_sigma2 = float(np.log(2.0 * np.pi * sigma2))
    log_tau2 = float(np.log(tau2))

    log_marginal = np.empty(n_models, dtype=np.float64)
    mu_full = np.zeros((n_models, p), dtype=np.float64)
    var_full = np.zeros((n_models, p), dtype=np.float64)

    for m in range(n_models):
        bits = np.array([(m >> b) & 1 for b in range(p)], dtype=bool)
        p_gamma = int(bits.sum())
        if p_gamma == 0:
            log_marginal[m] = -0.5 * N * log_2pi_sigma2 - 0.5 * yty * inv_sigma2
            continue
        Xa = X[:, bits]
        XtX = Xa.T @ Xa
        precision = XtX * inv_sigma2 + np.eye(p_gamma) * inv_tau2
        L = np.linalg.cholesky(precision)
        Xt_y = Xa.T @ y
        z = np.linalg.solve(L, Xt_y * inv_sigma2)
        mu_a = np.linalg.solve(L.T, z)
        log_det_Sigma = -2.0 * float(np.sum(np.log(np.diag(L))))
        L_inv = np.linalg.solve(L, np.eye(p_gamma))
        var_a = np.einsum("ji,ji->i", L_inv, L_inv)
        mu_prec_mu = float(np.dot(z, z))
        log_marginal[m] = (
            -0.5 * N * log_2pi_sigma2
            - 0.5 * p_gamma * log_tau2
            + 0.5 * log_det_Sigma
            - 0.5 * yty * inv_sigma2
            + 0.5 * mu_prec_mu
        )
        mu_full[m, bits] = mu_a
        var_full[m, bits] = var_a
    return log_marginal, mu_full, var_full


def enumerate_full_posterior(
    X: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    tau2: float,
    prior_inclusion: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form ground truth via enumeration over ``2**p`` inclusion
    patterns.

    Returns three length-``p`` arrays: ``P[gamma_k = 1 | y]``,
    ``E[beta_k | y]``, ``SD[beta_k | y]``.
    """
    N, p = X.shape
    log_marginal, mu_full, var_full = _per_gamma_moments(X, y, sigma2, tau2)
    log_pi_inc = float(np.log(prior_inclusion))
    log_pi_exc = float(np.log1p(-prior_inclusion))
    log_post = np.empty_like(log_marginal)
    masks = np.empty((log_marginal.size, p), dtype=bool)
    for m in range(log_marginal.size):
        bits = np.array([(m >> b) & 1 for b in range(p)], dtype=bool)
        masks[m] = bits
        n_in = int(bits.sum())
        log_post[m] = log_marginal[m] + n_in * log_pi_inc + (p - n_in) * log_pi_exc
    log_post -= float(np.logaddexp.reduce(log_post))
    post = np.exp(log_post)
    inclusion = (masks * post[:, None]).sum(axis=0)
    e_beta = (mu_full * post[:, None]).sum(axis=0)
    e_beta_sq = ((mu_full ** 2 + var_full) * post[:, None]).sum(axis=0)
    sd_beta = np.sqrt(np.maximum(e_beta_sq - e_beta ** 2, 0.0))
    return inclusion, e_beta, sd_beta


@dataclass
class PolynomialProblem:
    """Trans-dim polynomial-regression benchmark.

    Attributes
    ----------
    loglike, prior_transform, ndim, groups_kwargs
        Standard daedalus problem surface. Each polynomial degree gets
        its own ``Group`` with one continuous parameter (the
        coefficient).
    log_prior_continuous_factory
        ``log_prior_continuous_factory(i)`` returns the log-density of
        the independent ``N(0, tau2)`` slab on the i-th coefficient,
        for users supplying a custom :class:`BirthProposal`.
    inclusion_prob_true, e_beta_true, sd_beta_true
        Length-``K_max + 1`` arrays of closed-form ground truth from
        enumeration over ``2**(K_max + 1)`` inclusion patterns.
    true_active
        Boolean array of length ``K_max + 1``; ``True`` where the data
        was generated with a non-zero coefficient.
    true_coefficients
        Length-``K_max + 1`` array; the coefficients used to generate
        the synthetic data (zero on inactive degrees).
    sigma2, tau2
        Fixed nuisance variance and prior slab variance.
    K_max
        Maximum polynomial degree included (so ``p = K_max + 1`` groups
        total).
    x, y
        The synthetic dataset.
    group_names
        Per-degree labels of the form ``"deg0"``, ``"deg1"``, ....
    """

    loglike: Callable[[np.ndarray, np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    groups_kwargs: list[dict]
    log_prior_continuous_factory: Callable[[int], Callable[[np.ndarray], float]]
    inclusion_prob_true: np.ndarray
    e_beta_true: np.ndarray
    sd_beta_true: np.ndarray
    true_active: np.ndarray
    true_coefficients: np.ndarray
    sigma2: float
    tau2: float
    K_max: int
    x: np.ndarray
    y: np.ndarray
    group_names: list[str]


def make_problem(
    K_max: int = 6,
    n: int = 50,
    sigma: float = 0.5,
    tau2: float = 25.0,
    true_coefficients: dict[int, float] | None = None,
    prior_inclusion: float = 0.5,
    data_seed: int = 0,
) -> PolynomialProblem:
    """Build the polynomial-regression trans-dim benchmark.

    Parameters
    ----------
    K_max
        Maximum polynomial degree; the chain explores degrees
        ``0..K_max``.
    n
        Number of observations on a uniform grid in ``[-1, 1]``.
    sigma
        Standard deviation of the Gaussian noise on ``y``.
    tau2
        Variance of the independent Gaussian slab prior on each
        coefficient. Default 25 (broad; SD = 5 covers the default true
        coefficients comfortably).
    true_coefficients
        Mapping degree -> coefficient value. Missing degrees default to
        zero. Default: ``{0: 1.0, 1: 2.0, 3: -3.0}``.
    prior_inclusion
        Prior probability that any given degree is active. 0.5 -> uniform
        over inclusion vectors.
    data_seed
        Seed for the noise realisation. Fixed per problem so all chains
        see identical data.

    Returns
    -------
    PolynomialProblem
        Bundle of likelihood, prior, group declarations, and closed-form
        ground truth.
    """
    if true_coefficients is None:
        true_coefficients = {0: 1.0, 1: 2.0, 3: -3.0}

    p = K_max + 1
    true_beta = np.zeros(p, dtype=np.float64)
    for k, v in true_coefficients.items():
        if k < 0 or k > K_max:
            raise ValueError(
                f"true coefficient at degree {k} out of range [0, {K_max}]"
            )
        true_beta[k] = float(v)
    true_active = true_beta != 0.0

    x = np.linspace(-1.0, 1.0, n)
    X = _design_matrix(x, K_max)
    rng = np.random.default_rng(data_seed)
    y_clean = X @ true_beta
    y = y_clean + rng.normal(0.0, sigma, size=n)

    sigma2 = float(sigma) ** 2
    sd_slab = float(np.sqrt(tau2))

    inv_2sigma2 = 0.5 / sigma2
    log_2pi_sigma2 = float(np.log(2.0 * np.pi * sigma2))
    base_log_const = -0.5 * n * log_2pi_sigma2

    def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
        # Inactive coefficients are pinned to off (=0) by daedalus, so
        # ``X @ beta`` is the correct prediction.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred = X @ beta
            resid = y - pred
            ssr = float(np.dot(resid, resid))
        if not np.isfinite(ssr):
            return -np.inf
        return base_log_const - inv_2sigma2 * ssr

    from scipy.stats import norm as _norm

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return _norm.ppf(np.clip(u, 1e-15, 1.0 - 1e-15)) * sd_slab

    group_names = [f"deg{k}" for k in range(p)]
    groups_kwargs = [
        dict(
            name=group_names[k],
            params=[k],
            off_values=np.array([0.0]),
            inclusion_prior=prior_inclusion,
        )
        for k in range(p)
    ]

    log_const_slab = -0.5 * float(np.log(2.0 * np.pi * tau2))
    inv_2tau2 = 0.5 / tau2

    def log_prior_continuous_factory(_i: int):
        def log_prior(beta_for_group: np.ndarray) -> float:
            b = float(beta_for_group[0])
            return log_const_slab - inv_2tau2 * b * b
        return log_prior

    inclusion_prob_true, e_beta_true, sd_beta_true = enumerate_full_posterior(
        X, y, sigma2, tau2, prior_inclusion
    )

    return PolynomialProblem(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=p,
        groups_kwargs=groups_kwargs,
        log_prior_continuous_factory=log_prior_continuous_factory,
        inclusion_prob_true=inclusion_prob_true,
        e_beta_true=e_beta_true,
        sd_beta_true=sd_beta_true,
        true_active=true_active,
        true_coefficients=true_beta,
        sigma2=sigma2,
        tau2=float(tau2),
        K_max=K_max,
        x=x,
        y=y,
        group_names=group_names,
    )
