"""Diabetes-dataset variable selection benchmark (van den Bergh+ 2026, Table 1).

Loads the Efron, Hastie, Johnstone & Tibshirani (2004) diabetes dataset
(N = 442 patients, p = 10 predictors: AGE, SEX, BMI, BP, S1-S6) and
constructs a Bayesian linear regression with the Jeffreys-Zellner-Siow
(JZS) prior described in Section 2.1 of van den Bergh+ 2026.

The 1024 = 2^10 candidate models are enumerable, so posterior inclusion
probabilities are known to machine precision and serve as the ground truth
for daedalus validation. Required pass criterion: |P_estimated(gamma_k=1) -
P_truth(gamma_k=1)| < 0.02 for every predictor.

Setup follows the marginal approach: beta is integrated out analytically
under the JZS conjugate g-prior, leaving the chain to sample only over the
inclusion vector gamma. The likelihood becomes the Bayes factor of the
candidate model relative to the null. To keep the chain on a non-empty
continuous space, we add a single dummy uniform parameter that does not
enter the likelihood; ndim = 1 is the smallest daedalus currently supports.

Ground truth from van den Bergh+ 2026 Table 1:

    AGE 0.079   SEX 0.987   BMI 1.000   BP 1.000   S1 0.661
    S2  0.453   S3  0.515   S4  0.257   S5  1.000   S6  0.125
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

PREDICTOR_NAMES: tuple[str, ...] = (
    "AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6",
)

GROUND_TRUTH_INCLUSION: dict[str, float] = {
    "AGE": 0.079,
    "SEX": 0.987,
    "BMI": 1.000,
    "BP": 1.000,
    "S1": 0.661,
    "S2": 0.453,
    "S3": 0.515,
    "S4": 0.257,
    "S5": 1.000,
    "S6": 0.125,
}


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the Efron+ 2004 diabetes dataset.

    Uses scikit-learn's `load_diabetes` (X is mean-centered with each
    column scaled to unit norm; y is centered). The JZS Bayes factor
    depends on the predictors only through the coefficient of
    determination R^2, which is invariant under linear column scaling, so
    sklearn's standardisation does not affect inclusion probabilities.
    """
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    X = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.float64)
    y = y - y.mean()
    return X, y


def jzs_log_bf(N: int, p: int, R2: float) -> float:
    """Log Bayes factor of model M_j (with p predictors and coefficient of
    determination R2) vs the null model M_0 under the Jeffreys-Zellner-Siow
    prior.

    Uses Eq. (1) of van den Bergh+ 2026:

        BF_j0 = int_0^inf (1+g)^((N-1-p)/2) * (1+(1-R2)g)^(-(N-1)/2) * pi(g) dg

    where pi(g) = Inv-Gamma(1/2, n/2). The integrand is computed in
    log-space and accumulated via `scipy.integrate.quad`. Returns 0.0
    for the null model (p = 0).
    """
    if p == 0:
        return 0.0
    if not 0.0 <= R2 < 1.0:
        raise ValueError(f"R2 must be in [0, 1), got {R2}")

    n = float(N)
    # log of (n/2)^(1/2) / Gamma(1/2). Gamma(1/2) = sqrt(pi).
    log_norm = 0.5 * float(np.log(n / 2.0)) - 0.5 * float(np.log(np.pi))
    half_n_minus_1_minus_p = 0.5 * (N - 1 - p)
    half_n_minus_1 = 0.5 * (N - 1)
    one_minus_R2 = 1.0 - R2

    def integrand(g: float) -> float:
        if g <= 0.0:
            return 0.0
        log_val = (
            half_n_minus_1_minus_p * float(np.log1p(g))
            - half_n_minus_1 * float(np.log1p(one_minus_R2 * g))
            + log_norm
            - 1.5 * float(np.log(g))
            - n / (2.0 * g)
        )
        return float(np.exp(log_val))

    val, _ = quad(integrand, 0.0, np.inf, limit=200)
    if val <= 0.0:
        return float("-inf")
    return float(np.log(val))


def _r_squared(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Coefficient of determination from regressing y on the columns of X
    selected by the boolean `mask`. Assumes y is mean-centered.

    Uses an SVD-based pseudo-inverse with explicit rank truncation; the
    `np.errstate` block silences the harmless overflow warnings emitted by
    NumPy when degenerate diabetes subsets land on extreme singular values.
    """
    if not mask.any():
        return 0.0
    Xa = X[:, mask]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        U, S, Vt = np.linalg.svd(Xa, full_matrices=False)
        if S.size == 0:
            return 0.0
        cutoff = S.max() * max(Xa.shape) * np.finfo(Xa.dtype).eps
        keep = S > cutoff
        if not keep.any():
            return 0.0
        S_inv = np.where(keep, 1.0 / np.where(keep, S, 1.0), 0.0)
        beta_hat = Vt.T @ (S_inv * (U.T @ y))
        resid = y - Xa @ beta_hat
        ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y, y))
    if ss_tot == 0.0 or not np.isfinite(ss_res):
        return 0.0
    return 1.0 - ss_res / ss_tot


def enumerate_inclusion_probabilities(
    X: np.ndarray, y: np.ndarray, prior_inclusion: float = 0.5
) -> dict[str, float]:
    """Enumerate all 2^p models and return the marginal inclusion probability
    for each predictor under the JZS prior + uniform model prior.
    """
    N, p = X.shape
    n_models = 1 << p
    log_post = np.empty(n_models, dtype=np.float64)
    masks = np.empty((n_models, p), dtype=bool)
    log_pi_inc = float(np.log(prior_inclusion))
    log_pi_exc = float(np.log1p(-prior_inclusion))
    for m in range(n_models):
        bits = np.array([(m >> b) & 1 for b in range(p)], dtype=bool)
        masks[m] = bits
        R2 = _r_squared(X, y, bits)
        log_bf = jzs_log_bf(N, int(bits.sum()), R2)
        log_prior = float(bits.sum()) * log_pi_inc + float(p - bits.sum()) * log_pi_exc
        log_post[m] = log_bf + log_prior
    log_post -= float(np.logaddexp.reduce(log_post))
    post = np.exp(log_post)
    inclusion = (masks * post[:, None]).sum(axis=0)
    return {name: float(p_inc) for name, p_inc in zip(PREDICTOR_NAMES, inclusion)}


@dataclass
class DiabetesProblem:
    loglike: Callable[[np.ndarray, np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    groups_kwargs: list[dict]  # ready to feed into daedalus.Group(**kwargs)
    inclusion_prob_true: dict[str, float]


@dataclass
class DiabetesProblemFull:
    """Diabetes variable selection with explicit beta coordinates.

    Unlike :class:`DiabetesProblem`, this variant does *not* marginalise
    beta out. Each predictor's regression coefficient is sampled, so the
    chain explores the full ``(beta, gamma)`` posterior. Closed-form
    enumeration provides ground truth for inclusion probabilities and
    posterior moments of beta. This is the setup that actually exercises
    :class:`GaussianRWBirth` (which is a no-op on the marginalised
    problem because each group has zero continuous parameters).

    Attributes
    ----------
    loglike, prior_transform, ndim, groups_kwargs
        Pass straight to ``daedalus.NestedSampler``. Each predictor gets its
        own ``Group`` with one continuous parameter (the regression
        coefficient).
    log_prior_continuous_factory
        ``log_prior_continuous_factory(i)`` returns the log-density of the
        independent Zellner slab for predictor ``i``. Required when the
        user supplies a custom :class:`BirthProposal` (the package no
        longer cancels the continuous prior with the proposal density).
    inclusion_prob_true, e_beta_true, sd_beta_true
        Closed-form ground truth from enumeration over all ``2**p`` models.
    sigma2, tau2
        Fixed nuisance scales used in the likelihood and slab prior.
    """

    loglike: Callable[[np.ndarray, np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    groups_kwargs: list[dict]
    log_prior_continuous_factory: Callable[[int], Callable[[np.ndarray], float]]
    inclusion_prob_true: dict[str, float]
    e_beta_true: dict[str, float]
    sd_beta_true: dict[str, float]
    sigma2: float
    tau2: float


def _per_gamma_moments(
    X: np.ndarray, y: np.ndarray, sigma2: float, tau2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-gamma posterior log-marginal-likelihood and beta moments.

    For each of the 2**p inclusion patterns gamma, computes:

    * ``log_marginal[m]`` = log p(y | gamma_m) under
      y ~ N(X beta, sigma2 I) and beta_active ~ N(0, tau2 I).
    * ``mu[m]`` of shape (p,) — E[beta_i | y, gamma_m]; zero where
      gamma_{m,i} = 0 because beta_i is pinned to the off-value.
    * ``var[m]`` of shape (p,) — Var[beta_i | y, gamma_m]; zero where
      gamma_{m,i} = 0.

    The Gaussian conjugate posterior under the active subset is
    ``beta_active | y, gamma ~ N(Sigma X_a^T y / sigma2, Sigma)`` with
    ``Sigma = (X_a^T X_a / sigma2 + I / tau2)^-1``. The marginal
    likelihood log p(y | gamma) follows from Woodbury + the matrix
    determinant lemma; in particular,

        log p(y | gamma) = -N/2 log(2 pi sigma2) - p_gamma/2 log(tau2)
                           + 1/2 log |Sigma| - 1/(2 sigma2) y^T y
                           + 1/2 mu^T Sigma^{-1} mu.
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
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            XtX = Xa.T @ Xa
        precision = XtX * inv_sigma2 + np.eye(p_gamma) * inv_tau2
        # Cholesky for both inverse and log-det (numerically stable).
        L = np.linalg.cholesky(precision)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            Xt_y = Xa.T @ y
        # Solve precision @ mu = Xt_y / sigma2  ->  mu_a
        z = np.linalg.solve(L, Xt_y * inv_sigma2)
        mu_a = np.linalg.solve(L.T, z)
        # log |Sigma| = -2 sum log diag L
        log_det_Sigma = -2.0 * float(np.sum(np.log(np.diag(L))))
        # Diagonal of Sigma = (precision^-1)_{ii} via solving precision X = e_i
        # cheaply: invert L (lower-triangular) and assemble Sigma's diagonal.
        L_inv = np.linalg.solve(L, np.eye(p_gamma))
        var_a = np.einsum("ji,ji->i", L_inv, L_inv)
        # mu^T precision mu
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
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Closed-form ground truth for the conjugate Gaussian / independent-
    Zellner variable-selection setup.

    Returns three dictionaries keyed by predictor name:
    ``P[gamma_i = 1 | y]``, ``E[beta_i | y]``, ``SD[beta_i | y]``.
    Computed by exact enumeration over all ``2**p`` inclusion patterns;
    safe for ``p <= 16`` or so.
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
    return (
        {name: float(inclusion[i]) for i, name in enumerate(PREDICTOR_NAMES)},
        {name: float(e_beta[i]) for i, name in enumerate(PREDICTOR_NAMES)},
        {name: float(sd_beta[i]) for i, name in enumerate(PREDICTOR_NAMES)},
    )


def make_problem_full(
    prior_inclusion: float = 0.5,
    g: float | None = None,
    sigma2: float | None = None,
) -> DiabetesProblemFull:
    """Diabetes variable selection with an explicit beta state.

    Likelihood: ``y | beta, sigma2 ~ N(X beta, sigma2 I)``. Slab prior on
    each active coefficient: ``beta_i | gamma_i = 1 ~ N(0, tau2)`` with
    ``tau2 = g * sigma2``. Spike: ``beta_i | gamma_i = 0`` is fixed at 0.
    Nuisance scales ``sigma2`` and ``g`` are plug-in defaults (residual
    variance from the OLS full-model fit; ``g = N`` for the unit-
    information prior of Kass & Wasserman 1995) but may be overridden.

    Each predictor becomes a one-parameter ``Group``, so the trans-dim
    kernel proposes a real continuous coefficient on every flip. The
    setup deliberately uses an *independent* Zellner slab (diagonal
    covariance) rather than the JZS form
    ``Cov(beta) = g sigma2 (X_gamma^T X_gamma)^{-1}`` because the latter
    is gamma-dependent and would force the user-supplied prior_transform
    to know which other coordinates are active. Independent Zellner is in
    the same conjugate family and admits the same closed-form
    enumeration (see :func:`enumerate_full_posterior`).

    The numbers returned here will *not* match VdB+26 Table 1 exactly --
    Table 1 is full JZS with sigma2 and g treated as nuisance random
    variables, not plug-ins. This benchmark validates that daedalus
    correctly samples the joint ``(beta, gamma)`` posterior under a
    closely related conjugate setup with closed-form truth. The original
    :func:`make_problem` benchmark remains the direct Table 1 (left
    column) validation.
    """
    X, y = load_dataset()
    N, p = X.shape

    if sigma2 is None:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta_full
        sigma2 = float(np.dot(resid, resid)) / float(N)
    if g is None:
        g = float(N)
    tau2 = g * sigma2
    sd_slab = float(np.sqrt(tau2))

    # Cache the inclusion + moments enumeration (cheap at p = 10) and
    # also the per-gamma E[beta|gamma], Var[beta|gamma] tables so the
    # likelihood hot loop is a single dict lookup.
    log_marginal, mu_full, var_full = _per_gamma_moments(X, y, sigma2, tau2)

    # The daedalus chain only needs loglike(beta, gamma); we hand it the
    # full Gaussian likelihood directly. beta is in [-some range] R^p.
    inv_2sigma2 = 0.5 / sigma2
    log_2pi_sigma2 = float(np.log(2.0 * np.pi * sigma2))
    base_log_const = -0.5 * N * log_2pi_sigma2

    def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
        # Off-value override has already pinned inactive beta_i to 0.
        # The Gaussian prior on beta is heavy-tailed under prior_transform,
        # so occasional draws produce overflow in X @ beta; the constraint
        # L > L* will reject these anyway, so we just silence the warning.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred = X @ beta
            resid = y - pred
            ssr = float(np.dot(resid, resid))
        if not np.isfinite(ssr):
            return -np.inf
        return base_log_const - inv_2sigma2 * ssr

    # prior_transform pushes Uniform(0,1)^p to the independent slab
    # N(0, tau2) per coordinate. We use scipy's Gaussian inverse-CDF.
    from scipy.stats import norm as _norm

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return _norm.ppf(np.clip(u, 1e-15, 1.0 - 1e-15)) * sd_slab

    groups_kwargs = [
        dict(
            name=name,
            params=[i],
            off_values=np.array([0.0]),
            inclusion_prior=prior_inclusion,
        )
        for i, name in enumerate(PREDICTOR_NAMES)
    ]

    log_const_slab = -0.5 * float(np.log(2.0 * np.pi * tau2))
    inv_2tau2 = 0.5 / tau2

    def log_prior_continuous_factory(i: int):
        def log_prior(beta_for_group: np.ndarray) -> float:
            # Independent Zellner slab; only one coordinate per group.
            b = float(beta_for_group[0])
            return log_const_slab - inv_2tau2 * b * b
        return log_prior

    inclusion_prob_true, e_beta_true, sd_beta_true = enumerate_full_posterior(
        X, y, sigma2, tau2, prior_inclusion
    )

    return DiabetesProblemFull(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=p,
        groups_kwargs=groups_kwargs,
        log_prior_continuous_factory=log_prior_continuous_factory,
        inclusion_prob_true=inclusion_prob_true,
        e_beta_true=e_beta_true,
        sd_beta_true=sd_beta_true,
        sigma2=sigma2,
        tau2=tau2,
    )


def make_problem(prior_inclusion: float = 0.5) -> DiabetesProblem:
    """Construct the marginal-JZS diabetes setup ready for daedalus.

    Returns a problem with one dummy continuous parameter (uniform on
    [0, 1], does not enter the likelihood) and ten toggleable groups
    (one per predictor, each with no continuous parameters attached).

    With p = 10 predictors there are only 1024 possible gamma vectors,
    each with a fixed log-BF (R^2 is a deterministic function of gamma);
    the table is precomputed at construction so the chain's hot loop is
    a single dict lookup per likelihood evaluation.

    Example:
        from daedalus import NestedSampler, Group
        from daedalus.benchmarks import diabetes

        problem = diabetes.make_problem()
        groups = [Group(**kwargs) for kwargs in problem.groups_kwargs]
        sampler = NestedSampler(
            loglike=problem.loglike,
            prior_transform=problem.prior_transform,
            ndim=problem.ndim,
            groups=groups,
            ...
        )
    """
    X, y = load_dataset()
    N, p = X.shape

    # Precompute log_BF for every possible gamma vector. p = 10 -> 1024 values.
    log_bf_table: dict[int, float] = {}
    for m in range(1 << p):
        bits = np.array([(m >> b) & 1 for b in range(p)], dtype=bool)
        R2 = _r_squared(X, y, bits)
        log_bf_table[m] = jzs_log_bf(N, int(bits.sum()), R2)

    pow2 = np.array([1 << b for b in range(p)], dtype=np.int64)

    def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
        # beta is the dummy continuous parameter; ignored.
        # gamma -> integer key via bitpacking.
        key = int(np.dot(gamma.astype(np.int64), pow2))
        return log_bf_table[key]

    def prior_transform(u: np.ndarray) -> np.ndarray:
        # Identity -- u is the dummy and doesn't affect the likelihood.
        return u.copy()

    groups_kwargs = [
        dict(
            name=name,
            params=[],            # no continuous params attached to the group
            off_values=np.empty(0),
            inclusion_prior=prior_inclusion,
        )
        for name in PREDICTOR_NAMES
    ]

    inclusion_prob_true = enumerate_inclusion_probabilities(X, y, prior_inclusion)

    return DiabetesProblem(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=1,
        groups_kwargs=groups_kwargs,
        inclusion_prob_true=inclusion_prob_true,
    )
