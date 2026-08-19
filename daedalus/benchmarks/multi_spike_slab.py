"""Multi-group spike-and-slab: closed-form trans-dim benchmark with two
toggleable groups and analytic per-gamma evidence.

A direct generalisation of :mod:`spike_slab` from one toggleable group
to two, designed so that every quantity daedalus produces ---
inclusion probabilities, joint log Z, per-model evidences
:math:`Z_\\gamma`, per-coord posterior moments --- has a closed-form
ground truth.

Setup
-----
Two toggleable scalar coordinates :math:`\\beta_0, \\beta_1` with
inclusion indicators :math:`\\gamma_0, \\gamma_1 \\in \\{0,1\\}`. When
:math:`\\gamma_i = 0`, :math:`\\beta_i = 0`; when :math:`\\gamma_i = 1`,
the slab prior is :math:`\\beta_i \\sim \\mathcal{N}(0, \\sigma_{\\rm slab}^2)`.

Likelihood is a diagonal Gaussian on a single observation pair
:math:`(y_0, y_1)`,

.. math::
   \\mathcal{L}(\\bm\\beta, \\bm\\gamma)
   = \\prod_{i=0,1}
     \\mathcal{N}(y_i;\\, \\gamma_i \\beta_i, \\sigma^2),

so the active components couple to their corresponding observation
and inactive components contribute the noise-only term :math:`\\mathcal{N}(y_i;0,\\sigma^2)`.

Closed forms
------------
Per-:math:`\\gamma` model marginal likelihood (i.e., per-model
evidence as reported by :meth:`Results.model_evidences`, which by
convention *excludes* the inclusion prior :math:`P(\\gamma)`):

.. math::
   Z_\\gamma \\;=\\; \\int \\mathcal{L}(\\bm y \\mid \\bm\\beta, \\bm\\gamma)\\,
                          \\pi(\\bm\\beta \\mid \\bm\\gamma)\\, d\\bm\\beta
            \\;=\\; \\prod_{i=0,1} f_{\\gamma_i}(y_i),

where

.. math::
   f_0(y) = \\mathcal{N}(y; 0, \\sigma^2),
   \\qquad
   f_1(y) = \\mathcal{N}(y; 0, \\sigma^2 + \\sigma_{\\rm slab}^2).

Joint evidence (which *does* fold in the inclusion prior
:math:`P(\\gamma) = \\prod_i p_i^{\\gamma_i}(1-p_i)^{1-\\gamma_i}`):

.. math::
   Z \\;=\\; \\sum_\\gamma P(\\gamma)\\, Z_\\gamma.

Per-coordinate posterior moments under :math:`\\gamma_i = 1`:

.. math::
   \\mathbb{E}[\\beta_i \\mid y_i, \\gamma_i = 1]
   = \\frac{\\sigma_{\\rm slab}^2}{\\sigma^2 + \\sigma_{\\rm slab}^2}\\, y_i,
   \\qquad
   \\mathrm{Var}[\\beta_i \\mid y_i, \\gamma_i = 1]
   = \\frac{\\sigma^2 \\sigma_{\\rm slab}^2}{\\sigma^2 + \\sigma_{\\rm slab}^2}.

The marginal posterior :math:`P(\\beta_i \\mid y)` integrates over
:math:`\\gamma`,

.. math::
   \\mathbb{E}[\\beta_i \\mid y]
   = P(\\gamma_i = 1 \\mid y) \\, \\mathbb{E}[\\beta_i \\mid y_i, \\gamma_i = 1].

These are the exact analytic targets the daedalus chain is required to
match.

Why this benchmark
------------------
This is the first daedalus benchmark that *directly* validates per-
model evidences :math:`\\hat Z_\\gamma` from
:meth:`Results.model_evidences` against an analytic reference (rather
than against internal consistency relations). It also stress-tests the
two-group joint inclusion-vector posterior in a regime where each
group is independently informative, complementing the three-group-
and-up validations that come from diabetes and polynomial-regression.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class MultiSpikeSlabProblem:
    loglike: Callable[[np.ndarray, np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    groups_kwargs: list[dict]
    log_prior_continuous_factory: Callable[[int], Callable[[np.ndarray], float]]
    # Closed-form ground truth.
    sigma: float
    sigma_slab: float
    inclusion_priors: np.ndarray
    y: np.ndarray
    log_Z_true: float
    log_Z_gamma_true: dict[tuple[bool, ...], float]
    inclusion_prob_true: np.ndarray
    e_beta_true: np.ndarray
    sd_beta_true: np.ndarray


def _per_gamma_log_evidence(
    y: np.ndarray,
    sigma: float,
    sigma_slab: float,
) -> dict[tuple[bool, ...], float]:
    """Per-model log-evidence ``log Z_gamma = log p(y | gamma)`` for each
    of the 2**g inclusion configurations. *Excludes* the inclusion prior
    ``P(gamma)``: this matches the daedalus :meth:`Results.model_evidences`
    convention, where ``log Z_gamma`` is the per-model marginal likelihood
    used in Bayes-factor computations.
    """
    g = y.size
    sigma2 = sigma * sigma
    sigma2_total = sigma2 + sigma_slab * sigma_slab
    log_f0 = norm.logpdf(y, loc=0.0, scale=sigma)
    log_f1 = norm.logpdf(y, loc=0.0, scale=float(np.sqrt(sigma2_total)))
    out: dict[tuple[bool, ...], float] = {}
    for mask in range(1 << g):
        gamma = tuple(bool((mask >> i) & 1) for i in range(g))
        ll_data = sum(
            (log_f1[i] if gamma[i] else log_f0[i]) for i in range(g)
        )
        out[gamma] = float(ll_data)
    return out


def make_problem(
    y: tuple[float, float] = (2.0, 3.5),
    sigma: float = 1.0,
    sigma_slab: float = 3.0,
    inclusion_priors: tuple[float, float] = (0.5, 0.5),
) -> MultiSpikeSlabProblem:
    """Build a 2-group spike-and-slab problem with closed-form ground truth.

    Defaults give: y = (2.0, 3.5), sigma = 1, sigma_slab = 3, prior
    inclusion 0.5 each. This puts y_0 in the "moderate evidence for
    inclusion" regime (P(gamma_0=1|y) ~ 0.7) and y_1 in the "strong
    evidence" regime (P(gamma_1=1|y) ~ 0.99), so the benchmark covers
    both cases simultaneously.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    inc = np.asarray(inclusion_priors, dtype=np.float64)
    if y_arr.size != inc.size:
        raise ValueError(
            f"y has {y_arr.size} components but inclusion_priors has "
            f"{inc.size}; lengths must match"
        )
    g = y_arr.size
    if not np.all((inc > 0.0) & (inc < 1.0)):
        raise ValueError("inclusion_priors must lie strictly in (0, 1)")
    if sigma <= 0.0 or sigma_slab <= 0.0:
        raise ValueError("sigma and sigma_slab must be positive")

    # Closed-form ground truth. log Z_gamma here is the per-model
    # marginal likelihood (excludes the inclusion prior); the joint
    # evidence is computed by folding in P(gamma) below.
    log_Z_gamma = _per_gamma_log_evidence(y_arr, sigma, sigma_slab)
    log_inc = np.log(inc)
    log_exc = np.log1p(-inc)
    log_joint_terms = []
    for gamma_key, log_Z_g in log_Z_gamma.items():
        gamma_arr = np.asarray(gamma_key, dtype=bool)
        log_P_gamma = float(np.where(gamma_arr, log_inc, log_exc).sum())
        log_joint_terms.append(log_P_gamma + log_Z_g)
    log_Z = float(np.logaddexp.reduce(np.array(log_joint_terms)))
    # Posterior model probabilities P(gamma | y) = P(gamma) Z_gamma / Z.
    P_gamma = {
        gamma_key: float(np.exp(
            float(np.where(np.asarray(gamma_key, dtype=bool), log_inc, log_exc).sum())
            + log_Z_g - log_Z
        ))
        for gamma_key, log_Z_g in log_Z_gamma.items()
    }
    # Marginal inclusion probabilities.
    inclusion = np.zeros(g, dtype=np.float64)
    for gamma, p in P_gamma.items():
        for i, bit in enumerate(gamma):
            if bit:
                inclusion[i] += p
    # Per-coord posterior moments. E[beta_i | y, gamma_i=1] is shrinkage
    # toward 0; Var[...] is the Gaussian conjugate update; marginal
    # posterior moments fold in the inclusion probability.
    sigma2 = sigma * sigma
    sigma2_slab = sigma_slab * sigma_slab
    shrinkage = sigma2_slab / (sigma2 + sigma2_slab)
    var_beta_active = sigma2 * sigma2_slab / (sigma2 + sigma2_slab)
    e_beta_active = shrinkage * y_arr
    e_beta = inclusion * e_beta_active
    e_beta_sq_active = e_beta_active ** 2 + var_beta_active
    e_beta_sq = inclusion * e_beta_sq_active
    sd_beta = np.sqrt(np.maximum(e_beta_sq - e_beta ** 2, 0.0))

    # Define the daedalus-facing model.
    sd_slab = sigma_slab
    inv_2sigma2 = 0.5 / sigma2
    log_const_data = -0.5 * float(g * np.log(2.0 * np.pi * sigma2))
    log_const_slab = -0.5 * float(np.log(2.0 * np.pi * sigma2_slab))
    inv_2sigma2_slab = 0.5 / sigma2_slab

    def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
        # Off-value override pins inactive beta_i to 0; but defensively
        # construct the predicted mean from gamma * beta.
        resid = y_arr - gamma.astype(np.float64) * beta
        return log_const_data - inv_2sigma2 * float(np.dot(resid, resid))

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return norm.ppf(np.clip(u, 1e-15, 1.0 - 1e-15)) * sd_slab

    groups_kwargs = [
        dict(
            name=f"x{i}",
            params=[i],
            off_values=np.array([0.0]),
            inclusion_prior=float(inc[i]),
        )
        for i in range(g)
    ]

    def log_prior_continuous_factory(i: int):
        def log_prior(beta_for_group: np.ndarray) -> float:
            b = float(beta_for_group[0])
            return log_const_slab - inv_2sigma2_slab * b * b
        return log_prior

    return MultiSpikeSlabProblem(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=g,
        groups_kwargs=groups_kwargs,
        log_prior_continuous_factory=log_prior_continuous_factory,
        sigma=sigma,
        sigma_slab=sigma_slab,
        inclusion_priors=inc,
        y=y_arr,
        log_Z_true=log_Z,
        log_Z_gamma_true=log_Z_gamma,
        inclusion_prob_true=inclusion,
        e_beta_true=e_beta,
        sd_beta_true=sd_beta,
    )
