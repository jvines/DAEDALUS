"""Staged-refinement helper for trans-dim posteriors.

.. warning::

    EXPERIMENTAL. This module is not used to produce any published
    result, has no test coverage, and is not exported from the
    package namespace. :func:`make_within_gamma_problem` assumes a
    coordinate-separable ``prior_transform``.

The two-stage workflow:

1. Run a trans-dim survey chain (standard MoMS-NS) to identify the
   gamma-configurations with substantial posterior mass.
2. For each of the top-K gamma-configurations, run a *continuous-only*
   NS chain with gamma fixed and the inactive groups pinned to their
   off-values. Each such chain is standard daedalus NS, fully validated.
3. Combine the per-gamma Z's into the joint Z:

       log Z = logsumexp_gamma [ log P(gamma) + log Z_gamma ]

Per-gamma inclusion probabilities derive trivially from the per-gamma
Z's and the inclusion prior:

       P(gamma | y) = exp[ log P(gamma) + log Z_gamma - log Z ]

and the per-component inclusion marginal is

       P(gamma_k = 1 | y) = sum_{gamma: gamma_k = 1} P(gamma | y).

Unlike multi-modal trans-dim sampling at finite ``n_live``, this
workflow makes per-gamma exploration *independent*, so each gamma's
Z and posterior are recovered without seed-dependence in their
basin. The price is that the set of gamma-configurations evaluated is
prescribed (from the survey) rather than discovered by the chain.

.. warning::

    EXPERIMENTAL. This module is not used to produce any published
    result and is not part of the supported public API of
    :mod:`daedalus`. Its helpers (``WithinGammaProblem``,
    ``combine_per_gamma_Z``, ``make_within_gamma_problem``,
    ``select_top_gammas_from_chain``) are no longer exported from the
    package namespace.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .groups import Group


def _identity(x):
    return x


@dataclass
class WithinGammaProblem:
    """A continuous-only NS problem with one specific gamma fixed.

    Use with :class:`daedalus.NestedSampler` (groups=()) to sample the
    within-model continuous posterior at this gamma.
    """

    gamma_tuple: tuple[bool, ...]
    loglike: Callable[[np.ndarray, np.ndarray], float]  # signature matches NS API
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim_free: int


def make_within_gamma_problem(
    loglike_full: Callable[[np.ndarray, np.ndarray], float],
    prior_transform_full: Callable[[np.ndarray], np.ndarray],
    ndim_full: int,
    groups: Sequence[Group],
    gamma_tuple: tuple[bool, ...],
) -> WithinGammaProblem:
    """Build a continuous-only NS problem at fixed gamma.

    The returned ``loglike`` and ``prior_transform`` operate on a
    reduced ``ndim_free`` parameter vector containing only the active
    groups' parameters plus any always-on (not-in-any-group) parameters.
    """
    if len(gamma_tuple) != len(groups):
        raise ValueError(
            f"gamma_tuple length {len(gamma_tuple)} != n_groups {len(groups)}"
        )

    in_some_group: set[int] = set()
    for g in groups:
        in_some_group.update(int(p) for p in g.params)
    always_on = sorted(d for d in range(ndim_full) if d not in in_some_group)

    active_indices: list[int] = list(always_on)
    for k, on in enumerate(gamma_tuple):
        if on:
            active_indices.extend(int(p) for p in groups[k].params)
    active_indices = sorted(set(active_indices))
    active_arr = np.asarray(active_indices, dtype=np.intp)

    gamma_arr = np.asarray(gamma_tuple, dtype=bool)
    # Cache off-values per inactive group for fast apply.
    inactive_group_params = [
        np.asarray(groups[k].params, dtype=np.intp)
        for k, on in enumerate(gamma_tuple)
        if not on
    ]
    inactive_group_off = [
        np.asarray(groups[k].off_values, dtype=np.float64)
        for k, on in enumerate(gamma_tuple)
        if not on
    ]

    ndim_free = active_arr.size

    def loglike(beta_free: np.ndarray, gamma_unused: np.ndarray) -> float:
        beta_full = np.zeros(ndim_full, dtype=np.float64)
        beta_full[active_arr] = beta_free
        for j in range(len(inactive_group_params)):
            beta_full[inactive_group_params[j]] = inactive_group_off[j]
        return float(loglike_full(beta_full, gamma_arr))

    def prior_transform(u_free: np.ndarray) -> np.ndarray:
        # Round-trip through the full prior_transform with placeholder
        # values for the inactive coords (they don't affect the
        # likelihood because off-values overwrite them downstream).
        u_full = np.zeros(ndim_full, dtype=np.float64)
        u_full[active_arr] = u_free
        beta_full = np.asarray(prior_transform_full(u_full), dtype=np.float64)
        return beta_full[active_arr]

    return WithinGammaProblem(
        gamma_tuple=gamma_tuple,
        loglike=loglike,
        prior_transform=prior_transform,
        ndim_free=ndim_free,
    )


@dataclass
class StagedResult:
    """Output of staged-refinement combine step."""

    log_Z: float
    log_Z_err: float
    per_gamma_log_Z: dict[tuple[bool, ...], float]
    per_gamma_log_Z_err: dict[tuple[bool, ...], float]
    per_gamma_posterior: dict[tuple[bool, ...], float]
    inclusion_probabilities: dict[str, float]
    n_gamma_evaluated: int


def combine_per_gamma_Z(
    per_gamma_log_Z: dict[tuple[bool, ...], float],
    per_gamma_log_Z_err: dict[tuple[bool, ...], float],
    inclusion_priors: np.ndarray,
    group_names: Sequence[str],
) -> StagedResult:
    """Combine per-gamma log-evidences into a joint log Z.

    log Z = logsumexp_gamma [ log P(gamma) + log Z_gamma ]
    P(gamma | y) = exp[ log P(gamma) + log Z_gamma - log Z ]
    P(gamma_k = 1 | y) = sum_{gamma: gamma_k=1} P(gamma | y)

    The joint log_Z_err is propagated by summing the per-gamma errors
    weighted by their P(gamma | y) contributions.
    """
    log_inc = np.log(inclusion_priors)
    log_exc = np.log1p(-inclusion_priors)

    log_terms = []
    gammas = list(per_gamma_log_Z.keys())
    for gamma_tuple in gammas:
        gamma_arr = np.asarray(gamma_tuple, dtype=bool)
        log_prior_gamma = float(np.where(gamma_arr, log_inc, log_exc).sum())
        log_terms.append(log_prior_gamma + per_gamma_log_Z[gamma_tuple])

    log_Z = float(np.logaddexp.reduce(np.array(log_terms)))

    # P(gamma | y) for each evaluated gamma
    posterior: dict[tuple[bool, ...], float] = {}
    for gamma_tuple, lt in zip(gammas, log_terms):
        posterior[gamma_tuple] = float(np.exp(lt - log_Z))

    # Per-component inclusion probability
    inc_marginal = {name: 0.0 for name in group_names}
    for gamma_tuple, p in posterior.items():
        for k, name in enumerate(group_names):
            if gamma_tuple[k]:
                inc_marginal[name] += p

    # Aggregate joint Z err: weighted sum of per-gamma errs
    err_sq = 0.0
    for gamma_tuple in gammas:
        w = posterior[gamma_tuple]
        e = per_gamma_log_Z_err[gamma_tuple]
        err_sq += (w * e) ** 2
    log_Z_err = float(np.sqrt(err_sq))

    return StagedResult(
        log_Z=log_Z,
        log_Z_err=log_Z_err,
        per_gamma_log_Z=dict(per_gamma_log_Z),
        per_gamma_log_Z_err=dict(per_gamma_log_Z_err),
        per_gamma_posterior=posterior,
        inclusion_probabilities=inc_marginal,
        n_gamma_evaluated=len(gammas),
    )


def select_top_gammas_from_chain(
    gamma_dead: np.ndarray,
    log_weights: np.ndarray,
    top_k: int = 20,
    min_posterior: float = 1e-3,
) -> list[tuple[bool, ...]]:
    """Select the top-K gamma-configurations by importance-weighted
    posterior mass from a trans-dim survey chain's dead points.

    Parameters
    ----------
    gamma_dead
        (n_dead, n_groups) bool array of gamma-tuples per dead point.
    log_weights
        (n_dead,) log importance weights per dead point.
    top_k
        Maximum number of gamma-configurations to return.
    min_posterior
        Threshold on per-gamma posterior mass; configurations below
        this threshold are dropped even if they fit in the top-K.

    Returns
    -------
    List of gamma-tuples, ordered by decreasing posterior mass.
    """
    w = np.exp(log_weights - float(np.logaddexp.reduce(log_weights)))
    # Per-gamma posterior mass
    from collections import defaultdict

    mass: dict[tuple[bool, ...], float] = defaultdict(float)
    for i in range(gamma_dead.shape[0]):
        gt = tuple(bool(x) for x in gamma_dead[i])
        mass[gt] += float(w[i])

    ranked = sorted(mass.items(), key=lambda item: -item[1])
    chosen = [
        gt for gt, m in ranked[:top_k] if m >= min_posterior
    ]
    return chosen
