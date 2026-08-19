"""Importance-sampled Rao-Blackwell inclusion-probability estimator.

The exact conditional inclusion probability at a dead point
``(beta, gamma)`` for group ``k`` is

    P(gamma_k = 1 | beta_{-k}, gamma_{-k}, y)
        = p_k * Z_on / [(1 - p_k) * L_off + p_k * Z_on]

where

    Z_on  = integral pi_k(beta_k) * L(beta_{-k}, beta_k, gamma_k = 1,
                                       gamma_{-k}) d beta_k
    L_off = L(beta_{-k}, beta_k = beta_k^off, gamma_k = 0, gamma_{-k})

For Gaussian-conjugate models ``Z_on`` is closed-form (see
:meth:`daedalus.results.Results.rao_blackwell_inclusion`). For
non-conjugate models (Keplerians, transits, line-fitting), this module
estimates ``Z_on`` by importance sampling using the group's birth
proposal as the IS distribution.  Any :class:`daedalus.BirthProposal`
satisfying detailed balance is a valid IS proposal by construction;
the GLS- and BLS-informed births shipped with the package concentrate
their density on the periodogram peaks of the residuals, which is
exactly where ``Z_on`` has support, so the IS estimator typically
needs only a few tens of samples to converge.

Usage
-----
The returned ``logits_fn`` plugs into the existing
:meth:`Results.rao_blackwell_inclusion` API:

    from daedalus.rao_blackwell import make_is_logit_fn
    logits_fn = make_is_logit_fn(
        loglike=my_loglike, groups=my_groups, n_samples=64, seed=0
    )
    inc = results.rao_blackwell_inclusion(logits_fn)

``groups`` must be the same :class:`daedalus.Group` instances passed to
the chain; their ``birth_proposal`` and ``log_prior_continuous``
attributes are read directly.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from .state import State

if TYPE_CHECKING:
    from .groups import Group


def make_is_logit_fn(
    loglike: Callable[[np.ndarray, np.ndarray], float],
    groups: Sequence["Group"],
    n_samples: int = 64,
    seed: int | None = None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Build a Rao-Blackwell conditional-logit callable backed by an
    importance-sampling estimate of the within-model marginal likelihood.

    Parameters
    ----------
    loglike
        The same ``loglike(beta, gamma) -> float`` callable used by the
        chain.
    groups
        Sequence of :class:`Group` instances. Each group must have
        ``birth_proposal`` and ``log_prior_continuous`` set (those are
        the IS proposal and prior used here).
    n_samples
        Number of IS samples per dead point per group. The IS estimator
        of ``log Z_on`` has Monte-Carlo standard error roughly
        ``ess^{-1/2}`` where ``ess`` is the effective sample size of the
        importance weights; for the GLS-informed birth on a smooth
        periodic signal ``n_samples = 32`` is usually sufficient.
    seed
        RNG seed for the IS sampling. The same seed is used across all
        dead points (within-call deterministic).

    Returns
    -------
    A callable ``logits_fn(beta, gamma) -> (g,) array`` that returns the
    Rao-Blackwell conditional log-odds, ``log P(gamma_k = 1 | beta,
    gamma, y) - log P(gamma_k = 0 | beta, gamma, y)``, for each group
    ``k``. Pass it to :meth:`Results.rao_blackwell_inclusion`.
    """
    groups = tuple(groups)
    if not all(
        g.birth_proposal is not None and g.log_prior_continuous is not None
        for g in groups
    ):
        raise ValueError(
            "make_is_logit_fn requires every group to have a birth_proposal "
            "and log_prior_continuous; the IS proposal is the birth and the "
            "prior is the user-supplied continuous prior on the group's "
            "active parameters."
        )

    # Pre-extract per-group bookkeeping for the hot loop.
    g_params = [np.asarray(g.params, dtype=np.intp) for g in groups]
    g_off = [np.asarray(g.off_values, dtype=np.float64) for g in groups]
    g_log_p_on = np.array(
        [float(np.log(g.inclusion_prior)) for g in groups], dtype=np.float64
    )
    g_log_p_off = np.array(
        [float(np.log1p(-g.inclusion_prior)) for g in groups], dtype=np.float64
    )

    def _apply_off_values(beta: np.ndarray, gamma: np.ndarray) -> None:
        for k, params in enumerate(g_params):
            if not gamma[k]:
                beta[params] = g_off[k]

    rng = np.random.default_rng(seed)

    def logits_fn(beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        n_groups = len(groups)
        logits = np.empty(n_groups, dtype=np.float64)

        for k in range(n_groups):
            group = groups[k]
            params_k = g_params[k]
            off_k = g_off[k]

            # log L_off: pin group k to off-values; gamma_{-k} unchanged
            # from the dead point's configuration.
            beta_off = beta.copy()
            beta_off[params_k] = off_k
            gamma_off = gamma.copy()
            gamma_off[k] = False
            _apply_off_values(beta_off, gamma_off)
            log_L_off = float(loglike(beta_off, gamma_off))

            # IS proposal state: gamma reflects the (gamma_k = 0,
            # gamma_{-k}) configuration so the birth's residual_fn /
            # active_periods_fn see the right active set.
            base_beta = beta.copy()
            base_beta[params_k] = off_k
            _apply_off_values(base_beta, gamma_off)
            is_state = State(
                u=np.zeros_like(base_beta),
                beta=base_beta,
                gamma=gamma_off.copy(),
            )

            # Draw n_samples from the birth and accumulate
            # log(pi * L / q) for each.
            beta_eval = beta.copy()
            gamma_on = gamma.copy()
            gamma_on[k] = True
            log_weights = np.empty(n_samples, dtype=np.float64)

            for j in range(n_samples):
                result = group.birth_proposal.propose(group, is_state, rng)
                proposed = np.asarray(result.proposed_beta, dtype=np.float64)
                log_q = float(result.log_q_forward)
                log_pi = float(group.log_prior_continuous(proposed))
                beta_eval[params_k] = proposed
                _apply_off_values(beta_eval, gamma_on)
                log_L = float(loglike(beta_eval, gamma_on))
                log_weights[j] = log_pi + log_L - log_q

            # log Z_on ~ logsumexp(log_weights) - log(N)
            log_Z_on = float(logsumexp(log_weights)) - float(np.log(n_samples))

            # logit_k = log p_k - log(1-p_k) + log Z_on - log L_off
            logits[k] = (
                g_log_p_on[k] - g_log_p_off[k] + log_Z_on - log_L_off
            )

        return logits

    return logits_fn
