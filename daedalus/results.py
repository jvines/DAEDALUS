"""Results object returned by `NestedSampler.run_nested`.

Carries the dead-point chain, importance-resampled posterior samples, log
evidence, and trans-dim summaries (per-model probabilities, per-group
inclusion probabilities). Mirrors `dynesty.results.Results` where it makes
sense, but adds gamma-aware fields for trans-dim outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Results:
    samples: np.ndarray  # (n_resample, ndim) — equal-weighted posterior draws of beta
    gamma: np.ndarray    # (n_resample, n_groups) — bool; empty if no toggleable groups
    log_likelihoods: np.ndarray
    log_weights: np.ndarray
    log_Z: float
    log_Z_err: float
    H: float
    n_iter: int
    n_calls: int
    group_names: Sequence[str] = field(default_factory=list)
    inclusion_priors: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def eff(self) -> float:
        """NS efficiency: dead-point yield per likelihood evaluation."""
        if self.n_calls <= 0:
            return 0.0
        return float(self.n_iter / self.n_calls)

    def model_probabilities(self) -> dict[tuple[bool, ...], float]:
        """Posterior probability of each visited gamma configuration.

        Computed from the equal-weight posterior resample. For trans-dim
        runs with `groups=()`, returns an empty dict.
        """
        if self.gamma.shape[1] == 0:
            return {}
        out: dict[tuple[bool, ...], float] = {}
        n = self.gamma.shape[0]
        for row in self.gamma:
            key = tuple(bool(x) for x in row)
            out[key] = out.get(key, 0.0) + 1.0 / n
        return out

    def inclusion_probabilities(self) -> dict[str, float]:
        """Marginal posterior P(gamma_k = 1 | data) for each declared group.

        Computed from the equal-weight posterior resample. For trans-dim
        runs with `groups=()`, returns an empty dict.
        """
        if self.gamma.shape[1] == 0:
            return {}
        means = self.gamma.mean(axis=0)
        return {name: float(p) for name, p in zip(self.group_names, means)}

    def rao_blackwell_inclusion(
        self,
        gamma_conditional_logits: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> dict[str, float]:
        """Rao-Blackwellised marginal inclusion probabilities.

        The empirical-frequency estimator (`inclusion_probabilities()`) is
        sensitive to the trans-dim acceptance asymmetry of the constrained-
        MCMC kernel: at high `\\mathcal{L}^\\star` the add move admits any
        fresh continuous draw with `\\mathcal{L} > \\mathcal{L}^\\star`
        while the delete move requires `\\mathcal{L}(\\theta_k^{\\rm off})
        > \\mathcal{L}^\\star`, which is harder, biasing the chain toward
        active configurations. This estimator removes that asymmetry by
        marginalising `\\gamma` analytically at each dead point,

            P(gamma_k = 1 | y) = sum_i w_i * P(gamma_k = 1 | beta_i, y),

        where the conditional is supplied by the user (analytical for
        Gaussian-conjugate models; for general models it must be obtained
        by integrating out the within-model continuous parameters of group
        `k` or estimated by importance sampling).

        Parameters
        ----------
        gamma_conditional_logits
            Callable returning a `(g,)` array of conditional log-odds,
            `log P(gamma_k = 1 | beta, gamma, y) - log P(gamma_k = 0 |
            beta, gamma, y)`, for each group `k` at the given dead point
            `(beta, gamma)`. Signature: ``f(beta, gamma) -> np.ndarray``.

        Returns
        -------
        dict[str, float] mapping each group name to its Rao-Blackwellised
        marginal inclusion probability.
        """
        if self.gamma.shape[1] == 0:
            return {}
        n = self.samples.shape[0]
        g = self.gamma.shape[1]
        p_sums = np.zeros(g, dtype=np.float64)
        for i in range(n):
            logits = np.asarray(
                gamma_conditional_logits(self.samples[i], self.gamma[i]),
                dtype=np.float64,
            )
            if logits.shape != (g,):
                raise ValueError(
                    f"gamma_conditional_logits returned shape {logits.shape}, "
                    f"expected ({g},)"
                )
            # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
            p_sums += 0.5 * (1.0 + np.tanh(0.5 * logits))
        return {name: float(p / n) for name, p in zip(self.group_names, p_sums)}

    def model_evidences(self) -> dict[tuple[bool, ...], tuple[float, float]]:
        """Per-model log evidence and standard error for each visited gamma.

        The headline ``log_Z`` is the **joint** evidence over the full
        trans-dim state space: ``Z = sum_gamma P(gamma) Z_gamma``. Per-
        model evidences ``Z_gamma`` are recovered from the chain via

            log Z_gamma = log P(gamma | y) + log Z - log P(gamma),

        where ``P(gamma | y)`` is the chain's posterior frequency for that
        gamma configuration, ``Z`` is the joint evidence, and ``P(gamma)``
        is the prior on the inclusion vector (computed from the per-group
        ``inclusion_priors``).

        Returns a dict mapping ``tuple(bool, ...)`` -> ``(log_Z_gamma,
        log_Z_gamma_err)``. The error combines the joint-evidence error
        from Skilling's recursion in quadrature with the Monte-Carlo
        binomial error on ``log P(gamma | y)``:

            err(log Z_gamma)^2 = log_Z_err^2 + (1 - p_gamma) / (p_gamma * N)

        where ``p_gamma = P(gamma | y)`` is the chain frequency and
        ``N`` is the resample size.

        For a trans-dim run with ``groups=()``, returns an empty dict.
        Models that the chain never visited are not in the dict (their
        evidence is consistent with -inf at the chain's resolution; if
        you need to bound such evidences, raise ``n_live`` and re-run).

        Bayes factors between trans-dim configurations need only the
        chain's gamma frequencies (the joint Z and the inclusion priors
        cancel):

            log BF(gamma_1, gamma_2) = log_Z_gamma_1 - log_Z_gamma_2
                                    = (log P(gamma_1 | y) - log P(gamma_2 | y))
                                    - (log P(gamma_1) - log P(gamma_2)).
        """
        if self.gamma.shape[1] == 0:
            return {}
        if self.inclusion_priors.size != self.gamma.shape[1]:
            raise RuntimeError(
                f"inclusion_priors length ({self.inclusion_priors.size}) does "
                f"not match number of groups ({self.gamma.shape[1]}); cannot "
                f"compute per-model evidences"
            )
        log_inc = np.log(self.inclusion_priors)
        log_exc = np.log1p(-self.inclusion_priors)
        n = self.gamma.shape[0]
        # Empirical posterior frequencies of each visited gamma.
        counts: dict[tuple[bool, ...], int] = {}
        for row in self.gamma:
            key = tuple(bool(x) for x in row)
            counts[key] = counts.get(key, 0) + 1
        out: dict[tuple[bool, ...], tuple[float, float]] = {}
        for key, n_g in counts.items():
            p_g = n_g / n
            gamma_arr = np.asarray(key, dtype=bool)
            log_prior_gamma = float(
                np.where(gamma_arr, log_inc, log_exc).sum()
            )
            log_Z_gamma = float(np.log(p_g)) + self.log_Z - log_prior_gamma
            # MC variance of log p̂ for binomial p̂ = n_g/N is (1-p)/(p*N);
            # add the joint-Z error in quadrature.
            mc_var = (1.0 - p_g) / (p_g * n)
            err = float(np.sqrt(self.log_Z_err ** 2 + mc_var))
            out[key] = (log_Z_gamma, err)
        return out

    def save(self, path: str) -> None:
        """Persist results to disk as a NumPy .npz archive.

        The archive is portable across daedalus versions that share the
        same `Results` field set; group_names are stored as a string array.
        """
        np.savez(
            path,
            samples=self.samples,
            gamma=self.gamma.astype(np.uint8),  # bool -> uint8 for portability
            log_likelihoods=self.log_likelihoods,
            log_weights=self.log_weights,
            log_Z=np.asarray([self.log_Z], dtype=np.float64),
            log_Z_err=np.asarray([self.log_Z_err], dtype=np.float64),
            H=np.asarray([self.H], dtype=np.float64),
            n_iter=np.asarray([self.n_iter], dtype=np.int64),
            n_calls=np.asarray([self.n_calls], dtype=np.int64),
            group_names=np.asarray(list(self.group_names), dtype=object),
            inclusion_priors=np.asarray(self.inclusion_priors, dtype=np.float64),
        )


def load_results(path: str) -> Results:
    """Inverse of `Results.save`."""
    arch = np.load(path, allow_pickle=True)
    # inclusion_priors was added after the initial save format; older
    # archives don't have it. Default to empty so existing snapshots
    # still load (model_evidences() will then raise a clear error if
    # called, rather than silently returning wrong numbers).
    if "inclusion_priors" in arch.files:
        inclusion_priors = np.asarray(arch["inclusion_priors"], dtype=np.float64)
    else:
        inclusion_priors = np.empty(0, dtype=np.float64)
    return Results(
        samples=arch["samples"],
        gamma=arch["gamma"].astype(bool),
        log_likelihoods=arch["log_likelihoods"],
        log_weights=arch["log_weights"],
        log_Z=float(arch["log_Z"][0]),
        log_Z_err=float(arch["log_Z_err"][0]),
        H=float(arch["H"][0]),
        n_iter=int(arch["n_iter"][0]),
        n_calls=int(arch["n_calls"][0]),
        group_names=list(arch["group_names"]),
        inclusion_priors=inclusion_priors,
    )
