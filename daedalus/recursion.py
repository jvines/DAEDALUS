"""Skilling's nested sampling bookkeeping.

Maintains the running prior-volume estimate X_i, the log-evidence Z, the
information H, and the ordered sequence of dead points and their weights.
The recursion operates on log-quantities throughout for numerical stability:

    log X_i  = -i / n_live                                   (mean estimate)
    log dX_i = log X_{i-1} + log1p(-exp(-1/n_live))          (interval mass)
    log Z    = logsumexp_i (log L_i + log dX_i)
    H        = sum_i p_i * (log L_i - log Z),    p_i = exp(log dX_i + log L_i - log Z)

After termination, the remaining `n_live` live points are added with equal
residual weight log dX_residual = log X_final - log n_live (the live-point
distribution is approximately uniform on [0, X_final], so each point gets
mass X_final / n_live).

Termination criterion (Skilling 2006, §6): stop when

    log(X_curr * max_live(L)) - log Z < log(dlogz)

i.e. the remaining live points cannot add more than `dlogz` to log Z under
the worst case where every live point sits at the current maximum likelihood.

Independent of MoMS / trans-dim — the recursion sees only an ordered
sequence of log-likelihoods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SkillingAccumulator:
    n_live: int
    log_Z: float = -np.inf
    H: float = 0.0
    log_X_prev: float = 0.0
    n_iter: int = 0
    dead_log_likelihoods: list[float] = field(default_factory=list)
    dead_log_weights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_live <= 0:
            raise ValueError(f"n_live must be positive, got {self.n_live}")
        self._inv_n: float = 1.0 / self.n_live
        # log(X_{i-1} - X_i) - log X_{i-1} = log(1 - exp(-1/n_live))
        self._log_shrink: float = float(np.log1p(-np.exp(-self._inv_n)))

    def _shrink_for(self, n_live: int) -> tuple[float, float]:
        """Return (1/n_live, log(1 - exp(-1/n_live))) for a specified n_live.
        Used when the live count varies across iterations (gamma-adaptive
        MoMS-NS).
        """
        if n_live <= 0:
            raise ValueError(f"n_live must be positive, got {n_live}")
        inv_n = 1.0 / n_live
        log_shrink = float(np.log1p(-np.exp(-inv_n)))
        return inv_n, log_shrink

    def add_dead(self, log_L: float, n_live: int | None = None) -> float:
        """Account for one dead point at log-likelihood `log_L`. Returns log_dX.

        If ``n_live`` is provided, the X-shrinkage at this iteration is
        computed for that count rather than the accumulator's default. This
        is required for gamma-adaptive MoMS-NS where the live cloud size
        grows during the run.
        """
        if n_live is None:
            inv_n = self._inv_n
            log_shrink = self._log_shrink
        else:
            inv_n, log_shrink = self._shrink_for(n_live)
        log_dX = self.log_X_prev + log_shrink
        log_w = log_dX + log_L

        # Update H using the recurrence
        #   H_new = (Z_old / Z_new) * (H_old + log Z_old) +
        #           p_new * log L  -  log Z_new
        # with p_new = exp(log_w - log_Z_new). Numerically robust at startup
        # (log Z_old = -inf -> ratio = 0, p_new = 1).
        log_Z_new = float(np.logaddexp(self.log_Z, log_w))
        if np.isfinite(self.log_Z):
            ratio = float(np.exp(self.log_Z - log_Z_new))
            p_new = float(np.exp(log_w - log_Z_new))
            self.H = ratio * (self.H + self.log_Z) + p_new * log_L - log_Z_new
        else:
            # First contribution: Z_new = exp(log_w), p_new = 1.
            self.H = log_L - log_Z_new
        self.log_Z = log_Z_new

        self.dead_log_likelihoods.append(float(log_L))
        self.dead_log_weights.append(float(log_dX))

        # Volume shrinkage X_i = X_{i-1} * exp(-1/n_live(i)).
        self.log_X_prev -= inv_n
        self.n_iter += 1
        return log_dX

    def add_remaining_live(self, log_L_values: np.ndarray) -> None:
        """Account for the residual live points after termination.

        Each live point receives equal log-weight log dX_residual =
        log X_final - log n_live_current. Updates log Z and H in one
        batched step. Uses the actual number of values passed, so the
        gamma-adaptive case where the live cloud has grown is handled
        transparently.
        """
        log_L_values = np.asarray(log_L_values, dtype=float)
        n_live_current = log_L_values.size
        if n_live_current == 0:
            return
        log_dX = self.log_X_prev - np.log(n_live_current)
        log_w = log_dX + log_L_values

        log_Z_new = float(np.logaddexp(self.log_Z, float(np.logaddexp.reduce(log_w))))
        if np.isfinite(self.log_Z):
            ratio = float(np.exp(self.log_Z - log_Z_new))
            p_new = np.exp(log_w - log_Z_new)
            self.H = (
                ratio * (self.H + self.log_Z)
                + float(np.sum(p_new * log_L_values))
                - log_Z_new
            )
        else:
            p_new = np.exp(log_w - log_Z_new)
            self.H = float(np.sum(p_new * log_L_values)) - log_Z_new
        self.log_Z = log_Z_new

        for log_L in log_L_values:
            self.dead_log_likelihoods.append(float(log_L))
            self.dead_log_weights.append(float(log_dX))

    def termination_gap(self, log_L_max_live: float) -> float:
        """log(X_curr * max_L_live) - log Z. Sampler stops when this < log(dlogz)."""
        return self.log_X_prev + log_L_max_live - self.log_Z

    @property
    def log_Z_err(self) -> float:
        """Skilling's NS error estimate: sqrt(H / n_live) on log Z."""
        if self.H <= 0.0 or self.n_iter == 0:
            return float("inf")
        return float(np.sqrt(self.H / self.n_live))
