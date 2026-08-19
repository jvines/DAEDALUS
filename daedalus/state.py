"""Joint MoMS state representation.

A `State` is a point on the augmented MoMS state space
    X = [0, 1]^ndim x R^ndim x {0, 1}^n_groups
parameterised by:
    u     -- unit-cube coordinates; the working space for within-model NS
             updates. Under any user-supplied `prior_transform`, the prior
             on u is uniform on the unit cube, which makes the constrained-
             MCMC M-H acceptance ratio trivial in u-space.
    beta  -- physical parameter vector; what the user's `loglike` sees. For
             active coordinates this equals `prior_transform(u)`; for
             inactive coordinates (gamma_k = 0) the corresponding entries
             are pinned to their off-values.
    gamma -- inclusion indicators (boolean); empty array when no toggleable
             groups are declared (continuous-only NS).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _empty_bool() -> np.ndarray:
    return np.empty(0, dtype=bool)


@dataclass
class State:
    u: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray = field(default_factory=_empty_bool)
    log_likelihood: float = -np.inf
    log_prior: float = 0.0

    def copy(self) -> "State":
        return State(
            u=self.u.copy(),
            beta=self.beta.copy(),
            gamma=self.gamma.copy(),
            log_likelihood=self.log_likelihood,
            log_prior=self.log_prior,
        )

    @property
    def ndim(self) -> int:
        return int(self.beta.size)

    @property
    def n_groups(self) -> int:
        return int(self.gamma.size)
