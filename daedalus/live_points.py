"""Struct-of-arrays storage for the live point cloud.

The live cloud is the per-iteration hot data: we read `argmin(log_L)` and
`max(log_L)` every iteration, replace one point per iteration, and (when a
bound is fitted) read the full `(n_live, ndim)` matrix. Using parallel
NumPy arrays keeps these reductions in tight C loops; the per-point `State`
object is constructed only at the boundary between the NS loop and the
constrained-MCMC kernel, which the user might supply in pure Python.

Memory layout (contiguous, row-major):
    u             : (n_live, ndim)        float64  -- unit-cube coords
    beta          : (n_live, ndim)        float64  -- physical params
    gamma         : (n_live, n_groups)    bool     -- zero-cols when g = 0
    log_likelihood: (n_live,)             float64
    log_prior     : (n_live,)             float64

All arrays are pre-allocated at construction; `replace` writes in-place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import State


@dataclass
class LivePoints:
    u: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    log_likelihood: np.ndarray
    log_prior: np.ndarray

    @classmethod
    def empty(cls, n_live: int, ndim: int, n_groups: int) -> "LivePoints":
        return cls(
            u=np.empty((n_live, ndim), dtype=np.float64),
            beta=np.empty((n_live, ndim), dtype=np.float64),
            gamma=np.empty((n_live, n_groups), dtype=bool),
            log_likelihood=np.full(n_live, -np.inf, dtype=np.float64),
            log_prior=np.zeros(n_live, dtype=np.float64),
        )

    @property
    def n_live(self) -> int:
        return int(self.u.shape[0])

    @property
    def ndim(self) -> int:
        return int(self.u.shape[1])

    @property
    def n_groups(self) -> int:
        return int(self.gamma.shape[1])

    def argmin_log_L(self) -> int:
        return int(np.argmin(self.log_likelihood))

    def max_log_L(self) -> float:
        return float(np.max(self.log_likelihood))

    def snapshot(self, idx: int) -> State:
        """Copy the point at `idx` into a stand-alone State (safe for mutation)."""
        return State(
            u=self.u[idx].copy(),
            beta=self.beta[idx].copy(),
            gamma=self.gamma[idx].copy(),
            log_likelihood=float(self.log_likelihood[idx]),
            log_prior=float(self.log_prior[idx]),
        )

    def replace(self, idx: int, state: State) -> None:
        """Write `state` into slot `idx` in-place."""
        np.copyto(self.u[idx], state.u)
        np.copyto(self.beta[idx], state.beta)
        np.copyto(self.gamma[idx], state.gamma)
        self.log_likelihood[idx] = state.log_likelihood
        self.log_prior[idx] = state.log_prior

    def append(self, state: State) -> int:
        """Grow the live cloud by appending a new state.

        Returns the index of the new live point. Used by the gamma-adaptive
        speciation step that allocates fresh live-point budget to
        underrepresented gamma-configurations. Reallocation is via
        ``np.concatenate`` (O(n_live) per append); speciation is
        infrequent so this is not a hot path.
        """
        self.u = np.concatenate([self.u, state.u[None, :]], axis=0)
        self.beta = np.concatenate([self.beta, state.beta[None, :]], axis=0)
        self.gamma = np.concatenate(
            [self.gamma, state.gamma[None, :]], axis=0
        )
        self.log_likelihood = np.concatenate(
            [self.log_likelihood, np.array([state.log_likelihood])]
        )
        self.log_prior = np.concatenate(
            [self.log_prior, np.array([state.log_prior])]
        )
        return int(self.u.shape[0] - 1)
