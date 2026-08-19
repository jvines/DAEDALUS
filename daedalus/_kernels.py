"""Numba-JIT inner kernels.

These functions are the small numerical bits called millions of times
inside the constrained-MCMC and bound machinery. Each is pure NumPy
that numba can compile straight to LLVM, eliminating Python call
overhead. Cached to disk via ``@njit(cache=True)`` so import-time
compile cost is paid once.

Public functions:
    outside_cube_nonperiodic(u, periodic_mask) -> bool
    wrap_periodic(u, periodic_indices) -> None  (in place)
    mahalanobis_sq(L_inv, x, mean) -> float
    apply_off_values_flat(beta, gamma, group_starts, group_param_idx,
                          group_off_values_flat) -> None  (in place)

`_kernels.py` keeps no module-level state. The user-facing layer
(samplers, bounds, sampler) calls these with raw arrays.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def outside_cube_nonperiodic(u: np.ndarray, periodic_mask: np.ndarray) -> bool:
    """Return True if any non-periodic coordinate is outside [0, 1]."""
    n = u.size
    if periodic_mask.size == 0:
        for i in range(n):
            if u[i] < 0.0 or u[i] > 1.0:
                return True
        return False
    for i in range(n):
        if not periodic_mask[i]:
            if u[i] < 0.0 or u[i] > 1.0:
                return True
    return False


@njit(cache=True, fastmath=True)
def wrap_periodic(u: np.ndarray, periodic_indices: np.ndarray) -> None:
    """Reduce listed coords modulo 1 in place."""
    for k in range(periodic_indices.size):
        i = periodic_indices[k]
        u[i] -= np.floor(u[i])


@njit(cache=True, fastmath=True)
def mahalanobis_sq(L_inv: np.ndarray, x: np.ndarray, mean: np.ndarray) -> float:
    """(x - mean)^T (L L^T)^-1 (x - mean) computed via L_inv (x - mean)."""
    n = x.size
    s = 0.0
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += L_inv[i, j] * (x[j] - mean[j])
        s += acc * acc
    return s


@njit(cache=True, fastmath=True)
def apply_off_values_flat(
    beta: np.ndarray,
    gamma: np.ndarray,
    group_starts: np.ndarray,
    group_param_idx: np.ndarray,
    group_off_values_flat: np.ndarray,
) -> None:
    """Override beta entries for inactive groups with their off-values.

    Uses a CSR-like layout:
      - group_starts: shape (n_groups + 1,), the prefix-sum of group sizes
        so group k owns slots [group_starts[k], group_starts[k+1])
      - group_param_idx: flat array of beta indices, indexed by the slot
      - group_off_values_flat: flat array of off-values, same indexing

    For each k with gamma[k] = False, writes beta[group_param_idx[s]] =
    group_off_values_flat[s] for every slot s in the group's slice.
    """
    n_groups = gamma.size
    for k in range(n_groups):
        if gamma[k]:
            continue
        s0 = group_starts[k]
        s1 = group_starts[k + 1]
        for s in range(s0, s1):
            beta[group_param_idx[s]] = group_off_values_flat[s]
