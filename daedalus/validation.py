"""Birth-proposal consistency validation.

When a user supplies a custom :class:`BirthProposal` they must also supply
``Group.log_prior_continuous`` -- the package can no longer cancel the
continuous prior with the proposal density (cf. ``birth_proposals.py``
docstring and ``docs/algorithm.md``). The user-supplied
density is supposed to be the pushforward of ``Uniform[0,1)^|G_k|`` under
``prior_transform`` restricted to ``Group.params``. If it isn't, the
M-H ratio is silently wrong and the chain is miscalibrated -- without
the package raising any error.

This module provides :func:`validate_birth_consistency`, a deterministic
change-of-variables check that catches the common failure modes:

  1. **Wrong constant**: user multiplied the density by a normaliser
     that doesn't match the prior_transform's volume change.
  2. **Wrong shape**: user supplied a Gaussian density when the
     prior_transform pushes through a uniform, or vice versa.
  3. **Wrong support**: user's density returns ``-inf`` somewhere the
     prior_transform actually puts samples.

The check works under the standard *coordinate-separable* assumption on
``prior_transform``: each output coord depends only on one input coord
(the dynesty convention). For coupled transforms (e.g., correlated
multivariate Gaussian via a Cholesky factor), :func:`validate_birth_
consistency` will report the coupling and skip the Jacobian check.

The :class:`NestedSampler` constructor calls this on each Group with a
custom birth_proposal at construction time and raises ``BirthConsistency
Warning`` on detected mismatches. Disable with ``validate_births=False``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .groups import Group


class BirthConsistencyWarning(UserWarning):
    """Raised when a Group's log_prior_continuous doesn't match the
    pushforward of prior_transform on the group's coords."""


def _detect_coordinate_coupling(
    prior_transform: Callable[[np.ndarray], np.ndarray],
    ndim: int,
    group_params: tuple[int, ...],
    rng: np.random.Generator,
    fd_step: float,
    n_probes: int,
    deriv_threshold: float = 1e-6,
) -> list[int]:
    """Probe whether perturbing a u-coord *outside* the group's slot
    moves any of the group's beta coords. Returns the list of coupling
    coords (empty if separable).

    A coord is flagged as coupling when the finite-difference derivative
    ``d beta_{out} / d u_{in}`` exceeds ``deriv_threshold`` in magnitude
    -- the threshold is on the derivative, not the raw delta, so it's
    independent of the chosen ``fd_step``.
    """
    out_set = set(group_params)
    in_idx = [j for j in range(ndim) if j not in out_set]
    coupling: list[int] = []
    for _ in range(n_probes):
        u = rng.uniform(size=ndim)
        beta0 = np.asarray(prior_transform(u), dtype=np.float64)
        for j_in in in_idx:
            u_p = u.copy()
            u_p[j_in] += fd_step
            beta_p = np.asarray(prior_transform(u_p), dtype=np.float64)
            for j_out in group_params:
                d_beta = (beta_p[j_out] - beta0[j_out]) / fd_step
                if abs(d_beta) > deriv_threshold and j_in not in coupling:
                    coupling.append(j_in)
    return coupling


def _jacobian_log_abs_det(
    prior_transform: Callable[[np.ndarray], np.ndarray],
    u: np.ndarray,
    beta: np.ndarray,
    group_params: tuple[int, ...],
    fd_step: float,
) -> float | None:
    """Central-difference log|det(d beta_{G_k} / d u_{G_k})| under the
    coordinate-separable assumption: the Jacobian sub-matrix is diagonal
    so the determinant is the product of per-coord derivatives.

    Central differences (error O(h^2)) rather than forward (O(h)) so
    that steep transforms like ``scipy.stats.norm.ppf`` near the
    distribution tails don't blow up the residual. Returns None if any
    derivative is degenerate (zero) or if either probe falls outside
    the unit cube where the user's prior_transform is undefined.
    """
    log_det = 0.0
    for j in group_params:
        u_minus = u.copy()
        u_plus = u.copy()
        u_minus[j] -= fd_step
        u_plus[j] += fd_step
        # Refuse to evaluate prior_transform outside [0, 1]; fall back
        # to forward-difference if we'd overshoot the boundary.
        if u_minus[j] < 0.0 or u_plus[j] > 1.0:
            beta_p = np.asarray(prior_transform(u_plus if u_plus[j] <= 1.0 else u_minus), dtype=np.float64)
            d_beta = (beta_p[j] - beta[j]) / (
                fd_step if u_plus[j] <= 1.0 else -fd_step
            )
        else:
            beta_minus = np.asarray(prior_transform(u_minus), dtype=np.float64)
            beta_plus = np.asarray(prior_transform(u_plus), dtype=np.float64)
            d_beta = (beta_plus[j] - beta_minus[j]) / (2.0 * fd_step)
        if d_beta == 0.0 or not np.isfinite(d_beta):
            return None
        log_det += float(np.log(abs(d_beta)))
    return log_det


def validate_birth_consistency(
    group: "Group",
    prior_transform: Callable[[np.ndarray], np.ndarray],
    ndim: int,
    n_samples: int = 200,
    seed: int = 0,
    fd_step: float = 1e-5,
    tol: float = 1e-3,
    n_coupling_probes: int = 10,
) -> dict:
    """Deterministic change-of-variables check on a Group's user-supplied
    ``log_prior_continuous``.

    Procedure: for each of ``n_samples`` ``u`` draws,

        1. Compute ``beta = prior_transform(u)``.
        2. Evaluate ``log_p_user = group.log_prior_continuous(beta[group.params])``.
        3. Estimate ``log_det = log|det(d beta_{G_k} / d u_{G_k})|`` by
           finite-differencing each coord (assumes separability).
        4. The pushforward density at ``beta_{G_k}`` is then
           ``-log_det`` (with the uniform-on-cube source density of 1).
        5. Residual ``r = log_p_user - (-log_det)`` should be near zero.

    Returns a dict with:

      ``passed``                 — bool, True if max_residual < ``tol`` and no
                                  support violations and no coupling detected.
      ``max_residual``           — max |r| across the sample (NaN-safe).
      ``mean_residual``          — mean of r (signed); a constant offset here
                                  indicates a missing log-normaliser.
      ``n_support_violations``   — count of samples with non-finite log_p_user.
      ``coupling_coords``        — list of u-indices outside group.params that
                                  perturb beta inside group.params.
      ``residuals``              — full per-sample residuals array.
      ``message``                — human-readable summary; empty if passed.

    Notes
    -----
    * The default ``tol = 1e-3`` allows for finite-difference noise in
      the log-Jacobian. Tighten by passing a smaller ``tol`` and a
      smaller ``fd_step`` if you want a stricter check.
    * If ``coupling_coords`` is non-empty, the Jacobian check is skipped
      (the separability assumption is violated) and the function returns
      ``passed=False`` with an explanatory message. The user must verify
      consistency manually for coupled transforms.
    """
    if group.log_prior_continuous is None:
        raise ValueError(
            f"Group {group.name!r} has no log_prior_continuous set; "
            "nothing to validate"
        )
    group_params = tuple(int(p) for p in group.params)
    rng = np.random.default_rng(seed)

    coupling = _detect_coordinate_coupling(
        prior_transform, ndim, group_params, rng, fd_step, n_coupling_probes
    )
    if coupling:
        return {
            "passed": False,
            "max_residual": float("nan"),
            "mean_residual": float("nan"),
            "n_support_violations": 0,
            "coupling_coords": coupling,
            "residuals": np.empty(0, dtype=np.float64),
            "message": (
                f"prior_transform couples u-coords {coupling} into "
                f"beta entries of group {group.name!r}. The change-of-"
                f"variables check assumes coordinate separability and "
                f"cannot evaluate the Jacobian; verify log_prior_continuous "
                f"manually for coupled transforms."
            ),
        }

    residuals = np.full(n_samples, np.nan, dtype=np.float64)
    n_support_violations = 0
    n_jacobian_skipped = 0
    for i in range(n_samples):
        u = rng.uniform(size=ndim)
        beta = np.asarray(prior_transform(u), dtype=np.float64)
        beta_g = beta[list(group_params)]
        log_p_user = float(group.log_prior_continuous(beta_g))
        if not np.isfinite(log_p_user):
            n_support_violations += 1
            continue
        log_det = _jacobian_log_abs_det(
            prior_transform, u, beta, group_params, fd_step
        )
        if log_det is None:
            n_jacobian_skipped += 1
            continue
        residuals[i] = log_p_user - (-log_det)

    finite = residuals[~np.isnan(residuals)]
    max_err = float(np.max(np.abs(finite))) if finite.size > 0 else float("inf")
    mean_err = float(np.mean(finite)) if finite.size > 0 else float("nan")

    msgs: list[str] = []
    passed = True
    if n_support_violations > 0:
        passed = False
        msgs.append(
            f"{n_support_violations}/{n_samples} samples returned -inf from "
            f"log_prior_continuous; user's claimed support does not cover "
            f"the prior_transform's range"
        )
    if max_err > tol:
        passed = False
        if abs(mean_err) > 0.5 * max_err:
            msgs.append(
                f"residuals are biased by {mean_err:+.4f} (max {max_err:.4f}); "
                f"this typically indicates a missing log-normalisation constant"
            )
        else:
            msgs.append(
                f"max residual {max_err:.4f} exceeds tolerance {tol}; "
                f"log_prior_continuous shape disagrees with the pushforward"
            )
    if n_jacobian_skipped > 0:
        msgs.append(
            f"{n_jacobian_skipped}/{n_samples} samples had a degenerate "
            f"Jacobian (zero derivative under finite-difference); these "
            f"were skipped"
        )

    return {
        "passed": passed,
        "max_residual": max_err,
        "mean_residual": mean_err,
        "n_support_violations": n_support_violations,
        "coupling_coords": coupling,
        "residuals": residuals,
        "message": "; ".join(msgs),
    }


def warn_if_birth_inconsistent(
    group: "Group",
    prior_transform: Callable[[np.ndarray], np.ndarray],
    ndim: int,
    n_samples: int = 200,
    seed: int = 0,
    tol: float = 1e-3,
) -> None:
    """Run :func:`validate_birth_consistency` and emit a
    :class:`BirthConsistencyWarning` if it fails. Used by
    :class:`NestedSampler.__init__` when ``validate_births=True``.

    Skips silently if the group has no custom ``birth_proposal`` (the
    default uniform-u path doesn't use ``log_prior_continuous`` and
    needs no validation)."""
    if group.birth_proposal is None or group.log_prior_continuous is None:
        return
    report = validate_birth_consistency(
        group, prior_transform, ndim, n_samples=n_samples, seed=seed, tol=tol
    )
    if not report["passed"]:
        warnings.warn(
            f"Birth-proposal consistency check failed for group "
            f"{group.name!r}: {report['message']}. "
            f"This means M-H ratios for trans-dim moves on this group "
            f"are likely wrong and the chain may be miscalibrated. "
            f"Disable this check with NestedSampler(validate_births=False).",
            BirthConsistencyWarning,
            stacklevel=3,
        )
