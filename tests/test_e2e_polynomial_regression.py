"""E2E validation of daedalus on the polynomial-regression trans-dim demo.

The data is generated as ``y = 1 + 2 x - 3 x^3 + N(0, 0.5^2)`` on n = 50
points in [-1, 1], with an explored degree range 0..6. The chain must
recover the true sparsity pattern (degrees {0, 1, 3} active; {2, 4, 5,
6} inactive) AND track the closed-form per-degree posterior.
"""

from __future__ import annotations

import numpy as np
import pytest

import daedalus
from daedalus.benchmarks import polynomial_regression as polyreg


SEEDS = (101, 202, 303)


def _summarise(results: daedalus.Results) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-degree (P_inclusion, E[beta], SD[beta]) from a run."""
    p_inc = results.gamma.mean(axis=0)
    e_beta = results.samples.mean(axis=0)
    sd_beta = results.samples.std(axis=0)
    return p_inc, e_beta, sd_beta


@pytest.mark.benchmark
@pytest.mark.slow
def test_polynomial_regression_recovers_true_degrees() -> None:
    """Multi-seed run of daedalus on the polynomial-regression problem.

    Asserts:
      * P(gamma_k = 1) > 0.7 for the truly-active degrees (0, 1, 3).
      * P(gamma_k = 1) < 0.5 for the truly-inactive degrees (2, 4, 5, 6).
      * Mean per-degree inclusion prob within 0.1 of closed-form truth.
      * Mean per-degree E[beta] within 0.5 of closed-form truth.
    """
    problem = polyreg.make_problem()
    p = problem.K_max + 1

    p_runs = np.empty((len(SEEDS), p), dtype=float)
    e_runs = np.empty_like(p_runs)
    sd_runs = np.empty_like(p_runs)

    for i, seed in enumerate(SEEDS):
        groups = [daedalus.Group(**kw) for kw in problem.groups_kwargs]
        sampler = daedalus.NestedSampler(
            loglike=problem.loglike,
            prior_transform=problem.prior_transform,
            ndim=problem.ndim,
            groups=groups,
            bound="single",
            sample="rwalk",
            n_live=600,
            seed=seed,
        )
        results = sampler.run_nested(
            dlogz=0.5,
            n_mcmc=40,
            transdim_fraction=0.5,
            show_progress=False,
        )
        p_runs[i], e_runs[i], sd_runs[i] = _summarise(results)

    p_mean = p_runs.mean(axis=0)
    e_mean = e_runs.mean(axis=0)
    sd_mean = sd_runs.mean(axis=0)
    p_seedstd = p_runs.std(axis=0, ddof=1)
    e_seedstd = e_runs.std(axis=0, ddof=1)

    truth_P = problem.inclusion_prob_true
    truth_E = problem.e_beta_true
    truth_SD = problem.sd_beta_true

    print()
    print("=== polynomial-regression validation ===")
    print(
        f"{'deg':<4} {'true_beta':>10} {'active':>7}  "
        f"{'P_truth':>8} {'P_mean':>8} {'P_seedSD':>9}   "
        f"{'E_truth':>9} {'E_mean':>9} {'E_seedSD':>9}   "
        f"{'SD_truth':>9} {'SD_mean':>9}"
    )
    for k in range(p):
        print(
            f"{k:<4} {problem.true_coefficients[k]:>10.3f} "
            f"{str(bool(problem.true_active[k])):>7}  "
            f"{truth_P[k]:>8.4f} {p_mean[k]:>8.4f} {p_seedstd[k]:>9.4f}   "
            f"{truth_E[k]:>9.3f} {e_mean[k]:>9.3f} {e_seedstd[k]:>9.3f}   "
            f"{truth_SD[k]:>9.3f} {sd_mean[k]:>9.3f}"
        )

    failures: list[str] = []

    # Hard recovery thresholds on the true sparsity pattern.
    active_lo = 0.7
    inactive_hi = 0.5
    active_idx = (0, 1, 3)
    inactive_idx = (2, 4, 5, 6)
    for k in active_idx:
        if not p_mean[k] > active_lo:
            failures.append(
                f"deg{k}: P_mean = {p_mean[k]:.4f} not > {active_lo} (truly active)"
            )
    for k in inactive_idx:
        if not p_mean[k] < inactive_hi:
            failures.append(
                f"deg{k}: P_mean = {p_mean[k]:.4f} not < {inactive_hi} (truly inactive)"
            )

    # Match-the-enumeration tolerances (per-coordinate, multi-seed mean).
    tol_P = 0.1
    tol_E = 0.5
    for k in range(p):
        err_P = abs(p_mean[k] - truth_P[k])
        if err_P > tol_P:
            failures.append(
                f"deg{k}: |P_mean - P_truth| = {err_P:.4f} > {tol_P}"
            )
        err_E = abs(e_mean[k] - truth_E[k])
        if err_E > tol_E:
            failures.append(
                f"deg{k}: |E_mean - E_truth| = {err_E:.4f} > {tol_E}"
            )

    assert not failures, (
        "polynomial-regression deviations exceed tolerance:\n  "
        + "\n  ".join(failures)
    )
