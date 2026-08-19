"""Tests for the Skilling NS bookkeeping.

The recursion is independent of the constrained-MCMC kernel, MoMS proposals,
bounds, and live-point management — it sees only an ordered sequence of
log-likelihoods and accumulates log-evidence + information. We pin behaviour
on closed-form cases so any future change to the math is caught immediately.
"""

from __future__ import annotations

import numpy as np
import pytest

from daedalus.recursion import SkillingAccumulator


def test_n_live_must_be_positive() -> None:
    with pytest.raises(ValueError):
        SkillingAccumulator(n_live=0)
    with pytest.raises(ValueError):
        SkillingAccumulator(n_live=-3)


def test_initial_state_is_zero_evidence_unit_volume() -> None:
    acc = SkillingAccumulator(n_live=10)
    assert acc.log_Z == -np.inf
    assert acc.n_iter == 0
    assert acc.log_X_prev == 0.0
    assert acc.H == 0.0


def test_first_dead_point_emits_correct_log_dX() -> None:
    """log dX_1 = log X_0 + log(1 - exp(-1/n_live))."""
    n_live = 10
    acc = SkillingAccumulator(n_live=n_live)
    log_dX = acc.add_dead(log_L=0.0)
    expected = np.log1p(-np.exp(-1.0 / n_live))
    assert np.isclose(log_dX, expected)


def test_first_dead_point_log_Z_equals_log_dX_for_unit_likelihood() -> None:
    acc = SkillingAccumulator(n_live=10)
    log_dX = acc.add_dead(log_L=0.0)
    assert np.isclose(acc.log_Z, log_dX)


def test_log_X_shrinks_geometrically() -> None:
    """After n iterations, log X_n = -n / n_live."""
    n_live = 100
    n_steps = 50
    acc = SkillingAccumulator(n_live=n_live)
    for _ in range(n_steps):
        acc.add_dead(log_L=0.0)
    assert np.isclose(acc.log_X_prev, -n_steps / n_live)


def test_unit_likelihood_evidence_matches_closed_form() -> None:
    """For log L = 0 throughout, log Z = log(1 - exp(-n_iter / n_live)) — the
    accumulated prior mass swept by the dead points.
    """
    n_live = 100
    n_iter = 200
    acc = SkillingAccumulator(n_live=n_live)
    for _ in range(n_iter):
        acc.add_dead(log_L=0.0)
    expected = np.log1p(-np.exp(-n_iter / n_live))
    assert np.isclose(acc.log_Z, expected, atol=1e-12)


def test_dead_point_book_is_consistent_with_iteration_count() -> None:
    acc = SkillingAccumulator(n_live=50)
    for i in range(7):
        acc.add_dead(log_L=float(i))
    assert acc.n_iter == 7
    assert len(acc.dead_log_likelihoods) == 7
    assert len(acc.dead_log_weights) == 7
    assert acc.dead_log_likelihoods == [float(i) for i in range(7)]


def test_termination_gap_for_unit_likelihood() -> None:
    """gap = log X_curr + log L_max_live - log Z. Stops when gap < log(dlogz)."""
    n_live = 100
    n_iter = 100
    acc = SkillingAccumulator(n_live=n_live)
    for _ in range(n_iter):
        acc.add_dead(log_L=0.0)
    gap = acc.termination_gap(log_L_max_live=0.0)
    expected = -n_iter / n_live - acc.log_Z
    assert np.isclose(gap, expected)


def test_add_remaining_live_for_unit_likelihood_completes_evidence() -> None:
    """log Z should converge to log(prior_volume * L_mean) = 0 for L = 1."""
    n_live = 50
    n_iter = 500
    acc = SkillingAccumulator(n_live=n_live)
    for _ in range(n_iter):
        acc.add_dead(log_L=0.0)
    remaining = np.zeros(n_live)
    acc.add_remaining_live(remaining)
    assert np.isclose(acc.log_Z, 0.0, atol=1e-3)


def test_add_remaining_live_dead_book_grows_by_n_live() -> None:
    n_live = 7
    acc = SkillingAccumulator(n_live=n_live)
    for _ in range(10):
        acc.add_dead(log_L=0.0)
    acc.add_remaining_live(np.zeros(n_live))
    assert len(acc.dead_log_likelihoods) == 10 + n_live
    assert len(acc.dead_log_weights) == 10 + n_live


def test_information_for_unit_likelihood_is_negative_log_Z() -> None:
    """H = E_post[log L] - log Z. For L=1 throughout, E[log L] = 0, so H = -log Z."""
    n_live = 100
    n_iter = 300
    acc = SkillingAccumulator(n_live=n_live)
    for _ in range(n_iter):
        acc.add_dead(log_L=0.0)
    acc.add_remaining_live(np.zeros(n_live))
    assert np.isclose(acc.H, -acc.log_Z, atol=1e-6)


def test_log_Z_err_predicts_sqrt_H_over_n_live() -> None:
    n_live = 100
    n_iter = 300
    acc = SkillingAccumulator(n_live=n_live)
    for _ in range(n_iter):
        acc.add_dead(log_L=float(i := 0))  # use a fixed log_L > -inf
        del i
    acc.add_remaining_live(np.zeros(n_live))
    assert np.isclose(acc.log_Z_err, np.sqrt(acc.H / n_live), atol=0.0)


def test_log_Z_err_returns_inf_when_no_dead_points() -> None:
    acc = SkillingAccumulator(n_live=10)
    assert acc.log_Z_err == float("inf")


def test_evidence_for_geometric_likelihood_matches_analytic_closed_form() -> None:
    """Sweep log L = -i for i = 0, 1, 2, ..., n_iter. With L = X_i, the evidence
    accumulates as sum_i X_{i-1} * (1 - exp(-1/n_live)) * exp(-i / n_live).

    This pins the recursion's shape under a non-trivial likelihood profile.
    """
    n_live = 200
    n_iter = 1000
    acc = SkillingAccumulator(n_live=n_live)
    inv_n = 1.0 / n_live
    log_shrink = np.log1p(-np.exp(-inv_n))

    expected_log_Z_terms = []
    for i in range(n_iter):
        log_X_prev = -i * inv_n
        log_dX = log_X_prev + log_shrink
        log_L = -float(i)
        expected_log_Z_terms.append(log_dX + log_L)
        acc.add_dead(log_L=log_L)

    expected_log_Z = float(np.logaddexp.reduce(expected_log_Z_terms))
    assert np.isclose(acc.log_Z, expected_log_Z, atol=1e-12)
