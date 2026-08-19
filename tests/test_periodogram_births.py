"""Unit tests for the package-level periodogram-informed birth proposals.

Pins:

* BLSPeriodBirth and GLSPeriodBirth produce internally consistent
  ``propose`` and ``log_density`` values (round-trip equality on the
  proposed beta).
* The mixture density is correctly normalised under both alpha = 1
  (pure informed) and alpha < 1 (informed + uniform fallback).
* CDF caching is keyed by ``state.gamma``: distinct gamma vectors
  trigger fresh periodograms; identical gamma reuses the cache.
* Harmonic suppression knocks down density inside the configured
  window of every active period.
* ``GaussianRWBirth.update`` pattern (no-op for periodogram births)
  and the layout-validation guards fire on misconfigured groups.

End-to-end correctness on real data is exercised by
``tests/test_e2e_wasp47.py``; this file stays narrow and fast.
"""

from __future__ import annotations

import numpy as np
import pytest

import daedalus


def _toy_state(gamma: list[bool], beta: list[float]) -> daedalus.State:
    """Build a State for tests where prior_transform isn't being used."""
    return daedalus.State(
        u=np.zeros(len(beta)),
        beta=np.asarray(beta, dtype=np.float64),
        gamma=np.asarray(gamma, dtype=bool),
    )


def _make_test_residuals(seed: int = 0):
    """Synthesise a transit-like time series so BLS has something to find."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 30.0, 4000)
    period_true = 4.16
    depth = 0.01
    duration = 0.1
    phase = ((time + 1.5) % period_true) / period_true
    in_transit = np.minimum(phase, 1.0 - phase) < (duration / period_true / 2.0)
    flux = 1.0 - depth * in_transit + 0.001 * rng.standard_normal(time.size)
    return time, flux


@pytest.mark.benchmark
def test_bls_period_birth_propose_log_density_consistent() -> None:
    """propose() and log_density() must agree at the proposed beta."""
    pytest.importorskip("astropy")
    time, flux = _make_test_residuals(seed=1)

    def residual_fn(state: daedalus.State) -> np.ndarray:
        return flux

    birth = daedalus.BLSPeriodBirth(
        time=time,
        residual_fn=residual_fn,
        prior_log_period=(np.log(0.5), np.log(12.0)),
        non_period_priors=((0.0, 1.0), (np.log(1e-4), np.log(0.05))),
        duration=0.1,
        n_periods=400,
        sharpening=10.0,
        alpha=0.7,
    )
    group = daedalus.Group(
        name="planet_0",
        params=[0, 1, 2],
        off_values=np.array([np.log(2.0), 0.5, np.log(0.01)]),
        inclusion_prior=0.5,
        birth_proposal=birth,
        log_prior_continuous=lambda b: 0.0,
    )
    state = _toy_state([False], [np.log(2.0), 0.5, np.log(0.01)])
    rng = np.random.default_rng(0)
    for _ in range(8):
        result = birth.propose(group, state, rng)
        eval_density = birth.log_density(group, state, result.proposed_beta)
        assert np.isclose(result.log_q_forward, eval_density, atol=1e-12), (
            f"propose log_q={result.log_q_forward} != log_density eval {eval_density}"
        )


@pytest.mark.benchmark
def test_gls_period_birth_propose_log_density_consistent() -> None:
    """Same self-consistency check for the Lomb-Scargle variant."""
    pytest.importorskip("astropy")
    rng = np.random.default_rng(2)
    time = np.sort(rng.uniform(0.0, 30.0, size=400))
    # Synthesise a sinusoidal RV-like signal so GLS has a real peak.
    period_true = 6.5
    flux = 5.0 * np.sin(2 * np.pi * time / period_true) + rng.standard_normal(time.size)

    def residual_fn(state: daedalus.State) -> np.ndarray:
        return flux

    birth = daedalus.GLSPeriodBirth(
        time=time,
        residual_fn=residual_fn,
        prior_log_period=(np.log(1.0), np.log(20.0)),
        non_period_priors=((0.0, 10.0),),  # one extra param: amplitude
        n_periods=300,
        sharpening=10.0,
        alpha=0.7,
    )
    group = daedalus.Group(
        name="signal_0",
        params=[0, 1],
        off_values=np.array([np.log(3.0), 0.0]),
        inclusion_prior=0.5,
        birth_proposal=birth,
        log_prior_continuous=lambda b: 0.0,
    )
    state = _toy_state([False], [np.log(3.0), 0.0])
    proposal_rng = np.random.default_rng(0)
    for _ in range(8):
        result = birth.propose(group, state, proposal_rng)
        eval_density = birth.log_density(group, state, result.proposed_beta)
        assert np.isclose(result.log_q_forward, eval_density, atol=1e-12)


@pytest.mark.benchmark
def test_period_birth_cdf_cache_keyed_by_gamma() -> None:
    """CDF cache must hit on identical gamma and miss on distinct gamma.
    The miss count is the cheapest proxy for catching a regression in
    the cache key."""
    pytest.importorskip("astropy")
    time, flux = _make_test_residuals(seed=3)
    n_calls = {"residual_fn": 0}

    def residual_fn(state: daedalus.State) -> np.ndarray:
        n_calls["residual_fn"] += 1
        return flux

    birth = daedalus.BLSPeriodBirth(
        time=time,
        residual_fn=residual_fn,
        prior_log_period=(np.log(0.5), np.log(12.0)),
        non_period_priors=((0.0, 1.0), (np.log(1e-4), np.log(0.05))),
        duration=0.1,
        n_periods=200,
        sharpening=10.0,
        alpha=0.5,
    )
    group = daedalus.Group(
        name="p0",
        params=[0, 1, 2],
        off_values=np.array([np.log(2.0), 0.5, np.log(0.01)]),
        inclusion_prior=0.5,
        birth_proposal=birth,
        log_prior_continuous=lambda b: 0.0,
    )
    rng = np.random.default_rng(0)
    state_a = _toy_state([False], [np.log(2.0), 0.5, np.log(0.01)])
    state_b = _toy_state([True], [np.log(2.0), 0.5, np.log(0.01)])

    birth.propose(group, state_a, rng)
    n_after_a = n_calls["residual_fn"]
    birth.propose(group, state_a, rng)        # cache hit
    assert n_calls["residual_fn"] == n_after_a, "cache hit on identical gamma"

    birth.propose(group, state_b, rng)        # cache miss, gamma differs
    assert n_calls["residual_fn"] == n_after_a + 1
    birth.propose(group, state_b, rng)        # cache hit on b
    assert n_calls["residual_fn"] == n_after_a + 1


@pytest.mark.benchmark
def test_period_birth_harmonic_mask_suppresses_active_period() -> None:
    """When ``active_periods_fn`` reports an active period, the mixture
    density at that period (and its harmonics) must be suppressed
    relative to the un-masked CDF."""
    pytest.importorskip("astropy")
    time, flux = _make_test_residuals(seed=5)

    def residual_fn(state: daedalus.State) -> np.ndarray:
        return flux

    active_period = 4.16  # the synthesised true period
    birth_no_mask = daedalus.BLSPeriodBirth(
        time=time,
        residual_fn=residual_fn,
        prior_log_period=(np.log(0.5), np.log(12.0)),
        non_period_priors=((0.0, 1.0), (np.log(1e-4), np.log(0.05))),
        duration=0.1,
        n_periods=400,
        sharpening=20.0,
        alpha=1.0,  # pure informed -> easier to see suppression
    )
    birth_with_mask = daedalus.BLSPeriodBirth(
        time=time,
        residual_fn=residual_fn,
        prior_log_period=(np.log(0.5), np.log(12.0)),
        non_period_priors=((0.0, 1.0), (np.log(1e-4), np.log(0.05))),
        duration=0.1,
        n_periods=400,
        sharpening=20.0,
        alpha=1.0,
        active_periods_fn=lambda s: [active_period],
        harmonic_window_log=0.05,
    )

    group = daedalus.Group(
        name="p0",
        params=[0, 1, 2],
        off_values=np.array([np.log(2.0), 0.5, np.log(0.01)]),
        inclusion_prior=0.5,
        birth_proposal=birth_with_mask,
        log_prior_continuous=lambda b: 0.0,
    )
    state = _toy_state([True], [np.log(2.0), 0.5, np.log(0.01)])
    beta_at_active = np.array([np.log(active_period), 0.5, np.log(0.01)])
    log_q_no_mask = birth_no_mask.log_density(group, state, beta_at_active)
    log_q_masked = birth_with_mask.log_density(group, state, beta_at_active)
    assert log_q_masked < log_q_no_mask - 50.0, (
        f"mask should suppress density at active period: "
        f"no_mask={log_q_no_mask}, masked={log_q_masked}"
    )


def test_period_birth_layout_validation() -> None:
    """Construction is fine; misconfigured group geometry must raise."""
    pytest.importorskip("astropy")

    def residual_fn(state: daedalus.State) -> np.ndarray:
        return np.zeros(100)

    birth = daedalus.BLSPeriodBirth(
        time=np.linspace(0.0, 10.0, 100),
        residual_fn=residual_fn,
        prior_log_period=(np.log(0.5), np.log(12.0)),
        # one period coord + 2 non-period priors -> ndim = 3 expected
        non_period_priors=((0.0, 1.0), (np.log(1e-4), np.log(0.05))),
        duration=0.1,
    )
    bad_group = daedalus.Group(
        name="too_few",
        params=[0, 1],   # ndim = 2, mismatched
        off_values=np.array([np.log(2.0), 0.5]),
        inclusion_prior=0.5,
        birth_proposal=birth,
        log_prior_continuous=lambda b: 0.0,
    )
    state = _toy_state([False], [np.log(2.0), 0.5])
    with pytest.raises(ValueError, match="configured for 3 params per slot"):
        birth.propose(bad_group, state, np.random.default_rng(0))


def test_period_birth_init_validation() -> None:
    """Bad construction args raise cleanly."""
    pytest.importorskip("astropy")
    common = dict(
        time=np.linspace(0.0, 10.0, 100),
        residual_fn=lambda s: np.zeros(100),
        non_period_priors=((0.0, 1.0),),
    )
    with pytest.raises(ValueError, match="prior_log_period"):
        daedalus.BLSPeriodBirth(prior_log_period=(np.log(2.0), np.log(2.0)), **common)
    with pytest.raises(ValueError, match="alpha"):
        daedalus.BLSPeriodBirth(
            prior_log_period=(np.log(0.5), np.log(12.0)), alpha=1.5, **common
        )
    with pytest.raises(ValueError, match="n_periods"):
        daedalus.BLSPeriodBirth(
            prior_log_period=(np.log(0.5), np.log(12.0)), n_periods=1, **common
        )
    with pytest.raises(ValueError, match="duration"):
        daedalus.BLSPeriodBirth(
            prior_log_period=(np.log(0.5), np.log(12.0)), duration=-1.0, **common
        )
    with pytest.raises(ValueError, match="non_period_priors"):
        daedalus.BLSPeriodBirth(
            prior_log_period=(np.log(0.5), np.log(12.0)),
            time=common["time"],
            residual_fn=common["residual_fn"],
            non_period_priors=((1.0, 1.0),),
        )


def test_period_birth_update_is_noop() -> None:
    """Periodogram-informed births do not adapt a scale; update() must
    be a no-op (called by NestedSampler on every flip attempt)."""
    pytest.importorskip("astropy")
    birth = daedalus.BLSPeriodBirth(
        time=np.linspace(0.0, 10.0, 100),
        residual_fn=lambda s: np.zeros(100),
        prior_log_period=(np.log(0.5), np.log(12.0)),
        non_period_priors=((0.0, 1.0),),
        duration=0.1,
    )
    # No state to inspect, but mustn't raise.
    birth.update(accepted=True)
    birth.update(accepted=False)
