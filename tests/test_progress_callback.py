"""Tests for the structured progress hook on ``NestedSampler.run_nested``.

The hook exists so a caller can observe an NS run without owning a terminal:
``on_progress`` receives an ``NSProgress`` record of the raw per-iteration
state at the same site that feeds the tqdm bar. The contract these tests pin:

  (a) it is purely additive -- a run without the hook is bit-identical to the
      pre-hook sampler, and attaching the hook does not perturb the chain;
  (b) it fires exactly once per NS iteration, including under ``max_iter``
      termination and on the trans-dim path;
  (c) it carries ``log_dlogz`` (the log of the stopping target), not ``dlogz``;
  (d) the ``gap`` the hook sees on the final call is the same value the
      termination test broke on;
  (e) bar and hook are independent and may both be active;
  (f) an exception raised by the callback propagates out of ``run_nested``.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import warnings

import numpy as np
import pytest

import daedalus
from daedalus.benchmarks import gaussians, spike_slab

# Sandbox BLAS emits harmless RuntimeWarnings on some ops.
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Golden values recorded from the pre-hook sampler (daedalus 0.0.1.dev0) on the
# fixtures below, before ``on_progress`` existed. The hook must not perturb the
# chain in any way, so these stay frozen: a change here is a regression, not a
# number to re-record.
GOLDEN_GAUSS_LOG_Z = -6.05949810949364
GOLDEN_GAUSS_N_ITER = 527
GOLDEN_GAUSS_N_CALLS = 7440
GOLDEN_GAUSS_SAMPLES_SHA256 = (
    "b34e5349305bc5d8b466fd180b25fe2c723febb6230cc39b765118310fe24d01"
)

GOLDEN_TRANSDIM_LOG_Z = -4.724362943038906
GOLDEN_TRANSDIM_N_ITER = 397
GOLDEN_TRANSDIM_N_CALLS = 5244
GOLDEN_TRANSDIM_SAMPLES_SHA256 = (
    "09df8e718e91f755f0a229dc23955f630d1690cee60d09132d80b86f547bca94"
)

GAUSS_N_LIVE = 100


def _digest(arr: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(arr, dtype=np.float64).tobytes()
    ).hexdigest()


def _gauss_sampler() -> daedalus.NestedSampler:
    problem = gaussians.make_problem(ndim=2, prior_half_width=10.0)
    return daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="single",
        sample="rwalk",
        n_live=GAUSS_N_LIVE,
        seed=7,
    )


def _run(*, dlogz: float = 0.5, show_progress: bool = False, **kwargs):
    return _gauss_sampler().run_nested(
        dlogz=dlogz, n_mcmc=15, show_progress=show_progress, **kwargs
    )


def _transdim_sampler() -> daedalus.NestedSampler:
    problem = spike_slab.make_problem(prior_half_width=10.0, inclusion_prior=0.5)
    g = daedalus.Group(
        name="beta_1", params=[1], off_values=np.array([0.0]), inclusion_prior=0.5
    )
    return daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=[g],
        bound="single",
        sample="rwalk",
        n_live=GAUSS_N_LIVE,
        seed=11,
    )


def _run_transdim(**kwargs):
    return _transdim_sampler().run_nested(
        dlogz=0.5, n_mcmc=15, transdim_fraction=0.4, show_progress=False, **kwargs
    )


# ---------------- public surface ------------------------------------------


def test_nsprogress_is_public_with_the_documented_fields() -> None:
    assert hasattr(daedalus, "NSProgress")
    assert "NSProgress" in daedalus.__all__
    assert daedalus.NSProgress._fields == (
        "n_iter",
        "log_Z",
        "log_Z_err",
        "gap",
        "log_L_star",
        "n_calls",
        "log_dlogz",
        "n_live",
    )


def test_on_progress_is_keyword_only_and_defaults_to_none() -> None:
    """Downstream callers feature-detect the hook through this signature probe."""
    sig = inspect.signature(daedalus.NestedSampler.run_nested)
    assert "on_progress" in sig.parameters
    param = sig.parameters["on_progress"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None
    # show_progress is untouched and remains independently settable.
    assert sig.parameters["show_progress"].default is True


# ---------------- no regression -------------------------------------------


def test_default_run_reproduces_pre_hook_golden_values() -> None:
    """on_progress=None must leave the chain bit-identical to the pre-hook code."""
    r = _run()
    assert r.log_Z == GOLDEN_GAUSS_LOG_Z
    assert r.n_iter == GOLDEN_GAUSS_N_ITER
    assert r.n_calls == GOLDEN_GAUSS_N_CALLS
    assert _digest(r.samples) == GOLDEN_GAUSS_SAMPLES_SHA256


def test_default_transdim_run_reproduces_pre_hook_golden_values() -> None:
    r = _run_transdim()
    assert r.log_Z == GOLDEN_TRANSDIM_LOG_Z
    assert r.n_iter == GOLDEN_TRANSDIM_N_ITER
    assert r.n_calls == GOLDEN_TRANSDIM_N_CALLS
    assert _digest(r.samples) == GOLDEN_TRANSDIM_SAMPLES_SHA256


def test_attaching_the_hook_does_not_perturb_the_chain() -> None:
    """Observation-only: no RNG draws, no likelihood calls, no mutation."""
    baseline = _run()
    events: list = []
    hooked = _run(on_progress=events.append)

    assert events, "hook never fired"
    assert hooked.log_Z == baseline.log_Z
    assert hooked.log_Z_err == baseline.log_Z_err
    assert hooked.H == baseline.H
    assert hooked.n_iter == baseline.n_iter
    assert hooked.n_calls == baseline.n_calls
    assert np.array_equal(hooked.samples, baseline.samples)
    assert np.array_equal(hooked.log_likelihoods, baseline.log_likelihoods)
    assert np.array_equal(hooked.log_weights, baseline.log_weights)


# ---------------- cadence -------------------------------------------------


def test_hook_fires_exactly_once_per_iteration() -> None:
    events: list = []
    r = _run(on_progress=events.append)
    # Exactly, not off by one: the callback fires after acc.add_dead() has
    # already incremented n_iter and before the termination break, so the
    # final call is included and the post-loop add_remaining_live() -- which
    # does not advance n_iter -- adds none. Same relation insertion_recorder
    # holds (see test_diagnostics.py).
    assert len(events) == r.n_iter


def test_hook_fires_once_per_iteration_under_max_iter_termination() -> None:
    """The max_iter break sits at the top of the loop, after that iteration's
    callback has already fired -- so the count is still exact."""
    events: list = []
    r = _run(max_iter=25, on_progress=events.append)
    assert r.n_iter == 25
    assert len(events) == 25


def test_hook_fires_on_the_transdim_path() -> None:
    events: list = []
    r = _run_transdim(on_progress=events.append)
    assert len(events) == r.n_iter
    assert all(e.n_live == GAUSS_N_LIVE for e in events)


# ---------------- payload -------------------------------------------------


def test_hook_state_is_monotonic_and_finite() -> None:
    events: list = []
    r = _run(on_progress=events.append)

    assert [e.n_iter for e in events] == list(range(1, r.n_iter + 1))
    assert all(math.isfinite(e.gap) for e in events)
    assert all(math.isfinite(e.log_L_star) for e in events)
    n_calls = [e.n_calls for e in events]
    assert all(b >= a for a, b in zip(n_calls, n_calls[1:]))
    assert n_calls[0] >= 1
    log_Z = [e.log_Z for e in events]
    assert all(math.isfinite(z) for z in log_Z)
    assert all(b >= a for a, b in zip(log_Z, log_Z[1:]))
    assert all(e.n_live == GAUSS_N_LIVE for e in events)


def test_hook_reports_raw_accumulator_state_not_final_results() -> None:
    """The record is live state mid-run, not a rewrite of the finished run."""
    events: list = []
    r = _run(on_progress=events.append)
    last = events[-1]

    assert last.n_iter == r.n_iter
    assert last.n_calls == r.n_calls
    # add_remaining_live() folds the surviving live points in *after* the
    # loop, so the last observed log_Z is strictly below the final evidence.
    assert last.log_Z < r.log_Z


@pytest.mark.parametrize("dlogz", [0.5, 1.0])
def test_hook_ships_log_dlogz_not_dlogz(dlogz: float) -> None:
    """dlogz=1.0 is the discriminating case: log_dlogz is 0.0, not 1.0."""
    events: list = []
    _run(dlogz=dlogz, on_progress=events.append)
    observed = {e.log_dlogz for e in events}
    assert len(observed) == 1, "log_dlogz must be constant for a run"
    (log_dlogz,) = observed
    assert log_dlogz == pytest.approx(math.log(dlogz), rel=0, abs=1e-15)
    assert not math.isclose(log_dlogz, dlogz)


def test_final_gap_is_the_value_the_termination_test_broke_on() -> None:
    events: list = []
    _run(on_progress=events.append)
    assert events[-1].gap < events[-1].log_dlogz
    # ...and no earlier call already satisfied it, i.e. the hook observes the
    # same gap the loop tested rather than a stale or recomputed one.
    assert all(e.gap >= e.log_dlogz for e in events[:-1])


# ---------------- coexistence and error propagation ------------------------


def test_bar_and_hook_both_fire(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tqdm")
    bar_calls: list = []
    original = daedalus.NestedSampler._update_progress_bar

    def spy(self, *args, **kwargs):
        bar_calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(daedalus.NestedSampler, "_update_progress_bar", spy)

    events: list = []
    r = _run(show_progress=True, on_progress=events.append)
    assert len(bar_calls) == r.n_iter
    assert len(events) == r.n_iter


@pytest.mark.parametrize("show_progress", [False, True])
def test_raising_callback_propagates_out_of_run_nested(show_progress: bool) -> None:
    """Documented behaviour: the library does not swallow consumer bugs."""

    def boom(_progress) -> None:
        raise RuntimeError("consumer sink exploded")

    with pytest.raises(RuntimeError, match="consumer sink exploded"):
        _run(show_progress=show_progress, on_progress=boom)
