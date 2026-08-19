"""Tests for the public NS diagnostics (daedalus.diagnostics).

Covers:
  (a) insertion_index_test PASSES on a known-uniform synthetic sequence;
  (b) it FAILS on a deliberately skewed (under-mixed) sequence;
  (c) end-to-end: a small NS chain run with insertion_recorder=[] yields a
      sequence the test runs on cleanly;
  (d) multirun_logZ_error returns the correct mean/std on a toy list.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import daedalus
from daedalus.diagnostics import (
    InsertionTestResult,
    insertion_index_test,
    multirun_logZ_error,
)
from daedalus.benchmarks import gaussians

# Sandbox BLAS emits harmless RuntimeWarnings on some ops; silence them so the
# diagnostic tests stay focused on assertions.
warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_public_api_exposes_diagnostics() -> None:
    assert daedalus.insertion_index_test is insertion_index_test
    assert daedalus.InsertionTestResult is InsertionTestResult
    assert daedalus.multirun_logZ_error is multirun_logZ_error
    for name in ("insertion_index_test", "InsertionTestResult",
                 "multirun_logZ_error"):
        assert name in daedalus.__all__


def test_insertion_index_test_passes_on_uniform_sequence() -> None:
    """A genuine U{0,...,n_live-1} sequence must pass: KS p>0.05, |z|<3.

    n_live is kept large relative to the sample count so the KS test does not
    resolve the discreteness of the index grid (the continuity-corrected
    fractions sit on a 1/n_live staircase; with n_live small and N huge the
    KS test legitimately distinguishes that staircase from the continuous
    uniform). Real NS runs sit in this regime, so this mirrors actual use.
    """
    n_live = 2000
    rng = np.random.default_rng(12345)
    indices = rng.integers(0, n_live, size=8000)
    res = insertion_index_test(indices, n_live)

    assert isinstance(res, InsertionTestResult)
    assert res.ks_pvalue > 0.05, f"KS p={res.ks_pvalue:.3g} should be > 0.05"
    assert abs(res.z_mean) < 3.0, f"|z_mean|={abs(res.z_mean):.2f} should be < 3"
    assert res.verdict.startswith("consistent with uniform")
    # mean fraction ~ 0.5 for correct sampling.
    assert abs(res.mean_fraction - 0.5) < 0.02


def test_insertion_index_test_fails_on_skewed_sequence() -> None:
    """A deliberately under-mixed (low-skewed) sequence must be flagged.

    Drawing indices that pile up near the bottom of the live set mimics a
    kernel whose newborns never escape a near-threshold donor: mean fraction
    well below 0.5, z_mean strongly negative, KS reject.
    """
    n_live = 2000
    rng = np.random.default_rng(999)
    # min of two uniform draws is left-skewed on {0,...,n_live-1}: E[index]
    # well below the uniform mean of (n_live-1)/2.
    a = rng.integers(0, n_live, size=8000)
    b = rng.integers(0, n_live, size=8000)
    indices = np.minimum(a, b)
    res = insertion_index_test(indices, n_live)

    assert res.ks_pvalue < 0.05, f"KS p={res.ks_pvalue:.3g} should reject"
    assert res.z_mean < -3.0, f"z_mean={res.z_mean:.2f} should be < -3"
    assert res.verdict.startswith("UNDER-MIXING")
    assert "low" in res.verdict


def test_insertion_index_test_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty"):
        insertion_index_test(np.array([], dtype=float), n_live=100)


@pytest.mark.benchmark
def test_insertion_recorder_end_to_end() -> None:
    """Run a small NS chain with insertion_recorder=[] and confirm the test
    runs on the recorded per-iteration indices.
    """
    problem = gaussians.make_problem(ndim=2, prior_half_width=10.0)
    n_live = 200
    recorder: list = []
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="single",
        sample="rwalk",
        n_live=n_live,
        seed=3,
    )
    results = sampler.run_nested(
        dlogz=1.0, n_mcmc=25, insertion_recorder=recorder,
    )

    assert len(recorder) == results.n_iter
    indices = np.asarray(recorder)
    # Each recorded index is the newborn's rank among the n_live-1 survivors,
    # so it lives in {0, ..., n_live-1}.
    assert indices.min() >= 0
    assert indices.max() <= n_live - 1

    res = insertion_index_test(indices, n_live)
    assert isinstance(res, InsertionTestResult)
    assert res.n_iter == len(recorder)
    assert res.n_live == n_live
    assert np.isfinite(res.ks_pvalue)
    assert np.isfinite(res.z_mean)


def test_multirun_logZ_error_toy_list() -> None:
    logZ = [10.0, 12.0, 14.0]
    mean, std = multirun_logZ_error(logZ)
    assert mean == pytest.approx(12.0)
    # sample std with ddof=1 of [10,12,14] is 2.0.
    assert std == pytest.approx(2.0)


def test_multirun_logZ_error_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    logZ = rng.normal(-42.0, 0.7, size=15)
    mean, std = multirun_logZ_error(logZ)
    assert mean == pytest.approx(float(np.mean(logZ)))
    assert std == pytest.approx(float(np.std(logZ, ddof=1)))


def test_multirun_logZ_error_needs_two_runs() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        multirun_logZ_error([1.0])
