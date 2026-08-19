"""Asteroseismic peak-bagging on real Kepler data.

Bundled PDS: 4 stitched short-cadence Kepler quarters of KIC 6603624
("Saxo", Lund+ 2017 LEGACY sample). Trans-dim NS over 8 candidate
modes auto-detected from the spectrum. Pinning behaviour:

  - At least one slot strongly confirmed (P(gamma=1) > 0.9) -- the
    dominant 2936 uHz peak (or its closest neighbour) is unmissable.
  - At least three slots in the oscillation band [2700, 3100] uHz
    confirmed -- the chain is finding the mode envelope rather than
    fitting a single peak.
"""

from __future__ import annotations

import pytest

import daedalus
from daedalus.benchmarks import asteroseismic


@pytest.mark.benchmark
@pytest.mark.slow
def test_kic6603624_peak_bagging_recovers_oscillation_envelope() -> None:
    problem = asteroseismic.make_problem(n_candidates=8)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=problem.groups,
        bound="single",
        sample="rwalk",
        n_live=400,
        seed=42,
    )
    results = sampler.run_nested(
        dlogz=0.5, n_mcmc=30, transdim_fraction=0.3, bound_update_interval=10
    )
    inc = results.inclusion_probabilities()
    high = sum(1 for p in inc.values() if p > 0.9)
    assert high >= 1, (
        f"At least one mode should be strongly detected; got {high} "
        f"with P(in) > 0.9. Full inclusion: {inc}"
    )
    # Modes in [2700, 3100] uHz are the oscillation envelope of KIC 6603624.
    in_envelope_strong = 0
    for k, name in enumerate([g.name for g in problem.groups]):
        nu = problem.candidate_frequencies[k]
        if 2700 <= nu <= 3100 and inc[name] > 0.5:
            in_envelope_strong += 1
    assert in_envelope_strong >= 3, (
        f"Expected >= 3 oscillation-band modes with P(in) > 0.5, got "
        f"{in_envelope_strong}. Inclusion: {inc}"
    )
