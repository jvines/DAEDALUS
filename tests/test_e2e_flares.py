"""Stellar flare counting on real TESS data of AU Mic.

Bundled data: TESS Sector 1 SPOC PDCSAP for AU Mic, a young flaring
M dwarf with multiple ~1-5% amplitude events per day. Each flare
candidate (auto-detected as a >4 sigma peak in the detrended flux)
becomes a toggleable group; trans-dim NS confirms which are real.

Expected behaviour: at SNR threshold >= 4 sigma, every detected
candidate should be a real flare (AU Mic's events are unmissable),
so all P(gamma=1) should be ~1. Test asserts that fraction.
"""

from __future__ import annotations

import pytest

import daedalus
from daedalus.benchmarks import flares


@pytest.mark.benchmark
@pytest.mark.slow
def test_au_mic_flare_recovery() -> None:
    problem = flares.make_problem(
        n_candidates=5,        # fewer candidates -> faster test
        snr_threshold=4.0,
    )
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=problem.groups,
        bound="single",
        sample="rwalk",
        n_live=200,
        seed=42,
    )
    results = sampler.run_nested(
        dlogz=0.5, n_mcmc=20, transdim_fraction=0.3, bound_update_interval=10
    )
    inc = results.inclusion_probabilities()
    n_high = sum(1 for p in inc.values() if p > 0.95)
    assert n_high == len(problem.candidate_peaks), (
        f"Expected all {len(problem.candidate_peaks)} >4-sigma flare "
        f"candidates to be confirmed, got {n_high}. Inclusion: {inc}"
    )
