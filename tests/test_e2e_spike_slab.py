"""Quantitative end-to-end validation of MoMS-NS on a synthetic spike-and-slab.

This is the smallest trans-dim problem with a closed-form posterior
inclusion probability and log-evidence -- the right place to pin the MoMS
flip kernel before scaling up to the diabetes benchmark with the JZS prior.
"""

from __future__ import annotations

import numpy as np
import pytest

import daedalus
from daedalus.benchmarks import spike_slab


@pytest.mark.benchmark
def test_spike_slab_inclusion_prob_within_tolerance() -> None:
    problem = spike_slab.make_problem(prior_half_width=10.0, inclusion_prior=0.5)
    g = daedalus.Group(
        name="beta_1", params=[1], off_values=np.array([0.0]), inclusion_prior=0.5
    )
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=[g],
        bound="single",
        sample="rwalk",
        n_live=400,
        seed=2026,
    )
    results = sampler.run_nested(dlogz=0.1, n_mcmc=40, transdim_fraction=0.4)

    inc = results.inclusion_probabilities()["beta_1"]
    truth = problem.inclusion_prob_true
    err = abs(inc - truth)
    assert err < 0.05, (
        f"P(gamma_beta_1 = 1) = {inc:.4f} vs analytical {truth:.4f} "
        f"(diff {err:.4f})"
    )


@pytest.mark.benchmark
def test_spike_slab_log_Z_within_tolerance() -> None:
    problem = spike_slab.make_problem(prior_half_width=10.0, inclusion_prior=0.5)
    g = daedalus.Group(
        name="beta_1", params=[1], off_values=np.array([0.0]), inclusion_prior=0.5
    )
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=[g],
        bound="single",
        sample="rwalk",
        n_live=400,
        seed=2026,
    )
    results = sampler.run_nested(dlogz=0.1, n_mcmc=40, transdim_fraction=0.4)
    err = abs(results.log_Z - problem.log_Z_true)
    assert err < 0.5, (
        f"log Z = {results.log_Z:.4f} vs analytical {problem.log_Z_true:.4f} "
        f"(diff {err:.4f}); NS error estimate {results.log_Z_err:.4f}"
    )
