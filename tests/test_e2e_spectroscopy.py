"""End-to-end MoMS-NS validation on a synthetic emission-line spectrum.

This is the first astrophysical demo: a multi-line spectrum where each
candidate emission line at a known wavelength has unknown amplitude and
width and toggleable presence. The chain must (a) recover the truly
present lines as P(gamma=1) ~ 1, (b) correctly suppress the absent
lines as P(gamma=1) ~ 0.

Friendly to the default uniform-u birth -- the line center is fixed,
so we don't need a periodogram-informed proposal.
"""

from __future__ import annotations

import pytest

import daedalus
from daedalus.benchmarks import spectroscopy


@pytest.mark.benchmark
def test_spectroscopy_recovers_true_lines() -> None:
    problem = spectroscopy.make_problem(
        n_candidates=5,
        n_true=3,
        n_wavelengths=200,
        noise_sigma=0.1,
        seed=0,
    )
    groups = [daedalus.Group(**kwargs) for kwargs in problem.groups_kwargs]
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=groups,
        bound="single",
        sample="rwalk",
        n_live=500,
        seed=42,
    )
    results = sampler.run_nested(
        dlogz=0.5, n_mcmc=50, transdim_fraction=0.4, bound_update_interval=5
    )

    inc = results.inclusion_probabilities()
    failures = []
    for k, name in enumerate(g["name"] for g in problem.groups_kwargs):
        p = inc[name]
        truth = bool(problem.true_active[k])
        if truth and p < 0.9:
            failures.append(
                f"true line {name}: P(gamma=1) = {p:.3f}, expected > 0.9"
            )
        if (not truth) and p > 0.5:
            failures.append(
                f"absent line {name}: P(gamma=1) = {p:.3f}, expected < 0.5"
            )
    assert not failures, "Spectroscopy line recovery failures:\n" + "\n".join(failures)
