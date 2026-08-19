"""End-to-end MoMS-NS validation against the diabetes benchmark.

This is the load-bearing trans-dim validation: posterior inclusion
probabilities for all 10 predictors must agree with the enumerated
ground truth (van den Bergh+ 2026 Table 1) to within 0.02. Run takes
~12 seconds at n_live=2000.

Setup uses the marginal-JZS approach: beta is integrated out
analytically, the chain samples gamma alongside one dummy continuous
parameter. The likelihood is precomputed as a 1024-entry table.
"""

from __future__ import annotations

import pytest

import daedalus
from daedalus.benchmarks import diabetes


@pytest.mark.benchmark
@pytest.mark.slow
def test_diabetes_inclusion_probs_match_table_1_within_0_02() -> None:
    problem = diabetes.make_problem(prior_inclusion=0.5)
    groups = [daedalus.Group(**kwargs) for kwargs in problem.groups_kwargs]

    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=groups,
        bound="none",
        sample="rwalk",
        n_live=2000,
        seed=2026,
    )
    results = sampler.run_nested(dlogz=0.1, n_mcmc=200, transdim_fraction=1.0)

    inc = results.inclusion_probabilities()
    failures = []
    for name in diabetes.PREDICTOR_NAMES:
        truth = problem.inclusion_prob_true[name]
        ours = inc[name]
        diff = abs(ours - truth)
        if diff >= 0.02:
            failures.append(f"{name}: daedalus={ours:.4f} truth={truth:.4f} diff={diff:.4f}")
    assert not failures, "Inclusion probability deviations >=0.02:\n" + "\n".join(
        failures
    )


@pytest.mark.benchmark
def test_diabetes_enumeration_matches_paper_table_1() -> None:
    """Sanity check that our JZS BF + enumeration reproduces the MoMS paper
    Table 1. This validates the *truth* against which the trans-dim NS
    test compares -- if this fails, our daedalus NS results would be
    compared to a wrong reference.
    """
    problem = diabetes.make_problem(prior_inclusion=0.5)
    failures = []
    for name in diabetes.PREDICTOR_NAMES:
        ours = problem.inclusion_prob_true[name]
        paper = diabetes.GROUND_TRUTH_INCLUSION[name]
        diff = abs(ours - paper)
        if diff >= 0.005:
            failures.append(
                f"{name}: enumeration={ours:.4f} paper={paper:.4f} diff={diff:.4f}"
            )
    assert not failures, "JZS BF enumeration disagrees with paper Table 1:\n" + "\n".join(
        failures
    )
