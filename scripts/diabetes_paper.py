"""One-off diabetes-JZS run for the paper Table 4.

Settings mirror tests/test_e2e_diabetes.py. Prints daedalus inclusion
probabilities alongside enumerated truth so the paper table can be
populated.
"""
from __future__ import annotations

import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import daedalus  # noqa: E402
from daedalus.benchmarks import diabetes  # noqa: E402


def main() -> None:
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
    res = sampler.run_nested(dlogz=0.1, n_mcmc=200, transdim_fraction=1.0,
                             show_progress=False)
    inc = res.inclusion_probabilities()
    print(f"log Z = {res.log_Z:.4f} +/- {res.log_Z_err:.4f}")
    print(f"{'predictor':<8} {'truth':>9} {'daedalus':>10} {'|diff|':>9}")
    for name in diabetes.PREDICTOR_NAMES:
        truth = problem.inclusion_prob_true[name]
        ours = inc[name]
        print(f"{name:<8} {truth:>9.4f} {ours:>10.4f} {abs(ours-truth):>9.4f}")


if __name__ == "__main__":
    main()
