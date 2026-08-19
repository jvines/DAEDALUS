"""Diabetes-JZS dummy-coordinate ablation for the §3.5 robustness claim.

The diabetes table is constructed on a marginalised state where the
chain samples only over gamma; daedalus currently requires ndim >= 1
so a single dummy continuous coordinate is carried along that does not
enter the likelihood. The §3.5 prose claims the inclusion-vector
marginals are insensitive to the dummy's prior width; this script runs
the same chain at three dummy-prior widths and reports the Table 4
predictor-wise inclusion probabilities.

Widths swept: 0.1, 1.0, 10.0 (uniform prior on [0, width]). The
likelihood is unaffected; what changes is the chain's within-model RW
proposal scale and the RNG sequence consumed by the dummy updates.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import daedalus  # noqa: E402
from daedalus.benchmarks import diabetes  # noqa: E402


def run_one(width: float) -> dict[str, float]:
    base = diabetes.make_problem(prior_inclusion=0.5)

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return u * width

    groups = [daedalus.Group(**kwargs) for kwargs in base.groups_kwargs]
    sampler = daedalus.NestedSampler(
        loglike=base.loglike,
        prior_transform=prior_transform,
        ndim=base.ndim,
        groups=groups,
        bound="none",
        sample="rwalk",
        n_live=2000,
        seed=2026,
    )
    res = sampler.run_nested(
        dlogz=0.1, n_mcmc=200, transdim_fraction=1.0, show_progress=False
    )
    return res.inclusion_probabilities()


def main() -> None:
    widths = (0.1, 1.0, 10.0)
    truth = diabetes.make_problem(prior_inclusion=0.5).inclusion_prob_true
    runs = {w: run_one(w) for w in widths}

    print()
    header = ["predictor", "truth"] + [f"w={w}" for w in widths] + ["max|d-d|"]
    print(("{:<10}" + " {:>9}" * (1 + len(widths)) + " {:>10}").format(*header))
    for name in diabetes.PREDICTOR_NAMES:
        vals = [runs[w][name] for w in widths]
        max_spread = max(vals) - min(vals)
        row = [name, f"{truth[name]:.4f}"] + [f"{v:.4f}" for v in vals]
        row += [f"{max_spread:.4f}"]
        print(("{:<10}" + " {:>9}" * (1 + len(widths)) + " {:>10}").format(*row))

    overall_max = max(
        max(runs[w][name] for w in widths) - min(runs[w][name] for w in widths)
        for name in diabetes.PREDICTOR_NAMES
    )
    print()
    print(f"max inclusion spread across widths = {overall_max:.4f}")


if __name__ == "__main__":
    main()
