"""High-g scaling extension: single-run n_live sweep at the hardest
cell (K_max=20, K_true=10, g=21).

Companion to scripts/polynomial_scaling.py. Demonstrates the
1/sqrt(N_eff) tightening of the binomial floor on per-coefficient
inclusion probabilities as n_live (and hence the dead-point count
N_eff) grows.

For each n_live, runs ONE MoMS-NS chain and records the
per-coefficient inclusion vector and the closed-form reference.
Saves to scripts/polynomial_scaling_extra.npz.

Run from project root:
    python scripts/polynomial_scaling_extra.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import daedalus  # noqa: E402
from daedalus.benchmarks import polynomial_regression  # noqa: E402


def _truth_dict(K_max: int, K_true: int, rng: np.random.Generator) -> dict[int, float]:
    if K_true == 3 and K_max >= 3:
        return {0: 1.0, 1: 2.0, 3: -3.0}
    degrees = rng.choice(np.arange(K_max + 1), size=K_true, replace=False)
    coeffs: dict[int, float] = {}
    for d in degrees:
        sign = rng.choice([-1.0, 1.0])
        mag = float(rng.uniform(1.0, 3.0))
        coeffs[int(d)] = sign * mag
    return coeffs


def _run_one(K_max: int, K_true: int, n_live: int, seed: int = 2026) -> dict:
    rng = np.random.default_rng(2026)  # truth fixed
    truth = _truth_dict(K_max, K_true, rng)
    problem = polynomial_regression.make_problem(
        K_max=K_max, n=80, sigma=0.5, tau2=25.0,
        true_coefficients=truth, data_seed=0,
    )
    groups = [daedalus.Group(**kw) for kw in problem.groups_kwargs]
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=groups,
        bound="single", sample="rwalk", n_live=n_live, seed=seed,
    )
    t0 = time.time()
    r = sampler.run_nested(
        dlogz=0.3, n_mcmc=40, transdim_fraction=0.4,
        bound_update_interval=10, show_progress=False,
    )
    dt = time.time() - t0
    inc_chain = r.gamma.astype(bool).mean(axis=0)
    return {
        "K_max": K_max, "K_true": K_true,
        "n_live": n_live,
        "log_Z": float(r.log_Z),
        "log_Z_err": float(r.log_Z_err),
        "n_iter": int(r.log_likelihoods.size),
        "n_dead": int(r.gamma.shape[0]),
        "inclusion_chain": inc_chain.astype(float),
        "inclusion_truth": np.asarray(problem.inclusion_prob_true).astype(float),
        "true_active": [int(d) for d in sorted(truth.keys())],
        "dt_seconds": dt,
    }


def main() -> None:
    K_max, K_true = 20, 10
    n_live_grid = [500, 1000, 2000]

    cells = []
    for n_live in n_live_grid:
        print(f"\n[K_max={K_max}, K_true={K_true}, n_live={n_live}]",
              flush=True)
        r = _run_one(K_max, K_true, n_live)
        cells.append(r)
        inc_diff = np.abs(r["inclusion_chain"] - r["inclusion_truth"])
        print(f"  log Z = {r['log_Z']:+.2f} +/- {r['log_Z_err']:.2f}, "
              f"n_iter = {r['n_iter']}, n_dead = {r['n_dead']}, "
              f"max |dP| = {inc_diff.max():.4f}, "
              f"mean |dP| = {inc_diff.mean():.4f}, "
              f"dt = {r['dt_seconds']:.0f} s",
              flush=True)

    out_path = os.path.join(HERE, "polynomial_scaling_extra.npz")
    np.savez_compressed(out_path, cells=np.asarray(cells, dtype=object))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
