"""High-g scaling sweep on the polynomial-regression benchmark.

For each (K_max, K_true) cell, runs one MoMS-NS chain and records:
    log_Z, log_Z_err, n_iter, n_unique_gamma,
    inclusion probabilities (per coefficient),
    closed-form inclusion probabilities (per coefficient),
    top-50 model-probability dict,
    cumulative posterior mass in top-N configs.

The sweep maps how the chain's resolution of the inclusion-vector
posterior degrades with g = K_max + 1 at fixed n_live * n_iter.

Run from project root:
    python scripts/polynomial_scaling.py
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
    """Return a dense-truth coefficient dict with K_true active degrees.

    For K_true == 3 we use the canonical (0, 1, 3) -> (1, 2, -3) so the
    K_max=6 control case matches the §4 toy run. Otherwise we pick K_true
    distinct degrees uniformly and assign coefficients ~ Uniform([-3, -1]
    union [+1, +3]) so they're all comfortably above the noise floor.
    """
    if K_true == 3 and K_max >= 3:
        return {0: 1.0, 1: 2.0, 3: -3.0}
    degrees = rng.choice(np.arange(K_max + 1), size=K_true, replace=False)
    coeffs: dict[int, float] = {}
    for d in degrees:
        sign = rng.choice([-1.0, 1.0])
        mag = float(rng.uniform(1.0, 3.0))
        coeffs[int(d)] = sign * mag
    return coeffs


def _run_one(K_max: int, K_true: int, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
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
        bound="single", sample="rwalk", n_live=500, seed=seed,
    )
    t0 = time.time()
    r = sampler.run_nested(
        dlogz=0.3, n_mcmc=40, transdim_fraction=0.4,
        bound_update_interval=10, show_progress=False,
    )
    dt = time.time() - t0

    # Inclusion-vector posterior: weight each dead point by its
    # importance weight (already normalised in r.gamma after resampling).
    gamma = r.gamma.astype(bool)
    n_samples = gamma.shape[0]
    # Unique gamma count
    gamma_packed = np.packbits(gamma, axis=1)
    _, counts = np.unique(gamma_packed, axis=0, return_counts=True)
    n_unique = int(counts.size)

    # Top-N model probabilities
    mp = r.model_probabilities()
    items = sorted(mp.items(), key=lambda kv: -kv[1])
    top50 = items[:50]
    cum_mass = np.cumsum([p for _, p in items])
    # Number of configs to reach 90% / 99% / 99.9% posterior mass
    n_for_90 = int(np.searchsorted(cum_mass, 0.90) + 1)
    n_for_99 = int(np.searchsorted(cum_mass, 0.99) + 1)
    n_for_999 = int(np.searchsorted(cum_mass, 0.999) + 1)

    # Per-coefficient inclusion probabilities + closed-form reference
    inc_chain = gamma.mean(axis=0)
    inc_truth = problem.inclusion_prob_true

    out = {
        "K_max": K_max,
        "K_true": K_true,
        "g": K_max + 1,
        "config_space_size": 2 ** (K_max + 1),
        "true_active": [int(d) for d in sorted(truth.keys())],
        "true_coefficients": truth,
        "log_Z": float(r.log_Z),
        "log_Z_err": float(r.log_Z_err),
        "n_iter": int(r.log_likelihoods.size),
        "n_dead": int(n_samples),
        "n_unique_gamma": n_unique,
        "n_for_90pct": n_for_90,
        "n_for_99pct": n_for_99,
        "n_for_999pct": n_for_999,
        "top50_configs": [
            (tuple(int(b) for b in key), float(p)) for key, p in top50
        ],
        "inclusion_chain": inc_chain.astype(float),
        "inclusion_truth": np.asarray(inc_truth).astype(float),
        "dt_seconds": dt,
    }
    return out


def main() -> None:
    K_max_grid = [6, 10, 15, 20]
    cells = []
    for K_max in K_max_grid:
        K_true_grid = sorted(set([3, max(3, K_max // 2)]))
        for K_true in K_true_grid:
            print(f"\n[{K_max=}, {K_true=}, g={K_max+1}, "
                  f"|cfg|=2^{K_max+1}={2**(K_max+1)}]", flush=True)
            res = _run_one(K_max, K_true)
            cells.append(res)
            print(f"  log Z = {res['log_Z']:+.2f} +/- {res['log_Z_err']:.2f}, "
                  f"n_iter = {res['n_iter']}, "
                  f"n_unique_gamma = {res['n_unique_gamma']} / "
                  f"{res['config_space_size']}, "
                  f"top-N for 90%/99%/99.9% mass = "
                  f"{res['n_for_90pct']}/{res['n_for_99pct']}/"
                  f"{res['n_for_999pct']}, "
                  f"dt = {res['dt_seconds']:.0f} s",
                  flush=True)
            inc_diff = np.abs(res["inclusion_chain"] - res["inclusion_truth"])
            print(f"  max |P_chain(gamma_k) - P_truth(gamma_k)| = "
                  f"{inc_diff.max():.4f}, mean = {inc_diff.mean():.4f}",
                  flush=True)

    out_path = os.path.join(HERE, "polynomial_scaling.npz")
    np.savez_compressed(out_path, cells=np.asarray(cells, dtype=object))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
