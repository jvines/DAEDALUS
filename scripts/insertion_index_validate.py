"""Validate the insertion-index diagnostic on a controlled problem.

Two checks:

  (A) A correctly-mixing reference: a low-dim Gaussian with generous n_mcmc
      should give mean insertion fraction ~0.5 and a non-significant KS.

  (B) A deliberately under-mixed run: the SAME Gaussian in high dim with a
      tiny n_mcmc should skew the insertion fraction LOW and reject KS.

If the diagnostic separates these two cases, it is a valid under-mixing
meter and we can trust it on HD 10180.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import daedalus
from daedalus.samplers import RandomWalkSampler
from insertion_index import insertion_index_test


def gaussian_problem(ndim, sigma=0.1):
    """Unit-cube prior, Gaussian likelihood centred at 0.5, width sigma."""
    def loglike(beta):
        d = beta - 0.5
        return float(-0.5 * np.sum(d * d) / (sigma * sigma))

    def prior_transform(u):
        return u.copy()

    return loglike, prior_transform


def run_case(label, ndim, n_mcmc, n_live, seed, sigma=0.1):
    loglike, prior_transform = gaussian_problem(ndim, sigma)
    sampler = daedalus.NestedSampler(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=ndim,
        sample=RandomWalkSampler(target_accept=0.5, proposal="ball"),
        n_live=n_live,
        seed=seed,
    )
    rec: list[int] = []
    res = sampler.run_nested(
        dlogz=0.5, n_mcmc=n_mcmc, show_progress=False,
        insertion_recorder=rec,
    )
    idx = np.asarray(rec, dtype=int)
    test = insertion_index_test(idx, n_live)
    print(f"\n=== {label}: ndim={ndim} n_mcmc={n_mcmc} n_live={n_live} seed={seed}")
    print(f"  n_iter recorded     : {test.n_iter}")
    print(f"  logZ                : {res.log_Z:.3f} +/- {res.log_Z_err:.3f}")
    print(f"  mean insert fraction: {test.mean_fraction:.4f} "
          f"(SE {test.mean_fraction_se:.4f}; 0.5 = correct)")
    print(f"  z(mean vs 0.5)      : {test.z_mean:+.2f}")
    print(f"  KS stat / p         : {test.ks_stat:.4f} / {test.ks_pvalue:.3e}")
    print(f"  rolling min p       : {test.rolling_min_pvalue:.3e} "
          f"(window {test.rolling_window}, {test.rolling_n_windows} windows, "
          f"frac low {test.rolling_frac_windows_low:.2f})")
    print(f"  index range observed: [{idx.min()}, {idx.max()}] of [0,{n_live-1}]")
    print(f"  VERDICT             : {test.verdict}")
    return test


if __name__ == "__main__":
    # (A) Well-mixed reference: low dim, generous mixing.
    run_case("WELL-MIXED REF", ndim=2, n_mcmc=50, n_live=400, seed=1)
    run_case("WELL-MIXED REF", ndim=2, n_mcmc=50, n_live=400, seed=2)

    # (B) Deliberately under-mixed: high dim, starved n_mcmc.
    run_case("UNDER-MIXED", ndim=23, n_mcmc=2, n_live=400, seed=1)
    run_case("UNDER-MIXED", ndim=23, n_mcmc=2, n_live=400, seed=2)

    # (C) Intermediate: high dim, n_mcmc=40 (the HD 10180 within-model budget)
    run_case("HD10180-BUDGET (23d, n_mcmc=40)", ndim=23, n_mcmc=40,
             n_live=400, seed=1)
