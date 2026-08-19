"""Insertion-index test on the HD 10180 labelled MoMS problem.

Runs the EXACT labelled problem from hd10180_lovis_labelled.py (same loglike,
prior_transform, groups, periodic, RandomWalkSampler, n_mcmc=40,
transdim_fraction=0.35) but at a low n_live where under-mixing is worse and
the run is cheap. Records the per-iteration insertion index via the
``insertion_recorder`` hook added to NestedSampler.run_nested, then tests
those indices for U{0..n_live-1} uniformity (Fowlie, Handley & Su 2020).

HARPS data is fetched from VizieR and cached by hd10180_lovis_labelled.load_harps()
so repeated runs don't hit the network.
"""
from __future__ import annotations

import argparse
import os
import sys
import time as tm

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import daedalus
import hd10180_lovis_labelled as lab
from daedalus.samplers import RandomWalkSampler

from hd10180_lovis_labelled import build_problem, NDIM
from insertion_index import insertion_index_test


def load_data():
    d = _load_harps_npz()
    return d["t"], d["rv"], d["err"]


def run_one(seed, n_live, n_mcmc=40, transdim_fraction=0.35, bound="multi"):
    t, rv, err = load_data()
    loglike, prior_transform, groups, periodic_idx = build_problem(t, rv, err)
    sampler = daedalus.NestedSampler(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=NDIM,
        groups=groups,
        bound=bound,
        sample=RandomWalkSampler(target_accept=0.5, proposal="ball"),
        n_live=n_live,
        seed=seed,
        periodic=periodic_idx,
        validate_births=False,
    )
    rec: list[int] = []
    t0 = tm.time()
    res = sampler.run_nested(
        dlogz=0.3, n_mcmc=n_mcmc, transdim_fraction=transdim_fraction,
        show_progress=False, insertion_recorder=rec,
    )
    elapsed = tm.time() - t0
    idx = np.asarray(rec, dtype=int)
    test = insertion_index_test(idx, n_live)
    return res, test, idx, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-live", type=int, default=300)
    ap.add_argument("--n-mcmc", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--transdim-fraction", type=float, default=0.35)
    ap.add_argument("--save-idx", action="store_true")
    args = ap.parse_args()

    print("HD 10180 labelled insertion-index test")
    print(f"n_live={args.n_live} n_mcmc={args.n_mcmc} "
          f"transdim_fraction={args.transdim_fraction} ndim={NDIM}")
    print("=" * 72)

    logZs = []
    means = []
    for seed in args.seeds:
        res, test, idx, elapsed = run_one(
            seed, args.n_live, n_mcmc=args.n_mcmc,
            transdim_fraction=args.transdim_fraction,
        )
        logZs.append(res.log_Z)
        means.append(test.mean_fraction)
        print(f"\n--- seed {seed}  ({elapsed:.1f}s, {test.n_iter} iters)")
        print(f"  logZ                : {res.log_Z:.3f} +/- {res.log_Z_err:.3f}"
              f"  (Skilling err)")
        print(f"  mean insert fraction: {test.mean_fraction:.4f} "
              f"(SE {test.mean_fraction_se:.4f}; 0.5 = correct)")
        print(f"  z(mean vs 0.5)      : {test.z_mean:+.2f}")
        print(f"  KS stat / p         : {test.ks_stat:.4f} / {test.ks_pvalue:.3e}")
        print(f"  rolling min p       : {test.rolling_min_pvalue:.3e} "
              f"(window {test.rolling_window}, {test.rolling_n_windows} win, "
              f"frac low {test.rolling_frac_windows_low:.2f})")
        print(f"  index range         : [{idx.min()},{idx.max()}] of "
              f"[0,{args.n_live-1}]")
        print(f"  VERDICT             : {test.verdict}")
        if args.save_idx:
            np.save(
                os.path.join(HERE, f"insidx_hd10180_seed{seed}_"
                             f"nlive{args.n_live}_nmcmc{args.n_mcmc}.npy"),
                idx,
            )

    logZs = np.asarray(logZs)
    print("\n" + "=" * 72)
    print(f"ACROSS-SEED logZ spread: std = {logZs.std(ddof=1):.3f} nats "
          f"(range {logZs.min():.2f} .. {logZs.max():.2f})")
    print(f"Mean insert fraction across seeds: "
          f"{np.mean(means):.4f} +/- {np.std(means, ddof=1):.4f}")


def _load_harps_npz():
    """HARPS RVs as a dict with keys t/rv/err; fetches and caches on first use."""
    t, rv, err = lab.load_harps()
    return {"t": t, "rv": rv, "err": err}


if __name__ == "__main__":
    main()
