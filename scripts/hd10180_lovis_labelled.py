"""HD 10180 as a LABELLED candidate-confirmation problem.

Reframes the HD 10180 analysis from "discover up to N planets in a
wide period prior" to the question MoMS-NS is actually designed for:
"given the seven Lovis (2011) candidate periods b/c/d/e/f/g/h, which
are robustly supported by the HARPS data?"

Each slot is non-exchangeable: slot 0 ↔ b, slot 1 ↔ c, …, slot 6 ↔ h.
The slot's period prior is a tight window centred on the Lovis
literature value (±5% in P), so within-γ permutation modes are
eliminated by construction.

This makes the problem a clean MoMS application: a vector of binary
inclusion indicators (γ_b, γ_c, …, γ_h), each labelling a distinct
named candidate. Single-cloud trans-dim NS with multi-ellipsoid bound
should converge seed-stably, recovering per-candidate detection
posteriors P(γ_k = 1 | y).
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
from daedalus.groups import Group
from daedalus.samplers import RandomWalkSampler


# Lovis 2011 published periods (days)
LOVIS_PERIODS = {
    "b": 1.17766,
    "c": 5.75979,
    "d": 16.3570,
    "e": 49.745,
    "f": 122.74,
    "g": 601.2,
    "h": 2229.0,
}
SLOT_NAMES = ("b", "c", "d", "e", "f", "g", "h")
N_SLOTS = len(SLOT_NAMES)

# Per-slot period window: ±5% multiplicative
P_WINDOW_FRAC = 0.05

# Shared priors (all slots have the same K, t0_phase prior; only P is
# slot-labelled).
LOG_K_LO, LOG_K_HI = float(np.log(0.3)), float(np.log(20.0))
LOG_JIT_LO, LOG_JIT_HI = float(np.log(0.1)), float(np.log(20.0))
GAMMA_LO, GAMMA_HI = -30.0, 30.0
NDIM = 3 * N_SLOTS + 2  # 23
GAMMA_OFFSET_IDX = 3 * N_SLOTS
JIT_IDX = 3 * N_SLOTS + 1


HARPS_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "hd10180_harps_cache.npz")


def load_harps(cache: str = HARPS_CACHE):
    """Return the Lovis+ 2011 HARPS RVs of HD 10180 as ``(t, rv, err)``.

    Fetches VizieR J/A+A/528/A112 on first use and caches the result to
    ``cache`` so the other reproduction scripts share one download.
    Times are days from the first epoch; RVs and errors are m/s with the
    mean RV removed.
    """
    if os.path.exists(cache):
        d = np.load(cache)
        return d["t"], d["rv"], d["err"]
    from astroquery.vizier import Vizier
    v = Vizier()
    v.ROW_LIMIT = -1
    tab = v.get_catalogs("J/A+A/528/A112")[0]
    bjd = np.asarray(tab["BJD"], dtype=np.float64)
    rv_kms = np.asarray(tab["RV"], dtype=np.float64)
    err_kms = np.asarray(tab["e_RV"], dtype=np.float64)
    rv = (rv_kms - rv_kms.mean()) * 1000.0
    err = err_kms * 1000.0
    t = bjd - bjd.min()
    np.savez(cache, t=t, rv=rv, err=err)
    return t, rv, err


def make_slot_period_priors():
    """Return list of (log_P_lo, log_P_hi) per slot."""
    out = []
    for name in SLOT_NAMES:
        P_lit = LOVIS_PERIODS[name]
        P_lo = P_lit * (1.0 - P_WINDOW_FRAC)
        P_hi = P_lit * (1.0 + P_WINDOW_FRAC)
        out.append((float(np.log(P_lo)), float(np.log(P_hi))))
    return out


def build_problem(t, rv, err):
    var_data = err * err
    model_buf = np.empty_like(rv)
    slot_priors = make_slot_period_priors()

    def loglike(beta, gamma):
        model_buf[:] = float(beta[GAMMA_OFFSET_IDX])
        for k in range(N_SLOTS):
            if not gamma[k]:
                continue
            P = float(np.exp(beta[3 * k]))
            t0_phase = float(beta[3 * k + 1])
            K = float(np.exp(beta[3 * k + 2]))
            model_buf[:] += K * np.cos(2.0 * np.pi * (t - t0_phase * P) / P)
        resid = rv - model_buf
        jit2 = float(np.exp(2.0 * float(beta[JIT_IDX])))
        sigma2 = var_data + jit2
        return float(-0.5 * np.sum(resid * resid / sigma2 + np.log(sigma2)))

    def prior_transform(u_):
        v_ = np.empty_like(u_)
        for k, (log_P_lo, log_P_hi) in enumerate(slot_priors):
            v_[3 * k] = log_P_lo + (log_P_hi - log_P_lo) * u_[3 * k]
            v_[3 * k + 1] = u_[3 * k + 1]
            v_[3 * k + 2] = LOG_K_LO + (LOG_K_HI - LOG_K_LO) * u_[3 * k + 2]
        v_[GAMMA_OFFSET_IDX] = GAMMA_LO + (GAMMA_HI - GAMMA_LO) * u_[GAMMA_OFFSET_IDX]
        v_[JIT_IDX] = LOG_JIT_LO + (LOG_JIT_HI - LOG_JIT_LO) * u_[JIT_IDX]
        return v_

    # Off-value for each slot: take the centre of its period window so
    # the inactive state is sensible.
    groups = []
    for k, (log_P_lo, log_P_hi) in enumerate(slot_priors):
        groups.append(Group(
            name=f"slot_{SLOT_NAMES[k]}",
            params=[3 * k, 3 * k + 1, 3 * k + 2],
            off_values=np.array([
                0.5 * (log_P_lo + log_P_hi), 0.0,
                0.5 * (LOG_K_LO + LOG_K_HI),
            ]),
            inclusion_prior=0.5,
        ))

    periodic_idx = [3 * k + 1 for k in range(N_SLOTS)]
    return loglike, prior_transform, groups, periodic_idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-live", type=int, default=1000)
    parser.add_argument("--dlogz", type=float, default=0.3)
    parser.add_argument("--n-mcmc", type=int, default=40)
    parser.add_argument("--transdim-fraction", type=float, default=0.35)
    parser.add_argument("--bound", type=str, default="multi",
                        choices=["single", "multi"])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    print("[lovis-labelled] loading HARPS data", flush=True)
    t, rv, err = load_harps()

    loglike, prior_transform, groups, periodic_idx = build_problem(t, rv, err)
    print(f"[lovis-labelled] {N_SLOTS} slots, NDIM={NDIM}, "
          f"period windows = ±{P_WINDOW_FRAC*100:.0f}% around Lovis lit",
          flush=True)
    for k, name in enumerate(SLOT_NAMES):
        Plo, Phi = np.exp(prior_transform(np.zeros(NDIM))[3 * k]), \
                   np.exp(prior_transform(np.ones(NDIM))[3 * k])
        print(f"  slot_{name}: P ∈ [{Plo:.3f}, {Phi:.3f}] d (lit {LOVIS_PERIODS[name]:.3f})")

    t0 = tm.time()
    sampler = daedalus.NestedSampler(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=NDIM,
        groups=groups,
        bound=args.bound,
        sample=RandomWalkSampler(target_accept=0.5, proposal="ball"),
        n_live=args.n_live,
        seed=args.seed,
        periodic=periodic_idx,
        validate_births=False,
    )
    results = sampler.run_nested(
        dlogz=args.dlogz, n_mcmc=args.n_mcmc,
        transdim_fraction=args.transdim_fraction,
        show_progress=True,
    )
    elapsed = tm.time() - t0

    print(f"\n[lovis-labelled] DONE in {elapsed/60:.1f} min | "
          f"log Z = {results.log_Z:.3f} +/- {results.log_Z_err:.3f}",
          flush=True)
    inc = results.inclusion_probabilities()
    print("\n[lovis-labelled] per-candidate detection posteriors:", flush=True)
    print(f"  {'name':>4} {'P_lit':>10} {'P(γ=1|y)':>12}")
    for name in SLOT_NAMES:
        print(f"  {name:>4} {LOVIS_PERIODS[name]:>10.3f} "
              f"{inc[f'slot_{name}']:>12.3f}")

    n_planets = results.gamma.sum(axis=1)
    n = results.gamma.shape[0]
    print("\n[lovis-labelled] N_planet posterior:")
    for n_p in range(N_SLOTS + 1):
        p = float((n_planets == n_p).mean())
        if p > 0.01:
            print(f"  N = {n_p}: {p:.3f}")

    out_path = args.out or os.path.join(
        ROOT, "scripts",
        f"hd10180_lovis_labelled_seed{args.seed}_nlive{args.n_live}.npz",
    )
    np.savez(
        out_path,
        log_Z=results.log_Z,
        log_Z_err=results.log_Z_err,
        samples=results.samples,
        gamma=results.gamma,
        log_likelihoods=results.log_likelihoods,
        log_weights=results.log_weights,
        inclusion=np.array([inc[f"slot_{n}"] for n in SLOT_NAMES]),
        slot_names=np.array(SLOT_NAMES),
        lovis_periods=np.array([LOVIS_PERIODS[n] for n in SLOT_NAMES]),
        seed=args.seed,
        n_live=args.n_live,
        elapsed_seconds=elapsed,
    )
    print(f"\n[lovis-labelled] saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
