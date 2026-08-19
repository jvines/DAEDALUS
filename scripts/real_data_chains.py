"""Run all four real-data trans-dim chains for paper §5 and dump results.

Produces ``scripts/real_data_results.npz`` containing inclusion
probabilities, model-probability dicts, log-evidence summaries,
sample subsets, and gamma chains for:

    1. WASP-47 tight-prior confirmatory transit fit
    2. KIC 6603624 asteroseismic peak bagging
    3. AU Mic TESS flare counting
    4. SDSS star-forming galaxy emission-line trans-dim

Run from project root:
    python scripts/real_data_chains.py
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
from daedalus.benchmarks import (  # noqa: E402
    asteroseismic,
    flares,
    spectroscopy_real,
    wasp47,
)


def _summarise(results, label: str) -> dict:
    inc = results.inclusion_probabilities()
    out = {
        "label": label,
        "log_Z": float(results.log_Z),
        "log_Z_err": float(results.log_Z_err),
        "inclusion": {k: float(v) for k, v in inc.items()},
        "n_iter": int(results.log_likelihoods.size),
    }
    try:
        mp = results.model_probabilities()
        out["model_probs"] = {tuple(k): float(v) for k, v in mp.items()}
    except Exception as exc:
        out["model_probs"] = {}
        out["model_probs_error"] = repr(exc)
    sub_n = min(results.samples.shape[0], 4000)
    rng = np.random.default_rng(0)
    sub = rng.choice(results.samples.shape[0], size=sub_n, replace=False)
    out["samples_sub"] = results.samples[sub].astype(np.float32)
    out["gamma_sub"] = results.gamma[sub].astype(bool)
    return out


def run_wasp47() -> dict:
    print("\n[1/4] WASP-47 tight-prior trans-dim...", flush=True)
    t0 = time.time()
    problem = wasp47.make_problem_tight()
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=problem.groups,
        bound="single",
        sample="rwalk",
        n_live=500,
        seed=42,
    )
    results = sampler.run_nested(
        dlogz=0.3, n_mcmc=40, transdim_fraction=0.3,
        bound_update_interval=10, show_progress=False,
    )
    out = _summarise(results, "WASP-47 (tight prior, b/d/e)")
    out["candidate_names"] = list(problem.candidate_names)
    out["n_planets_posterior"] = wasp47.n_planets_posterior(results)
    out["dt_seconds"] = time.time() - t0
    print(f"    log Z = {out['log_Z']:.2f} +/- {out['log_Z_err']:.2f}, "
          f"{out['n_iter']} iter, {out['dt_seconds']:.0f} s", flush=True)
    print(f"    inclusion: {out['inclusion']}", flush=True)
    print(f"    N_p posterior: {out['n_planets_posterior']}", flush=True)
    return out


def run_asteroseismic() -> dict:
    print("\n[2/4] KIC 6603624 peak bagging...", flush=True)
    t0 = time.time()
    problem = asteroseismic.make_problem(n_candidates=10)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=problem.groups,
        bound="single",
        sample="rwalk",
        n_live=500,
        seed=42,
    )
    results = sampler.run_nested(
        dlogz=0.3, n_mcmc=40, transdim_fraction=0.3,
        bound_update_interval=10, show_progress=False,
    )
    out = _summarise(results, "KIC 6603624 peak bagging")
    out["candidate_frequencies"] = list(problem.candidate_frequencies)
    out["dt_seconds"] = time.time() - t0
    print(f"    log Z = {out['log_Z']:.2f} +/- {out['log_Z_err']:.2f}, "
          f"{out['n_iter']} iter, {out['dt_seconds']:.0f} s", flush=True)
    print(f"    inclusion: {out['inclusion']}", flush=True)
    return out


def run_flares() -> dict:
    print("\n[3/4] AU Mic flare counting...", flush=True)
    t0 = time.time()
    problem = flares.make_problem(n_candidates=8, snr_threshold=4.0)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=problem.groups,
        bound="single",
        sample="rwalk",
        n_live=400,
        seed=42,
    )
    results = sampler.run_nested(
        dlogz=0.3, n_mcmc=30, transdim_fraction=0.3,
        bound_update_interval=10, show_progress=False,
    )
    out = _summarise(results, "AU Mic flares (TESS S1)")
    out["candidate_peaks"] = list(problem.candidate_peaks)
    out["dt_seconds"] = time.time() - t0
    print(f"    log Z = {out['log_Z']:.2f} +/- {out['log_Z_err']:.2f}, "
          f"{out['n_iter']} iter, {out['dt_seconds']:.0f} s", flush=True)
    print(f"    inclusion: {out['inclusion']}", flush=True)
    return out


def run_sdss() -> dict:
    print("\n[4/4] SDSS galaxy emission-line trans-dim...", flush=True)
    t0 = time.time()
    problem = spectroscopy_real.make_problem()
    groups = [daedalus.Group(**kw) for kw in problem.groups_kwargs]
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=groups,
        bound="single",
        sample="rwalk",
        n_live=400,
        seed=42,
    )
    results = sampler.run_nested(
        dlogz=0.3, n_mcmc=30, transdim_fraction=0.5,
        bound_update_interval=10, show_progress=False,
    )
    out = _summarise(results, "SDSS galaxy emission lines")
    out["line_names"] = list(problem.line_names)
    out["redshift"] = float(problem.z)
    out["dt_seconds"] = time.time() - t0
    print(f"    log Z = {out['log_Z']:.2f} +/- {out['log_Z_err']:.2f}, "
          f"{out['n_iter']} iter, {out['dt_seconds']:.0f} s", flush=True)
    print(f"    inclusion: {out['inclusion']}", flush=True)
    return out


def main() -> None:
    out = {
        "wasp47": run_wasp47(),
        "kic6603624": run_asteroseismic(),
        "au_mic": run_flares(),
        "sdss": run_sdss(),
    }
    npz_path = os.path.join(HERE, "real_data_results.npz")
    np.savez_compressed(npz_path, results=out)
    print(f"\nWrote: {npz_path}")


if __name__ == "__main__":
    main()
