"""Fixed-dimensional cross-check of the HD 10180 candidate-b detection.

Runs plain (non-trans-dimensional) nested sampling on the 7-planet
(b + c..h) and 6-planet (c..h) models with priors identical to the
labelled trans-dim analysis: same per-slot windows, same K/jitter/offset
priors, same DE kernel and chain hyperparameters. The inactive b slot in
the 6-planet model keeps its (ignored) prior coordinates so both runs
integrate over the same 23-dimensional unit cube and the two evidences
share an identical prior measure.

With the other six candidates at P(gamma=1|y) = 1.000 and even inclusion
priors, the trans-dim inclusion odds for b equal Z_7/Z_6 exactly, so
this pair measures the b Bayes factor independently of the trans-dim
kernel's recorded-bit statistics.

Writes scripts/hd10180_fixed_dim_bf.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import daedalus  # noqa: E402
from daedalus.samplers import DifferentialEvolutionSampler  # noqa: E402
import hd10180_lovis_labelled as lab  # noqa: E402
from insertion_index import insertion_index_test  # noqa: E402

N_LIVE = 600
SEEDS = (42,)


def run_one(spec):
    model, seed = spec
    d = _load_harps_npz()
    t, rv, err = d["t"], d["rv"], d["err"]
    loglike, prior_transform, _groups, periodic = lab.build_problem(t, rv, err)
    gamma_fixed = np.ones(lab.N_SLOTS, dtype=bool)
    if model == "six":
        gamma_fixed[0] = False  # b off; c..h on

    def ll(beta, gamma):
        return loglike(beta, gamma_fixed)

    sampler = daedalus.NestedSampler(
        loglike=ll, prior_transform=prior_transform, ndim=lab.NDIM,
        bound="multi",
        sample=DifferentialEvolutionSampler(target_accept=0.234),
        n_live=N_LIVE, seed=seed, periodic=periodic)
    rec: list[int] = []
    t0 = time.time()
    res = sampler.run_nested(dlogz=0.3, n_mcmc=115, show_progress=False,
                             insertion_recorder=rec)
    el = time.time() - t0
    it = insertion_index_test(np.asarray(rec, float), N_LIVE)
    return {"model": model, "seed": seed, "logZ": float(res.log_Z),
            "logZ_err": float(res.log_Z_err),
            "maxL": float(np.max(res.log_likelihoods)),
            "ks_p": float(it.ks_pvalue), "z_mean": float(it.z_mean),
            "elapsed": el}


def main() -> None:
    specs = [(m, s) for m in ("seven", "six") for s in SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for r in ex.map(run_one, specs):
            rows.append(r)
            print(f"{r['model']:5s} s{r['seed']}: logZ={r['logZ']:.3f}"
                  f"+/-{r['logZ_err']:.3f} maxL={r['maxL']:.2f} "
                  f"ks_p={r['ks_p']:.3g} z={r['z_mean']:+.2f} "
                  f"{r['elapsed']:.0f}s", flush=True)
    for s in SEEDS:
        z7 = next(r for r in rows if r["model"] == "seven" and r["seed"] == s)
        z6 = next(r for r in rows if r["model"] == "six" and r["seed"] == s)
        bf = z7["logZ"] - z6["logZ"]
        err = float(np.hypot(z7["logZ_err"], z6["logZ_err"]))
        odds = np.exp(bf)
        print(f"seed {s}: ln BF(b) = {bf:+.2f} +/- {err:.2f}  "
              f"-> P(b) = {odds/(1+odds):.3f}", flush=True)
    with open(os.path.join(HERE, "hd10180_fixed_dim_bf.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=1)


def _load_harps_npz():
    """HARPS RVs as a dict with keys t/rv/err; fetches and caches on first use."""
    t, rv, err = lab.load_harps()
    return {"t": t, "rv": rv, "err": err}


if __name__ == "__main__":
    main()
