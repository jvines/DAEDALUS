"""Confirmatory real-data chains: KIC 6603624, AU Mic, SDSS (paper Section 5).

Runs the three confirmatory applications at the dimension-aware default
chain budget n_mcmc = 5 * ndim, seed 42, and writes their entries into
scripts/real_data_results.npz (produced by real_data_chains.py), recording
the insertion-index validity test for each run.

Run from project root:  python scripts/real_data_confirmatory.py
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import daedalus  # noqa: E402
from daedalus.benchmarks import asteroseismic, flares, spectroscopy_real  # noqa: E402
from daedalus.samplers import DifferentialEvolutionSampler  # noqa: E402
from insertion_index import insertion_index_test  # noqa: E402
from real_data_chains import _summarise  # noqa: E402

NPZ = os.path.join(HERE, "real_data_results.npz")


def run_app(name):
    if name == "kic6603624":
        prob = asteroseismic.make_problem(n_candidates=10)
        groups, kernel, nlive, bound, transdim = prob.groups, "rwalk", 500, "single", 0.3
        extra = {"candidate_frequencies": list(prob.candidate_frequencies)}
    elif name == "au_mic":
        prob = flares.make_problem(n_candidates=8, snr_threshold=4.0)
        groups, kernel, nlive, bound, transdim = prob.groups, "rwalk", 400, "single", 0.3
        extra = {"candidate_peaks": list(prob.candidate_peaks)}
    elif name == "sdss":
        prob = spectroscopy_real.make_problem()
        groups = [daedalus.Group(**kw) for kw in prob.groups_kwargs]
        kernel = DifferentialEvolutionSampler(target_accept=0.234)
        nlive, bound, transdim = 500, "multi", 0.3
        extra = {"line_names": list(prob.line_names), "redshift": float(prob.z)}
    sampler = daedalus.NestedSampler(
        loglike=prob.loglike, prior_transform=prob.prior_transform, ndim=prob.ndim,
        groups=groups, bound=bound, sample=kernel, n_live=nlive, seed=42)
    rec: list[int] = []
    t0 = time.time()
    res = sampler.run_nested(dlogz=0.3, transdim_fraction=transdim,  # n_mcmc=None -> 5*ndim
                             show_progress=False, insertion_recorder=rec)
    dt = time.time() - t0
    out = _summarise(res, name)
    out.update(extra)
    it = insertion_index_test(np.asarray(rec, float), nlive)
    out["insertion"] = {"z_mean": float(it.z_mean), "ks_p": float(it.ks_pvalue),
                        "mean_fraction": float(it.mean_fraction)}
    out["ndim"] = int(prob.ndim); out["n_mcmc"] = max(25, 5 * prob.ndim)
    out["dt_seconds"] = dt
    return name, out


def main():
    if not os.path.exists(NPZ):
        raise SystemExit(
            f"missing {NPZ}\n"
            "Run scripts/real_data_chains.py first -- it produces the "
            "real_data_results.npz that this script patches."
        )
    shutil.copy2(NPZ, NPZ + ".bak")
    all_res = np.load(NPZ, allow_pickle=True)["results"].item()
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(run_app, n) for n in ("kic6603624", "au_mic", "sdss")]
        for f in as_completed(futs):
            name, out = f.result()
            all_res[name] = out
            inc = {k: round(v, 3) for k, v in out["inclusion"].items()}
            print(f"[{name}] ndim={out['ndim']} n_mcmc={out['n_mcmc']} "
                  f"logZ={out['log_Z']:.2f}+/-{out['log_Z_err']:.2f} "
                  f"ins z={out['insertion']['z_mean']:+.2f} ks_p={out['insertion']['ks_p']:.2g} "
                  f"{out['dt_seconds']:.0f}s", flush=True)
            print(f"    inclusion: {inc}", flush=True)
    np.savez_compressed(NPZ, results=np.array(all_res, dtype=object))
    print(f"\npatched kic6603624/au_mic/sdss in {NPZ} (backup .bak)")


if __name__ == "__main__":
    main()
