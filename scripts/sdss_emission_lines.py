"""SDSS emission-line chain with the documented paper configuration.

The paper (Section: SDSS star-forming galaxy) adopts the
differential-evolution within-model kernel for SDSS because the
random-walk kernel fails the insertion-index test on this correlated
line-fit likelihood. This script is the provenance for the SDSS entry of
real_data_results.npz: DE kernel, multi-ellipsoid bound, n_live = 500,
n_mcmc = 40, dlogz = 0.3, transdim_fraction = 0.3, seed 42, with the
insertion-index statistics recorded alongside the chain outputs.

Patches ONLY the 'sdss' entry of scripts/real_data_results.npz (a backup
copy is written first).
"""
from __future__ import annotations

import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import daedalus  # noqa: E402
from daedalus.benchmarks import spectroscopy_real  # noqa: E402
from daedalus.samplers import DifferentialEvolutionSampler  # noqa: E402
from insertion_index import insertion_index_test  # noqa: E402

NPZ = os.path.join(HERE, "real_data_results.npz")
N_LIVE = 500


def main() -> None:
    if not os.path.exists(NPZ):
        raise SystemExit(
            f"missing {NPZ}\n"
            "Run scripts/real_data_chains.py first -- it produces the "
            "real_data_results.npz that this script patches."
        )
    problem = spectroscopy_real.make_problem()
    groups = [daedalus.Group(**kw) for kw in problem.groups_kwargs]
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=groups,
        bound="multi",
        sample=DifferentialEvolutionSampler(target_accept=0.234),
        n_live=N_LIVE,
        seed=42,
    )
    rec: list[int] = []
    t0 = time.time()
    results = sampler.run_nested(
        dlogz=0.3, n_mcmc=40, transdim_fraction=0.3,
        show_progress=False, insertion_recorder=rec,
    )
    dt = time.time() - t0
    it = insertion_index_test(np.asarray(rec, float), N_LIVE)

    inc = results.inclusion_probabilities()
    out = {
        "label": "SDSS galaxy emission lines (DE, documented config)",
        "log_Z": float(results.log_Z),
        "log_Z_err": float(results.log_Z_err),
        "inclusion": {k: float(v) for k, v in inc.items()},
        "n_iter": int(results.log_likelihoods.size),
        "model_probs": {},
        "line_names": list(problem.line_names),
        "redshift": float(problem.z),
        "dt_seconds": dt,
        "config": {
            "sample": "de", "bound": "multi", "n_live": N_LIVE,
            "n_mcmc": 40, "dlogz": 0.3, "transdim_fraction": 0.3,
            "seed": 42,
        },
        "insertion": {
            "mean_fraction": float(it.mean_fraction),
            "z_mean": float(it.z_mean),
            "ks_p": float(it.ks_pvalue),
            "n_recorded": int(len(rec)),
        },
    }
    try:
        mp = results.model_probabilities()
        out["model_probs"] = {tuple(k): float(v) for k, v in mp.items()}
    except Exception as exc:  # pragma: no cover
        out["model_probs_error"] = repr(exc)
    sub_n = min(results.samples.shape[0], 4000)
    rng = np.random.default_rng(0)
    sub = rng.choice(results.samples.shape[0], size=sub_n, replace=False)
    out["samples_sub"] = results.samples[sub].astype(np.float32)
    out["gamma_sub"] = results.gamma[sub].astype(bool)

    print(f"logZ = {out['log_Z']:.2f} +/- {out['log_Z_err']:.2f}; "
          f"{out['n_iter']} iter; {dt:.0f} s")
    print(f"insertion: mean_frac={it.mean_fraction:.4f} "
          f"z={it.z_mean:+.2f} ks_p={it.ks_pvalue:.3g}")
    for k, v in out["inclusion"].items():
        print(f"  {k}: {v:.4f}")

    shutil.copy2(NPZ, NPZ + ".bak")
    all_results = np.load(NPZ, allow_pickle=True)["results"].item()
    all_results["sdss"] = out
    np.savez_compressed(NPZ, results=np.array(all_results, dtype=object))
    print(f"patched sdss entry of {NPZ} (backup at .bak)")


if __name__ == "__main__":
    main()
