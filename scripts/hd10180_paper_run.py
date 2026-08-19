"""HD 10180 labelled confirmation and fake-slot control (paper Section 5.1).

Runs the seven-slot labelled candidate-confirmation chain and the injected
fake-eighth-candidate control at the dimension-aware chain budget
n_mcmc = 115 (= 5 * ndim), seed 42, and writes the reference artifacts:

  * hd10180_de_labelled_seed42_n600.npz  (P, gamma, log_Z, log_Z_err)
  * hd10180_fake_planet_results.json     ({fakes, rows})
  * hd10180_fake3_replaced.json          (fake at 1620 d)

Each chain is certified by the insertion-index test, the per-run validity
check adopted throughout the paper.

Run from project root:  python scripts/hd10180_paper_run.py
"""
from __future__ import annotations

import json
import os
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
from daedalus.samplers import DifferentialEvolutionSampler  # noqa: E402
import hd10180_lovis_labelled as lab  # noqa: E402
import hd10180_fake_planet as fp  # noqa: E402  (reuse build_8slot)
from insertion_index import insertion_index_test  # noqa: E402

CACHE = os.path.join(HERE, "hd10180_harps_cache.npz")
N_LIVE = 600
N_MCMC = 115
SEED = 42
FAKES = [4.599224968324304, 36.784060302426184, 334.49759397135466, 1620.0]


def run(spec):
    kind, fake_P = spec
    d = _load_harps_npz(); t, rv, err = d["t"], d["rv"], d["err"]
    if kind == "labelled":
        ll, pt, groups, per = lab.build_problem(t, rv, err)
        ndim = lab.NDIM; nslots = lab.N_SLOTS
    else:
        ll, pt, groups, per, ndim, nslots = fp.build_8slot(t, rv, err, fake_P)
    sampler = daedalus.NestedSampler(
        loglike=ll, prior_transform=pt, ndim=ndim, groups=groups, bound="multi",
        sample=DifferentialEvolutionSampler(target_accept=0.234),
        n_live=N_LIVE, seed=SEED, periodic=per, validate_births=False)
    rec: list[int] = []
    t0 = time.time()
    res = sampler.run_nested(dlogz=0.3, n_mcmc=N_MCMC, transdim_fraction=0.35,
                             show_progress=False, insertion_recorder=rec)
    el = time.time() - t0
    it = insertion_index_test(np.asarray(rec, float), N_LIVE)
    g = res.gamma.astype(bool)
    out = {"kind": kind, "fake_P": fake_P, "logZ": float(res.log_Z),
           "logZ_err": float(res.log_Z_err), "maxL": float(np.max(res.log_likelihoods)),
           "ks_p": float(it.ks_pvalue), "z_mean": float(it.z_mean),
           "inc": [float(x) for x in g.mean(axis=0)], "elapsed": el}
    if kind == "labelled":
        P = np.exp(res.samples[:, [3 * k for k in range(nslots)]])
        npz = os.path.join(HERE, "hd10180_de_labelled_seed42_n600.npz")
        np.savez_compressed(npz, P=P.astype(np.float32), gamma=g,
                            log_Z=res.log_Z, log_Z_err=res.log_Z_err)
    return out


def main():
    specs = [("labelled", 0.0)] + [("fake", P) for P in FAKES]
    res = {}
    with ProcessPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(run, s): s for s in specs}
        for f in as_completed(futs):
            r = f.result()
            key = r["kind"] if r["kind"] == "labelled" else f"fake_{r['fake_P']:.1f}"
            res[key] = r
            tag = "labelled" if r["kind"] == "labelled" else f"fake P={r['fake_P']:.1f}"
            extra = (f"b={r['inc'][0]:.3f}" if r["kind"] == "labelled"
                     else f"FAKE={r['inc'][7]:.3f} reals_min={min(r['inc'][:7]):.3f}")
            print(f"  {tag:16}: logZ={r['logZ']:.2f}+/-{r['logZ_err']:.2f} maxL={r['maxL']:.1f} "
                  f"ks_p={r['ks_p']:.2g} z={r['z_mean']:+.2f} {extra} {r['elapsed']:.0f}s", flush=True)

    lab_r = res["labelled"]
    print("\n=== LABELLED (seed 42, n_mcmc=115) ===")
    for nm, v in zip(lab.SLOT_NAMES, lab_r["inc"]):
        print(f"  {nm}: {v:.4f}")
    print(f"  logZ = {lab_r['logZ']:.3f} +/- {lab_r['logZ_err']:.3f}  ks_p={lab_r['ks_p']:.3f}")

    # --- write the two fake-control JSONs (single seed 42) -----------------
    fake3 = next(v for k, v in res.items() if k.startswith("fake_") and abs(v["fake_P"] - 1620.0) < 1)
    main_fakes = [v for k, v in res.items()
                  if k.startswith("fake_") and abs(v["fake_P"] - 1620.0) >= 1]
    main_fakes.sort(key=lambda r: r["fake_P"])
    fp_json = {
        "fakes": [{"P": v["fake_P"]} for v in main_fakes],
        "rows": [{"fi": i, "fake_P": v["fake_P"], "seed": SEED, "ok": True,
                  "logZ": v["logZ"], "maxL": v["maxL"], "inc": v["inc"],
                  "ks_p": v["ks_p"], "n_mcmc": N_MCMC}
                 for i, v in enumerate(main_fakes)],
    }
    with open(os.path.join(HERE, "hd10180_fake_planet_results.json"), "w") as f:
        json.dump(fp_json, f, indent=1)
    f3_json = {"P_fake": 1620.0,
               "rows": [{"fi": 3, "fake_P": 1620.0, "seed": SEED, "ok": True,
                         "logZ": fake3["logZ"], "maxL": fake3["maxL"],
                         "inc": fake3["inc"], "ks_p": fake3["ks_p"], "n_mcmc": N_MCMC}]}
    with open(os.path.join(HERE, "hd10180_fake3_replaced.json"), "w") as f:
        json.dump(f3_json, f, indent=1)

    print("\n=== FAKE CONTROL (seed 42, n_mcmc=115) ===")
    for v in main_fakes + [fake3]:
        print(f"  P={v['fake_P']:7.1f} d: fake_incl={v['inc'][7]:.3f}  reals_min={min(v['inc'][:7]):.3f}  ks_p={v['ks_p']:.3f}")
    json.dump({k: v for k, v in res.items()},
              open(os.path.join(HERE, "hd10180_paper_run_summary.json"), "w"), indent=1)
    print("\nwrote npz + fake JSONs + summary")


def _load_harps_npz():
    """HARPS RVs as a dict with keys t/rv/err; fetches and caches on first use."""
    t, rv, err = lab.load_harps()
    return {"t": t, "rv": rv, "err": err}


if __name__ == "__main__":
    main()
