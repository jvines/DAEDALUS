"""daedalus Fourier-harmonic selection on SX Car (paper Section 5.2).

White-noise model, same setup as the other harmonic applications. Recovers a
monotonic harmonic ladder certified by the insertion-index test.

Consumes the reduced light curve written by scripts/sx_car_reduce.py.

Run from project root:  python scripts/sx_car_run.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import daedalus  # noqa: E402
from daedalus.groups import Group  # noqa: E402
from daedalus.samplers import RandomWalkSampler  # noqa: E402
from insertion_index import insertion_index_test  # noqa: E402

DATA = os.path.join(ROOT, "daedalus", "benchmarks", "data", "sx_car_tess.npz")
N_HARM = 15


def main():
    d = np.load(DATA)
    flux, flux_err = d["flux"], d["flux_err"]
    t = d["time"] - d["time"].min()
    P = float(d["period"])
    var = flux_err * flux_err
    NDIM = 2 * N_HARM + 2
    MFI, JI = 2 * N_HARM, 2 * N_HARM + 1
    bc = np.array([np.cos(n * 2 * np.pi / P * t) for n in range(1, N_HARM + 1)])
    bs = np.array([np.sin(n * 2 * np.pi / P * t) for n in range(1, N_HARM + 1)])
    mbuf = np.empty_like(flux)

    def loglike(beta, gamma):
        mbuf[:] = float(beta[MFI])
        for n in range(N_HARM):
            if gamma[n]:
                mbuf[:] += float(beta[2 * n]) * bc[n] + float(beta[2 * n + 1]) * bs[n]
        r = flux - mbuf
        s2 = var + float(np.exp(2.0 * float(beta[JI])))
        return float(-0.5 * np.sum(r * r / s2 + np.log(s2)))

    def prior_transform(u):
        v = np.empty_like(u)
        for n in range(N_HARM):
            v[2 * n] = -0.5 + u[2 * n]
            v[2 * n + 1] = -0.5 + u[2 * n + 1]
        v[MFI] = 0.8 + 0.4 * u[MFI]
        v[JI] = np.log(1e-5) + (np.log(1e-1) - np.log(1e-5)) * u[JI]
        return v

    def lp(b):
        if abs(b[0]) > 0.5 or abs(b[1]) > 0.5:
            return float("-inf")
        return 0.0

    groups = [Group(name=f"h{n+1}", params=[2 * n, 2 * n + 1], off_values=np.zeros(2),
                    inclusion_prior=0.5, log_prior_continuous=lp) for n in range(N_HARM)]
    s = daedalus.NestedSampler(
        loglike=loglike, prior_transform=prior_transform, ndim=NDIM, groups=groups,
        bound="multi", sample=RandomWalkSampler(target_accept=0.5, proposal="ball"),
        n_live=500, seed=42)
    rec: list[int] = []
    res = s.run_nested(dlogz=0.5, transdim_fraction=0.3, show_progress=False,
                       insertion_recorder=rec)
    it = insertion_index_test(np.asarray(rec, float), 500)
    inc = res.gamma.mean(axis=0)
    g, bsamp = res.gamma, res.samples
    amps = {}
    for n in range(N_HARM):
        on = g[:, n].astype(bool)
        if on.sum() > 10:
            a = np.sqrt(bsamp[on, 2 * n] ** 2 + bsamp[on, 2 * n + 1] ** 2)
            amps[n + 1] = float(np.median(a))
    print(f"SX CAR (P={P:.5f}): logZ={res.log_Z:.2f} +/- {res.log_Z_err:.2f}  "
          f"insertion z={it.z_mean:+.2f} ks_p={it.ks_pvalue:.3g} "
          f"mean_frac={it.mean_fraction:.3f}", flush=True)
    print("inclusions:", " ".join(f"{i+1}:{v:.2f}" for i, v in enumerate(inc)), flush=True)
    print(f"N active (>0.5): {int((inc > 0.5).sum())}", flush=True)
    print("amplitudes:", " ".join(f"A{n}={a:.4f}" for n, a in amps.items()), flush=True)
    if 1 in amps:
        for n in (2, 3):
            if n in amps:
                print(f"R{n}1 = {amps[n] / amps[1]:.3f}", flush=True)
    np.savez(os.path.join(HERE, "sx_car_results.npz"),
             inclusion=inc, log_Z=res.log_Z, log_Z_err=res.log_Z_err,
             insertion_z=it.z_mean, ks_p=it.ks_pvalue, samples=bsamp, gamma=g,
             period=P)


if __name__ == "__main__":
    main()
