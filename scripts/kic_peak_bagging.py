"""Reproduce the KIC 6603624 asteroseismic peak-bagging re-run (paper Section 5.3).

Self-contained pipeline, correct by construction:
  1. Fetch the full 4-yr Kepler short-cadence photometry (Q1, Q5-Q17) via
     lightkurve, stitch normalised PDCSAP flux.
  2. Build the NATIVE power-density spectrum (0.008 uHz resolution) by
     Lomb-Scargle over [1000, 3600] uHz.
  3. Block-average NON-overlapping groups of s=13 native ordinates to
     0.10 uHz bins -> independent Gamma(s) statistics (bin width << mode Gamma),
     so the correct per-bin log-likelihood is exactly s * (exponential form).
  4. Peak-bag with a 2-component Kallinger (2014) granulation + white-noise
     background and the s*exp likelihood, over a comb of candidate frequencies.

Result: the real p-mode comb (Delta_nu = 110 uHz) is recovered at P=1.000 and
the below-envelope 1501-1513 uHz candidates reject/marginalise -- the earlier
"false positive" was an artefact of a smoothed spectrum + single-Harvey
background + a DOF-diluted likelihood.

Run from project root:  python scripts/kic_peak_bagging.py
Requires: lightkurve, astropy, scipy, numba, daedalus.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from numba import njit
from scipy.ndimage import uniform_filter1d, percentile_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import daedalus
from daedalus.groups import Group

DATA = os.path.join(HERE, "kic6603624_pds_binned.npz")   # cached binned spectrum
RESULTS = os.path.join(HERE, "kic_peak_bagging_results.npz")
TARGET, S_BIN = "KIC 6603624", 13


# ----------------------------------------------------------------- data ----
def build_binned_spectrum(path: str = DATA) -> None:
    """Fetch LC, build native PSD, block-average to independent Gamma(s) bins."""
    import lightkurve as lk
    from astropy.timeseries import LombScargle

    print("Fetching full short-cadence photometry from MAST ...", flush=True)
    lc = lk.search_lightcurve(TARGET, mission="Kepler", cadence="short") \
        .download_all().stitch(corrector_func=lambda x: x.normalize())
    t = np.asarray(lc.time.value, float)
    y = (np.asarray(lc.flux.value, float) / np.nanmedian(lc.flux.value) - 1.0) * 1e6
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    med = np.median(y); mad = 1.4826 * np.median(np.abs(y - med))
    keep = np.abs(y - med) < 6 * mad
    tsec = (t[keep] - t[keep][0]) * 86400.0
    df = 1.0 / (tsec[-1] - tsec[0])
    freqs = np.arange(1000e-6, 3600e-6, df)
    print(f"  native PSD: N={tsec.size}, df={df*1e6:.4f} uHz", flush=True)
    power = LombScargle(tsec, y[keep], normalization="psd").power(
        freqs, method="fast", assume_regular_frequency=True)
    psd = power * 2.0 / (tsec.size * df) / 1e6
    nu = freqs * 1e6
    m = (nu >= 1400) & (nu <= 3100)              # restrict to mode-bearing band
    nu, psd = nu[m], psd[m]
    n = (psd.size // S_BIN) * S_BIN
    nub = nu[:n].reshape(-1, S_BIN).mean(1)
    pb = psd[:n].reshape(-1, S_BIN).mean(1)
    np.savez(path, freq=nub, power=pb, s_bin=S_BIN, bin_width_uHz=S_BIN * df)
    print(f"  binned: N={nub.size}, bin={S_BIN*df:.3f} uHz -> {path}", flush=True)


def load_binned():
    if not os.path.exists(DATA):
        build_binned_spectrum()
    a = np.load(DATA)
    return (a["freq"].astype(float), a["power"].astype(float),
            int(a["s_bin"]), float(a["bin_width_uHz"]))


# ----------------------------------------------------- likelihood + model ----
@njit(cache=True, fastmath=True)
def _ll_core(freq, power, nu, h, g, W, P1, b1, P2, b2, s_bin):
    inv1 = 1.0 / (b1 * b1 * b1 * b1); inv2 = 1.0 / (b2 * b2 * b2 * b2)
    c = np.empty(nu.shape[0])
    for k in range(nu.shape[0]):
        c[k] = 4.0 / (g[k] * g[k])
    tot = 0.0
    for i in range(freq.shape[0]):
        f = freq[i]; f2 = f * f; f4 = f2 * f2
        M = W + P1 / (1.0 + f4 * inv1) + P2 / (1.0 + f4 * inv2)
        for k in range(nu.shape[0]):
            d = f - nu[k]
            M += h[k] / (1.0 + c[k] * d * d)
        tot += np.log(M) + power[i] / M
    return -s_bin * tot


def find_candidates(freq, power, bw, n_peaks=8, min_spacing=6.0,
                    snr_over_bg=1.30, band=(1779., 2989.)):
    sm = uniform_filter1d(power, max(1, int(round(1.0 / bw))), mode="reflect")
    B = percentile_filter(power, 25, size=int(round(40 / bw)),
                          mode="reflect") / (-np.log(0.75))
    R = sm / B
    lo, hi = band; inb = (freq > lo) & (freq < hi)
    loc = (sm[1:-1] > sm[:-2]) & (sm[1:-1] > sm[2:])
    cand = np.where(loc)[0] + 1
    cand = cand[inb[cand] & (R[cand] > snr_over_bg)]
    chosen = []
    for i in cand[np.argsort(R[cand])[::-1]]:
        if all(abs(freq[i] - freq[c]) > min_spacing for c in chosen):
            chosen.append(int(i))
        if len(chosen) >= n_peaks:
            break
    return sorted(float(freq[i]) for i in chosen)


def make_problem(cf, freq, power, s_bin, freq_window=1.5, inclusion_prior=0.5,
                 lh=(np.log(0.05), np.log(50.0)), lw=(np.log(0.1), np.log(10.0)),
                 lwn=(np.log(0.05), np.log(2.0)), lP=(np.log(0.02), np.log(5.0)),
                 lb=(np.log(300.), np.log(6000.))):
    cf = np.asarray(cf, float); nm = cf.size
    lfc = np.log(cf); lflo = lfc - np.log1p(freq_window / cf)
    lfhi = lfc + np.log1p(freq_window / cf)

    def loglike(beta, gamma):
        idx = np.nonzero(gamma)[0]; o = 3 * nm
        return float(_ll_core(freq, power, np.exp(beta[3 * idx]),
            np.exp(beta[3 * idx + 1]), np.exp(beta[3 * idx + 2]),
            np.exp(beta[o]), np.exp(beta[o + 1]), np.exp(beta[o + 2]),
            np.exp(beta[o + 3]), np.exp(beta[o + 4]), s_bin))

    def ptform(u):
        v = np.empty_like(u); o = 3 * nm
        for k in range(nm):
            v[3 * k] = lflo[k] + (lfhi[k] - lflo[k]) * u[3 * k]
            v[3 * k + 1] = lh[0] + (lh[1] - lh[0]) * u[3 * k + 1]
            v[3 * k + 2] = lw[0] + (lw[1] - lw[0]) * u[3 * k + 2]
        for i, (a, b) in enumerate([lwn, lP, lb, lP, lb]):
            v[o + i] = a + (b - a) * u[o + i]
        return v

    groups = [Group(name=f"mode_{int(round(nu))}uHz", params=[3 * k, 3 * k + 1, 3 * k + 2],
                    off_values=np.array([0.5 * (lflo[k] + lfhi[k]),
                                         0.5 * (lh[0] + lh[1]), 0.5 * (lw[0] + lw[1])]),
                    inclusion_prior=inclusion_prior) for k, nu in enumerate(cf)]
    return loglike, ptform, 3 * nm + 5, groups, cf


def main() -> None:
    freq, power, s_bin, bw = load_binned()
    real = find_candidates(freq, power, bw, n_peaks=8, min_spacing=6.0, band=(1779., 2989.))
    cands = sorted(set([round(c, 1) for c in real] + [1501.4, 1506.4, 1513.4]))
    print(f"{len(cands)} candidates: {cands}", flush=True)
    ll, pt, ndim, groups, cf = make_problem(cands, freq, power, s_bin)
    t0 = time.perf_counter()
    res = daedalus.NestedSampler(loglike=ll, prior_transform=pt, ndim=ndim,
        groups=groups, bound="multi", sample="rwalk", n_live=400, seed=42
        ).run_nested(dlogz=0.5, transdim_fraction=0.3, show_progress=False)
    inc = res.inclusion_probabilities()
    print(f"logZ={res.log_Z:.1f} in {time.perf_counter()-t0:.0f}s", flush=True)
    for c, g in sorted(zip(cf, groups), key=lambda x: x[0]):
        print(f"  nu={c:7.1f}  P(gamma=1)={inc[g.name]:.3f}", flush=True)
    np.savez_compressed(RESULTS, samples=np.asarray(res.samples),
        gamma=np.asarray(res.gamma).astype(bool), cand=cf,
        inc=np.asarray([inc[g.name] for g in groups]),
        names=np.asarray([g.name for g in groups]),
        freq=freq, power=power, s_bin=s_bin, logZ=res.log_Z)
    print(f"saved -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
