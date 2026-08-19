"""Reduce SX Car TESS-SPOC photometry (Sectors 63+64, 200 s, contiguous) into a
clean Fourier-harmonic dataset (paper Section 5.2).

SX Car (Gaia DR3 5350977643913585280) is a fundamental-mode classical Cepheid,
P~4.860 d, G=8.79, unsaturated in TESS (brightness-dependent scatter ratio
~1.3). We use SAP flux (preserves the multi-day pulsation; QLP KSPSAP
over-detrends it), per-sector median-normalise, quality-mask, pin the period by
maximising the 10-harmonic variance explained, and save (time, flux, flux_err).
"""
from __future__ import annotations

import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import lightkurve as lk  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
import astropy.units as u  # noqa: E402

RA, DE = 161.52428, -57.54752
P_GAIA = 4.86012
R21_GAIA, R31_GAIA = 0.413, 0.156
N_HARM = 10
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "daedalus", "benchmarks", "data", "sx_car_tess.npz")


def design(t, P, nh):
    cols = [np.ones_like(t)]
    for n in range(1, nh + 1):
        cols += [np.cos(n * 2 * np.pi / P * t), np.sin(n * 2 * np.pi / P * t)]
    return np.column_stack(cols)


def ve_at(t, f, P):
    M = design(t, P, N_HARM)
    c, *_ = np.linalg.lstsq(M, f, rcond=None)
    r = f - M @ c
    return 1 - r.var() / f.var()


def main():
    c = SkyCoord(RA * u.deg, DE * u.deg)
    sr = lk.search_lightcurve(c, mission="TESS", author="TESS-SPOC", radius=10 * u.arcsec)
    sr = sr[[int(e) <= 200 for e in sr.table["exptime"]]]   # 200 s sectors 63,64
    print(f"using {len(sr)} TESS-SPOC 200s products", flush=True)
    coll = sr.download_all()
    ts, fs, es = [], [], []
    for lc in coll:
        t = np.asarray(lc.time.value, float)
        f = np.asarray(lc["sap_flux"].value, float)
        e = np.asarray(lc["sap_flux_err"].value, float)
        q = np.asarray(lc["quality"].value)
        m = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (q == 0)
        t, f, e = t[m], f[m], e[m]
        med = np.median(f)
        ts.append(t); fs.append(f / med); es.append(e / med)
        print(f"  sector chunk: {len(t)} pts, baseline {t.max()-t.min():.1f} d", flush=True)
    t = np.concatenate(ts); f = np.concatenate(fs); e = np.concatenate(es)
    o = np.argsort(t); t, f, e = t[o], f[o], e[o]
    t = t - t.min()

    # sigma-clip gross outliers on the harmonic residual (cosmic rays etc.)
    M = design(t, P_GAIA, N_HARM)
    cf, *_ = np.linalg.lstsq(M, f, rcond=None)
    r = f - M @ cf
    keep = np.abs(r) < 5 * np.std(r)
    t, f, e = t[keep], f[keep], e[keep]
    print(f"after 5-sigma clip: {len(t)} points, baseline {t.max():.1f} d", flush=True)

    # pin period: coarse grid then golden refine, maximise 10-harmonic varExp
    grid = np.linspace(P_GAIA * 0.997, P_GAIA * 1.003, 6001)
    P = grid[np.argmax([ve_at(t, f, g) for g in grid])]
    for half in (1e-4, 1e-5, 1e-6):
        fine = np.linspace(P - half, P + half, 201)
        P = fine[np.argmax([ve_at(t, f, g) for g in fine])]
    ve = ve_at(t, f, P)

    # amplitude metrics at the pinned period
    M = design(t, P, N_HARM)
    cf, *_ = np.linalg.lstsq(M, f, rcond=None)
    resid = f - M @ cf
    A = np.array([np.hypot(cf[2 * n - 1], cf[2 * n]) for n in range(1, N_HARM + 1)])
    hi = f > np.median(f)
    bratio = resid[hi].std() / resid[~hi].std()
    print(f"\nSX Car: P={P:.6f} d (Gaia {P_GAIA})  varExp={ve:.4f}  "
          f"resid_rms={resid.std():.4f}  brRatio={bratio:.2f}", flush=True)
    print(f"  A1={A[0]:.4f}  R21={A[1]/A[0]:.3f} (Gaia {R21_GAIA})  "
          f"R31={A[2]/A[0]:.3f} (Gaia {R31_GAIA})  SNR={A[0]/resid.std():.1f}", flush=True)
    ph = (t / P) % 1.0
    print(f"  phase coverage: {np.histogram(ph, 10)[0].min()} pts in sparsest "
          f"of 10 phase bins (uniform~{len(t)//10})", flush=True)

    np.savez(OUT, time=t, flux=f, flux_err=e, period=P, period_gaia=P_GAIA,
             r21_gaia=R21_GAIA, r31_gaia=R31_GAIA, star="SX Car",
             gaia_dr3="5350977643913585280")
    print(f"\nsaved {os.path.abspath(OUT)}  ({len(t)} points)", flush=True)


if __name__ == "__main__":
    main()
