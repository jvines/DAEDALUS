"""HD 10180 discrimination control: 7 real Lovis candidates + 1 FAKE slot.

The decisive rebuttal to the circularity critique. If the tight-window labelled
confirmation merely rubber-stamps whatever it is handed, then an 8th slot whose
period window sits on NO real signal would also fire. If instead the method
discriminates, the fake slot's inclusion P(gamma_fake=1|y) collapses toward the
prior-driven floor while the 7 real planets stay ~1.

Rigor:
1. Pre-whiten the 7 Lovis planets (linear least-squares fit of cos/sin amplitudes
   at the FIXED literature periods + offset) -> residuals.
2. GLS periodogram of the residuals; place several FAKE periods at genuinely
   low-power points, away from any Lovis period and its low-order harmonics.
3. For each fake position: 8-slot DE problem (7 Lovis tight windows + the fake
   tight window), multi-seed, report all 8 inclusions + the residual-periodogram
   power at the fake period (to document it is a low-power region).
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
from daedalus.groups import Group  # noqa: E402
from daedalus.samplers import DifferentialEvolutionSampler  # noqa: E402
import hd10180_lovis_labelled as lab  # noqa: E402
from insertion_index import insertion_index_test  # noqa: E402

CACHE = os.path.join(HERE, "hd10180_harps_cache.npz")
N_LIVE = 600
SEEDS = [42, 43, 44]
WIN = lab.P_WINDOW_FRAC  # +/-5%
LOVIS_P = [lab.LOVIS_PERIODS[n] for n in lab.SLOT_NAMES]   # 7 periods


def get_data():
    d = _load_harps_npz(); return d["t"], d["rv"], d["err"]


def prewhiten_residuals(t, rv, err):
    """LS fit offset + (cos,sin) per Lovis planet at fixed periods. Return resid."""
    cols = [np.ones_like(t)]
    for P in LOVIS_P:
        ph = 2 * np.pi * t / P
        cols += [np.cos(ph), np.sin(ph)]
    A = np.vstack(cols).T
    w = 1.0 / err
    coef, *_ = np.linalg.lstsq(A * w[:, None], rv * w, rcond=None)
    return rv - A @ coef


def gls_power(t, y, err, periods):
    """Floating-mean GLS power at each period (Zechmeister & Kurster 2009, simplified)."""
    w = 1.0 / err**2
    w /= w.sum()
    yy = y - np.sum(w * y)
    YY = np.sum(w * yy**2)
    out = np.empty_like(periods)
    for i, P in enumerate(periods):
        ph = 2 * np.pi * t / P
        c = np.cos(ph); s = np.sin(ph)
        c -= np.sum(w * c); s -= np.sum(w * s)
        YC = np.sum(w * yy * c); YS = np.sum(w * yy * s)
        CC = np.sum(w * c * c); SS = np.sum(w * s * s); CS = np.sum(w * c * s)
        D = CC * SS - CS**2
        if D <= 0:
            out[i] = 0.0; continue
        out[i] = (SS * YC**2 + CC * YS**2 - 2 * CS * YC * YS) / (D * YY)
    return out


def is_clean(P):
    """True if P is >5% from every Lovis period and its 0.5/2/3x harmonics."""
    for Pl in LOVIS_P:
        for h in (1.0, 0.5, 2.0, 1.0 / 3.0, 3.0):
            if abs(np.log(P) - np.log(Pl * h)) < np.log(1.0 + 2 * WIN):
                return False
    return True


def pick_fakes(t, resid, err, n_fakes=4):
    grid = np.exp(np.linspace(np.log(2.0), np.log(2500.0), 4000))
    power = gls_power(t, resid, err, grid)
    clean = np.array([is_clean(P) for P in grid])
    # candidate low-power clean periods in 4 broad regions
    regions = [(2, 12), (20, 100), (150, 900), (900, 2200)]
    fakes = []
    for lo, hi in regions:
        m = clean & (grid >= lo) & (grid <= hi)
        if not m.any():
            continue
        idx = np.where(m)[0]
        j = idx[np.argmin(power[idx])]
        fakes.append((float(grid[j]), float(power[j])))
    return fakes[:n_fakes], (grid, power)


def build_8slot(t, rv, err, fake_P):
    """7 Lovis tight windows + 1 fake tight window. Returns NS pieces."""
    windows = [(P * (1 - WIN), P * (1 + WIN)) for P in LOVIS_P] + \
              [(fake_P * (1 - WIN), fake_P * (1 + WIN))]
    logw = [(float(np.log(lo)), float(np.log(hi))) for lo, hi in windows]
    NS = len(logw)  # 8
    GOFF = 3 * NS; JIT = 3 * NS + 1; NDIM = 3 * NS + 2
    var = err * err; mbuf = np.empty_like(rv)
    LK_LO, LK_HI = lab.LOG_K_LO, lab.LOG_K_HI

    def loglike(beta, gamma):
        mbuf[:] = float(beta[GOFF])
        for k in range(NS):
            if not gamma[k]:
                continue
            P = float(np.exp(beta[3 * k])); t0 = float(beta[3 * k + 1])
            K = float(np.exp(beta[3 * k + 2]))
            mbuf[:] += K * np.cos(2.0 * np.pi * (t - t0 * P) / P)
        r = rv - mbuf
        s2 = var + float(np.exp(2.0 * float(beta[JIT])))
        return float(-0.5 * np.sum(r * r / s2 + np.log(s2)))

    def prior_transform(u):
        v = np.empty_like(u)
        for k, (lo, hi) in enumerate(logw):
            v[3 * k] = lo + (hi - lo) * u[3 * k]
            v[3 * k + 1] = u[3 * k + 1]
            v[3 * k + 2] = LK_LO + (LK_HI - LK_LO) * u[3 * k + 2]
        v[GOFF] = lab.GAMMA_LO + (lab.GAMMA_HI - lab.GAMMA_LO) * u[GOFF]
        v[JIT] = lab.LOG_JIT_LO + (lab.LOG_JIT_HI - lab.LOG_JIT_LO) * u[JIT]
        return v

    groups = []
    for k, (lo, hi) in enumerate(logw):
        groups.append(Group(
            name=f"slot_{k}", params=[3 * k, 3 * k + 1, 3 * k + 2],
            off_values=np.array([0.5 * (lo + hi), 0.0, 0.5 * (LK_LO + LK_HI)]),
            inclusion_prior=0.5))
    periodic = [3 * k + 1 for k in range(NS)]
    return loglike, prior_transform, groups, periodic, NDIM, NS


def run_chain(spec):
    fi, fake_P, seed, t, rv, err = spec
    try:
        ll, pt, groups, per, ndim, NS = build_8slot(t, rv, err, fake_P)
        sampler = daedalus.NestedSampler(
            loglike=ll, prior_transform=pt, ndim=ndim, groups=groups, bound="multi",
            sample=DifferentialEvolutionSampler(target_accept=0.234), n_live=N_LIVE,
            seed=seed, periodic=per, validate_births=False)
        rec: list[int] = []
        t0 = time.time()
        res = sampler.run_nested(dlogz=0.3, n_mcmc=40, transdim_fraction=0.35,
                                 show_progress=False, insertion_recorder=rec)
        el = time.time() - t0
        idx = np.asarray(rec, float)
        it = insertion_index_test(idx, N_LIVE) if idx.size > 50 else None
        inc = res.gamma.mean(axis=0)  # 8 inclusions; slot 7 = fake
        return {"fi": fi, "fake_P": fake_P, "seed": seed, "ok": True,
                "logZ": float(res.log_Z), "maxL": float(np.max(res.log_likelihoods)),
                "inc": [float(x) for x in inc], "elapsed": el,
                "ks_p": float(it.ks_pvalue) if it else float("nan")}
    except Exception as e:
        import traceback
        return {"fi": fi, "fake_P": fake_P, "seed": seed, "ok": False,
                "err": repr(e)[:150], "tb": traceback.format_exc()[-400:]}


def main():
    t, rv, err = get_data()
    resid = prewhiten_residuals(t, rv, err)
    print(f"[fake] residual rms after 7-planet prewhiten: {resid.std():.2f} m/s "
          f"(raw {rv.std():.2f})", flush=True)
    fakes, _ = pick_fakes(t, resid, err)
    # Scale reference: strongest RAW-RV periodogram power at the Lovis periods
    # (the real planet signal strength), NOT the residual power (which is ~0
    # there by construction after prewhitening).
    real_pow = max(float(gls_power(t, rv, err, np.array([P]))[0]) for P in LOVIS_P)
    print(f"[fake] strongest RAW periodogram power among the 7 Lovis periods = {real_pow:.4f}")
    print("[fake] chosen fake periods (low residual power, harmonic-clean):")
    for i, (P, pw) in enumerate(fakes):
        print(f"   fake{i}: P={P:.2f} d  resid_power={pw:.4f}  ({pw/real_pow*100:.1f}% of real-peak scale)")
    print(flush=True)

    specs = [(i, P, s, t, rv, err) for i, (P, _) in enumerate(fakes) for s in SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=9) as ex:
        futs = [ex.submit(run_chain, s) for s in specs]
        for k, f in enumerate(as_completed(futs)):
            r = f.result(); rows.append(r)
            if r.get("ok"):
                print(f"  [{k+1:2d}/{len(specs)}] fake{r['fi']} P={r['fake_P']:.1f} s{r['seed']} "
                      f"logZ={r['logZ']:.2f} maxL={r['maxL']:.1f} "
                      f"FAKE_incl={r['inc'][7]:.3f}  reals_min={min(r['inc'][:7]):.3f} "
                      f"{r['elapsed']:.0f}s", flush=True)
            else:
                print(f"  [{k+1:2d}] fake{r['fi']} s{r['seed']} FAIL {r['err']}\n{r.get('tb','')}", flush=True)

    print("\n" + "=" * 100)
    nm = list(lab.SLOT_NAMES) + ["FAKE"]
    print("Per-fake-position inclusion (mean over seeds): b c d e f g h | FAKE")
    summ = []
    for i, (P, pw) in enumerate(fakes):
        cr = [r for r in rows if r.get("ok") and r["fi"] == i]
        if not cr:
            print(f" fake{i} P={P:.1f}: all failed"); continue
        inc = np.array([r["inc"] for r in cr]).mean(axis=0)
        fake_seeds = [r["inc"][7] for r in cr]
        summ.append({"fake_P": P, "resid_power": pw, "fake_incl_mean": float(inc[7]),
                     "fake_incl_per_seed": fake_seeds,
                     "reals_min_mean": float(inc[:7].min()), "n_seed": len(cr)})
        reals = " ".join(f"{inc[k]:.2f}" for k in range(7))
        print(f" fake{i} P={P:7.1f}d (pw {pw:.3f}): reals[{reals}] | FAKE={inc[7]:.3f} "
              f"(per-seed {','.join(f'{v:.2f}' for v in fake_seeds)})")
    print("=" * 100)
    print("DISCRIMINATION PASS = FAKE inclusion << real inclusions (ideally near the "
          "0.5-prior-driven floor or below), reals stay ~1.")
    json.dump({"fakes": [{"P": P, "resid_power": pw} for P, pw in fakes],
               "real_peak_power": real_pow, "summary": summ,
               "rows": [{k: v for k, v in r.items() if k != 'tb'} for r in rows]},
              open(os.path.join(HERE, "hd10180_fake_planet_results.json"), "w"), indent=1)


def _load_harps_npz():
    """HARPS RVs as a dict with keys t/rv/err; fetches and caches on first use."""
    t, rv, err = lab.load_harps()
    return {"t": t, "rv": rv, "err": err}


if __name__ == "__main__":
    main()
