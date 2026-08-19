"""HD 10180 DE campaign: discovery (wide prior + GLS births) AND labelled
(tight windows, no births) in one harness, both with the DE kernel and
insertion-index QC.

--mode discovery : 8 anonymous slots, log-uniform P in [1,3000] d, GLS births
                   (sharpening=5, alpha=0.5). Evaluate the PERMUTATION-INVARIANT
                   period-marginal across seeds (the right deliverable for an
                   exchangeable-slot blind search). Seed-stable recovery of the
                   Lovis periods here would moot the candidate-confirmation
                   circularity entirely.
--mode labelled  : 7 tight +/-5% slots, DE only (GLS N/A on tight windows).
                   Test whether DE + higher n_live removes the mode-sticking and
                   collapses across-seed logZ to the analytic budget.
"""
from __future__ import annotations

import argparse
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
LOVIS = dict(zip(lab.SLOT_NAMES,
                 [lab.LOVIS_PERIODS[n] for n in lab.SLOT_NAMES]))

# Discovery: wide shared prior, 8 anonymous slots.
DISC_NSLOTS = 8
LOG_P_LO, LOG_P_HI = float(np.log(1.0)), float(np.log(3000.0))
LOG_K_LO, LOG_K_HI = float(np.log(0.5)), float(np.log(50.0))
LOG_JIT_LO, LOG_JIT_HI = float(np.log(0.1)), float(np.log(20.0))
GAMMA_LO, GAMMA_HI = -30.0, 30.0


def get_data():
    d = _load_harps_npz()
    return d["t"], d["rv"], d["err"]


def build_discovery(t, rv, err):
    N = DISC_NSLOTS
    GOFF = 3 * N
    JIT = 3 * N + 1
    NDIM = 3 * N + 2
    var = err * err
    mbuf = np.empty_like(rv)
    rbuf = np.empty_like(rv)

    def _model(beta, gamma, out):
        out[:] = float(beta[GOFF])
        for k in range(N):
            if not gamma[k]:
                continue
            P = float(np.exp(beta[3 * k])); t0 = float(beta[3 * k + 1])
            K = float(np.exp(beta[3 * k + 2]))
            out[:] += K * np.cos(2.0 * np.pi * (t - t0 * P) / P)

    def loglike(beta, gamma):
        _model(beta, gamma, mbuf)
        resid = rv - mbuf
        s2 = var + float(np.exp(2.0 * float(beta[JIT])))
        return float(-0.5 * np.sum(resid * resid / s2 + np.log(s2)))

    def prior_transform(u):
        v = np.empty_like(u)
        for k in range(N):
            v[3 * k] = LOG_P_LO + (LOG_P_HI - LOG_P_LO) * u[3 * k]
            v[3 * k + 1] = u[3 * k + 1]
            v[3 * k + 2] = LOG_K_LO + (LOG_K_HI - LOG_K_LO) * u[3 * k + 2]
        v[GOFF] = GAMMA_LO + (GAMMA_HI - GAMMA_LO) * u[GOFF]
        v[JIT] = LOG_JIT_LO + (LOG_JIT_HI - LOG_JIT_LO) * u[JIT]
        return v

    log_pi = -np.log(LOG_P_HI - LOG_P_LO) - np.log(1.0) - np.log(LOG_K_HI - LOG_K_LO)

    def lp(b):
        if (b[0] < LOG_P_LO or b[0] > LOG_P_HI or b[1] < 0 or b[1] > 1
                or b[2] < LOG_K_LO or b[2] > LOG_K_HI):
            return float("-inf")
        return log_pi

    def mk_resid(kk):
        def _r(state):
            g = state.gamma.copy(); g[kk] = False
            _model(state.beta, g, rbuf); return rv - rbuf
        return _r

    def mk_active(kk):
        def _a(state):
            return [float(np.exp(state.beta[3 * j])) for j in range(N)
                    if state.gamma[j] and j != kk]
        return _a

    groups = []
    for k in range(N):
        birth = daedalus.GLSPeriodBirth(
            time=t, residual_fn=mk_resid(k), prior_log_period=(LOG_P_LO, LOG_P_HI),
            non_period_priors=((0.0, 1.0), (LOG_K_LO, LOG_K_HI)),
            nterms=1, period_param_offset=0, active_periods_fn=mk_active(k),
            n_periods=2500, sharpening=5.0, alpha=0.5,
            harmonic_ratios=(1 / 3, 0.5, 1.0, 2.0, 3.0), harmonic_window_log=0.08)
        groups.append(Group(
            name=f"slot_{k}", params=[3 * k, 3 * k + 1, 3 * k + 2],
            off_values=np.array([0.5 * (LOG_P_LO + LOG_P_HI), 0.0,
                                 0.5 * (LOG_K_LO + LOG_K_HI)]),
            inclusion_prior=0.5, birth_proposal=birth, log_prior_continuous=lp))
    periodic = [3 * k + 1 for k in range(N)]
    return loglike, prior_transform, groups, periodic, NDIM


def run_chain(spec):
    mode, seed, n_live, t, rv, err = spec
    try:
        if mode == "discovery":
            ll, pt, groups, per, ndim = build_discovery(t, rv, err)
            nslots = DISC_NSLOTS
        else:
            ll, pt, groups, per = lab.build_problem(t, rv, err)
            ndim = lab.NDIM; nslots = lab.N_SLOTS
        sampler = daedalus.NestedSampler(
            loglike=ll, prior_transform=pt, ndim=ndim, groups=groups,
            bound="multi", sample=DifferentialEvolutionSampler(target_accept=0.234),
            n_live=n_live, seed=seed, periodic=per, validate_births=False)
        rec: list[int] = []
        t0 = time.time()
        res = sampler.run_nested(dlogz=0.3, transdim_fraction=0.35,  # n_mcmc=None -> 5*ndim
                                 show_progress=False, insertion_recorder=rec)
        el = time.time() - t0
        idx = np.asarray(rec, float)
        it = insertion_index_test(idx, n_live) if idx.size > 50 else None
        # period-marginal data: P per slot + gamma (downsampled to cap size)
        g = res.gamma.astype(bool)
        P = np.exp(res.samples[:, [3 * k for k in range(nslots)]])
        out = os.path.join(HERE, f"hd10180_de_{mode}_seed{seed}_n{n_live}.npz")
        np.savez_compressed(out, P=P.astype(np.float32), gamma=g,
                            log_Z=res.log_Z, log_Z_err=res.log_Z_err)
        nplan = int(np.median(g.sum(axis=1)))
        return {"mode": mode, "seed": seed, "ok": True, "logZ": float(res.log_Z),
                "logZ_err": float(res.log_Z_err),
                "maxL": float(np.max(res.log_likelihoods)), "elapsed": el,
                "ks_p": float(it.ks_pvalue) if it else float("nan"),
                "z_mean": float(it.z_mean) if it else float("nan"),
                "n_med": nplan, "npz": out}
    except Exception as e:
        import traceback
        return {"mode": mode, "seed": seed, "ok": False, "err": repr(e)[:150],
                "tb": traceback.format_exc()[-500:]}


def period_marginal(npzs, tol_nat=0.05):
    """Permutation-invariant per-Lovis period-marginal inclusion, per seed."""
    out = {}
    for f in npzs:
        d = np.load(f); P = d["P"].astype(float); g = d["gamma"].astype(bool)
        n = P.shape[0]
        row = {}
        for nm, Plit in LOVIS.items():
            lp = np.log(Plit)
            has = np.zeros(n, bool)
            for k in range(P.shape[1]):
                has |= g[:, k] & (np.abs(np.log(P[:, k]) - lp) < tol_nat)
            row[nm] = float(has.mean())
        out[f] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["discovery", "labelled", "both"], default="both")
    ap.add_argument("--seeds", default="42,43,44,45,46,47")
    ap.add_argument("--n-live-disc", type=int, default=500)
    ap.add_argument("--n-live-lab", type=int, default=600)
    ap.add_argument("--serial", action="store_true",
                    help="run chains in-process (avoids ProcessPoolExecutor "
                         "deadlock when launched detached/nohup)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    t, rv, err = get_data()
    specs = []
    if args.mode in ("discovery", "both"):
        specs += [("discovery", s, args.n_live_disc, t, rv, err) for s in seeds]
    if args.mode in ("labelled", "both"):
        specs += [("labelled", s, args.n_live_lab, t, rv, err) for s in seeds]
    print(f"[campaign] {len(specs)} chains: {args.mode}; seeds={seeds}; "
          f"n_live disc={args.n_live_disc} lab={args.n_live_lab}", flush=True)
    rows = []

    def _report(i, r):
        if r.get("ok"):
            print(f"  [{i+1:2d}/{len(specs)}] {r['mode']:9s} s{r['seed']} "
                  f"logZ={r['logZ']:.2f} maxL={r['maxL']:.1f} ks_p={r['ks_p']:.1e} "
                  f"z={r['z_mean']:+.1f} N~{r['n_med']} {r['elapsed']:.0f}s", flush=True)
        else:
            print(f"  [{i+1:2d}/{len(specs)}] {r['mode']} s{r['seed']} FAIL {r['err']}\n{r.get('tb','')}", flush=True)

    if args.serial:
        for i, s in enumerate(specs):
            r = run_chain(s); rows.append(r); _report(i, r)
    else:
        with ProcessPoolExecutor(max_workers=12) as ex:
            futs = [ex.submit(run_chain, s) for s in specs]
            for i, f in enumerate(as_completed(futs)):
                r = f.result(); rows.append(r); _report(i, r)

    print("\n" + "=" * 100)
    for mode in (["discovery", "labelled"] if args.mode == "both" else [args.mode]):
        cr = [r for r in rows if r.get("ok") and r["mode"] == mode]
        if not cr:
            print(f"{mode}: all failed"); continue
        lz = np.array([r["logZ"] for r in cr]); er = np.array([r["logZ_err"] for r in cr])
        std = float(lz.std(ddof=1)) if len(lz) > 1 else 0.0
        gmax = max(r["maxL"] for r in cr)
        reach = np.mean([r["maxL"] >= gmax - 3 for r in cr])
        insp = np.mean([(r["ks_p"] > 0.05) and (abs(r["z_mean"]) < 3) for r in cr])
        print(f"{mode}: meanLogZ={lz.mean():.2f} stdLogZ={std:.2f} anaErr={er.mean():.3f} "
              f"ratio={std/er.mean():.1f} ins_pass={insp:.2f} reachPk={reach:.2f} "
              f"N_med={[r['n_med'] for r in sorted(cr,key=lambda x:x['seed'])]}")
        if mode == "discovery":
            pm = period_marginal([r["npz"] for r in cr])
            print("  PERMUTATION-INVARIANT period-marginal inclusion per Lovis candidate, per seed:")
            print("   cand  " + "  ".join(f"s{r['seed']}" for r in sorted(cr, key=lambda x: x['seed'])))
            for nm in lab.SLOT_NAMES:
                vals = [pm[r["npz"]][nm] for r in sorted(cr, key=lambda x: x['seed'])]
                print(f"   {nm:>4}  " + "  ".join(f"{v:.2f}" for v in vals)
                      + f"   (mean {np.mean(vals):.2f}, std {np.std(vals):.2f})")
    print("=" * 100)
    json.dump({"rows": [{k: v for k, v in r.items() if k != 'tb'} for r in rows]},
              open(os.path.join(HERE, f"hd10180_de_campaign_{args.mode}.json"), "w"), indent=1)


def _load_harps_npz():
    """HARPS RVs as a dict with keys t/rv/err; fetches and caches on first use."""
    t, rv, err = lab.load_harps()
    return {"t": t, "rv": rv, "err": err}


if __name__ == "__main__":
    main()
