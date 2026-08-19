"""SBC bias diagnostic: scan n_live to localize the source of the
inclusion-marginal 5-7% over-coverage observed in scripts/sbc_paper.py.

The hypothesis is that the bias is binomial-limited at the live-point
count rather than an algorithmic artefact: if so, raising n_live should
reduce it. If it persists, the bias is structural and needs a different
explanation. We run M=400 trials at a single n_live setting (configurable)
and emit the standard SBC summary.

Outputs scripts/sbc_diagnostic_nlive{N}_M{M}.npz with the same fields
as sbc_paper.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import daedalus  # noqa: E402
from daedalus.benchmarks import sbc_toy  # noqa: E402


N_MCMC = 25
TRANSDIM_FRACTION = 0.5
DLOGZ = 0.5
MIN_ACTIVE_SAMPLES = 30
PROGRESS_EVERY = 25


def run_one_trial(problem, trial_seed, n_live):
    rng = np.random.default_rng(trial_seed)
    gamma_true, beta_true = problem.sample_truth(rng)
    y = problem.sample_data(beta_true, rng)
    chain = problem.make_chain_problem(y)
    groups = [daedalus.Group(**kw) for kw in chain.groups_kwargs]
    sampler = daedalus.NestedSampler(
        loglike=chain.loglike,
        prior_transform=chain.prior_transform,
        ndim=chain.ndim,
        groups=groups,
        bound="single",
        sample="rwalk",
        n_live=n_live,
        seed=trial_seed,
    )
    results = sampler.run_nested(
        dlogz=DLOGZ,
        n_mcmc=N_MCMC,
        transdim_fraction=TRANSDIM_FRACTION,
        show_progress=False,
    )
    return gamma_true, beta_true, results


def run_sbc(problem, M, n_live):
    g = problem.n_groups
    ranks = [[] for _ in range(g)]
    n_filtered = [0 for _ in range(g)]
    inc_probs = np.empty((M, g), dtype=np.float64)
    gamma_truths = np.empty((M, g), dtype=bool)
    t0 = time.time()
    for t in range(M):
        gamma_true, beta_true, results = run_one_trial(
            problem, trial_seed=10_000 + t, n_live=n_live
        )
        gamma_truths[t] = gamma_true
        inc_probs[t] = results.gamma.mean(axis=0)
        for k in range(g):
            if not gamma_true[k]:
                continue
            active_mask = results.gamma[:, k]
            n_active = int(active_mask.sum())
            if n_active < MIN_ACTIVE_SAMPLES:
                n_filtered[k] += 1
                continue
            beta_active = results.samples[active_mask, k]
            r = float(np.sum(beta_active < beta_true[k])) / float(n_active)
            ranks[k].append(r)
        if (t + 1) % PROGRESS_EVERY == 0 or (t + 1) == M:
            elapsed = time.time() - t0
            rate = (t + 1) / elapsed
            eta = (M - t - 1) / rate if rate > 0 else float("nan")
            print(
                f"  [{t + 1:4d}/{M}]  n_live={n_live}  elapsed={elapsed:6.1f}s  "
                f"rate={rate:5.2f} trial/s  ETA={eta:6.1f}s",
                flush=True,
            )
    return {
        "ranks": [np.asarray(r) for r in ranks],
        "n_filtered": np.asarray(n_filtered, dtype=np.int64),
        "inc_probs": inc_probs,
        "gamma_truths": gamma_truths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=400)
    parser.add_argument("--n-live", type=int, default=800)
    args = parser.parse_args()

    problem = sbc_toy.make_sbc_problem(
        ndim=2, N=12, sigma2=1.0, tau2=4.0, inclusion_prior=0.5
    )
    print(f"[sbc diagnostic] M={args.M}  n_live={args.n_live}  "
          f"n_mcmc={N_MCMC}  transdim_fraction={TRANSDIM_FRACTION}", flush=True)

    out = run_sbc(problem, M=args.M, n_live=args.n_live)

    target = problem.inclusion_prior
    g = problem.n_groups
    print("\n--- continuous-beta rank uniformity ---", flush=True)
    for k in range(g):
        r = out["ranks"][k]
        ks_stat, ks_p = stats.kstest(r, "uniform")
        print(f"  coord x{k}: n_eff={r.size}  KS p={ks_p:.3f}", flush=True)
    print(f"\n--- inclusion-marginal calibration (target = {target}) ---", flush=True)
    for k in range(g):
        col = out["inc_probs"][:, k]
        mean_p = float(col.mean())
        se_M = float(col.std(ddof=1)) / float(np.sqrt(args.M))
        z = (mean_p - target) / se_M if se_M > 0 else 0.0
        emp = float(out["gamma_truths"][:, k].mean())
        print(f"  coord x{k}: <P(gamma=1|y)> = {mean_p:.4f} +- {se_M:.4f}  "
              f"(z={z:+.2f}  empirical truth rate={emp:.3f})", flush=True)

    out_path = os.path.join(
        HERE, f"sbc_diagnostic_nlive{args.n_live}_M{args.M}.npz"
    )
    save_kwargs = {f"ranks_x{k}": out["ranks"][k] for k in range(g)}
    save_kwargs["inc_probs"] = out["inc_probs"]
    save_kwargs["gamma_truths"] = out["gamma_truths"]
    save_kwargs["n_filtered"] = out["n_filtered"]
    save_kwargs["n_live"] = np.array([args.n_live], dtype=np.int64)
    np.savez(out_path, **save_kwargs)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
