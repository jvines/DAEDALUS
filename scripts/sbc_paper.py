"""Paper-grade Simulation-Based Calibration (SBC) run for daedalus.

Mirrors the test suite SBC procedure (tests/test_sbc.py) but runs at a
larger trial count M (default 400) to drive down per-coordinate
diagnostic noise to a level appropriate for a paper appendix.

Per-trial procedure (Talts et al. 2020, arXiv:1804.06788):

    1. Draw (gamma_true, beta_true) ~ prior.
    2. Draw y ~ p(y | beta_true).
    3. Run daedalus to get N posterior draws of (beta, gamma).
    4. Per coordinate i: when gamma_true_i = 1 and the chain has at
       least MIN_ACTIVE_SAMPLES samples with gamma_i = 1, record the
       fractional rank of beta_true_i within those active beta_i samples.
       Also record the chain's marginal P(gamma_i = 1 | y).

Reproducibility: trial seeds are 10000 + t (matches tests/test_sbc.py).

Outputs:
    scripts/sbc_paper_M{M}.npz with arrays
        ranks_x0, ranks_x1, inc_probs (M, g), gamma_truths (M, g),
        n_filtered (g,)
    Console summary: per-coord KS p, chi2 p (10 bins), inclusion mean
    +- SE_M, z from the prior.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy import stats

# Make sure we can import the in-tree daedalus package without installation.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import daedalus  # noqa: E402
from daedalus.benchmarks import sbc_toy  # noqa: E402


N_LIVE = 200
N_MCMC = 25
TRANSDIM_FRACTION = 0.5
DLOGZ = 0.5
MIN_ACTIVE_SAMPLES = 30
PROGRESS_EVERY = 25


def run_one_trial(
    problem: sbc_toy.SBCProblem, trial_seed: int
) -> tuple[np.ndarray, np.ndarray, "daedalus.Results"]:
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
        n_live=N_LIVE,
        seed=trial_seed,
    )
    results = sampler.run_nested(
        dlogz=DLOGZ,
        n_mcmc=N_MCMC,
        transdim_fraction=TRANSDIM_FRACTION,
        show_progress=False,
    )
    return gamma_true, beta_true, results


def run_sbc(problem: sbc_toy.SBCProblem, M: int) -> dict:
    g = problem.n_groups
    ranks: list[list[float]] = [[] for _ in range(g)]
    n_filtered: list[int] = [0 for _ in range(g)]
    inc_probs = np.empty((M, g), dtype=np.float64)
    gamma_truths = np.empty((M, g), dtype=bool)
    t0 = time.time()
    for t in range(M):
        gamma_true, beta_true, results = run_one_trial(
            problem, trial_seed=10_000 + t
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
                f"  [{t + 1:4d}/{M}]  elapsed={elapsed:6.1f}s  "
                f"rate={rate:5.2f} trial/s  ETA={eta:6.1f}s",
                flush=True,
            )
    return {
        "ranks": [np.asarray(r) for r in ranks],
        "n_filtered": np.asarray(n_filtered, dtype=np.int64),
        "inc_probs": inc_probs,
        "gamma_truths": gamma_truths,
    }


def summarize(out: dict, problem: sbc_toy.SBCProblem, M: int) -> None:
    target = problem.inclusion_prior
    g = problem.n_groups
    print()
    print(f"=== SBC paper run: M = {M} ===")
    print()
    print("--- continuous-beta rank uniformity (10-bin chi2 + KS) ---")
    for k in range(g):
        r = out["ranks"][k]
        n_eff = r.size
        ks_stat, ks_p = stats.kstest(r, "uniform")
        hist, _ = np.histogram(r, bins=10, range=(0.0, 1.0))
        expected = n_eff / 10.0
        chi2 = float(np.sum((hist - expected) ** 2 / max(expected, 1.0)))
        chi2_p = 1.0 - stats.chi2.cdf(chi2, df=9)
        print(
            f"  coord x{k}: n_eff={n_eff:4d}  filtered={out['n_filtered'][k]:3d}  "
            f"KS stat={ks_stat:.3f} p={ks_p:.3f}  "
            f"chi2={chi2:.2f} p={chi2_p:.3f}"
        )
        print(f"    bins: {hist.tolist()}  expected/bin ~= {expected:.1f}")
    print()
    print(f"--- inclusion-marginal calibration (target = {target}) ---")
    for k in range(g):
        col = out["inc_probs"][:, k]
        mean_p = float(col.mean())
        se_M = float(col.std(ddof=1)) / float(np.sqrt(M))
        z = (mean_p - target) / se_M if se_M > 0 else 0.0
        emp = float(out["gamma_truths"][:, k].mean())
        truth_se = float(np.sqrt(target * (1 - target) / M))
        z_truth = (emp - target) / truth_se if truth_se > 0 else 0.0
        print(
            f"  coord x{k}: <P(gamma=1|y)> = {mean_p:.4f} +- {se_M:.4f}  "
            f"(z = {z:+.2f} from {target})   "
            f"empirical truth rate = {emp:.3f} (z_truth = {z_truth:+.2f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=400, help="number of SBC trials")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output .npz path (default: scripts/sbc_paper_M{M}.npz)",
    )
    args = parser.parse_args()

    problem = sbc_toy.make_sbc_problem(
        ndim=2, N=12, sigma2=1.0, tau2=4.0, inclusion_prior=0.5
    )
    print(
        f"Running SBC: M={args.M}, ndim={problem.ndim}, N={problem.N}, "
        f"sigma2={problem.sigma2}, tau2={problem.tau2}, "
        f"inclusion_prior={problem.inclusion_prior}"
    )
    print(
        f"Sampler: n_live={N_LIVE}, n_mcmc={N_MCMC}, "
        f"transdim_fraction={TRANSDIM_FRACTION}, dlogz={DLOGZ}, "
        f"bound=single, sample=rwalk"
    )
    print()

    out = run_sbc(problem, M=args.M)
    summarize(out, problem, args.M)

    out_path = args.out or os.path.join(HERE, f"sbc_paper_M{args.M}.npz")
    g = problem.n_groups
    save_kwargs = {
        f"ranks_x{k}": out["ranks"][k] for k in range(g)
    }
    save_kwargs["inc_probs"] = out["inc_probs"]
    save_kwargs["gamma_truths"] = out["gamma_truths"]
    save_kwargs["n_filtered"] = out["n_filtered"]
    np.savez(out_path, **save_kwargs)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
