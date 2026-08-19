"""Validate that Rao-Blackwellised inclusion estimator removes the SBC bias.

Runs M=200 SBC trials at n_live=800 on the standard sbc_toy spike-and-slab
regression. For each trial, computes BOTH the empirical-frequency
inclusion probability (which is biased) AND the Rao-Blackwellised
inclusion probability (which should be unbiased), using the analytical
conditional logit for the Gaussian-conjugate orthonormal-design problem.

Outputs scripts/sbc_rao_blackwell_validate_M200.npz with arrays:
    inc_probs_empirical (M, g) -- chain's <P(gamma_k=1|y)>
    inc_probs_rb        (M, g) -- Rao-Blackwellised <P(gamma_k=1|y)>
    gamma_truths        (M, g) -- empirical truth rate per trial

Console summary: bias of each estimator against the empirical truth rate.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import daedalus  # noqa: E402
from daedalus.benchmarks import sbc_toy  # noqa: E402


M = 200
N_LIVE = 800
N_MCMC = 25
TRANSDIM_FRACTION = 0.5
DLOGZ = 0.5


def make_conditional_logits_fn(problem, y):
    """Build the per-trial conditional logit callable for the SBC toy.

    Under orthonormal design (X^T X = N I) and Gaussian slab beta_k ~
    N(0, tau^2), the conditional posterior log-odds for gamma_k is

        logit_k = log(p_k / (1 - p_k))
                  - 0.5 * log(1 + N tau^2 / sigma^2)
                  + 0.5 * (N z_k)^2 * tau^2 / (sigma^2 * (sigma^2 + N tau^2))

    where z_k = X_k^T y / N is the per-coordinate projection. The
    orthonormal design makes the conditional independent of beta_{-k} and
    gamma_{-k}, so it factorises across groups.
    """
    N = problem.N
    sigma2 = problem.sigma2
    tau2 = problem.tau2
    p = problem.inclusion_prior
    z = (problem.X.T @ y) / N
    log_prior_odds = np.log(p / (1.0 - p))
    log_marg_ratio = -0.5 * np.log(1.0 + N * tau2 / sigma2)
    quad_coef = 0.5 * (N * N) * tau2 / (sigma2 * (sigma2 + N * tau2))
    base_logits = log_prior_odds + log_marg_ratio + quad_coef * (z * z)

    def fn(beta, gamma):
        return base_logits

    return fn


def run_one_trial(problem, trial_seed):
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
    return gamma_true, y, results


def main() -> None:
    problem = sbc_toy.make_sbc_problem(
        ndim=2, N=12, sigma2=1.0, tau2=4.0, inclusion_prior=0.5
    )
    g = problem.n_groups
    inc_empirical = np.empty((M, g), dtype=np.float64)
    inc_rb = np.empty((M, g), dtype=np.float64)
    gamma_truths = np.empty((M, g), dtype=bool)

    t0 = time.time()
    for t in range(M):
        gamma_true, y, results = run_one_trial(problem, trial_seed=10_000 + t)
        gamma_truths[t] = gamma_true
        # Empirical-frequency estimator (current default).
        emp = results.inclusion_probabilities()
        inc_empirical[t] = [emp[f"x{k}"] for k in range(g)]
        # Rao-Blackwellised estimator.
        cond_fn = make_conditional_logits_fn(problem, y)
        rb = results.rao_blackwell_inclusion(cond_fn)
        inc_rb[t] = [rb[f"x{k}"] for k in range(g)]
        if (t + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{t + 1:3d}/{M}] elapsed={elapsed:6.1f}s "
                f"rate={(t + 1) / elapsed:.2f}/s",
                flush=True,
            )

    print()
    print(f"=== SBC validation: M={M} trials, n_live={N_LIVE} ===")
    print(f"{'coord':>6} {'emp_truth':>10} {'<P_emp>':>10} {'bias_emp':>10} "
          f"{'<P_rb>':>10} {'bias_rb':>10}")
    for k in range(g):
        truth = float(gamma_truths[:, k].mean())
        p_emp = float(inc_empirical[:, k].mean())
        p_rb = float(inc_rb[:, k].mean())
        print(f"  x{k:>4} {truth:>10.4f} {p_emp:>10.4f} "
              f"{p_emp - truth:>+10.4f} {p_rb:>10.4f} {p_rb - truth:>+10.4f}")

    out_path = os.path.join(HERE, f"sbc_rao_blackwell_validate_M{M}.npz")
    np.savez(
        out_path,
        inc_probs_empirical=inc_empirical,
        inc_probs_rb=inc_rb,
        gamma_truths=gamma_truths,
        n_live=np.array([N_LIVE], dtype=np.int64),
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
