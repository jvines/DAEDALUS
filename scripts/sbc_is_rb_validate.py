"""Validate the importance-sampled Rao-Blackwell estimator on the SBC toy.

The SBC toy is Gaussian-conjugate so the exact RB conditional logit is
closed-form (scripts/sbc_rao_blackwell_validate.py). This script
re-validates the IS-RB estimator (daedalus.rao_blackwell.make_is_logit_fn)
on the
same problem: IS-RB should match exact-RB within Monte-Carlo noise.

We use a Gaussian random-walk birth (the simplest valid birth for the
unidim spike-and-slab toy; the birth's q(beta_k) cancels with itself
in the IS ratio so we just need a defensible proposal density).

Outputs scripts/sbc_is_rb_validate_M{M}.npz with arrays:
    inc_empirical (M, g)
    inc_exact_rb  (M, g)
    inc_is_rb     (M, g)
    gamma_truths  (M, g)
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
from daedalus.rao_blackwell import make_is_logit_fn  # noqa: E402
from daedalus.benchmarks import sbc_toy  # noqa: E402


M = 100  # smaller than the exact-RB validator since IS adds wall-time
N_LIVE = 800
N_MCMC = 25
TRANSDIM_FRACTION = 0.5
DLOGZ = 0.5
N_IS_SAMPLES = 2048  # large because proposal=prior is wide vs concentrated L


def make_exact_logits_fn(problem, y):
    """Closed-form Gaussian-conjugate conditional logit for SBC toy."""
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
    """Run a single SBC trial with a Gaussian-RW birth at the prior
    slab width. This is the worst case for IS-RB (proposal = prior; on
    a concentrated likelihood the IS estimator has high variance) and
    is the right honest validation: if IS-RB converges to the exact-RB
    answer at large N_IS under a prior-as-proposal, the method is
    correct. On real applications (HD 10180) the GLS-informed birth
    is much better targeted and the same N_IS suffices."""

    rng = np.random.default_rng(trial_seed)
    gamma_true, beta_true = problem.sample_truth(rng)
    y = problem.sample_data(beta_true, rng)
    chain = problem.make_chain_problem(y)
    sd_slab = float(np.sqrt(problem.tau2))
    log_norm = -0.5 * np.log(2.0 * np.pi * problem.tau2)

    def make_lp():
        def _lp(b):
            x = float(b[0])
            return log_norm - 0.5 * (x * x) / problem.tau2
        return _lp

    groups = []
    for kw in chain.groups_kwargs:
        kwargs = dict(kw)
        kwargs["birth_proposal"] = daedalus.GaussianRWBirth(scale=sd_slab)
        kwargs["log_prior_continuous"] = make_lp()
        groups.append(daedalus.Group(**kwargs))

    sampler = daedalus.NestedSampler(
        loglike=chain.loglike,
        prior_transform=chain.prior_transform,
        ndim=chain.ndim,
        groups=groups,
        bound="single",
        sample="rwalk",
        n_live=N_LIVE,
        seed=trial_seed,
        validate_births=False,
    )
    results = sampler.run_nested(
        dlogz=DLOGZ,
        n_mcmc=N_MCMC,
        transdim_fraction=TRANSDIM_FRACTION,
        show_progress=False,
    )
    return gamma_true, y, groups, results, chain.loglike


def main() -> None:
    problem = sbc_toy.make_sbc_problem(
        ndim=2, N=12, sigma2=1.0, tau2=4.0, inclusion_prior=0.5
    )
    g = problem.n_groups
    inc_emp = np.empty((M, g))
    inc_exact = np.empty((M, g))
    inc_is = np.empty((M, g))
    gamma_truths = np.empty((M, g), dtype=bool)

    t0 = time.time()
    for t in range(M):
        gamma_true, y, groups, results, loglike = run_one_trial(
            problem, trial_seed=10_000 + t
        )
        gamma_truths[t] = gamma_true
        emp = results.inclusion_probabilities()
        inc_emp[t] = [emp[f"x{k}"] for k in range(g)]
        exact_fn = make_exact_logits_fn(problem, y)
        ex = results.rao_blackwell_inclusion(exact_fn)
        inc_exact[t] = [ex[f"x{k}"] for k in range(g)]
        is_fn = make_is_logit_fn(
            loglike=loglike,
            groups=groups,
            n_samples=N_IS_SAMPLES,
            seed=trial_seed_for_is(t),
        )
        is_inc = results.rao_blackwell_inclusion(is_fn)
        inc_is[t] = [is_inc[f"x{k}"] for k in range(g)]
        if (t + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{t + 1:3d}/{M}] elapsed={elapsed:6.1f}s "
                f"rate={(t + 1) / elapsed:.2f}/s",
                flush=True,
            )

    print()
    print(f"=== IS-RB validation: M={M} trials, n_live={N_LIVE}, "
          f"N_IS={N_IS_SAMPLES} ===")
    print(f"{'coord':>6} {'truth':>10} {'<P_emp>':>10} {'<P_exact>':>11} "
          f"{'<P_IS>':>10} {'b_emp':>9} {'b_exact':>9} {'b_IS':>9}")
    for k in range(g):
        truth = float(gamma_truths[:, k].mean())
        p_e = float(inc_emp[:, k].mean())
        p_x = float(inc_exact[:, k].mean())
        p_i = float(inc_is[:, k].mean())
        print(
            f"  x{k:>4} {truth:>10.4f} {p_e:>10.4f} {p_x:>11.4f} "
            f"{p_i:>10.4f} {p_e - truth:>+9.4f} {p_x - truth:>+9.4f} "
            f"{p_i - truth:>+9.4f}"
        )

    out_path = os.path.join(HERE, f"sbc_is_rb_validate_M{M}.npz")
    np.savez(
        out_path,
        inc_empirical=inc_emp,
        inc_exact_rb=inc_exact,
        inc_is_rb=inc_is,
        gamma_truths=gamma_truths,
        n_live=np.array([N_LIVE], dtype=np.int64),
        n_is=np.array([N_IS_SAMPLES], dtype=np.int64),
    )
    print(f"\nSaved: {out_path}")


def trial_seed_for_is(trial: int) -> int:
    """Distinct IS seed per trial to randomise the importance draws."""
    return 999_000 + trial


if __name__ == "__main__":
    main()
