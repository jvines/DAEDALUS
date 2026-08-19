"""Simulation-based calibration (SBC; Talts, Betancourt, Simpson, Vehtari 2020,
arXiv:1804.06788) of daedalus on a tiny synthetic spike-and-slab regression.

SBC is the gold-standard correctness test for Bayesian inference software.
Procedure, repeated for each of M trials:

    1. Draw (gamma_true, beta_true) ~ prior.
    2. Draw y ~ p(y | beta_true).
    3. Run daedalus to get N independent posterior draws of (beta, gamma).
    4. For each continuous beta_i, when gamma_true_i = 1, compute the
       fractional rank of beta_true_i within the posterior samples that
       have gamma_i = 1.

If the inference is calibrated, the M fractional ranks per coordinate are
uniformly distributed on [0, 1]. We check this with a Kolmogorov-Smirnov
test against U(0, 1).

We also report a binary calibration check on the marginal inclusion
posteriors: under simulation from the prior, the average of P(gamma_i = 1 | y)
across trials must equal the prior P(gamma_i = 1) = inclusion_prior. A
chain that systematically over- or under-includes a predictor would
violate this and the test would catch it.

This test is the only one in the suite that can detect *systematic*
posterior miscalibration -- the kind of bug where every individual run
looks plausible but the chain is consistently wrong in the same
direction. The diabetes-full benchmark (test_e2e_diabetes_full.py)
catches gross errors against a single ground-truth, but a calibrated
chain can fail diabetes for MC-noise reasons; only SBC, by aggregating
across truths, gives the unbiased correctness signal.

Runtime is dominated by the M daedalus runs. Default M = 80 with N_live =
200 takes ~5 minutes wall on Apple Silicon; mark as @slow.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import daedalus
from daedalus.benchmarks import sbc_toy


# SBC budget. M = 80 gives KS test power to detect ~0.15 deviation from
# uniform at alpha = 0.05 -- plenty for catching systematic bias, while
# keeping the runtime under 5 minutes per coord.
M_TRIALS = 80
N_LIVE = 200
N_MCMC = 25
TRANSDIM_FRACTION = 0.5
DLOGZ = 0.5
MIN_ACTIVE_SAMPLES = 30  # trials with fewer slab samples are dropped


def _run_one_trial(
    problem: sbc_toy.SBCProblem,
    trial_seed: int,
) -> tuple[np.ndarray, np.ndarray, daedalus.Results]:
    """One SBC trial: sample truth, simulate data, run daedalus. Returns
    (gamma_true, beta_true, results)."""
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


def _run_sbc_loop(problem: sbc_toy.SBCProblem) -> dict:
    """Run M trials, collecting per-coord fractional ranks and inclusion
    probabilities. Returns a dict of arrays keyed by coordinate index."""
    g = problem.n_groups
    # Per-coord: list of fractional ranks (one per trial where gamma_true_i=1
    # AND the chain has enough samples with gamma_i=1).
    ranks: list[list[float]] = [[] for _ in range(g)]
    n_filtered_too_few: list[int] = [0 for _ in range(g)]
    inc_probs_chain: np.ndarray = np.empty((M_TRIALS, g), dtype=np.float64)
    gamma_truths: np.ndarray = np.empty((M_TRIALS, g), dtype=bool)
    for t in range(M_TRIALS):
        gamma_true, beta_true, results = _run_one_trial(
            problem, trial_seed=10_000 + t
        )
        gamma_truths[t] = gamma_true
        inc_probs_chain[t] = results.gamma.mean(axis=0)
        for k in range(g):
            if not gamma_true[k]:
                continue
            active_mask = results.gamma[:, k]
            n_active = int(active_mask.sum())
            if n_active < MIN_ACTIVE_SAMPLES:
                n_filtered_too_few[k] += 1
                continue
            beta_active = results.samples[active_mask, k]
            r = float(np.sum(beta_active < beta_true[k])) / float(n_active)
            ranks[k].append(r)
    return {
        "ranks": [np.asarray(r) for r in ranks],
        "n_filtered_too_few": np.asarray(n_filtered_too_few),
        "inc_probs_chain": inc_probs_chain,
        "gamma_truths": gamma_truths,
    }


@pytest.mark.benchmark
@pytest.mark.slow
def test_sbc_continuous_beta_ranks_uniform() -> None:
    """Per-coordinate fractional ranks of beta_true within the slab posterior
    must be uniform on [0, 1] under simulation from the prior."""
    problem = sbc_toy.make_sbc_problem(
        ndim=2, N=12, sigma2=1.0, tau2=4.0, inclusion_prior=0.5
    )
    results = _run_sbc_loop(problem)
    ranks = results["ranks"]
    n_filtered = results["n_filtered_too_few"]

    print()
    print(f"=== SBC: continuous-beta rank uniformity (M = {M_TRIALS} trials) ===")
    failures: list[str] = []
    for k in range(problem.n_groups):
        r = ranks[k]
        n_eff = r.size
        ks_stat, ks_p = stats.kstest(r, "uniform")
        # Histogram for visual sanity (10 bins).
        hist, _ = np.histogram(r, bins=10, range=(0.0, 1.0))
        expected = n_eff / 10.0
        chi2 = float(np.sum((hist - expected) ** 2 / max(expected, 1.0)))
        chi2_p = 1.0 - stats.chi2.cdf(chi2, df=9)
        print(
            f"  coord x{k}: n_eff={n_eff}  filtered={n_filtered[k]}  "
            f"KS stat={ks_stat:.3f} p={ks_p:.3f}  chi2={chi2:.2f} p={chi2_p:.3f}"
        )
        print(f"    bins: {hist.tolist()}  expected per bin ~= {expected:.1f}")
        # Pass criterion: KS p > 0.01. The threshold is loose because we're
        # running M = 80 trials -- power is moderate, false-positive rate
        # under a calibrated chain is the standard alpha = 0.05. Setting
        # the threshold at 0.01 gives a safety margin against rare flakes
        # while still flagging genuine miscalibration (which under SBC
        # typically yields p < 1e-3).
        if ks_p < 0.01:
            failures.append(
                f"  coord x{k}: KS p = {ks_p:.4f} < 0.01 (n_eff = {n_eff})"
            )
    assert not failures, (
        "SBC fractional-rank uniformity failed:\n" + "\n".join(failures)
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_sbc_inclusion_marginal_calibrated() -> None:
    """Marginal inclusion calibration: under simulation from the prior,
    E[P(gamma_i = 1 | y)] = P(gamma_i = 1) = inclusion_prior. A
    systematic over- or under-inclusion bias would violate this."""
    problem = sbc_toy.make_sbc_problem(
        ndim=2, N=12, sigma2=1.0, tau2=4.0, inclusion_prior=0.5
    )
    results = _run_sbc_loop(problem)
    inc = results["inc_probs_chain"]  # shape (M, g)
    truths = results["gamma_truths"]  # shape (M, g)

    target = problem.inclusion_prior
    print()
    print(
        f"=== SBC: inclusion-marginal calibration (M = {M_TRIALS}, "
        f"target = {target}) ==="
    )
    failures: list[str] = []
    for k in range(problem.n_groups):
        mean_p = float(inc[:, k].mean())
        # Standard error of the mean; we treat each trial's chain
        # estimate as one observation. The chain's own MC variance
        # within a trial is small relative to the trial-to-trial signal
        # at M = 80, so SE_M = (sample std of inc) / sqrt(M) is the
        # relevant uncertainty.
        se_M = float(inc[:, k].std(ddof=1)) / float(np.sqrt(M_TRIALS))
        z = (mean_p - target) / se_M if se_M > 0.0 else 0.0
        empirical_truth_rate = float(truths[:, k].mean())
        print(
            f"  coord x{k}: <P(gamma=1|y)> = {mean_p:.3f} ± {se_M:.3f}  "
            f"(z = {z:+.2f} from target {target})  "
            f"empirical truth rate = {empirical_truth_rate:.3f}"
        )
        # Pass criterion: |z| < 3. Under a calibrated chain z ~ N(0, 1)
        # so |z| > 3 happens with prob 0.003 per coord. We additionally
        # require the empirical truth rate to be within 3 binomial SEs
        # of the prior, since that's a sanity check on the SBC ground
        # truth (not on daedalus).
        if abs(z) > 3.0:
            failures.append(
                f"  coord x{k}: <P(gamma=1|y)> = {mean_p:.4f} differs from "
                f"prior {target} by {z:+.2f} sigma"
            )
        truth_se = float(np.sqrt(target * (1 - target) / M_TRIALS))
        if abs(empirical_truth_rate - target) > 3.0 * truth_se:
            failures.append(
                f"  coord x{k}: empirical truth rate {empirical_truth_rate:.3f} "
                f"is {(empirical_truth_rate - target) / truth_se:+.2f} sigma from "
                f"prior {target} -- SBC ground truth is itself non-calibrated"
            )
    assert not failures, (
        "SBC inclusion-marginal calibration failed:\n" + "\n".join(failures)
    )
