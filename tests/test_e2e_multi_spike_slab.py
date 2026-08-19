"""End-to-end trans-dim NS validation with closed-form ground truth on
a two-group spike-and-slab problem. Validates everything daedalus
produces against analytic targets:

  * the joint log evidence ``log Z`` (Skilling recursion),
  * the per-group inclusion probabilities ``P(gamma_k = 1 | y)``,
  * the per-coordinate posterior moments ``E[beta_i | y]``,
    ``SD[beta_i | y]``,
  * the per-model evidences ``log Z_gamma`` for every visited
    inclusion configuration, via :meth:`Results.model_evidences`.

The last item is the load-bearing one: this is the first benchmark
that confronts daedalus's per-model evidence output with an analytic
reference (rather than internal-consistency relations).
"""

from __future__ import annotations

import numpy as np
import pytest

import daedalus
from daedalus.benchmarks import multi_spike_slab


@pytest.mark.benchmark
@pytest.mark.slow
def test_multi_spike_slab_recovers_closed_form() -> None:
    problem = multi_spike_slab.make_problem(
        y=(2.0, 3.5), sigma=1.0, sigma_slab=3.0, inclusion_priors=(0.5, 0.5)
    )

    # Multi-seed run for stable inclusion probabilities and per-gamma Z.
    seeds = (101, 202, 303, 404, 505)
    log_Z_runs = np.empty(len(seeds), dtype=np.float64)
    inc_runs = np.empty((len(seeds), problem.ndim), dtype=np.float64)
    e_beta_runs = np.empty_like(inc_runs)
    sd_beta_runs = np.empty_like(inc_runs)
    log_Z_gamma_runs: list[dict[tuple[bool, ...], float]] = []

    for k, seed in enumerate(seeds):
        groups = [daedalus.Group(**kw) for kw in problem.groups_kwargs]
        sampler = daedalus.NestedSampler(
            loglike=problem.loglike,
            prior_transform=problem.prior_transform,
            ndim=problem.ndim,
            groups=groups,
            bound="single",
            sample="rwalk",
            n_live=500,
            seed=seed,
        )
        results = sampler.run_nested(
            dlogz=0.5, n_mcmc=30, transdim_fraction=0.5, show_progress=False
        )
        log_Z_runs[k] = results.log_Z
        inc_runs[k] = results.gamma.mean(axis=0)
        e_beta_runs[k] = results.samples.mean(axis=0)
        sd_beta_runs[k] = results.samples.std(axis=0)
        log_Z_gamma_runs.append(
            {key: val[0] for key, val in results.model_evidences().items()}
        )

    log_Z_chain = float(np.mean(log_Z_runs))
    inc_chain = inc_runs.mean(axis=0)
    e_beta_chain = e_beta_runs.mean(axis=0)
    sd_beta_chain = sd_beta_runs.mean(axis=0)

    print()
    print("=== multi_spike_slab closed-form vs chain ===")
    print(f"log Z: truth = {problem.log_Z_true:.4f}, "
          f"chain = {log_Z_chain:.4f} +/- {log_Z_runs.std(ddof=1):.4f}")
    print(f"{'coord':<6} {'P_truth':>9} {'P_chain':>9} "
          f"{'E_truth':>9} {'E_chain':>9} {'SD_truth':>9} {'SD_chain':>9}")
    for i in range(problem.ndim):
        print(
            f"x{i:<5} "
            f"{problem.inclusion_prob_true[i]:>9.4f} {inc_chain[i]:>9.4f} "
            f"{problem.e_beta_true[i]:>9.4f} {e_beta_chain[i]:>9.4f} "
            f"{problem.sd_beta_true[i]:>9.4f} {sd_beta_chain[i]:>9.4f}"
        )
    print()
    print("per-gamma log Z_gamma (truth vs chain mean across visited seeds):")
    for gamma_key, log_Z_g_true in problem.log_Z_gamma_true.items():
        # Average across the seeds that visited this gamma.
        visited = [run.get(gamma_key) for run in log_Z_gamma_runs]
        visited = [v for v in visited if v is not None]
        if not visited:
            continue
        mean_chain = float(np.mean(visited))
        print(
            f"  gamma = {gamma_key}: truth = {log_Z_g_true:+.4f}, "
            f"chain = {mean_chain:+.4f}"
        )

    # Tolerances.
    log_Z_tol = 0.30                            # generous; H/n_live ~ 0.05
    inc_tol = 0.05                              # NS-level precision
    e_beta_tol = 0.20 * problem.sd_beta_true + 0.10
    sd_beta_tol = 0.20 * problem.sd_beta_true + 0.10
    log_Z_gamma_tol = 0.50                      # log Z error + binomial MC

    failures: list[str] = []

    # 1. Joint log Z.
    if abs(log_Z_chain - problem.log_Z_true) > log_Z_tol:
        failures.append(
            f"log Z: chain {log_Z_chain:.4f} - truth "
            f"{problem.log_Z_true:.4f} = "
            f"{log_Z_chain - problem.log_Z_true:+.4f} "
            f"(tol {log_Z_tol})"
        )

    # 2. Per-group inclusion probabilities.
    for i in range(problem.ndim):
        d = abs(inc_chain[i] - problem.inclusion_prob_true[i])
        if d > inc_tol:
            failures.append(
                f"P(gamma_{i}=1|y): chain {inc_chain[i]:.4f} vs truth "
                f"{problem.inclusion_prob_true[i]:.4f}, |diff| = {d:.4f}"
            )

    # 3. Posterior moments per coord.
    for i in range(problem.ndim):
        d_e = abs(e_beta_chain[i] - problem.e_beta_true[i])
        d_sd = abs(sd_beta_chain[i] - problem.sd_beta_true[i])
        if d_e > e_beta_tol[i]:
            failures.append(
                f"E[beta_{i}|y]: chain {e_beta_chain[i]:.4f} vs truth "
                f"{problem.e_beta_true[i]:.4f}, |diff| = {d_e:.4f} "
                f"(tol {e_beta_tol[i]:.4f})"
            )
        if d_sd > sd_beta_tol[i]:
            failures.append(
                f"SD[beta_{i}|y]: chain {sd_beta_chain[i]:.4f} vs truth "
                f"{problem.sd_beta_true[i]:.4f}, |diff| = {d_sd:.4f} "
                f"(tol {sd_beta_tol[i]:.4f})"
            )

    # 4. Per-gamma evidences. Average across seeds for each visited gamma;
    #    require agreement within log_Z_gamma_tol with the closed form.
    for gamma_key, log_Z_g_true in problem.log_Z_gamma_true.items():
        visited = [run.get(gamma_key) for run in log_Z_gamma_runs]
        visited = [v for v in visited if v is not None]
        if not visited:
            # Configuration not visited by any seed; this is fine for
            # vanishingly small posterior probability, only flag if the
            # truth says it should be visited often.
            P_g_true = float(np.exp(log_Z_g_true - problem.log_Z_true))
            if P_g_true > 0.05:
                failures.append(
                    f"gamma = {gamma_key} has truth posterior prob "
                    f"{P_g_true:.3f} but was visited by 0 seeds"
                )
            continue
        mean_chain = float(np.mean(visited))
        if abs(mean_chain - log_Z_g_true) > log_Z_gamma_tol:
            failures.append(
                f"log Z_gamma for {gamma_key}: chain mean {mean_chain:+.4f} "
                f"vs truth {log_Z_g_true:+.4f}, "
                f"|diff| = {abs(mean_chain - log_Z_g_true):.4f}"
            )

    assert not failures, (
        "multi_spike_slab closed-form validation failed:\n"
        + "\n".join("  - " + f for f in failures)
    )
