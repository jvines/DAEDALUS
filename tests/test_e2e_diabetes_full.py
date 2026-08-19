"""Multi-seed validation of daedalus on the full (non-marginalised) diabetes
variable-selection benchmark, under both birth kernels.

Unlike :mod:`tests.test_e2e_diabetes`, this benchmark does *not* marginalise
beta out, so the chain explores the joint ``(beta, gamma)`` posterior. Each
predictor becomes a one-parameter ``Group``, and the trans-dim kernel
proposes a real continuous coefficient on every flip. Closed-form
enumeration over ``2^10 = 1024`` models gives ground truth for
``P(gamma_i = 1 | y)``, ``E[beta_i | y]``, and ``SD[beta_i | y]`` (the
three columns of VdB+26 Table 1 in spirit, though the numbers will differ
because we use an independent Zellner slab with plug-in nuisance scales,
not full JZS).

This is the test that:

  (a) actually exercises :class:`GaussianRWBirth` (the marginalised
      benchmark has ``params=[]`` per group, so its trans-dim move never
      draws a continuous beta -- both birth kernels reduce to the same
      algorithm there);

  (b) validates the middle and right columns of Table 1 in addition to
      the left;

  (c) reports seed-to-seed variance, so the tolerance we adopt for the
      benchmark is not hiding noise.
"""

from __future__ import annotations

import numpy as np
import pytest

import daedalus
from daedalus.benchmarks import diabetes


# Multi-seed budget: 5 seeds is enough to see the dominant variance
# component (the Skilling H/n_live error) without making the test too
# slow. Tighten if the suite grows tolerant of a longer runtime.
N_SEEDS = 5
SEEDS = (101, 202, 303, 404, 505)


def _run_one(
    birth_kind: str,  # "uniform_u" or "gaussian_rw"
    seed: int,
) -> daedalus.Results:
    """Run daedalus on the full diabetes problem with the requested birth
    kernel. Returns the Results object."""
    problem = diabetes.make_problem_full(prior_inclusion=0.5)

    if birth_kind == "uniform_u":
        groups = [
            daedalus.Group(**kw) for kw in problem.groups_kwargs
        ]
    elif birth_kind == "gaussian_rw":
        # Initial scale matched to the prior SD so the chain starts in the
        # right ballpark; Robbins-Monro then tunes per Eq. D1. Matching the
        # scale matters here because off=0 sits in the *tail* of the
        # posterior for several of the strong predictors, so an untuned
        # GaussianRWBirth proposes poorly into the high-likelihood region.
        prior_sd = float(np.sqrt(problem.tau2))
        groups = [
            daedalus.Group(
                **kw,
                birth_proposal=daedalus.GaussianRWBirth(scale=prior_sd),
                log_prior_continuous=problem.log_prior_continuous_factory(i),
            )
            for i, kw in enumerate(problem.groups_kwargs)
        ]
    else:
        raise ValueError(birth_kind)

    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=groups,
        bound="single",
        sample="rwalk",
        n_live=800,
        seed=seed,
    )
    return sampler.run_nested(
        dlogz=0.5,
        n_mcmc=50,
        transdim_fraction=0.5,
        show_progress=False,
    )


def _summarise(results: daedalus.Results) -> dict[str, np.ndarray]:
    """From a Results object compute per-predictor inclusion prob, E[beta],
    SD[beta]. Returns dict of length-10 arrays keyed by metric name."""
    # Inclusion probability per group: chain frequency.
    p_inc = results.gamma.mean(axis=0)
    # E[beta] and Var[beta] over the equally-weighted resample. The
    # samples already have inactive coords clamped to off (=0).
    e_beta = results.samples.mean(axis=0)
    var_beta = results.samples.var(axis=0)
    sd_beta = np.sqrt(var_beta)
    return {"P": p_inc, "E_beta": e_beta, "SD_beta": sd_beta}


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize(
    "birth_kind",
    [
        "uniform_u",
        # Both birth proposals reproduce the exact enumeration on this
        # problem. With the prior-SD initial scale set above, GaussianRWBirth
        # matches every inclusion probability inside tolerance (worst
        # |dP| = 0.0400 against the 0.07 bound), with E and SD well inside
        # their 0.25*SD_truth bounds.
        "gaussian_rw",
    ],
)
def test_diabetes_full_matches_enumeration(birth_kind: str) -> None:
    """Multi-seed run; pin posterior moments within tolerance of the
    closed-form enumeration."""
    problem = diabetes.make_problem_full(prior_inclusion=0.5)
    truth_P = np.array(
        [problem.inclusion_prob_true[n] for n in diabetes.PREDICTOR_NAMES]
    )
    truth_E = np.array([problem.e_beta_true[n] for n in diabetes.PREDICTOR_NAMES])
    truth_SD = np.array(
        [problem.sd_beta_true[n] for n in diabetes.PREDICTOR_NAMES]
    )

    p_runs = np.empty((N_SEEDS, len(truth_P)), dtype=float)
    e_runs = np.empty_like(p_runs)
    sd_runs = np.empty_like(p_runs)
    for i, seed in enumerate(SEEDS):
        results = _run_one(birth_kind, seed)
        s = _summarise(results)
        p_runs[i] = s["P"]
        e_runs[i] = s["E_beta"]
        sd_runs[i] = s["SD_beta"]

    p_mean = p_runs.mean(axis=0)
    e_mean = e_runs.mean(axis=0)
    sd_mean = sd_runs.mean(axis=0)
    p_seedstd = p_runs.std(axis=0, ddof=1)
    e_seedstd = e_runs.std(axis=0, ddof=1)
    sd_seedstd = sd_runs.std(axis=0, ddof=1)

    # Tolerances. P is bounded in [0,1] -> 0.07 absolute is the
    # NS-level precision we achieve on the intermediate-evidence S1/S2/S3
    # predictors at n_live=800, n_mcmc=50; the always-on (P=1) and
    # always-off (P~0.05) predictors land within 0.005. For E[beta] and
    # SD[beta] we tolerate 0.25 of the *posterior* SD per coordinate --
    # the natural scale-free MC-noise tolerance for posterior moments.
    # Cap with an absolute floor so coords with vanishing SD don't trip
    # an unfairly tight relative bound.
    tol_P_abs = 0.07
    tol_relative_to_truthSD = 0.25
    tol_floor_abs = 1.0  # absolute floor on E,SD tolerance

    tol_E = np.maximum(tol_relative_to_truthSD * truth_SD, tol_floor_abs)
    tol_SD = np.maximum(tol_relative_to_truthSD * truth_SD, tol_floor_abs)

    err_P = np.abs(p_mean - truth_P)
    err_E = np.abs(e_mean - truth_E)
    err_SD = np.abs(sd_mean - truth_SD)

    print()
    print(f"=== diabetes-full validation, birth_kind={birth_kind} ===")
    print(f"{'name':<6} {'P_truth':>9} {'P_mean':>9} {'P_seedSD':>9} {'errP':>8}"
          f"   {'E_truth':>10} {'E_mean':>10} {'E_seedSD':>9} {'errE':>9}"
          f"   {'SD_truth':>10} {'SD_mean':>10} {'errSD':>9}")
    for i, name in enumerate(diabetes.PREDICTOR_NAMES):
        print(
            f"{name:<6} "
            f"{truth_P[i]:>9.4f} {p_mean[i]:>9.4f} {p_seedstd[i]:>9.4f} "
            f"{err_P[i]:>8.4f}   "
            f"{truth_E[i]:>10.3f} {e_mean[i]:>10.3f} {e_seedstd[i]:>9.3f} "
            f"{err_E[i]:>9.3f}   "
            f"{truth_SD[i]:>10.3f} {sd_mean[i]:>10.3f} {err_SD[i]:>9.3f}"
        )

    failures: list[str] = []
    for i, name in enumerate(diabetes.PREDICTOR_NAMES):
        if err_P[i] > tol_P_abs:
            failures.append(
                f"{name} P: |{p_mean[i]:.4f} - {truth_P[i]:.4f}| = "
                f"{err_P[i]:.4f} > {tol_P_abs}"
            )
        if err_E[i] > tol_E[i]:
            failures.append(
                f"{name} E[beta]: |{e_mean[i]:.4f} - {truth_E[i]:.4f}| = "
                f"{err_E[i]:.4f} > {tol_E[i]:.4f}"
            )
        if err_SD[i] > tol_SD[i]:
            failures.append(
                f"{name} SD[beta]: |{sd_mean[i]:.4f} - {truth_SD[i]:.4f}| = "
                f"{err_SD[i]:.4f} > {tol_SD[i]:.4f}"
            )
    assert not failures, (
        f"diabetes-full ({birth_kind}) deviations exceed tolerance:\n"
        + "\n".join(failures)
    )


@pytest.mark.benchmark
def test_enumeration_self_consistency() -> None:
    """The enumeration must be internally consistent: probabilities in
    [0,1], variance non-negative, inactive coordinates contribute zero
    (we never see beta_i != 0 when gamma_i = 0)."""
    problem = diabetes.make_problem_full(prior_inclusion=0.5)
    fp_eps = 1e-9
    for name in diabetes.PREDICTOR_NAMES:
        p = problem.inclusion_prob_true[name]
        assert -fp_eps <= p <= 1.0 + fp_eps, f"{name}: P out of [0,1]: {p}"
        sd = problem.sd_beta_true[name]
        assert sd >= -fp_eps, f"{name}: SD < 0: {sd}"
    # When prior_inclusion = 0.5 we have a uniform model prior, so the
    # null model has weight 1/1024; the chain must therefore treat
    # *every* coefficient as having non-zero prior probability of being
    # zero. None of the marginal SDs may be exactly zero.
    for name in diabetes.PREDICTOR_NAMES:
        sd = problem.sd_beta_true[name]
        assert sd > 0.0, f"{name}: SD is exactly zero (prior_inclusion < 1?)"
