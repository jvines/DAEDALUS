"""End-to-end MoMS-NS demo on real TESS data of WASP-47.

Single-planet trans-dim search over a 25-day TESS Sector 42 light
curve. With a BLS-informed birth proposal the chain consistently
detects a transit signal at WASP-47b's period (4.16 d) or its first
harmonic (2.08 d) -- the half-period mode is a known local maximum
of the box-transit likelihood when the model can absorb every other
"transit" as zero-depth. Recovering all three known WASP-47 planets
(b, d, e) jointly requires iterative BLS or a sharper proposal
mixture and is left to a v0.2 demo.

The test is the first real-data validation of the pluggable
BirthProposal infrastructure: with default uniform-u birth the chain
typically gets stuck on an arbitrary period far from any transit
signal; the BLS-informed birth concentrates trans-dim flips on
genuinely periodic structure in the data.
"""

from __future__ import annotations

import numpy as np
import pytest

import daedalus
from daedalus.benchmarks import wasp47


WASP47B_PERIOD = wasp47.KNOWN_PLANETS_DAYS["WASP-47b"]
ACCEPTABLE_PERIODS = (
    WASP47B_PERIOD,            # 4.1592 (true period)
    WASP47B_PERIOD / 2.0,      # 2.0796 (first harmonic, common BLS confusion)
    WASP47B_PERIOD * 2.0,      # 8.3184 (sub-harmonic)
)


def _within_tolerance(period: float, target: float, frac: float = 0.01) -> bool:
    return abs(period - target) / target < frac


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Empirical-frequency inclusion saturates on this problem: all four "
        "BLS control aliases (b/2, b*2, b/3, b*1.5) reach P(gamma=1) = 1.000 "
        "against the < 0.20 bound, while the three literature candidates "
        "pass. Cause is the add/delete asymmetry documented in "
        "daedalus/results.py: once L* exceeds the best likelihood the "
        "off-state can reach, DELETE can never satisfy L > L* again, so no "
        "slot can ever be switched off and every candidate stays active. "
        "Not fixable by rejuvenate_fraction, which re-proposes *within* the "
        "active model and never restores DELETE acceptance. The right "
        "instrument is Results.rao_blackwell_inclusion(), which needs an "
        "analytic gamma-conditional that a transit likelihood does not "
        "provide. Left as a strict xfail so it surfaces if either changes."
    ),
)
def test_wasp47_bls_selection_real_vs_control() -> None:
    """Multi-component trans-dim selection on real WASP-47 TESS data.

    Builds a candidate set of 3 literature periods (b, d, e) plus N
    *control* periods picked from BLS peaks of the raw multi-sector
    light curve with the literature-period windows masked out. The
    chain must:
      - activate every literature candidate: P(gamma = 1) > 0.95
      - reject every control candidate: P(gamma = 1) < 0.20
      - concentrate the model-size posterior on N_planets = 3

    Without controls in the candidate list the same test "passes"
    even if a literature candidate is replaced with a fake period --
    the chain has no symmetric way to declare a candidate
    unsupported. Adding BLS-derived controls (mostly aliases of
    WASP-47b, since b's 1.3% transit dominates the periodogram)
    gives the test discriminating power: the alias slots have the
    same transit-shape model freedom as the real slots, so their
    P(gamma = 1) only stays low if the joint fit really does
    explain b's transits with the b slot rather than absorbing
    structure into the alias slots.
    """
    lc = wasp47.load_lightcurve(wasp47.MULTISECTOR_LC)
    real_periods = tuple(wasp47.KNOWN_PLANETS_DAYS.values())
    controls = wasp47.find_control_candidates(
        time=lc["time"], flux=lc["flux"],
        real_periods=real_periods,
        n_controls=4,
    )
    # Match the controls' log-period half-width (~1.2%) to the tuned
    # literature windows (b: ~1.5%, d: ~1.1%, e: ~1.5%) so real and
    # control slots have comparable prior widths -- otherwise wider
    # control priors would unfairly penalise inclusion through Occam.
    control_tuples = tuple(
        (f"CTRL_{c.period:.4f}d", c.period, (np.exp(0.012) - 1.0) * c.period)
        for c in controls
    )
    candidates = wasp47.TIGHT_PRIOR_WINDOWS + control_tuples

    problem = wasp47.make_problem_tight(
        candidates=candidates,
        lc_path=wasp47.MULTISECTOR_LC,
        model="mandel_agol",
        bin_minutes=0.0,
        fit_jitter=True,
    )
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=problem.groups,
        bound="single",
        sample="rwalk",
        n_live=400,
        seed=42,
    )
    results = sampler.run_nested(
        dlogz=0.5, n_mcmc=30, transdim_fraction=0.3, bound_update_interval=10,
    )

    inc = results.inclusion_probabilities()
    real_names = tuple(p[0] for p in wasp47.TIGHT_PRIOR_WINDOWS)
    control_names = tuple(c[0] for c in control_tuples)

    failures: list[str] = []
    for name in real_names:
        if inc[name] < 0.95:
            failures.append(f"REAL  {name}: P(gamma=1) = {inc[name]:.3f}, expected > 0.95")
    for name in control_names:
        if inc[name] > 0.20:
            failures.append(f"CTRL  {name}: P(gamma=1) = {inc[name]:.3f}, expected < 0.20")
    if failures:
        full_inc = "\n".join(f"  {k}: {v:.3f}" for k, v in inc.items())
        raise AssertionError(
            "BLS-selection real-vs-control discrimination failed:\n"
            + "\n".join(failures)
            + "\n\nFull inclusion probabilities:\n"
            + full_inc
        )

    n_p = wasp47.n_planets_posterior(results)
    assert n_p[3] > 0.85, (
        f"P(N_p = 3 | data) = {n_p[3]:.3f}, expected > 0.85 -- model-size "
        f"posterior should concentrate on the truth (3 real planets in "
        f"a {len(candidates)}-candidate dictionary). Full N_p posterior: {n_p}"
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_wasp47_multi_planet_bls_birth_recovers_b_plus_a_distinct_period() -> None:
    """3-candidate trans-dim NS over the WASP-47 light curve.

    Iterative-residual BLS birth should:
      - Activate every candidate slot (P(gamma) ~ 1)
      - Have AT LEAST ONE slot recover WASP-47b at P = 4.16 d to 1%
      - Have AT LEAST ONE slot land on a *different* period (P > 6 d
        OR P < 2 d) -- showing the iterative residuals successfully
        steered the chain off b and onto another part of the
        periodogram. Joint b+d+e recovery is harder still and left
        for v0.2.
    """
    problem = wasp47.make_problem(
        n_planets=3, use_bls_birth=True, bls_sharpening=30.0
    )
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=problem.groups,
        bound="single",
        sample="rwalk",
        n_live=400,
        seed=42,
    )
    results = sampler.run_nested(
        dlogz=0.5, n_mcmc=30, transdim_fraction=0.5, bound_update_interval=10
    )

    inc = results.inclusion_probabilities()
    n_active = sum(1 for k in range(3) if inc[f"planet_{k}"] > 0.95)
    assert n_active >= 2, (
        f"Only {n_active}/3 slots active above 0.95; expected the chain to "
        f"prefer multi-planet models over single-planet on this data."
    )

    median_periods = []
    for k in range(3):
        mask = results.gamma[:, k]
        if mask.sum() < 50:
            continue
        median_periods.append(float(np.median(np.exp(results.samples[mask, 4 * k]))))

    # At least one slot finds WASP-47b at the right period (within 1%).
    if not any(_within_tolerance(p, WASP47B_PERIOD) for p in median_periods):
        raise AssertionError(
            f"No slot recovered WASP-47b's period (4.16 d) within 1%. "
            f"Recovered medians: {median_periods}"
        )

    # At least one slot lands on a NON-b period (away from b harmonics).
    distinct = [
        p for p in median_periods
        if not any(_within_tolerance(p, target, frac=0.05) for target in ACCEPTABLE_PERIODS)
    ]
    if not distinct:
        raise AssertionError(
            f"All 3 slots converged onto WASP-47b or its harmonics. "
            f"Iterative-residual BLS should have steered at least one slot "
            f"to a different signal. Recovered medians: {median_periods}"
        )


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Default configuration cannot recover the period: gamma = 1 is "
        "absorbing once L* exceeds the off-model's ceiling (measured: last "
        "accepted DELETE is flip 1104 of 44011), and the BLS birth proposal "
        "only fires on an OFF->ON flip, so the period freezes wherever the "
        "early birth left it and drifts to a spurious ~10.40 d mode -- "
        "~2100 nats below the true global optimum. The planet IS detected "
        "(P(gamma=1) > 0.95); it is the period that is wrong. Fixed by the "
        "within-model rejuvenation move -- see "
        "test_wasp47_single_planet_recovers_with_rejuvenation. Kept strict "
        "so that a kernel change which fixes the default path surfaces here."
    ),
)
def test_wasp47_single_planet_bls_birth_detects_transit() -> None:
    problem = wasp47.make_problem(n_planets=1, use_bls_birth=True)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=problem.groups,
        bound="single",
        sample="rwalk",
        n_live=300,
        seed=42,
    )
    results = sampler.run_nested(
        dlogz=0.5, n_mcmc=20, transdim_fraction=0.4, bound_update_interval=10
    )

    # The candidate slot should be active essentially always: WASP-47b's
    # transit signal is unmissable.
    p_inc = results.inclusion_probabilities()["planet_0"]
    assert p_inc > 0.95, (
        f"P(planet detected) = {p_inc:.4f}, expected > 0.95 -- BLS-informed "
        f"birth should find the transit signal in TESS Sector 42 of WASP-47"
    )

    # Median posterior period should land within 1% of the true period or
    # one of its first harmonics (the box-transit half-period local mode is
    # benign in this context: it still corresponds to a real periodic
    # signal at the correct rest of WASP-47b's transit times).
    mask = results.gamma[:, 0]
    periods = np.exp(results.samples[mask, 0])
    p_median = float(np.median(periods))
    if not any(_within_tolerance(p_median, target) for target in ACCEPTABLE_PERIODS):
        raise AssertionError(
            f"Recovered median period = {p_median:.4f}d does not match any of "
            f"the WASP-47b acceptable periods {ACCEPTABLE_PERIODS}; the chain "
            f"locked onto a non-physical local mode."
        )


@pytest.mark.benchmark
@pytest.mark.slow
def test_wasp47_single_planet_recovers_with_rejuvenation() -> None:
    """The within-model rejuvenation move recovers WASP-47b's true period.

    Companion to ``test_wasp47_single_planet_bls_birth_detects_transit``,
    which is xfail because the default configuration cannot: once ``L*``
    passes the off-model's ceiling, ``gamma = 1`` is absorbing, the BLS
    birth can never re-fire, and the period stays frozen at a spurious
    ~10.40 d mode.

    ``rejuvenate_fraction`` re-proposes the active slot's continuous block
    from the same BLS proposal without touching ``gamma``, restoring a
    mode-hopping move at any ``L*``. Uses the DE kernel, which is the
    cheapest kernel that solves this problem.

    Not a guarantee: over 6 seeds this recovers the global mode in 5, and
    the slice kernel reaches it too but at ~44x the likelihood calls. This
    test pins seed 42, which recovers P = 4.1585 d (max log L -4629.4,
    against the -4624.48 global optimum confirmed by multi-start
    optimisation).
    """
    problem = wasp47.make_problem(n_planets=1, use_bls_birth=True)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=problem.groups,
        bound="single",
        sample="de",
        n_live=300,
        seed=42,
    )
    results = sampler.run_nested(
        dlogz=0.5, n_mcmc=20, transdim_fraction=0.4,
        bound_update_interval=10, rejuvenate_fraction=0.3,
    )

    p_inc = results.inclusion_probabilities()["planet_0"]
    assert p_inc > 0.95, f"P(planet detected) = {p_inc:.4f}, expected > 0.95"

    mask = results.gamma[:, 0]
    p_median = float(np.median(np.exp(results.samples[mask, 0])))
    if not any(_within_tolerance(p_median, t) for t in ACCEPTABLE_PERIODS):
        raise AssertionError(
            f"Recovered median period = {p_median:.4f}d does not match any of "
            f"the WASP-47b acceptable periods {ACCEPTABLE_PERIODS}; "
            f"rejuvenation failed to break the chain out of the frozen mode."
        )
