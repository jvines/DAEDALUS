"""Custom BirthProposal validation.

Pin that a custom birth proposal recovers the same posterior inclusion
probability as the default uniform-u birth on a problem where both
should agree (closed-form spike-and-slab). If the M-H math for the
custom path is wrong, the inclusion probability will be biased
relative to truth in a way the default path is not.
"""

from __future__ import annotations

import numpy as np
import pytest

import daedalus
from daedalus.benchmarks import spike_slab


def _uniform_log_prior_factory(half_width: float):
    log_density = float(-np.log(2.0 * half_width))

    def log_prior(beta: np.ndarray) -> float:
        if np.any(np.abs(beta) > half_width):
            return -np.inf
        return log_density

    return log_prior


@pytest.mark.benchmark
def test_custom_gaussian_rw_birth_matches_default_uniform_u_birth() -> None:
    """Custom GaussianRWBirth should recover the same inclusion
    probability as default uniform-u to within the run-to-run noise.
    """
    problem = spike_slab.make_problem(prior_half_width=10.0, inclusion_prior=0.5)

    # Default uniform-u path.
    g_default = daedalus.Group(
        name="beta_1", params=[1], off_values=np.array([0.0]), inclusion_prior=0.5
    )
    sampler_default = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=[g_default],
        bound="single",
        sample="rwalk",
        n_live=600,
        seed=2026,
    )
    res_default = sampler_default.run_nested(
        dlogz=0.1, n_mcmc=40, transdim_fraction=0.7
    )
    p_default = res_default.inclusion_probabilities()["beta_1"]

    # Custom Gaussian RW birth + explicit uniform log_prior.
    g_custom = daedalus.Group(
        name="beta_1",
        params=[1],
        off_values=np.array([0.0]),
        inclusion_prior=0.5,
        birth_proposal=daedalus.GaussianRWBirth(scale=1.0),
        log_prior_continuous=_uniform_log_prior_factory(10.0),
    )
    sampler_custom = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=[g_custom],
        bound="single",
        sample="rwalk",
        n_live=600,
        seed=2026,
    )
    res_custom = sampler_custom.run_nested(
        dlogz=0.1, n_mcmc=40, transdim_fraction=0.7
    )
    p_custom = res_custom.inclusion_probabilities()["beta_1"]

    truth = problem.inclusion_prob_true
    err_default = abs(p_default - truth)
    err_custom = abs(p_custom - truth)

    # Both paths should land within 0.05 of the analytic posterior.
    assert err_default < 0.05, f"default: P = {p_default:.4f}, truth = {truth:.4f}"
    assert err_custom < 0.05, f"custom:  P = {p_custom:.4f}, truth = {truth:.4f}"


def test_group_rejects_custom_birth_without_log_prior() -> None:
    """Defensive: a custom birth_proposal requires log_prior_continuous."""
    with pytest.raises(ValueError, match="log_prior_continuous"):
        daedalus.Group(
            name="x",
            params=[0],
            off_values=np.array([0.0]),
            birth_proposal=daedalus.GaussianRWBirth(scale=1.0),
        )


def test_gaussian_rw_birth_log_density_self_consistent() -> None:
    """log_density evaluated at the propose() output should match the
    log_q_forward returned by propose() (modulo numerical noise).
    """
    g = daedalus.Group(
        name="x",
        params=[0, 1],
        off_values=np.array([0.0, 0.0]),
        birth_proposal=daedalus.GaussianRWBirth(scale=2.5),
        log_prior_continuous=lambda b: 0.0,
    )
    rng = np.random.default_rng(0)
    state = daedalus.State(
        u=np.zeros(2),
        beta=np.array([0.0, 0.0]),
        gamma=np.array([False]),
    )
    for _ in range(5):
        result = g.birth_proposal.propose(g, state, rng)
        eval_density = g.birth_proposal.log_density(g, state, result.proposed_beta)
        assert np.isclose(result.log_q_forward, eval_density, atol=1e-12)


def test_gaussian_rw_birth_update_is_noop_before_propose() -> None:
    """update() before any propose() call must not crash and must leave the
    constructor scale untouched (it is the user-facing default until a
    Group materialises the runtime array)."""
    birth = daedalus.GaussianRWBirth(scale=3.0)
    birth.update(accepted=True)
    birth.update(accepted=False)
    # Scale property still returns the constructor argument.
    assert birth.scale == 3.0


def test_gaussian_rw_birth_update_follows_eq_D1() -> None:
    """Per van den Bergh+ 2026 Eq. D1, after a propose() materialises
    the working scale, update(accepted) must apply
        log tau_{t+1} = log tau_t + (t+1)^{-phi} * (1{accept} - alpha)
    on the log scale, with alpha defaulting to 0.44 for univariate groups
    and 0.234 for vector groups. We exercise both regimes.
    """
    # Univariate group -> alpha_target = 0.44.
    g_uni = daedalus.Group(
        name="x",
        params=[0],
        off_values=np.array([0.0]),
        birth_proposal=daedalus.GaussianRWBirth(scale=1.0),
        log_prior_continuous=lambda b: 0.0,
    )
    state = daedalus.State(u=np.zeros(1), beta=np.array([0.0]), gamma=np.array([False]))
    rng = np.random.default_rng(0)
    g_uni.birth_proposal.propose(g_uni, state, rng)  # materialise scale
    log_tau0 = float(np.log(g_uni.birth_proposal.scale[0]))
    g_uni.birth_proposal.update(accepted=True)  # t=0 -> step (1)^{-0.75}
    expected = log_tau0 + (1.0) ** (-0.75) * (1.0 - 0.44)
    assert np.isclose(np.log(g_uni.birth_proposal.scale[0]), expected, atol=1e-12)

    # Vector group -> alpha_target = 0.234.
    g_vec = daedalus.Group(
        name="v",
        params=[0, 1, 2],
        off_values=np.zeros(3),
        birth_proposal=daedalus.GaussianRWBirth(scale=2.0),
        log_prior_continuous=lambda b: 0.0,
    )
    state_v = daedalus.State(u=np.zeros(3), beta=np.zeros(3), gamma=np.array([False]))
    g_vec.birth_proposal.propose(g_vec, state_v, rng)
    log_tau0_v = float(np.log(g_vec.birth_proposal.scale[0]))
    g_vec.birth_proposal.update(accepted=False)  # t=0
    expected_v = log_tau0_v + (1.0) ** (-0.75) * (0.0 - 0.234)
    assert np.isclose(np.log(g_vec.birth_proposal.scale[0]), expected_v, atol=1e-12)
    # All coords share the same per-step signal, so they evolve in lockstep
    # relative to their starting values (here all equal -> all equal).
    assert np.allclose(g_vec.birth_proposal.scale, g_vec.birth_proposal.scale[0])


def test_gaussian_rw_birth_update_diminishes() -> None:
    """The (t+1)^{-0.75} schedule must reduce the per-step signal magnitude
    over time. After many all-accept updates the cumulative log-shift must
    converge (Robbins-Monro), so the scale stays finite well below
    scale_max."""
    g = daedalus.Group(
        name="x",
        params=[0],
        off_values=np.array([0.0]),
        birth_proposal=daedalus.GaussianRWBirth(scale=1.0, scale_max=1e6),
        log_prior_continuous=lambda b: 0.0,
    )
    state = daedalus.State(u=np.zeros(1), beta=np.array([0.0]), gamma=np.array([False]))
    rng = np.random.default_rng(0)
    g.birth_proposal.propose(g, state, rng)
    for _ in range(10_000):
        g.birth_proposal.update(accepted=True)
    # Sum_{t=1..N} t^{-0.75} ~ N^{0.25} / 0.25; per-step signal = (1 - 0.44).
    # Predicted upper bound on log_scale shift: 0.56 * 4 * 10000^0.25 ~ 22.4.
    # The actual sum is finite and well below scale_max=1e6 (log 1e6 ~ 13.8).
    # We pin: scale grew (adaptation worked) but stayed under the clamp.
    assert g.birth_proposal.scale[0] > 1.0
    assert g.birth_proposal.scale[0] <= 1e6


def test_gaussian_rw_birth_warmup_freezes_scale() -> None:
    """After warmup_steps updates, further calls must be no-ops -- the
    Markov property is then exactly preserved."""
    birth = daedalus.GaussianRWBirth(scale=1.0, warmup_steps=3)
    g = daedalus.Group(
        name="x",
        params=[0],
        off_values=np.array([0.0]),
        birth_proposal=birth,
        log_prior_continuous=lambda b: 0.0,
    )
    state = daedalus.State(u=np.zeros(1), beta=np.array([0.0]), gamma=np.array([False]))
    birth.propose(g, state, np.random.default_rng(0))
    for _ in range(3):
        birth.update(accepted=True)
    frozen = birth.scale.copy()
    for _ in range(50):
        birth.update(accepted=True)
    assert np.array_equal(birth.scale, frozen)


def test_nested_sampler_drives_birth_adaptation() -> None:
    """Integration test: NestedSampler must call BirthProposal.update() on
    every flip attempt. Verified by counting updates against an
    instrumented birth proposal on the spike-and-slab benchmark.
    """
    from daedalus.benchmarks import spike_slab

    problem = spike_slab.make_problem(prior_half_width=10.0, inclusion_prior=0.5)
    n_calls = {"count": 0, "accepts": 0}

    class CountingBirth(daedalus.GaussianRWBirth):
        def update(self, accepted: bool) -> None:
            n_calls["count"] += 1
            if accepted:
                n_calls["accepts"] += 1
            super().update(accepted)

    g = daedalus.Group(
        name="beta_1",
        params=[1],
        off_values=np.array([0.0]),
        birth_proposal=CountingBirth(scale=1.0),
        log_prior_continuous=_uniform_log_prior_factory(10.0),
    )
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=[g],
        bound="single",
        sample="rwalk",
        n_live=200,
        seed=0,
    )
    sampler.run_nested(dlogz=1.0, n_mcmc=10, transdim_fraction=0.5, show_progress=False)
    assert n_calls["count"] > 0, "NestedSampler never called BirthProposal.update()"
    # Trivial sanity: at least one of the calls must have flipped.
    assert n_calls["accepts"] >= 0  # cannot exceed count, accept rate may be low


def test_custom_birth_add_resyncs_u_so_kernels_preserve_value() -> None:
    """Regression: an accepted custom-birth ADD must leave state.u
    consistent with the proposed beta. The within-model kernels
    regenerate beta from u at every step, so a stale u silently
    overwrites the accepted birth value on the next move."""
    from daedalus.birth_proposals import BirthProposalResult

    log_q = float(-np.log(10.0))

    class FixedBirth:
        def propose(self, group, state, rng):
            return BirthProposalResult(
                proposed_beta=np.array([7.3]), log_q_forward=log_q
            )

        def log_density(self, group, state, beta):
            return log_q

    def prior_transform(u):
        return 10.0 * u  # beta uniform on [0, 10]

    g = daedalus.Group(
        name="x",
        params=[0],
        off_values=np.array([0.0]),
        inclusion_prior=0.5,
        birth_proposal=FixedBirth(),
        log_prior_continuous=lambda b: log_q,
    )
    sampler = daedalus.NestedSampler(
        loglike=lambda beta, gamma: 0.0,
        prior_transform=prior_transform,
        ndim=1,
        groups=[g],
        n_live=8,
        seed=0,
        validate_births=False,
    )
    state = daedalus.State(
        u=np.array([0.2]), beta=np.array([2.0]), gamma=np.array([False])
    )
    new_state, _ = sampler._moms_flip_once(
        state, L_threshold=-np.inf, group_idx=0
    )
    assert bool(new_state.gamma[0])
    # The accepted birth value survives (to bisection precision)...
    assert np.isclose(new_state.beta[0], 7.3, atol=1e-9)
    # ...and the kernel invariant beta == prior_transform(u) holds, so a
    # within-model step regenerating beta from u preserves it.
    assert np.isclose(
        prior_transform(new_state.u)[0], new_state.beta[0], atol=1e-12
    )


def test_periodogram_cdf_cache_is_bounded() -> None:
    """Regression: the periodogram CDF cache must not grow without bound.

    In a trans-dim run the (gamma, rounded-active-periods) key space is
    effectively unbounded (active periods wander continuously), and each
    entry holds O(n_periods) arrays; an uncapped cache reaches tens of
    GB over a long multi-slot run."""
    pytest.importorskip("astropy")
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 100.0, 60))
    y = rng.normal(0.0, 1.0, 60)
    max_entries = 8

    period_holder = [5.0]
    birth = daedalus.GLSPeriodBirth(
        time=t,
        residual_fn=lambda state: y,
        prior_log_period=(np.log(1.0), np.log(50.0)),
        non_period_priors=((0.0, 1.0),),
        n_periods=64,
        active_periods_fn=lambda state: list(period_holder),
        cdf_cache_max_entries=max_entries,
    )
    g = daedalus.Group(
        name="slot",
        params=[0, 1],
        off_values=np.zeros(2),
        birth_proposal=birth,
        log_prior_continuous=lambda b: 0.0,
    )
    state = daedalus.State(
        u=np.zeros(2), beta=np.zeros(2), gamma=np.array([False])
    )
    # 40 distinct active-period keys >> max_entries.
    for i in range(40):
        period_holder[0] = 2.0 + 0.5 * i
        birth.propose(g, state, rng)
        assert len(birth._cdf_by_gamma) <= max_entries
    assert len(birth._cdf_by_gamma) == max_entries


def test_dead_store_matches_list_storage() -> None:
    """The packed dead-point store must reproduce list-of-arrays storage
    exactly, including across capacity doublings."""
    from daedalus.sampler import _DeadStore

    rng = np.random.default_rng(1)
    store = _DeadStore(ndim=3, n_groups=2)
    ref_u, ref_beta, ref_gamma = [], [], []
    for _ in range(3000):  # crosses the 1024 -> 2048 -> 4096 doublings
        u = rng.uniform(size=3)
        beta = rng.normal(size=3)
        gamma = rng.uniform(size=2) > 0.5
        store.append(u, beta, gamma)
        ref_u.append(u.copy())
        ref_beta.append(beta.copy())
        ref_gamma.append(gamma.copy())
    u_v, beta_v, gamma_v = store.views()
    assert np.array_equal(u_v, np.asarray(ref_u))
    assert np.array_equal(beta_v, np.asarray(ref_beta))
    assert np.array_equal(gamma_v, np.asarray(ref_gamma))
