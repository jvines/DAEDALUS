"""``n_mcmc`` is a per-replacement *sweep* budget, and the slice step-out
cannot silently truncate its bracket.

Two properties are pinned here.

**Sweep budget.** ``RandomSliceSampler.step_once`` performs ``n_slices``
slice sweeps, so before the fix an explicit ``n_mcmc=20`` bought 20 walks
from ``rwalk`` but 20 x 5 = 100 sweeps from ``rslice``. Every matched-
``n_mcmc`` kernel comparison therefore handed the slice kernel five times
the work, which is how the WASP-47 head-to-head came to report rslice as
"32x more expensive than dynesty" when the true like-for-like ratio is
~2.8x. ``n_mcmc`` is now converted to ``step_once`` calls through
``Sampler.sweeps_per_step`` at a single point in ``run_nested``, so the
budget means the same thing for every kernel.

**Doubling.** Stepping out linearly by a fixed ``init_scale`` costs
``O(slice_width / init_scale)`` likelihood calls, and on exhausting
``max_steps_out`` the old code simply kept the truncated bracket -- a
silently under-explored slice. Neal (2003) doubling reaches the same
bracket in ``O(log2)`` calls, but is only reversible with the acceptance
test of his Fig. 6, so the two ship together and are validated against a
target whose constrained conditional is known exactly.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import stats

import daedalus
from daedalus.benchmarks import gaussians as gb
from daedalus.bounds import NoBound
from daedalus.samplers import (
    DifferentialEvolutionSampler,
    RandomSliceSampler,
    RandomWalkSampler,
    SliceBracketTruncationWarning,
    UniformSampler,
)
from daedalus.state import State


# --------------------------------------------------------------- sweep budget


def test_sweeps_per_step_reports_kernel_granularity() -> None:
    """Only the slice kernel does more than one sweep per ``step_once``."""
    assert RandomWalkSampler().sweeps_per_step == 1
    assert DifferentialEvolutionSampler().sweeps_per_step == 1
    assert UniformSampler().sweeps_per_step == 1
    assert RandomSliceSampler(n_slices=5).sweeps_per_step == 5
    assert RandomSliceSampler(n_slices=1).sweeps_per_step == 1


def _count_step_once(sampler):
    """Wrap ``step_once`` so the number of invocations is observable."""
    calls = {"n": 0}
    inner = sampler.step_once

    def counting(*a, **kw):
        calls["n"] += 1
        return inner(*a, **kw)

    sampler.step_once = counting  # type: ignore[method-assign]
    return calls


@pytest.mark.parametrize("n_slices,expected_steps", [(5, 4), (2, 10), (1, 20)])
def test_explicit_n_mcmc_is_a_sweep_budget(n_slices, expected_steps) -> None:
    """``n_mcmc=20`` must buy ~20 sweeps whatever ``n_slices`` is.

    Before the fix this bought 20 * n_slices sweeps, i.e. 100 at the
    default n_slices=5.
    """
    prob = gb.make_problem(ndim=2, prior_half_width=10.0)
    smp = RandomSliceSampler(n_slices=n_slices)
    s = daedalus.NestedSampler(
        loglike=prob.loglike, prior_transform=prob.prior_transform, ndim=2,
        bound="single", sample=smp, n_live=60, seed=0,
    )
    counter = _count_step_once(smp)
    res = s.run_nested(dlogz=5.0, n_mcmc=20, show_progress=False)

    # res.n_iter counts replacements; log_likelihoods also carries the final
    # live cloud, so it is n_iter + n_live and must not be used here.
    steps_per_replacement = counter["n"] / res.n_iter
    assert steps_per_replacement == pytest.approx(expected_steps, rel=1e-9)
    # Total sweeps land on the requested budget (exactly, since 20 is
    # divisible by every n_slices tested here).
    assert steps_per_replacement * n_slices == pytest.approx(20, rel=1e-9)


def test_rslice_recommended_budget_is_expressed_in_sweeps() -> None:
    """The recommendation is a sweep count, not a ``step_once`` count.

    The division by ``n_slices`` now happens once inside ``run_nested``,
    so the recommendation itself is directly comparable to rwalk's.
    """
    smp = RandomSliceSampler(n_slices=5)
    for ndim in (2, 4, 11, 20):
        assert smp.recommended_n_mcmc(ndim) == max(25, 3 * ndim)
    # ... and it is independent of n_slices, which only sets granularity.
    assert (
        RandomSliceSampler(n_slices=1).recommended_n_mcmc(11)
        == RandomSliceSampler(n_slices=7).recommended_n_mcmc(11)
    )


def test_default_sweep_total_is_unchanged_by_the_refactor() -> None:
    """Regression guard on the shipped default.

    The old code returned ``ceil(max(25, 3*ndim) / n_slices)`` steps of
    ``n_slices`` sweeps. The new code returns the sweep target and divides
    in ``run_nested``. Both land on the same effective sweep total, which is
    what keeps existing default-configuration results reproducible.
    """
    n_slices = 5
    smp = RandomSliceSampler(n_slices=n_slices)
    for ndim in (2, 4, 5, 11, 20):
        old_steps = math.ceil(max(25, 3 * ndim) / n_slices)
        new_steps = math.ceil(smp.recommended_n_mcmc(ndim) / smp.sweeps_per_step)
        assert new_steps == old_steps


# ------------------------------------------------------------------- doubling


def _one_dim_slice_kernel(init_scale, doubling, max_steps_out=100):
    """A 1-D constrained-MCMC kernel whose target is known exactly.

    ``log L(x) = -((x - 0.5) / 0.1)**2`` on a uniform prior over u in [0, 1]
    with the identity prior transform. Restricted to ``log L > L*`` the
    constrained prior is *uniform* on the interval
    ``0.5 +- 0.1 * sqrt(-L*)``, so a correct kernel leaves that uniform
    distribution invariant.
    """
    smp = RandomSliceSampler(
        n_slices=1, init_scale=init_scale, doubling=doubling,
        max_steps_out=max_steps_out,
    )
    smp.bind(
        loglike=lambda beta, gamma: -(((beta[0] - 0.5) / 0.1) ** 2),
        prior_transform=lambda u: u,
        ndim=1,
    )
    return smp


def _sample_chain(smp, L_threshold, n_steps, seed):
    rng = np.random.default_rng(seed)
    bound = NoBound(1)
    u = np.array([0.5])
    state = State(u=u, beta=u.copy(), log_likelihood=0.0)
    out = []
    for _ in range(n_steps):
        state, _ = smp.step_once(state, L_threshold, bound, rng)
        out.append(float(state.u[0]))
    return np.asarray(out)


@pytest.mark.parametrize("doubling", [True, False])
def test_slice_kernel_leaves_the_constrained_prior_invariant(doubling) -> None:
    """Both step-out schemes must sample the constrained prior exactly.

    ``init_scale`` is deliberately 20x narrower than the slice so the
    doubling path is actually exercised (and so the linear path has to
    step out many times).
    """
    L_threshold = -4.0                    # slice is 0.5 +- 0.2
    lo, hi = 0.5 - 0.2, 0.5 + 0.2
    smp = _one_dim_slice_kernel(init_scale=0.01, doubling=doubling)
    x = _sample_chain(smp, L_threshold, n_steps=6000, seed=17)

    assert np.all(x > lo - 1e-9) and np.all(x < hi + 1e-9)
    p = stats.kstest((x - lo) / (hi - lo), "uniform").pvalue
    assert p > 0.005, f"doubling={doubling}: constrained prior not uniform, KS p={p:.4g}"


def test_doubling_reaches_the_bracket_far_cheaper_than_linear_stepout() -> None:
    """The point of doubling: O(log2) instead of O(width / init_scale)."""
    L_threshold = -4.0
    calls = {}
    for doubling in (False, True):
        smp = _one_dim_slice_kernel(init_scale=1e-3, doubling=doubling)
        rng = np.random.default_rng(5)
        bound = NoBound(1)
        u = np.array([0.5])
        state = State(u=u, beta=u.copy(), log_likelihood=0.0)
        total = 0
        # The fixed-step arm truncates by construction here; that it warns is
        # covered elsewhere, this test is only about the call count.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SliceBracketTruncationWarning)
            for _ in range(300):
                state, n = smp.step_once(state, L_threshold, bound, rng)
                total += n
        calls[doubling] = total / 300.0
    assert calls[True] < 0.5 * calls[False], (
        f"doubling {calls[True]:.1f} calls/sweep vs linear {calls[False]:.1f}"
    )


def test_fixed_step_stepout_is_the_default() -> None:
    """Doubling costs 1.6-1.8x the likelihood calls and changes the kernel for
    a whole run, so it is opt-in; the default stays on Neal Fig. 3."""
    assert RandomSliceSampler().doubling is False
    assert RandomSliceSampler().doubling_active is False


def test_bracket_truncation_warns_instead_of_passing_silently() -> None:
    """The failure this guards: ``max_steps_out`` exhausted with the endpoint
    still inside the slice means the sweep sampled a bracket smaller than its
    slice. It used to be indistinguishable from a healthy run."""
    # init_scale 1e-4 against a 0.4-wide slice needs ~2000 fixed steps but is
    # capped at 5.
    narrow = _one_dim_slice_kernel(init_scale=1e-4, doubling=False,
                                   max_steps_out=5)
    with pytest.warns(SliceBracketTruncationWarning, match="max_steps_out"):
        _sample_chain(narrow, -4.0, n_steps=50, seed=1)
    # Reported, not repaired: the kernel is unchanged.
    assert narrow.doubling_active is False

    # A bracket already wider than the slice never truncates and must stay
    # quiet -- otherwise the warning is noise on every healthy run.
    wide = _one_dim_slice_kernel(init_scale=1.0, doubling=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SliceBracketTruncationWarning)
        _sample_chain(wide, -4.0, n_steps=200, seed=1)


def test_auto_doubling_latches_on_after_the_stepout_cap_is_hit() -> None:
    """``doubling='auto'`` is the opt-in self-repair: same warning, but it
    also switches the rest of the run onto doubling."""
    wide = _one_dim_slice_kernel(init_scale=1.0, doubling="auto")
    _sample_chain(wide, -4.0, n_steps=200, seed=1)
    assert wide.doubling_active is False

    narrow = _one_dim_slice_kernel(init_scale=1e-4, doubling="auto",
                                   max_steps_out=5)
    with pytest.warns(SliceBracketTruncationWarning):
        _sample_chain(narrow, -4.0, n_steps=50, seed=1)
    assert narrow.doubling_active is True


def test_doubling_rejects_an_unrecognised_mode() -> None:
    with pytest.raises(ValueError, match="doubling must be"):
        RandomSliceSampler(doubling="yes")


def test_auto_doubling_recovers_the_slice_the_cap_would_have_truncated() -> None:
    """With a step-out cap too small for the slice, the linear scheme is
    confined near its starting point while auto-doubling still covers the
    whole constrained interval.

    ``max_steps_out=20`` is chosen to separate the two: the fixed-step scheme
    can only reach ``20 * 1e-4`` either side, whereas 20 doublings of the same
    ``init_scale`` span far more than the 0.4-wide slice. This is exactly the
    silent truncation the cap used to cause.
    """
    L_threshold = -4.0
    lo, hi = 0.3, 0.7
    kw = dict(init_scale=1e-4, max_steps_out=20)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SliceBracketTruncationWarning)
        linear = _sample_chain(
            _one_dim_slice_kernel(doubling=False, **kw), L_threshold, 200, seed=2
        )
        auto = _sample_chain(
            _one_dim_slice_kernel(doubling="auto", **kw), L_threshold, 200, seed=2
        )

    assert np.ptp(linear) < 0.15 * (hi - lo), (
        f"linear step-out unexpectedly explored {np.ptp(linear):.4f}"
    )
    assert np.ptp(auto) > 0.7 * (hi - lo), (
        f"auto-doubling only explored {np.ptp(auto):.4f} of {hi - lo}"
    )


@pytest.mark.benchmark
def test_forced_doubling_recovers_the_analytic_evidence() -> None:
    """End-to-end: forcing doubling on must not shift the evidence."""
    ndim = 5
    prob = gb.make_problem(ndim=ndim, prior_half_width=10.0)
    s = daedalus.NestedSampler(
        loglike=prob.loglike, prior_transform=prob.prior_transform, ndim=ndim,
        bound="multi", sample=RandomSliceSampler(doubling=True),
        n_live=400, seed=4,
    )
    res = s.run_nested(dlogz=0.1, show_progress=False)
    assert res.log_Z == pytest.approx(prob.log_Z_true, abs=0.5), (
        f"doubling logZ={res.log_Z:.3f} vs analytic {prob.log_Z_true:.3f}"
    )
