"""Main user-facing entry point: `NestedSampler`.

The single algorithm is MoMS-NS -- Skilling nested sampling on the joint
state space X = R^ndim x {0, 1}^n_groups, with a constrained-MCMC kernel
that mixes within-model continuous moves (handled by a `Sampler`) and
trans-dim moves (the MoMS flip kernel implemented here).

When `groups=()` (the default), no toggleable structure is declared, the
trans-dim kernel is quiescent, and the sampler runs as standard nested
sampling. When `groups` is non-empty, MoMS flips fire with frequency
`transdim_fraction` per constrained-MCMC step.

User-supplied `loglike` may accept either `(beta)` -- in which case the
chain's continuous coordinates fully describe the likelihood -- or
`(beta, gamma)` -- in which case the indicator vector is also passed,
allowing the user to evaluate likelihoods that depend explicitly on
which model components are active (e.g. variable-selection problems
where beta is integrated out analytically and only the active subset
matters).
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

from .bounds import BoundingRegion, make_bound
from .groups import Group
from .live_points import LivePoints
from .recursion import SkillingAccumulator
from .results import Results
from .samplers import Sampler, make_sampler
from .state import State


LogLikelihood = Callable[..., float]
PriorTransform = Callable[[np.ndarray], np.ndarray]


class NSProgress(NamedTuple):
    """Raw per-iteration state of a running NS chain.

    Handed to the ``on_progress`` callback of `NestedSampler.run_nested`
    once per iteration. This is deliberately the *unprocessed* state that
    also feeds the tqdm bar -- no percentage, no phase label, no rate, no
    formatted string. Presentation is the consumer's, so that daedalus
    does not acquire a display schema it would then have to keep forever.

    Note `log_dlogz` is ``log(dlogz)``, not ``dlogz``: it is the quantity
    the termination test actually compares `gap` against, so a consumer
    computes ``gap - log_dlogz`` to judge how far along the run is
    without having to guess whether this sampler's stopping target lives
    in log or linear space.
    """

    n_iter: int          # NS iterations completed (1-based at first call)
    log_Z: float         # running log evidence, live points not yet folded in
    log_Z_err: float     # Skilling's sqrt(H / n_live) estimate on log_Z
    gap: float           # log(X_curr * max_live L) - log Z; run stops when
                         # this drops below log_dlogz
    log_L_star: float    # current likelihood threshold (the dead point's L)
    n_calls: int         # cumulative likelihood evaluations
    log_dlogz: float     # the stopping target, log(dlogz) -- NOT dlogz
    n_live: int          # current live-cloud size (may grow if gamma_adaptive)


ProgressCallback = Callable[[NSProgress], None]


def _make_loglike_wrapper(loglike: LogLikelihood) -> Callable[[np.ndarray, np.ndarray], float]:
    """Wrap user loglike so it always accepts (beta, gamma).

    Detects whether the user's callable wants 1 or 2 positional args.
    """
    sig = inspect.signature(loglike)
    n_params = sum(
        1
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    )
    if n_params == 1:
        def wrapper(beta: np.ndarray, gamma: np.ndarray) -> float:
            return float(loglike(beta))
    elif n_params == 2:
        def wrapper(beta: np.ndarray, gamma: np.ndarray) -> float:
            return float(loglike(beta, gamma))
    else:
        raise ValueError(
            f"loglike must accept either (beta) or (beta, gamma); "
            f"detected {n_params} required positional args"
        )
    return wrapper


class _DeadStore:
    """Packed, amortised-doubling storage for the dead-point cloud.

    Storing each dead point as its own small ndarray in a Python list
    costs several times the raw data in per-object overhead and forces a
    full contiguous copy at result-build time; for million-iteration
    runs that is gigabytes. Packing rows into doubling 2-D arrays keeps
    memory at the raw-data floor and makes result construction a view.
    """

    def __init__(self, ndim: int, n_groups: int) -> None:
        self._cap = 1024
        self.u = np.empty((self._cap, ndim), dtype=np.float64)
        self.beta = np.empty((self._cap, ndim), dtype=np.float64)
        self.gamma = np.empty((self._cap, n_groups), dtype=np.bool_)
        self.n = 0

    def append(
        self, u: np.ndarray, beta: np.ndarray, gamma: np.ndarray
    ) -> None:
        if self.n == self._cap:
            self._cap *= 2
            for name in ("u", "beta", "gamma"):
                old = getattr(self, name)
                new = np.empty((self._cap, old.shape[1]), dtype=old.dtype)
                new[: self.n] = old[: self.n]
                setattr(self, name, new)
        self.u[self.n] = u
        self.beta[self.n] = beta
        self.gamma[self.n] = gamma
        self.n += 1

    def views(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.u[: self.n], self.beta[: self.n], self.gamma[: self.n]


class NestedSampler:
    def __init__(
        self,
        loglike: LogLikelihood,
        prior_transform: PriorTransform,
        ndim: int,
        groups: Sequence[Group] = (),
        bound: str | BoundingRegion = "multi",
        sample: str | Sampler = "rwalk",
        n_live: int = 400,
        periodic: Sequence[int] | None = None,
        seed: int | None = None,
        validate_births: bool = True,
    ) -> None:
        if ndim < 1:
            raise ValueError(f"ndim must be >= 1, got {ndim}")
        if n_live < 2:
            raise ValueError(f"n_live must be >= 2, got {n_live}")

        self.loglike = _make_loglike_wrapper(loglike)
        self.prior_transform = prior_transform
        self.ndim = ndim
        self.groups = tuple(groups)
        self.n_live = n_live
        self.periodic = tuple(periodic) if periodic is not None else ()
        self.rng = np.random.default_rng(seed)

        self.bound: BoundingRegion = (
            make_bound(bound, ndim) if isinstance(bound, str) else bound
        )
        self.sampler: Sampler = (
            make_sampler(sample) if isinstance(sample, str) else sample
        )

        self._validate_groups()

        # Catch the most common user error in custom-birth setups:
        # log_prior_continuous that doesn't match the pushforward of
        # prior_transform on the group's coords. Silent miscalibration
        # otherwise -- see daedalus/validation.py.
        if validate_births:
            from .validation import warn_if_birth_inconsistent

            for g in self.groups:
                warn_if_birth_inconsistent(
                    g, prior_transform, ndim, n_samples=200, seed=0
                )
        # Samplers see the (beta, gamma) -wrapped loglike so they don't need
        # to inspect signatures themselves; they also receive the off-value
        # applier so within-model proposals stay consistent with gamma when
        # some groups are inactive.
        self.sampler.bind(
            loglike=self.loglike,
            prior_transform=prior_transform,
            ndim=ndim,
            groups=self.groups,
            periodic=self.periodic,
            apply_off_values=self._apply_off_values,
        )

        # Pre-extract per-group inclusion priors and parameter index arrays
        # for the trans-dim hot loop (avoids per-iter attribute lookups).
        self._inclusion_priors = np.asarray(
            [g.inclusion_prior for g in self.groups], dtype=np.float64
        )
        self._log_inc = np.log(self._inclusion_priors) if self.groups else np.empty(0)
        self._log_exc = np.log1p(-self._inclusion_priors) if self.groups else np.empty(0)
        self._group_params = [np.asarray(g.params, dtype=np.intp) for g in self.groups]
        self._group_off_values = [g.off_values.copy() for g in self.groups]

        # Build a CSR-style flat layout of (group_idx -> beta indices,
        # off-values). Used by the numba-JIT apply_off_values kernel so the
        # spike-and-slab override doesn't iterate over Python objects in
        # the hot loop.
        sizes = [p.size for p in self._group_params]
        self._group_starts = np.zeros(len(self.groups) + 1, dtype=np.intp)
        if sizes:
            np.cumsum(sizes, out=self._group_starts[1:])
            self._group_param_idx = np.concatenate(self._group_params).astype(
                np.intp
            )
            self._group_off_values_flat = np.concatenate(
                self._group_off_values
            ).astype(np.float64)
        else:
            self._group_param_idx = np.empty(0, dtype=np.intp)
            self._group_off_values_flat = np.empty(0, dtype=np.float64)

        try:
            from . import _kernels  # noqa: F401

            self._apply_off_values_jit = _kernels.apply_off_values_flat
        except Exception:  # pragma: no cover
            self._apply_off_values_jit = None

    # ---------------- validation ------------------------------------------

    def _validate_groups(self) -> None:
        all_indices: set[int] = set()
        for g in self.groups:
            for i in g.params:
                if not (0 <= i < self.ndim):
                    raise ValueError(
                        f"group {g.name!r} references parameter index {i} "
                        f"outside [0, {self.ndim})"
                    )
                if i in all_indices:
                    raise ValueError(
                        f"parameter index {i} appears in more than one group; "
                        f"each toggleable parameter must belong to exactly one group"
                    )
                all_indices.add(i)

    # ---------------- prior bookkeeping -----------------------------------

    def _log_inclusion_prior(self, gamma: np.ndarray) -> float:
        if not self.groups:
            return 0.0
        return float(np.where(gamma, self._log_inc, self._log_exc).sum())

    def _apply_off_values(self, beta: np.ndarray, gamma: np.ndarray) -> None:
        if self._apply_off_values_jit is not None and self.groups:
            self._apply_off_values_jit(
                beta,
                gamma,
                self._group_starts,
                self._group_param_idx,
                self._group_off_values_flat,
            )
            return
        for k, group_params in enumerate(self._group_params):
            if not gamma[k]:
                beta[group_params] = self._group_off_values[k]

    # ---------------- live-point initialisation ---------------------------

    def _initialise_live_points(self) -> LivePoints:
        n_groups = len(self.groups)
        live = LivePoints.empty(self.n_live, self.ndim, n_groups)
        live.u[:] = self.rng.uniform(size=(self.n_live, self.ndim))
        if n_groups > 0:
            for k, p in enumerate(self._inclusion_priors):
                live.gamma[:, k] = self.rng.uniform(size=self.n_live) < p
        for i in range(self.n_live):
            beta = np.asarray(self.prior_transform(live.u[i]), dtype=np.float64)
            if n_groups > 0:
                self._apply_off_values(beta, live.gamma[i])
            live.beta[i] = beta
            live.log_likelihood[i] = self.loglike(beta, live.gamma[i])
            live.log_prior[i] = self._log_inclusion_prior(live.gamma[i])
        return live

    # ---------------- main NS loop ----------------------------------------

    def run_nested(
        self,
        dlogz: float = 0.5,
        n_mcmc: int | None = None,
        transdim_fraction: float = 0.3,
        max_iter: int | None = None,
        show_progress: bool = True,
        bound_update_interval: int | None = None,
        gamma_adaptive: bool = False,
        gamma_speciation_interval: int = 50,
        gamma_c_min_factor: float = 0.005,
        gamma_c_min_floor: int = 5,
        gamma_max_growth_factor: float = 4.0,
        gamma_max_promotions_per_call: int = 3,
        gamma_speciation_n_mcmc: int = 0,
        insertion_recorder: list | None = None,
        rejuvenate_fraction: float = 0.0,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> Results:
        """Run nested sampling until termination.

        Parameters
        ----------
        n_mcmc
            Per-replacement budget of elementary within-model MCMC
            *sweeps* -- random-walk / DE steps, or slice sweeps. It is
            quoted in sweeps rather than in ``step_once`` calls so that the
            same value buys the same mixing effort from every kernel:
            ``RandomSliceSampler.step_once`` performs ``n_slices`` sweeps,
            so the budget is divided by ``Sampler.sweeps_per_step`` (rounding
            up) before the loop runs. ``None`` (default) uses each kernel's
            own dimension-aware recommendation, ``recommended_n_mcmc(ndim)``.
        gamma_adaptive
            When True, enables gamma-adaptive MoMS-NS: the live cloud is
            periodically rebalanced across gamma-configurations by
            promoting donor live points into under-represented gamma-
            configurations via directed trans-dim moves. The cloud size
            grows by one per successful promotion; the Skilling X-recursion
            uses the running ``n_live(i)`` at each iteration. Off by
            default — the standard fixed-cloud MoMS-NS is unchanged.
        gamma_speciation_interval
            Run a speciation step every this many NS iterations.
        gamma_c_min_factor, gamma_c_min_floor
            A gamma-configuration is considered under-represented when its
            live-point count is below ``max(gamma_c_min_floor,
            gamma_c_min_factor * n_live_initial)``.
        gamma_max_growth_factor
            Cap the live cloud at ``gamma_max_growth_factor * n_live_initial``
            so the adaptive growth has a finite upper bound. The chain
            terminates speciation attempts once this cap is reached;
            subsequent NS iterations proceed at the capped cloud size.
        gamma_max_promotions_per_call
            Cap the number of promotion attempts per speciation step. The
            ``dead_gamma_seen`` set grows monotonically over the run, so
            without this cap the speciation work grows linearly with run
            length and eventually dominates. Default of 3 keeps each
            speciation step bounded; speciation fires every
            ``gamma_speciation_interval`` iterations so the chain still
            gets many promotion attempts overall.
        gamma_speciation_n_mcmc
            Number of within-model MCMC sweeps to mix a newly promoted
            live point before adding it to the cloud (same unit as
            ``n_mcmc``). Default 0: skip the mix and rely on subsequent NS
            iterations to decorrelate the appended point from its donor.
            Setting > 0 enables a mix but risks stalling on a stale bound
            covariance.
        rejuvenate_fraction
            Probability per within-model step of attempting a *rejuvenation*
            move: re-proposing an already-active group's continuous block
            from its ``BirthProposal`` without toggling ``gamma`` (see
            ``_rejuvenate_once``). Default 0.0 -- off, and the run is then
            bit-identical to a build without this feature.

            Use it for DISCOVERY, not for calibrated inference. Measured on
            the WASP-47 single-planet trans-dim benchmark with the DE kernel
            over 6 seeds, ``rejuvenate_fraction=0.3`` raises the number of
            runs reaching the true global mode (independently confirmed at
            max log L = -4624.48 by multi-start optimisation) from 2/6 to
            5/6, improves mean max log L by ~710 nats, and does so at 3.6%
            FEWER likelihood calls on average -- a better-converging chain
            terminates sooner. It is not a guarantee: one of the six seeds
            ended 125 nats worse than without it. For comparison the slice
            kernel also solves this problem but needs ~44x more likelihood
            calls.

            The cost: on the analytic spike-and-slab (closed-form log Z and
            inclusion probability, 5 seeds, GaussianRWBirth) the setting that
            fixes the mode search, 0.3, also shows a small but statistically
            real bias -- log Z -0.058 (4.3 sigma) and P(gamma=1) +0.022 --
            while 0.0 is unbiased (-0.0007, 4.3 sigma smaller than its own
            seed scatter). Fractions <= 0.1 show no detectable bias but also
            do not fix the mode search. So: rejuvenate to FIND the mode, then
            re-run with rejuvenate_fraction=0 (or a tight confirmatory prior,
            cf. ``benchmarks.wasp47.make_problem_tight``) to quote evidence
            and inclusion probabilities.
        on_progress
            Optional callback invoked once per NS iteration with an
            `NSProgress` record of the raw chain state — the same state
            the tqdm bar renders, handed over unformatted so a caller can
            drive logging or a UI without owning a terminal. Independent
            of ``show_progress``: either, both, or neither may be active.
            The callback is observation-only from the sampler's side (it
            draws no random numbers and evaluates no likelihoods), so a
            run is bit-identical with and without one attached. Exceptions
            raised inside the callback are *not* caught — they propagate
            out of ``run_nested`` rather than silently hiding a bug in the
            consumer's sink.
        """
        if dlogz <= 0.0:
            raise ValueError(f"dlogz must be positive, got {dlogz}")
        if not (0.0 <= transdim_fraction <= 1.0):
            raise ValueError(
                f"transdim_fraction must be in [0, 1], got {transdim_fraction}"
            )
        if bound_update_interval is None:
            # Refit the bound every ~n_live/20 iterations rather than every
            # iteration: over that span the live cloud contracts by only
            # ~exp(-1/20) ~ 5% in volume, so the enlarged bound stays a
            # valid (looser) superset and proposal efficiency is unchanged,
            # while the expensive multi-ellipsoid k-means refit runs ~20x
            # less often. Measured 2.5-3.5x wall-clock speedup on fixed-dim
            # benchmarks with no change in likelihood-call count or evidence.
            # The bound only sets the proposal region, so this never biases
            # the evidence; it is purely an efficiency knob. Pass an explicit
            # integer to override (e.g. 1 to refit every iteration).
            bound_update_interval = max(1, self.n_live // 20)
        if bound_update_interval < 1:
            raise ValueError(
                f"bound_update_interval must be >= 1, got {bound_update_interval}"
            )
        if n_mcmc is None:
            # Per-kernel default: the within-model decorrelation time grows
            # ~linearly with dimension for random-walk / DE (a fixed step
            # count under-mixes and biases the evidence high in high-d), so
            # each kernel reports its own dimension-aware recommendation
            # (rwalk/DE: 5*ndim; slice: fewer, dimension-robust sweeps;
            # uniform: 1). Pass an explicit integer to override.
            n_mcmc = self.sampler.recommended_n_mcmc(self.ndim)
        if n_mcmc < 1:
            raise ValueError(f"n_mcmc must be >= 1, got {n_mcmc}")
        # ``n_mcmc`` is a per-replacement budget of elementary MCMC *sweeps*,
        # which is not the same as a count of ``step_once`` calls: a slice
        # ``step_once`` performs ``n_slices`` sweeps, every other kernel one.
        # Converting here, at the single point where the budget enters the
        # loop, is what makes the same ``n_mcmc`` buy the same mixing effort
        # from every kernel. Without it an explicit ``n_mcmc`` silently gave
        # rslice ``n_slices`` times the work of rwalk, so kernel comparisons
        # at matched ``n_mcmc`` were not matched at all (this is the origin of
        # the reported "rslice needs 32x dynesty's likelihood calls" on the
        # WASP-47 benchmark; like-for-like it is ~2.8x).
        sweeps_per_step = max(1, int(self.sampler.sweeps_per_step))
        n_steps = max(1, -(-n_mcmc // sweeps_per_step))  # ceil div
        # Same conversion for the (opt-in, default-off) post-promotion mix, so
        # both budgets are quoted in the same unit.
        spec_n_steps = (
            0 if gamma_speciation_n_mcmc <= 0
            else max(1, -(-gamma_speciation_n_mcmc // sweeps_per_step))
        )
        if gamma_adaptive and not self.groups:
            raise ValueError(
                "gamma_adaptive=True requires at least one toggleable Group"
            )
        if gamma_speciation_interval < 1:
            raise ValueError(
                f"gamma_speciation_interval must be >= 1, got "
                f"{gamma_speciation_interval}"
            )
        if gamma_max_growth_factor < 1.0:
            raise ValueError(
                f"gamma_max_growth_factor must be >= 1, got "
                f"{gamma_max_growth_factor}"
            )

        live = self._initialise_live_points()
        acc = SkillingAccumulator(n_live=self.n_live)
        log_dlogz = float(np.log(dlogz))
        n_calls = self.n_live

        # gamma-adaptive: track the set of gamma-tuples ever seen in the
        # dead-point cloud (so we know which under-represented configs
        # are "real" — i.e. the chain has been there at least once).
        dead_gamma_seen: set[tuple[bool, ...]] = set()
        c_min = max(
            gamma_c_min_floor, int(round(gamma_c_min_factor * self.n_live))
        )
        max_n_live = int(round(gamma_max_growth_factor * self.n_live))
        gamma_bounds: dict | None = None

        dead = _DeadStore(self.ndim, len(self.groups))

        pbar = self._make_progress_bar(show_progress, max_iter)
        try:
            while True:
                if max_iter is not None and acc.n_iter >= max_iter:
                    break

                dead_idx = live.argmin_log_L()
                L_thr = float(live.log_likelihood[dead_idx])

                dead.append(
                    live.u[dead_idx], live.beta[dead_idx], live.gamma[dead_idx]
                )
                if gamma_adaptive:
                    dead_gamma_seen.add(tuple(bool(x) for x in live.gamma[dead_idx]))
                # Pass the CURRENT live count for X-shrinkage so adaptive
                # growth is reflected in the recursion immediately.
                acc.add_dead(L_thr, n_live=live.n_live)

                if (acc.n_iter - 1) % bound_update_interval == 0:
                    self.bound.fit(live.u, log_volume_prior=acc.log_X_prev)
                    # Hand the live cloud to samplers that need it (e.g. DE
                    # uses pairs of other live points to build proposals).
                    # Refresh on the same cadence as the bound -- the cloud's
                    # geometry only changes meaningfully at that rate.
                    set_live = getattr(self.sampler, "set_live_points", None)
                    if set_live is not None:
                        set_live(live.u)
                    if gamma_adaptive:
                        gamma_bounds = self._fit_per_gamma_bounds(
                            live, acc.log_X_prev
                        )

                donor_idx = self._pick_donor_dynamic(dead_idx, live.n_live)
                donor = live.snapshot(donor_idx)
                new_state, n_attempts = self._replacement_kernel(
                    donor, L_thr, n_steps, transdim_fraction,
                    gamma_bounds=gamma_bounds,
                    rejuvenate_fraction=rejuvenate_fraction,
                )
                n_calls += n_attempts

                # Insertion-index diagnostic (Fowlie, Handley & Su 2020).
                # The newborn's rank among the n_live-1 *surviving* live
                # points (the dying one at dead_idx is about to be
                # overwritten and is excluded). Under correct NS sampling
                # this rank is Uniform{0, ..., n_live-1}; under-mixing skews
                # it low. Recorded only when a recorder list is supplied, so
                # the default hot loop is untouched. O(n_live) per iter.
                if insertion_recorder is not None:
                    L_new = float(new_state.log_likelihood)
                    n_below = 0
                    for j in range(live.n_live):
                        if j == dead_idx:
                            continue
                        if live.log_likelihood[j] < L_new:
                            n_below += 1
                    insertion_recorder.append(n_below)

                live.replace(dead_idx, new_state)

                # gamma-adaptive speciation: every K iterations, attempt
                # to promote donor points into under-represented gamma-
                # configurations seen in dead points. n_live grows on each
                # successful promotion.
                if (
                    gamma_adaptive
                    and acc.n_iter % gamma_speciation_interval == 0
                    and live.n_live < max_n_live
                ):
                    n_promoted, calls_promoted = self._gamma_speciate(
                        live, L_thr, n_steps, c_min, max_n_live,
                        dead_gamma_seen,
                        max_promotions_per_call=gamma_max_promotions_per_call,
                        speciation_n_mcmc=spec_n_steps,
                    )
                    n_calls += calls_promoted

                gap = acc.termination_gap(live.max_log_L())
                if pbar is not None:
                    self._update_progress_bar(
                        pbar, acc, gap, L_thr, n_calls, log_dlogz,
                        n_live_current=live.n_live,
                    )
                if on_progress is not None:
                    on_progress(
                        NSProgress(
                            n_iter=acc.n_iter,
                            log_Z=acc.log_Z,
                            log_Z_err=acc.log_Z_err,
                            gap=gap,
                            log_L_star=L_thr,
                            n_calls=n_calls,
                            log_dlogz=log_dlogz,
                            n_live=live.n_live,
                        )
                    )
                if gap < log_dlogz:
                    break
        finally:
            if pbar is not None:
                pbar.close()

        for i in range(live.n_live):
            dead.append(live.u[i], live.beta[i], live.gamma[i])
        acc.add_remaining_live(live.log_likelihood)

        return self._build_results(acc, *dead.views(), n_calls)

    def _pick_donor(self, dead_idx: int) -> int:
        idx = int(self.rng.integers(self.n_live - 1))
        return idx if idx < dead_idx else idx + 1

    def _pick_donor_dynamic(self, dead_idx: int, n_live_current: int) -> int:
        """Like ``_pick_donor`` but uses the current live count (which may
        have grown under gamma-adaptive MoMS-NS) rather than the initial
        ``self.n_live``."""
        idx = int(self.rng.integers(n_live_current - 1))
        return idx if idx < dead_idx else idx + 1

    # ---------------- gamma-adaptive: per-gamma bounds --------------------

    def _fit_per_gamma_bounds(
        self,
        live: "LivePoints",
        log_X_prev: float,
    ) -> dict[tuple[bool, ...], "BoundingRegion"]:
        """Fit a gamma-scoped ellipsoid per gamma-tuple in the live cloud.

        The within-model proposal kernel reads its Cholesky factor from
        the bound, which sets the proposal geometry. With a single
        global bound the geometry is the union-of-all-gamma-configs
        ellipsoid; for any individual gamma-config that ellipsoid is
        wider than the local cloud requires, so proposal direction is
        wrong and the chain mixes inefficiently in that gamma-config.

        Each per-gamma bound is a `GammaScopedEllipsoid` fit only on the
        *active* u-coordinates of the gamma-config (the params of active
        groups plus always-on dims that aren't in any group). Inactive
        dims get an axis-aligned proposal width derived from the
        cluster's own spread along those dims. This makes the bound
        rank-stable with far fewer cluster points than a full-ndim
        ellipsoid fit would require.
        """
        from collections import defaultdict
        from .bounds import GammaScopedEllipsoid

        bounds_by_gamma: dict[tuple[bool, ...], "BoundingRegion"] = {}
        if not self.groups:
            return bounds_by_gamma

        # Always-on dims: those that aren't in any toggleable group.
        in_some_group: set[int] = set()
        for g in self.groups:
            in_some_group.update(int(p) for p in g.params)
        always_on_dims = [d for d in range(self.ndim) if d not in in_some_group]

        gamma_tuples = [tuple(bool(x) for x in row) for row in live.gamma]
        indices_by_gamma: dict[tuple[bool, ...], list[int]] = defaultdict(list)
        for i, g in enumerate(gamma_tuples):
            indices_by_gamma[g].append(i)

        # Need at least a couple of points to even compute a sample
        # spread for the inactive dims; ellipsoid fit on the active
        # subspace handles its own under-determination internally.
        MIN_CLUSTER = 3

        for gamma_tuple, indices in indices_by_gamma.items():
            if len(indices) < MIN_CLUSTER:
                continue
            active_dims = list(always_on_dims)
            for k, on in enumerate(gamma_tuple):
                if on:
                    active_dims.extend(int(p) for p in self.groups[k].params)
            try:
                bound = GammaScopedEllipsoid(
                    self.ndim, active_dims, enlargement=1.25
                )
                bound.fit(live.u[indices], log_volume_prior=log_X_prev)
                bounds_by_gamma[gamma_tuple] = bound
            except Exception:
                continue

        return bounds_by_gamma

    # ---------------- gamma-adaptive speciation ---------------------------

    def _gamma_speciate(
        self,
        live: "LivePoints",
        L_threshold: float,
        n_steps: int,
        c_min: int,
        max_n_live: int,
        dead_gamma_seen: set,
        max_promotions_per_call: int = 3,
        speciation_n_mcmc: int = 0,
    ) -> tuple[int, int]:
        """Rebalance the live cloud by promoting donor points into
        under-represented gamma-configurations.

        For each gamma-configuration with current live count < ``c_min``
        that has been visited in the dead-point cloud, find a well-populated
        one-flip neighbour, apply the directed trans-dim flip on the
        differing slot, run a short within-model mix, and append the
        result if it satisfies the likelihood constraint. The standard
        MoMS M-H acceptance is preserved so the appended point is a valid
        sample from the constrained prior.

        Returns ``(n_promoted, n_calls)``.
        """
        from collections import Counter

        n_groups = len(self.groups)
        n_calls = 0
        n_promoted = 0

        # Per-gamma threshold scales with the gamma-config's active
        # continuous dimensionality so that high-dim gamma-configs
        # require proportionally more live points to be considered
        # "well-populated" — a flat threshold under-resolves high-dim
        # configs (10 points in 12 dims is sparse) and over-resolves
        # low-dim ones.
        group_dims = np.array(
            [g.ndim for g in self.groups], dtype=np.int64
        )

        def c_min_for(gamma_tuple: tuple[bool, ...]) -> int:
            active_dim = int(
                sum(group_dims[k] for k, v in enumerate(gamma_tuple) if v)
            )
            # base + one live point per active continuous dim
            return max(c_min, c_min + active_dim)

        # Bin the current live cloud by gamma tuple.
        counts: Counter = Counter(
            tuple(bool(x) for x in row) for row in live.gamma
        )
        well_pop = {g: c for g, c in counts.items() if c >= c_min_for(g)}
        if not well_pop:
            return 0, 0

        # Identify under-represented gammas: seen in dead points but
        # under-populated in the live cloud (with dimensionality-aware
        # threshold).
        under_rep = [
            g for g in dead_gamma_seen if counts.get(g, 0) < c_min_for(g)
        ]
        if not under_rep:
            return 0, 0

        self.rng.shuffle(under_rep)
        # Cap the work per speciation call: even with a large
        # ``dead_gamma_seen`` set, only attempt the top-`max_promotions`
        # candidates per call. Speciation fires every
        # ``gamma_speciation_interval`` iterations, so over the course of a
        # run the chain gets ample opportunity to promote into every
        # plausible gamma-config without spending O(|dead_gamma_seen|)
        # work per call.
        under_rep = under_rep[:max_promotions_per_call]
        # Within-model mix steps after promotion. Default 0: skip the mix
        # entirely. The promoted point is already a valid sample from the
        # constrained prior (we checked L > L_threshold below), and the
        # subsequent NS iterations mix it as a regular live point. The
        # stale ``self.bound`` (fit before the promotion) can interact
        # badly with the within-model RW's safety-cap retry mechanism --
        # mixing here is more trouble than it's worth.
        spec_mix_steps = max(0, int(speciation_n_mcmc))

        for target_gamma in under_rep:
            if live.n_live >= max_n_live:
                break
            # Find well-populated donor differing by exactly one slot.
            donors = []
            for donor_gamma in well_pop:
                diff = [
                    k for k in range(n_groups)
                    if donor_gamma[k] != target_gamma[k]
                ]
                if len(diff) == 1:
                    donors.append((donor_gamma, diff[0]))
            if not donors:
                continue
            donor_gamma, flip_k = donors[
                int(self.rng.integers(len(donors)))
            ]
            # Pick a random live point in the donor gamma config.
            donor_indices = [
                i for i, row in enumerate(live.gamma)
                if tuple(bool(x) for x in row) == donor_gamma
            ]
            if not donor_indices:
                continue
            donor_idx = donor_indices[
                int(self.rng.integers(len(donor_indices)))
            ]
            seed = live.snapshot(donor_idx)
            # Directed trans-dim flip on slot ``flip_k`` followed by a
            # within-model mix. Standard M-H acceptance applies.
            new_state, calls = self._moms_flip_once(
                seed, L_threshold, group_idx=flip_k
            )
            n_calls += calls
            # Verify the flip landed on the intended gamma; if M-H rejected
            # or proposed a no-op, skip.
            if tuple(bool(x) for x in new_state.gamma) != target_gamma:
                continue
            if new_state.log_likelihood <= L_threshold:
                continue
            if spec_mix_steps > 0:
                new_state, mix_calls = self.sampler.step(
                    new_state, L_threshold, self.bound, self.rng,
                    n_steps=spec_mix_steps,
                )
                n_calls += mix_calls
                if new_state.log_likelihood <= L_threshold:
                    continue
            live.append(new_state)
            counts[target_gamma] = counts.get(target_gamma, 0) + 1
            n_promoted += 1

        return n_promoted, n_calls

    # ---------------- progress display ------------------------------------

    def _make_progress_bar(self, show: bool, max_iter: int | None):
        """Construct a tqdm progress bar if available, else None.

        Mirrors the dynesty live-status idiom: per-iteration postfix with
        log Z (and its NS error estimate), the current likelihood threshold,
        the running NS efficiency, and the dlogz gap that drives termination.
        Falls back silently when tqdm is not importable, so the default
        install (no [progress] extra) still works without a dependency.
        """
        if not show:
            return None
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover - graceful degrade
            return None
        return tqdm(
            total=max_iter,  # None for indeterminate; fills when known
            desc="Daedalus",
            unit="iter",
            dynamic_ncols=True,
            mininterval=0.5,
        )

    def _update_progress_bar(
        self,
        pbar,
        acc: SkillingAccumulator,
        gap: float,
        L_thr: float,
        n_calls: int,
        log_dlogz: float,
        n_live_current: int | None = None,
    ) -> None:
        pbar.update(1)
        eff = acc.n_iter / max(n_calls, 1)
        # Display the termination criterion as an estimate of the number
        # of NS iterations remaining. NS shrinks log X by 1/n_live per
        # iter, so once logL* has plateaued the remaining iters to drive
        # the gap below the target are ~ (gap - log_dlogz) * n_live.
        # While logL* is still climbing the estimate is meaningless, so
        # we display "warming up" instead of bogus numbers.
        gap_above = gap - log_dlogz
        n_live_for_proj = n_live_current or self.n_live
        if np.isfinite(gap_above) and 0.0 < gap_above < 8.0:
            iters_remaining = int(np.ceil(gap_above * n_live_for_proj))
            remaining_str = f"~{iters_remaining} iter to go"
            new_total = acc.n_iter + iters_remaining
            if pbar.total != new_total:
                pbar.total = new_total
        elif np.isfinite(gap_above) and gap_above <= 0.0:
            remaining_str = "terminating"
            if pbar.total is not None:
                pbar.total = None
        else:
            remaining_str = "warming up (logL* still climbing)"
            if pbar.total is not None:
                pbar.total = None
        postfix = {
            "logZ": f"{acc.log_Z:+.3f}",
            "+/-": f"{acc.log_Z_err:.3f}",
            "logL*": f"{L_thr:.2f}",
            "eff": f"{eff:.1%}",
            "remaining": remaining_str,
        }
        if n_live_current is not None and n_live_current != self.n_live:
            postfix["n_live"] = f"{n_live_current}"
        pbar.set_postfix(postfix, refresh=False)

    # ---------------- replacement kernel (mixes within-model + MoMS flips)

    def _replacement_kernel(
        self,
        donor: State,
        L_threshold: float,
        n_steps: int,
        transdim_fraction: float,
        gamma_bounds: dict | None = None,
        rejuvenate_fraction: float = 0.0,
    ) -> tuple[State, int]:
        """``n_steps`` is a count of ``sampler.step_once`` calls, already
        divided down from the caller's sweep budget by
        ``Sampler.sweeps_per_step`` -- not the user's ``n_mcmc``.
        """
        # Pure continuous-only path: defer to sampler.step which can amortise
        # batch setup (e.g. extracting the bound's Cholesky factor once).
        if not self.groups or transdim_fraction <= 0.0:
            return self.sampler.step(
                donor, L_threshold, self.bound, self.rng, n_steps=n_steps
            )

        state = donor
        n_calls = 0
        ran_within_model = False
        # NOTE: unlike ``RandomWalkSampler.step``, this loop has no
        # "at least one acceptance" guarantee, so a batch in which nothing
        # is accepted returns the donor untouched -- a duplicate live point.
        # Measured clone rate on the WASP-47 benchmark is 19-25% (roughly
        # flat in transdim_fraction). Adding step()'s walk-until-accept
        # guard here was tried and REVERTED: it cut the clone rate to ~0.2%
        # but broke test_diabetes_inclusion_probs_match_table_1_within_0_02
        # (a validated paper benchmark) and cost 48-74% more likelihood
        # calls, while the evidence bias it was meant to remove could not be
        # demonstrated on any analytic benchmark. Left as a known issue
        # rather than papered over with a fix that measurably made things
        # worse; any future attempt must keep that diabetes test green.
        for _ in range(n_steps):
            draw = self.rng.uniform()
            if draw < transdim_fraction:
                state, calls = self._moms_flip_once(state, L_threshold)
            elif draw < transdim_fraction + rejuvenate_fraction:
                state, calls = self._rejuvenate_once(state, L_threshold)
            else:
                # Route within-model proposal to the bound fit on the
                # current state's gamma-cluster (gamma-adaptive only).
                # The bound's Cholesky drives proposal geometry, so per-
                # gamma bounds give each gamma-config proposal directions
                # scaled to its own continuous structure rather than the
                # union-of-all-gammas ellipsoid that would otherwise apply.
                bound_for_step = self.bound
                if gamma_bounds:
                    gamma_key = tuple(bool(x) for x in state.gamma)
                    bound_for_step = gamma_bounds.get(gamma_key, self.bound)
                state, calls = self.sampler.step_once(
                    state, L_threshold, bound_for_step, self.rng
                )
                ran_within_model = True
            n_calls += calls
        if ran_within_model:
            self.sampler._on_batch_complete()
        return state, n_calls

    # ---------------- within-model rejuvenation (mode hopping) ------------

    def _rejuvenate_once(
        self, state: State, L_threshold: float, group_idx: int | None = None
    ) -> tuple[State, int]:
        """Re-propose an ACTIVE group's continuous block from its birth proposal.

        Once ``L*`` rises above the best likelihood the off-state can reach,
        DELETE can never satisfy ``L > L*`` again and ``gamma = 1`` becomes
        absorbing. The birth proposal only fires on an OFF->ON flip, so past
        that point the group's parameters move only by local within-model
        steps, which cannot cross a likelihood valley. On the WASP-47
        benchmark the last accepted DELETE is flip 1104 of 44011, and the
        period stays frozen wherever the early birth left it.

        This move re-draws the block from the same ``BirthProposal`` without
        touching ``gamma``, giving the chain a mode-hopping proposal at any
        ``L*``. Metropolis-Hastings; the proposal may condition on the
        current state, so the reverse density is evaluated against the
        POST-move state, as detailed balance requires:

            log alpha = [log pi(new) - log q(new | old)]
                      - [log pi(old) - log q(old | new)]

        Costs exactly one likelihood evaluation per attempt. See
        ``run_nested(rejuvenate_fraction=...)`` for the measured accuracy
        trade-off -- this is a discovery aid, not a calibrated-inference
        setting.
        """
        n_groups = len(self.groups)
        if n_groups == 0:
            return state, 0
        k = int(self.rng.integers(n_groups)) if group_idx is None else int(group_idx)
        group = self.groups[k]
        # Bind both up front: Group.__post_init__ already refuses a
        # birth_proposal without a matching log_prior_continuous (the
        # continuous prior no longer cancels against the proposal density),
        # so the second check is belt-and-braces -- but it also lets the
        # type checker see that neither is None below.
        birth = group.birth_proposal
        log_prior_continuous = group.log_prior_continuous
        if (
            birth is None
            or log_prior_continuous is None
            or not bool(state.gamma[k])
        ):
            return state, 0

        gp = self._group_params[k]
        old_beta_group = state.beta[gp].copy()

        proposal = birth.propose(group, state, self.rng)
        log_q_forward = float(proposal.log_q_forward)
        log_pi_old = float(log_prior_continuous(old_beta_group))
        log_pi_new = float(log_prior_continuous(proposal.proposed_beta))

        state.beta[gp] = proposal.proposed_beta
        log_q_reverse = float(birth.log_density(group, state, old_beta_group))
        log_alpha = (log_pi_new - log_q_forward) - (log_pi_old - log_q_reverse)

        log_L_new = self.loglike(state.beta, state.gamma)
        accepted = log_L_new > L_threshold and (
            log_alpha >= 0.0 or self.rng.uniform() < float(np.exp(log_alpha))
        )
        if accepted:
            state.log_likelihood = log_L_new
            self._sync_u_to_beta(state, gp)
            return state, 1
        state.beta[gp] = old_beta_group
        return state, 1

    # ---------------- MoMS flip kernel (default: uniform-u birth) ---------

    def _moms_flip_once(
        self,
        state: State,
        L_threshold: float,
        group_idx: int | None = None,
    ) -> tuple[State, int]:
        """One MoMS between-model flip on a uniformly-chosen group.

        Default proposal:
            ADD    -- redraw u over the group's parameter indices uniformly
                      from [0, 1); convert to beta via prior_transform.
            DELETE -- pin beta over the group's indices to off-values; u
                      coordinates are left as-is (stale; will be refreshed
                      on the next ADD or by within-model RW).

        With uniform-u birth, the proposal density q in u-space is 1 in both
        directions, so the M-H ratio reduces to the inclusion-prior ratio
        (the continuous prior cancels). The NS constraint then enforces
        L_new > L_threshold.

        Implementation note: we mutate `state` in place and revert on
        rejection. This avoids the per-flip State allocation + array
        copies that dominated the trans-dim hot path. log_prior is
        updated incrementally (one add/sub per flip) instead of
        recomputed from the full gamma vector.
        """
        n_groups = len(self.groups)
        if group_idx is None:
            k = int(self.rng.integers(n_groups))
        else:
            k = int(group_idx)
        group = self.groups[k]
        group_params = self._group_params[k]
        off_vals = self._group_off_values[k]

        old_gamma_k = bool(state.gamma[k])
        new_gamma_k = not old_gamma_k

        # Inclusion-prior contribution to the M-H log-ratio.
        log_alpha = (self._log_inc[k] - self._log_exc[k]) * (
            1.0 if new_gamma_k else -1.0
        )

        custom_birth = group.birth_proposal is not None

        # ---- propose ---------------------------------------------------
        # Default uniform-u birth: redraw u for the group's params and
        # push through prior_transform; the proposal density and the
        # user's continuous prior cancel exactly in beta-space.
        # Custom birth: draw beta directly via the user's BirthProposal;
        # the q_beta and pi_beta densities both contribute to log_alpha.
        if new_gamma_k:  # ADD
            if custom_birth:
                old_beta_group = state.beta[group_params].copy()
                proposal = group.birth_proposal.propose(group, state, self.rng)
                state.gamma[k] = True
                state.beta[group_params] = proposal.proposed_beta
                log_alpha += (
                    group.log_prior_continuous(proposal.proposed_beta)
                    - proposal.log_q_forward
                )
                old_u_group = None
                old_beta = None
            else:
                old_u_group = state.u[group_params].copy()
                old_beta = state.beta.copy()
                old_beta_group = None
                state.gamma[k] = True
                state.u[group_params] = self.rng.uniform(size=group_params.size)
                new_beta = np.asarray(
                    self.prior_transform(state.u), dtype=np.float64
                )
                state.beta[:] = new_beta
                self._apply_off_values(state.beta, state.gamma)
        else:  # DELETE
            old_beta_group = state.beta[group_params].copy()
            old_u_group = None
            old_beta = None
            if custom_birth:
                # Reverse density: birth proposal evaluated at the current
                # (active) beta values for the group, "from" the post-delete
                # state. log_density may inspect state.gamma (not yet
                # flipped), which is the correct conditioning for a reverse
                # birth originating from the new state.
                log_q_reverse = group.birth_proposal.log_density(
                    group, state, old_beta_group
                )
                log_pi = group.log_prior_continuous(old_beta_group)
                log_alpha += log_q_reverse - log_pi
            state.gamma[k] = False
            state.beta[group_params] = off_vals

        # ---- evaluate likelihood under the constraint ------------------
        log_L_new = self.loglike(state.beta, state.gamma)
        accepted = log_L_new > L_threshold and (
            log_alpha >= 0.0 or self.rng.uniform() < float(np.exp(log_alpha))
        )

        # Adaptive birth proposals (e.g. GaussianRWBirth) consume the
        # accept signal of every flip attempt that involves their scale.
        # Both ADD (forward draw) and DELETE (reverse density at current
        # beta) use the same proposal scale, so both directions feed the
        # same Robbins-Monro update (van den Bergh+ 2026, Eq. D1).
        if custom_birth:
            update_fn = getattr(group.birth_proposal, "update", None)
            if update_fn is not None:
                update_fn(accepted)

        if accepted:
            state.log_likelihood = log_L_new
            if new_gamma_k and custom_birth:
                # Custom births propose directly in beta-space, but the
                # within-model kernels regenerate beta from u at every
                # step; without re-syncing u the next within-model move
                # would silently overwrite the accepted birth value with
                # prior_transform(stale u).
                self._sync_u_to_beta(state, group_params)
            # Inclusion contribution to log_prior; the continuous-prior
            # delta is intentionally NOT folded into state.log_prior --
            # we treat log_prior as the inclusion-prior accumulator only,
            # consistent with the default uniform-u path.
            state.log_prior += (self._log_inc[k] - self._log_exc[k]) * (
                1.0 if new_gamma_k else -1.0
            )
            return state, 1

        # Reject: revert.
        state.gamma[k] = old_gamma_k
        if new_gamma_k:
            if custom_birth:
                state.beta[group_params] = old_beta_group
            else:
                state.u[group_params] = old_u_group
                state.beta[:] = old_beta
        else:
            state.beta[group_params] = old_beta_group
        return state, 1

    def _sync_u_to_beta(self, state: State, group_params: np.ndarray) -> None:
        """Set ``state.u[group_params]`` so that ``prior_transform(u)``
        reproduces ``state.beta[group_params]``.

        Coordinate-wise monotone bisection: exact (to bisection precision)
        for prior transforms that act separably per coordinate, which
        covers every transform shipped with the package. If the transform
        is not invertible this way, the residual is detected and a
        one-time warning is emitted (the birth value then remains
        transient, as it was before this sync existed).
        """
        u = state.u
        target = state.beta[group_params].copy()
        for j, p in enumerate(group_params):
            t_j = float(target[j])
            u[p] = 0.0
            v_lo = float(np.asarray(self.prior_transform(u))[p])
            u[p] = 1.0
            v_hi = float(np.asarray(self.prior_transform(u))[p])
            increasing = v_hi >= v_lo
            lo, hi = 0.0, 1.0
            for _ in range(54):  # interval ~5.6e-17 after 54 halvings
                mid = 0.5 * (lo + hi)
                u[p] = mid
                v = float(np.asarray(self.prior_transform(u))[p])
                if (v < t_j) == increasing:
                    lo = mid
                else:
                    hi = mid
            u[p] = 0.5 * (lo + hi)
        beta_new = np.asarray(self.prior_transform(u), dtype=np.float64)
        resid = float(np.max(np.abs(beta_new[group_params] - target)))
        # Adopt the u-consistent values (differ from the proposal at
        # bisection precision for separable transforms) so the kernel
        # invariant beta == prior_transform(u) holds exactly.
        state.beta[group_params] = beta_new[group_params]
        if resid > 1e-6 and not getattr(self, "_u_sync_warned", False):
            self._u_sync_warned = True
            warnings.warn(
                "custom-birth u-sync: prior_transform could not be "
                f"inverted (residual {resid:.2e}) on params "
                f"{list(group_params)}; the transform appears "
                "non-separable or non-monotone per coordinate, so "
                "custom-birth values may not persist across "
                "within-model moves.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ---------------- result construction ---------------------------------

    def _build_results(
        self,
        acc: SkillingAccumulator,
        dead_u: np.ndarray,
        dead_beta: np.ndarray,
        dead_gamma: np.ndarray,
        n_calls: int,
    ) -> Results:
        log_likelihoods = np.asarray(acc.dead_log_likelihoods, dtype=np.float64)
        log_weights = np.asarray(acc.dead_log_weights, dtype=np.float64)
        log_post = log_weights + log_likelihoods - acc.log_Z
        weights = np.exp(log_post - np.max(log_post))
        weights /= weights.sum()
        n_resample = log_likelihoods.size
        choices = self.rng.choice(n_resample, size=n_resample, p=weights)

        beta_arr = np.asarray(dead_beta, dtype=np.float64)
        gamma_arr = np.asarray(dead_gamma, dtype=bool)
        if gamma_arr.ndim == 2 and gamma_arr.shape[0] and gamma_arr.shape[1]:
            samples_gamma = gamma_arr[choices]
        else:
            samples_gamma = np.empty((n_resample, 0), dtype=bool)

        return Results(
            samples=beta_arr[choices],
            gamma=samples_gamma,
            log_likelihoods=log_likelihoods,
            log_weights=log_weights,
            log_Z=acc.log_Z,
            log_Z_err=acc.log_Z_err,
            H=acc.H,
            n_iter=acc.n_iter,
            n_calls=n_calls,
            group_names=[g.name for g in self.groups],
            inclusion_priors=self._inclusion_priors.copy(),
        )
