"""Per-gamma-population MoMS-NS.

Standard MoMS-NS in :class:`daedalus.NestedSampler` runs a single
global live cloud over the joint (beta, gamma) state space. On
problems with substantial slot-permutation symmetry — multiple
candidate components that compete for a fixed budget of toggleable
slots — the single cloud cannot maintain coverage of all the
competitive gamma-configurations simultaneously, and chains from
different seeds end up trapped in different basins.

This module implements per-gamma-population MoMS-NS as a separate
sampler. The chain maintains an independent live-point population
per gamma-configuration; each population has its own Skilling
volume-shrinkage recursion, its own X_gamma, its own log Z_gamma,
and its own gamma-scoped bound. Within-model proposals stay inside
a single population; trans-dim flips MIGRATE individual live points
between populations.

Final results combine populations via the standard
``Z = sum_gamma P(gamma) Z_gamma`` decomposition.

.. warning::

    EXPERIMENTAL. This module is not used to produce any published
    result and is not part of the supported public API of
    :mod:`daedalus`. It is not exported from the package namespace.
    Use :class:`daedalus.NestedSampler` for all supported work.

This is the architecture often associated with RJObject / DNest4 at a
high level (per-model populations) but without DNest4's likelihood-
level structure: each population runs ordinary Skilling NS on its
own gamma-config, and the trans-dim moves between them are the only
inter-population dynamics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from .bounds import (
    GammaScopedEllipsoid,
    GammaScopedMultiEllipsoid,
    NoBound,
)
from .bounds import BoundingRegion, make_bound
from .groups import Group
from .live_points import LivePoints
from .recursion import SkillingAccumulator
from .results import Results
from .samplers import Sampler, make_sampler
from .state import State

if TYPE_CHECKING:
    pass


LogLikelihood = Callable[[np.ndarray, np.ndarray], float]
PriorTransform = Callable[[np.ndarray], np.ndarray]


@dataclass
class GammaPopulation:
    """One per-gamma live-point population.

    Each population has its own Skilling NS recursion and its own
    gamma-scoped bound. The population's ``live.gamma`` is constant
    across all live points (every point in the population is at this
    gamma-tuple); only the continuous coordinates vary.
    """

    gamma_tuple: tuple[bool, ...]
    live: LivePoints
    acc: SkillingAccumulator
    bound: BoundingRegion
    active: bool = True  # set False when the population empties

    # Recording: dead points produced by this population. Accumulated
    # into the final Results at termination.
    dead_u: list[np.ndarray] = field(default_factory=list)
    dead_beta: list[np.ndarray] = field(default_factory=list)
    dead_log_likelihood: list[float] = field(default_factory=list)
    dead_log_weight: list[float] = field(default_factory=list)

    @property
    def n_live(self) -> int:
        return self.live.n_live


class MultiPopNestedSampler:
    """Per-gamma-population MoMS-NS sampler.

    API mirrors :class:`daedalus.NestedSampler` where it makes sense.
    Differences:

    * `run_nested` advances per-gamma populations independently and
      handles inter-population migrations via trans-dim moves.
    * Returned :class:`Results` carries the union of all populations'
      dead points, weighted by their (P(gamma) * exp(log_w_i / Z_gamma))
      contribution to the joint posterior.
    """

    def __init__(
        self,
        loglike: LogLikelihood,
        prior_transform: PriorTransform,
        ndim: int,
        groups: Sequence[Group],
        bound: str = "single",
        sample: str | Sampler = "rwalk",
        n_live: int = 200,
        n_initial_draws: int | None = None,
        min_init_per_pop: int | None = None,
        init_log_L_max_floor: float | None = None,
        periodic: Sequence[int] | None = None,
        seed: int | None = None,
        validate_births: bool = True,
    ) -> None:
        if ndim < 1:
            raise ValueError(f"ndim must be >= 1, got {ndim}")
        if n_live < 2:
            raise ValueError(f"n_live (per population) must be >= 2, got {n_live}")
        if not groups:
            raise ValueError(
                "MultiPopNestedSampler requires at least one toggleable Group"
            )

        self.loglike = loglike
        self.prior_transform = prior_transform
        self.ndim = ndim
        self.groups = tuple(groups)
        self.n_live_base = n_live
        if isinstance(bound, str):
            if bound not in {"single", "multi"}:
                raise ValueError(
                    f"bound must be 'single' or 'multi', got {bound!r}"
                )
            self.bound_kind = bound
        else:
            raise TypeError(
                "MultiPopNestedSampler.bound must be a string ('single' or 'multi'); "
                "bound objects are constructed internally per-γ-config"
            )
        self.n_initial_draws_override = n_initial_draws
        self.min_init_per_pop_override = min_init_per_pop
        self.init_log_L_max_floor = init_log_L_max_floor
        self.periodic = tuple(periodic) if periodic is not None else ()
        self.rng = np.random.default_rng(seed)

        self.bound_factory = (
            (lambda: make_bound(bound, ndim))
            if isinstance(bound, str)
            else (lambda: bound)
        )
        self.sampler: Sampler = (
            make_sampler(sample) if isinstance(sample, str) else sample
        )

        # Group bookkeeping (mirrors NestedSampler)
        self._inclusion_priors = np.asarray(
            [g.inclusion_prior for g in self.groups], dtype=np.float64
        )
        self._log_inc = np.log(self._inclusion_priors)
        self._log_exc = np.log1p(-self._inclusion_priors)
        self._group_params = [
            np.asarray(g.params, dtype=np.intp) for g in self.groups
        ]
        self._group_off_values = [g.off_values.copy() for g in self.groups]

        # Always-on dims (params not in any group)
        in_some_group: set[int] = set()
        for g in self.groups:
            in_some_group.update(int(p) for p in g.params)
        self._always_on_dims = sorted(
            d for d in range(self.ndim) if d not in in_some_group
        )

        self.sampler.bind(
            loglike=self.loglike,
            prior_transform=prior_transform,
            ndim=ndim,
            groups=self.groups,
            periodic=self.periodic,
            apply_off_values=self._apply_off_values,
        )

        self.populations: dict[tuple[bool, ...], GammaPopulation] = {}
        self._flat_log_L: dict[tuple[bool, ...], float] = {}
        self._flat_init_points: dict[tuple[bool, ...], list[np.ndarray]] = {}

    # ---------------- helpers ---------------------------------------------

    def _apply_off_values(self, beta: np.ndarray, gamma: np.ndarray) -> None:
        for k, group_params in enumerate(self._group_params):
            if not gamma[k]:
                beta[group_params] = self._group_off_values[k]

    def _log_inclusion_prior(self, gamma: np.ndarray) -> float:
        return float(np.where(gamma, self._log_inc, self._log_exc).sum())

    def _active_dims_for_gamma(
        self, gamma_tuple: tuple[bool, ...]
    ) -> list[int]:
        dims = list(self._always_on_dims)
        for k, on in enumerate(gamma_tuple):
            if on:
                dims.extend(int(p) for p in self._group_params[k])
        return sorted(dims)

    def _make_bound_for_gamma(
        self, gamma_tuple: tuple[bool, ...]
    ) -> BoundingRegion:
        """Create a fresh bound appropriate for a population at ``gamma_tuple``.

        With ``bound='single'`` (default), use ``GammaScopedEllipsoid`` —
        a single ellipsoid on the active dims plus axis-aligned inactive
        spread.

        With ``bound='multi'``, use ``GammaScopedMultiEllipsoid`` — a
        clustered (Feroz+ 2009) MultiEllipsoid on the active dims. This
        is the bound needed when a single γ-tuple has permutation-
        equivalent posterior modes (e.g. exchangeable Keplerian slots
        with N_active! modes per γ-config).
        """
        active = self._active_dims_for_gamma(gamma_tuple)
        if self.bound_kind == "multi":
            return GammaScopedMultiEllipsoid(
                self.ndim, active, enlargement=1.25,
            )
        return GammaScopedEllipsoid(
            self.ndim, active, enlargement=1.25,
        )

    # ---------------- initialisation --------------------------------------

    def _initialise_populations(self) -> None:
        """Draw an initial joint-prior cloud, then group by gamma-tuple."""
        n_groups = len(self.groups)
        if self.n_initial_draws_override is not None:
            n_initial = int(self.n_initial_draws_override)
        else:
            # Heuristic: ~3x the per-population threshold per gamma-tuple,
            # capped to scale linearly with 2^n_groups up to n_groups=8.
            # For n_groups > 8, raise n_initial_draws manually to ensure
            # each non-trivial gamma-tuple is populated.
            n_initial = max(
                self.n_live_base * (2 ** min(n_groups, 8)) * 3,
                self.n_live_base * 16,
            )
            n_initial = min(n_initial, self.n_live_base * 1024)

        u_init = self.rng.uniform(size=(n_initial, self.ndim))
        gamma_init = np.empty((n_initial, n_groups), dtype=bool)
        for k, p in enumerate(self._inclusion_priors):
            gamma_init[:, k] = self.rng.uniform(size=n_initial) < p

        # Push through prior_transform, apply off-values, evaluate
        # likelihood.
        beta_init = np.empty_like(u_init)
        log_L = np.empty(n_initial, dtype=np.float64)
        log_pi = np.empty(n_initial, dtype=np.float64)
        for i in range(n_initial):
            beta = np.asarray(self.prior_transform(u_init[i]), dtype=np.float64)
            self._apply_off_values(beta, gamma_init[i])
            beta_init[i] = beta
            log_L[i] = float(self.loglike(beta, gamma_init[i]))
            log_pi[i] = self._log_inclusion_prior(gamma_init[i])

        # Group by gamma_tuple
        gamma_tuples = [
            tuple(bool(x) for x in row) for row in gamma_init
        ]
        from collections import defaultdict

        indices_by_gamma: dict[tuple[bool, ...], list[int]] = defaultdict(list)
        for i, g in enumerate(gamma_tuples):
            indices_by_gamma[g].append(i)

        # Instantiate populations for gamma-tuples with at least
        # ``min_init_per_pop`` initial points (relaxed from a flat
        # ``n_live_base`` so the sampler is robust to small initial
        # clouds; downstream migrations grow populations further).
        #
        # Populations whose continuous likelihood is FLAT (max log_L ==
        # min log_L across the initial draw) are skipped: NS on a
        # constant likelihood never terminates because the strict
        # ``log_L_new > L_threshold`` condition is never met. Their
        # Z_gamma is computable analytically (L_const * prior_volume)
        # and folded into the joint Z at result-combination time.
        self._flat_log_L: dict[tuple[bool, ...], float] = {}
        # Beta vectors of the prior-sampled init points for each flat-L
        # gamma-config, retained so they can be injected into the joint
        # importance-resampled cloud in `_combine_results`. Without this,
        # flat-L mass contributes to log Z but not to per-point gamma
        # samples, biasing inclusion marginals.
        self._flat_init_points: dict[tuple[bool, ...], list[np.ndarray]] = {}
        if self.min_init_per_pop_override is not None:
            min_init_per_pop = int(self.min_init_per_pop_override)
        else:
            # Relaxed minimum: floor(n_live_base / 4) but at least 5.
            min_init_per_pop = max(5, self.n_live_base // 4)
        # Pre-compute a global L-floor for pruning if requested. The
        # floor is interpreted as "skip γ-configs whose best initial-
        # draw log L is below this value", since their joint posterior
        # contribution is negligible compared to other γ-configs.
        for gamma_tuple, indices in indices_by_gamma.items():
            if len(indices) < min_init_per_pop:
                continue
            # Truncate to n_live_base, or take all if cluster is smaller.
            indices = indices[: self.n_live_base]
            actual_n_live = len(indices)
            log_L_cluster = log_L[indices]
            log_L_max_cluster = float(log_L_cluster.max())
            if (
                self.init_log_L_max_floor is not None
                and log_L_max_cluster < self.init_log_L_max_floor
            ):
                # Below the user-supplied L-floor: skip running NS on
                # this γ-config. Its joint Z contribution is folded in
                # as if it were a "soft flat-L" pop using the best
                # initial-draw L (a conservative under-estimate of
                # Z_gamma, but it caps the compute cost for n_groups
                # >> 6 problems where most γ-tuples are negligible).
                self._flat_log_L[gamma_tuple] = log_L_max_cluster
                self._flat_init_points[gamma_tuple] = [
                    beta_init[idx].copy() for idx in indices
                ]
                continue
            log_L_spread = float(log_L_cluster.max() - log_L_cluster.min())
            if log_L_spread < 1e-12:
                # Constant-likelihood gamma-config: record its log_L
                # value (the constant) and skip running NS on it.
                self._flat_log_L[gamma_tuple] = float(log_L_cluster.max())
                self._flat_init_points[gamma_tuple] = [
                    beta_init[idx].copy() for idx in indices
                ]
                continue
            live = LivePoints.empty(
                actual_n_live, self.ndim, n_groups
            )
            for j, idx in enumerate(indices):
                live.u[j] = u_init[idx]
                live.beta[j] = beta_init[idx]
                live.gamma[j] = gamma_init[idx]
                live.log_likelihood[j] = log_L[idx]
                live.log_prior[j] = log_pi[idx]
            bound = self._make_bound_for_gamma(gamma_tuple)
            try:
                bound.fit(live.u, log_volume_prior=0.0)
            except Exception:
                bound = NoBound(self.ndim)
            acc = SkillingAccumulator(n_live=actual_n_live)
            pop = GammaPopulation(
                gamma_tuple=gamma_tuple,
                live=live,
                acc=acc,
                bound=bound,
            )
            self.populations[gamma_tuple] = pop

    # ---------------- per-population NS step ------------------------------

    def _advance_population(
        self,
        pop: GammaPopulation,
        L_threshold_step: float | None,
        n_mcmc: int,
        transdim_fraction: float,
        bound_update_interval: int = 1,
    ) -> tuple[bool, int]:
        """One NS step on a single population.

        Discards the worst live point as a dead point, then replaces it
        via within-model + trans-dim moves. Returns ``(advanced, n_calls)``.
        Returns ``advanced=False`` when the population was not actually
        stepped (e.g., empty or below minimum); in that case the
        population is also marked inactive so the outer loop doesn't
        spin on it forever.
        """
        if not pop.active or pop.live.n_live < 2:
            pop.active = False
            return False, 0

        dead_idx = pop.live.argmin_log_L()
        L_thr = float(pop.live.log_likelihood[dead_idx])
        # Skilling accounting for this dead point.
        log_dX = pop.acc.add_dead(L_thr, n_live=pop.live.n_live)
        pop.dead_u.append(pop.live.u[dead_idx].copy())
        pop.dead_beta.append(pop.live.beta[dead_idx].copy())
        pop.dead_log_likelihood.append(L_thr)
        pop.dead_log_weight.append(float(log_dX))

        # Refit bound only every ``bound_update_interval`` iterations.
        # MultiEllipsoid fits are expensive (k-means + recursive
        # splits); refitting every step makes per-γ-pop runs O(100x)
        # slower without improving statistics. The main NestedSampler
        # uses the same throttle.
        if (pop.acc.n_iter - 1) % bound_update_interval == 0:
            try:
                mask = np.ones(pop.live.n_live, dtype=bool)
                mask[dead_idx] = False
                if mask.sum() > 1:
                    pop.bound.fit(pop.live.u[mask], log_volume_prior=pop.acc.log_X_prev)
            except Exception:
                pass

        # Hand live cloud to the sampler if it wants it (e.g. DE).
        set_live = getattr(self.sampler, "set_live_points", None)
        if set_live is not None:
            set_live(pop.live.u)

        # Pick a donor uniformly from the surviving live points.
        donor_idx = int(self.rng.integers(pop.live.n_live - 1))
        if donor_idx >= dead_idx:
            donor_idx += 1
        donor = pop.live.snapshot(donor_idx)

        # Replacement kernel: within-model proposals only (we do trans-
        # dim moves at a separate step in the outer loop). This is the
        # key difference from single-cloud MoMS-NS — each population
        # advances purely within its own gamma-config here.
        new_state, calls = self.sampler.step(
            donor, L_thr, pop.bound, self.rng, n_steps=n_mcmc
        )

        # Replace the dead point.
        pop.live.replace(dead_idx, new_state)
        return True, calls

    # ---------------- migration via trans-dim flip ------------------------

    def _attempt_migration(
        self, n_mcmc: int
    ) -> int:
        """One trans-dim migration attempt: try to move a random live
        point from its current population to a neighbouring gamma-
        configuration.

        Returns number of likelihood calls used.
        """
        active_pops = [p for p in self.populations.values() if p.active]
        if len(active_pops) < 1:
            return 0
        pop = active_pops[int(self.rng.integers(len(active_pops)))]
        if pop.live.n_live < 3:
            # Don't drain a small population further; trans-dim moves
            # that empty a population at small n_live are catastrophic
            # for the X_gamma recursion.
            return 0

        # Pick a live point in this population.
        donor_idx = int(self.rng.integers(pop.live.n_live))
        seed = pop.live.snapshot(donor_idx)

        # Pick a random group k to flip.
        k = int(self.rng.integers(len(self.groups)))
        target_gamma = list(seed.gamma)
        target_gamma[k] = not target_gamma[k]
        target_tuple = tuple(bool(x) for x in target_gamma)

        # Run the standard MoMS flip on this slot only. We adapt the
        # NestedSampler's _moms_flip_once logic inline.
        new_state, calls = self._moms_flip_directed(
            seed, k, target_gamma[k]
        )

        # Migration only happens when the flip succeeded AND the new
        # point clears the destination population's L_threshold.
        if tuple(bool(x) for x in new_state.gamma) != target_tuple:
            return calls

        # Find / create the target population.
        target_pop = self.populations.get(target_tuple)
        if target_pop is None:
            # Spawn new population with this one point. Its NS
            # recursion starts at n_live=1; we'll let it grow naturally
            # via subsequent migrations.
            live = LivePoints.empty(1, self.ndim, len(self.groups))
            live.u[0] = new_state.u
            live.beta[0] = new_state.beta
            live.gamma[0] = new_state.gamma
            live.log_likelihood[0] = new_state.log_likelihood
            live.log_prior[0] = new_state.log_prior
            bound = self._make_bound_for_gamma(target_tuple)
            acc = SkillingAccumulator(n_live=1)
            self.populations[target_tuple] = GammaPopulation(
                gamma_tuple=target_tuple,
                live=live,
                acc=acc,
                bound=bound,
            )
            # The donor's population loses one live point (the donor
            # migrated away). We do NOT decrement its n_live here; we
            # just leave the donor slot in pop's live array since the
            # X recursion continues at n_live; instead we shrink pop's
            # live cloud.
            pop.live.u = np.delete(pop.live.u, donor_idx, axis=0)
            pop.live.beta = np.delete(pop.live.beta, donor_idx, axis=0)
            pop.live.gamma = np.delete(pop.live.gamma, donor_idx, axis=0)
            pop.live.log_likelihood = np.delete(
                pop.live.log_likelihood, donor_idx
            )
            pop.live.log_prior = np.delete(pop.live.log_prior, donor_idx)
            return calls

        # Existing target: only accept the migration if the new point's
        # L clears the destination's current threshold (the destination
        # population's worst live point's L). Otherwise reject — adding
        # a sub-threshold point would invalidate the NS invariant that
        # the live cloud lies above the current threshold.
        target_L_thr = float(target_pop.live.log_likelihood.min())
        if new_state.log_likelihood <= target_L_thr:
            return calls

        # Append to target population (grows its n_live).
        target_pop.live.append(new_state)

        # Remove from donor population.
        pop.live.u = np.delete(pop.live.u, donor_idx, axis=0)
        pop.live.beta = np.delete(pop.live.beta, donor_idx, axis=0)
        pop.live.gamma = np.delete(pop.live.gamma, donor_idx, axis=0)
        pop.live.log_likelihood = np.delete(pop.live.log_likelihood, donor_idx)
        pop.live.log_prior = np.delete(pop.live.log_prior, donor_idx)
        if pop.live.n_live < 2:
            pop.active = False
        return calls

    def _moms_flip_directed(
        self,
        state: State,
        k: int,
        target_on: bool,
    ) -> tuple[State, int]:
        """One MoMS M-H flip on group ``k``, proposing ``gamma_k = target_on``.

        Same M-H acceptance ratio as NestedSampler._moms_flip_once but
        the slot index is forced.
        """
        # Migrations only flip in the direction state.gamma[k] -> target_on.
        # If they match already, this is a no-op.
        if bool(state.gamma[k]) == bool(target_on):
            return state, 0

        group = self.groups[k]
        group_params = self._group_params[k]
        off_vals = self._group_off_values[k]
        new_gamma_k = bool(target_on)

        log_alpha = (self._log_inc[k] - self._log_exc[k]) * (
            1.0 if new_gamma_k else -1.0
        )

        custom_birth = group.birth_proposal is not None
        original_state_beta = state.beta.copy()
        original_state_u = state.u.copy()

        if new_gamma_k:  # ADD
            if custom_birth:
                proposal = group.birth_proposal.propose(group, state, self.rng)
                state.gamma[k] = True
                state.beta[group_params] = proposal.proposed_beta
                log_alpha += (
                    group.log_prior_continuous(proposal.proposed_beta)
                    - proposal.log_q_forward
                )
            else:
                state.gamma[k] = True
                state.u[group_params] = self.rng.uniform(size=group_params.size)
                new_beta = np.asarray(
                    self.prior_transform(state.u), dtype=np.float64
                )
                state.beta[:] = new_beta
                self._apply_off_values(state.beta, state.gamma)
        else:  # DELETE
            old_beta_group = state.beta[group_params].copy()
            if custom_birth:
                log_q_reverse = group.birth_proposal.log_density(
                    group, state, old_beta_group
                )
                log_pi = group.log_prior_continuous(old_beta_group)
                log_alpha += log_q_reverse - log_pi
            state.gamma[k] = False
            state.beta[group_params] = off_vals

        log_L_new = float(self.loglike(state.beta, state.gamma))
        accepted = (
            log_alpha >= 0.0 or self.rng.uniform() < float(np.exp(log_alpha))
        ) and np.isfinite(log_L_new)

        # NB: unlike the single-cloud MoMS flip we do NOT gate on
        # log_L_new > L_threshold here. The migration check happens at
        # the caller, against the destination population's L_threshold.

        if accepted:
            state.log_likelihood = log_L_new
            state.log_prior += (self._log_inc[k] - self._log_exc[k]) * (
                1.0 if new_gamma_k else -1.0
            )
            return state, 1

        # Reject: revert.
        state.gamma[k] = not new_gamma_k
        state.beta[:] = original_state_beta
        state.u[:] = original_state_u
        return state, 1

    # ---------------- main run loop ---------------------------------------

    def run_nested(
        self,
        dlogz: float = 0.5,
        n_mcmc: int = 25,
        transdim_per_iter: int = 0,
        max_iter_per_population: int | None = None,
        bound_update_interval: int = 25,
        show_progress: bool = True,
    ) -> Results:
        """Run per-population NS as independent static chains.

        Each gamma-population runs static Skilling-style NS to its own
        dlogz_gamma termination. Populations do NOT interact: there is
        no joint constrained measure across populations (each has its
        own threshold L*_gamma), so migration moves cannot be framed
        as detailed-balance M-H steps preserving stationarity.

        Trans-dim "migration" moves are therefore DISABLED by default
        (``transdim_per_iter=0``). Enabling them produces an upward
        ~0.5-nat bias in joint log Z on the SBC toy and is retained
        only for diagnostic experimentation.

        gamma-configurations are seeded entirely by the initial
        joint-prior draw in :meth:`_initialise_populations`. For
        high-N_groups problems, raise ``n_live`` so the initial draw
        scales accordingly.

        Parameters
        ----------
        dlogz
            Per-population termination: stop once the Skilling gap
            ``log(X * max_L_live) - log Z_gamma`` falls below
            ``log(dlogz)``.
        n_mcmc
            Within-model MCMC steps per replacement.
        transdim_per_iter
            Number of migration attempts per round. Default 0
            (recommended). Setting > 0 biases log Z upward; intended
            only for diagnostic comparison against the unbiased
            (migrations-off) baseline.
        max_iter_per_population
            Optional cap on per-population iterations.
        show_progress
            Print per-population progress every 100 rounds.
        """
        if dlogz <= 0.0:
            raise ValueError(f"dlogz must be positive, got {dlogz}")
        if transdim_per_iter < 0:
            raise ValueError(
                f"transdim_per_iter must be >= 0, got {transdim_per_iter}"
            )
        if transdim_per_iter > 0:
            import warnings
            warnings.warn(
                "transdim_per_iter > 0 enables migration moves that are "
                "not detailed-balance-preserving in the per-gamma-pop "
                "setting and produce an upward log Z bias (~0.5 nats on "
                "the SBC toy). Set to 0 for unbiased estimates.",
                stacklevel=2,
            )

        log_dlogz = float(np.log(dlogz))
        self._initialise_populations()
        if not self.populations and not self._flat_log_L:
            raise RuntimeError(
                "No populations met the initial-cloud threshold; raise "
                "n_live or check the prior_transform / loglike."
            )

        n_calls = 0
        round_idx = 0
        while True:
            round_idx += 1
            active_pops = [
                p for p in self.populations.values() if p.active
            ]
            if not active_pops:
                break

            done_count = 0
            for pop in active_pops:
                if (
                    max_iter_per_population is not None
                    and pop.acc.n_iter >= max_iter_per_population
                ):
                    pop.active = False
                    done_count += 1
                    continue
                gap = pop.acc.termination_gap(pop.live.max_log_L())
                if gap < log_dlogz:
                    pop.active = False
                    done_count += 1
                    continue
                advanced, calls = self._advance_population(
                    pop, L_threshold_step=None,
                    n_mcmc=n_mcmc, transdim_fraction=0.0,
                    bound_update_interval=bound_update_interval,
                )
                n_calls += calls

            # Trans-dim migration attempts.
            for _ in range(transdim_per_iter):
                calls = self._attempt_migration(n_mcmc=n_mcmc)
                n_calls += calls

            if done_count == len(active_pops):
                break

            if show_progress and round_idx % 100 == 0:
                n_alive = sum(
                    1 for p in self.populations.values() if p.active
                )
                total_dead = sum(
                    len(p.dead_log_likelihood)
                    for p in self.populations.values()
                )
                print(
                    f"[multipop] round {round_idx} | {len(self.populations)} pops "
                    f"({n_alive} active) | total dead pts {total_dead} | "
                    f"calls {n_calls}",
                    flush=True,
                )

        # Final: fold remaining live points into per-population Z.
        for pop in self.populations.values():
            if pop.live.n_live > 0:
                pop.acc.add_remaining_live(pop.live.log_likelihood)
                for i in range(pop.live.n_live):
                    pop.dead_u.append(pop.live.u[i].copy())
                    pop.dead_beta.append(pop.live.beta[i].copy())
                    pop.dead_log_likelihood.append(
                        float(pop.live.log_likelihood[i])
                    )
                    # Equal residual weight: log_X_prev - log(n_live)
                    pop.dead_log_weight.append(
                        float(pop.acc.log_X_prev - np.log(pop.live.n_live))
                    )

        return self._combine_results(n_calls)

    # ---------------- result combination -----------------------------------

    def _combine_results(self, n_calls: int) -> Results:
        """Combine per-population dead-point clouds into a single Results.

        For each population gamma, the per-dead-point importance weight
        on the joint posterior is

            w_i = P(gamma) * L_i * dX_gamma,i

        and the joint log Z is

            log Z = logsumexp_gamma ( log P(gamma) + log Z_gamma ).
        """
        all_log_w: list[float] = []
        all_log_L: list[float] = []
        all_beta: list[np.ndarray] = []
        all_gamma: list[np.ndarray] = []

        log_Z_terms: list[float] = []
        # Flat-likelihood gamma-configs (NS-skipped at init): their
        # Z_gamma is just the constant L times the unit prior volume,
        # so log Z_gamma = log L_const. We also inject the stored
        # prior-sampled points so the flat-L mass is REPRESENTED in
        # the importance-resampled joint cloud (otherwise inclusion
        # marginals are biased toward gamma-configs with non-flat L).
        for gamma_tuple, log_L_const in self._flat_log_L.items():
            log_p_gamma = self._log_inclusion_prior(
                np.array(gamma_tuple, dtype=bool)
            )
            log_Z_terms.append(log_p_gamma + log_L_const)
            flat_pts = self._flat_init_points.get(gamma_tuple, [])
            if flat_pts:
                gamma_arr_flat = np.array(gamma_tuple, dtype=bool)
                # Each stored prior-sampled point carries weight
                #   w_i = P(gamma) * L_const * (1/n_pts)
                # so total weight sums to P(gamma) * L_const = exp(log_Z_term).
                log_w_per_pt = log_p_gamma + log_L_const - float(
                    np.log(len(flat_pts))
                )
                for beta_pt in flat_pts:
                    all_log_w.append(log_w_per_pt)
                    all_log_L.append(log_L_const)
                    all_beta.append(beta_pt)
                    all_gamma.append(gamma_arr_flat.copy())

        for pop in self.populations.values():
            log_p_gamma = self._log_inclusion_prior(
                np.array(pop.gamma_tuple, dtype=bool)
            )
            log_Z_terms.append(log_p_gamma + pop.acc.log_Z)
            gamma_arr = np.array(pop.gamma_tuple, dtype=bool)
            for i in range(len(pop.dead_log_likelihood)):
                # Per-dead-point joint log weight =
                #   log P(gamma) + log L_i + log dX_gamma,i.
                # Recall the population's dead_log_weight is log dX_gamma,
                # so we add log L and log P(gamma) here.
                all_log_w.append(
                    log_p_gamma
                    + pop.dead_log_likelihood[i]
                    + pop.dead_log_weight[i]
                )
                all_log_L.append(pop.dead_log_likelihood[i])
                all_beta.append(pop.dead_beta[i])
                all_gamma.append(gamma_arr.copy())

        if not log_Z_terms:
            raise RuntimeError("No populations recorded any dead points.")

        log_Z = float(np.logaddexp.reduce(log_Z_terms))
        # Aggregate H from per-population Hs weighted by their joint
        # contribution. This is approximate but good enough for the
        # final NS error.
        # joint_w is over (flat-L pops then live pops); only the live
        # pops contribute meaningful H (flat-L pops have zero entropy).
        joint_w = np.array(
            [np.exp(lt - log_Z) for lt in log_Z_terms], dtype=np.float64
        )
        live_pop_count = len(self.populations)
        live_joint_w = joint_w[-live_pop_count:] if live_pop_count else np.zeros(0)
        Hs = np.array(
            [p.acc.H for p in self.populations.values()], dtype=np.float64
        )
        H = float(np.sum(live_joint_w * Hs)) if Hs.size else 0.0
        n_live_avg = max(
            float(np.mean([p.acc.n_live for p in self.populations.values()]))
            if self.populations else 1.0,
            1.0,
        )
        log_Z_err = float(np.sqrt(max(H, 0.0) / n_live_avg))

        # Importance resample to equal-weight posterior draws.
        log_w_arr = np.array(all_log_w, dtype=np.float64)
        log_L_arr = np.array(all_log_L, dtype=np.float64)
        beta_arr = np.array(all_beta, dtype=np.float64)
        gamma_arr_full = np.array(all_gamma, dtype=bool)

        # Normalise weights.
        w_log = log_w_arr - float(np.logaddexp.reduce(log_w_arr))
        w = np.exp(w_log)
        # Resample n_resample = sum(w)**2 / sum(w**2) (effective sample size)
        # multinomial draws with replacement.
        n_resample = int(min(max(int(np.sum(w) ** 2 / np.sum(w * w)), 10), 50_000))
        n_resample = max(n_resample, 1)
        idx = self.rng.choice(
            beta_arr.shape[0], size=n_resample, replace=True, p=w / w.sum()
        )
        samples_eq = beta_arr[idx]
        gamma_eq = gamma_arr_full[idx]

        n_iter_total = int(sum(p.acc.n_iter for p in self.populations.values()))

        return Results(
            samples=samples_eq,
            gamma=gamma_eq,
            log_likelihoods=log_L_arr[idx],
            log_weights=log_w_arr[idx],
            log_Z=log_Z,
            log_Z_err=log_Z_err,
            H=H,
            n_iter=n_iter_total,
            n_calls=n_calls,
            group_names=[g.name for g in self.groups],
            inclusion_priors=self._inclusion_priors,
        )
