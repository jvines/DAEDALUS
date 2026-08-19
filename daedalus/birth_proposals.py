"""User-pluggable birth proposals for trans-dim moves.

The MoMS flip kernel calls a `BirthProposal` whenever it attempts to flip
a group's inclusion indicator from 0 to 1. Each proposal supplies:

    propose(group, state, rng) -> BirthProposalResult
        Sample new beta values for the group's parameters (in physical
        beta-space), and report the proposal density at the sampled point.

    log_density(group, state, beta_query) -> float
        Evaluate the same proposal density at an arbitrary beta_query --
        needed for the reverse direction when the trans-dim kernel does a
        DELETE move and must compute the M-H ratio that would carry the
        chain back into the slab.

The full M-H acceptance ratio for an ADD move (gamma_k: 0 -> 1) under NS
is

    alpha_ADD = (pi_inc / (1 - pi_inc))
              * pi_beta(beta_new) / q_forward(beta_new)
              * 1[L_new > L_threshold]

where ``pi_beta`` is the user's continuous prior on the group's
parameters (supplied as ``Group.log_prior_continuous``) and
``q_forward`` is what `propose` returns. The DELETE direction uses
the symmetric formula with the proposal evaluated at the *current*
beta via ``log_density``.

Default behaviour (``Group.birth_proposal = None``):
    The chain falls back to the uniform-u birth, which draws u ~
    Uniform(0, 1) for the group's u-coordinates and pushes through the
    user's `prior_transform`. Under this proposal the q_beta and
    pi_beta densities cancel exactly (both are the push-forward of the
    uniform u-density), so the M-H ratio collapses to the inclusion
    ratio alone -- no log_prior_continuous needed.

When a custom birth is supplied, ``Group.log_prior_continuous`` MUST
also be supplied: the user is the only one who knows the spike-and-slab
prior shape on their parameters, and the cancellation that saves the
default path no longer holds.

Built-ins:
    GaussianRWBirth   -- adaptive Gaussian random walk in beta-space
                         centered on the off-values (Algorithm 2 of
                         van den Bergh+ 2026 generalised to vector
                         components).
    PriorDrawBirth    -- independent draw from a user-supplied sampler
                         (the proposal density equals the user's prior,
                         so q cancels with pi_beta and the M-H ratio
                         again reduces to the inclusion ratio).
    BLSPeriodBirth    -- Box Least Squares periodogram informed birth
                         for transit-style periodic signals; samples
                         log-period from peaks of a residual BLS
                         spectrum.
    GLSPeriodBirth    -- Generalized Lomb-Scargle periodogram informed
                         birth for sinusoidal-style signals (RV,
                         photometric variability, etc.).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol

import numpy as np

# ``np.trapezoid`` is the NumPy >= 2.0 spelling of the older ``np.trapz``.
# The declared floor is numpy>=2.0, so ``trapezoid`` is always present; the
# fallback keeps this module working if the floor is ever relaxed again.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

if TYPE_CHECKING:
    from .groups import Group
    from .state import State


@dataclass
class BirthProposalResult:
    proposed_beta: np.ndarray
    log_q_forward: float


class BirthProposal(Protocol):
    def propose(
        self,
        group: "Group",
        state: "State",
        rng: np.random.Generator,
    ) -> BirthProposalResult: ...

    def log_density(
        self,
        group: "Group",
        state: "State",
        beta_query: np.ndarray,
    ) -> float: ...

    def update(self, accepted: bool) -> None:
        """Optional adaptation hook called once per MoMS flip attempt.

        NestedSampler invokes ``update(accepted)`` after each ADD/DELETE
        accept-reject decision so that adaptive proposals (such as
        :class:`GaussianRWBirth`) can update internal scales. Proposals
        that do not adapt may omit this method or implement a no-op.
        """
        ...


class GaussianRWBirth:
    """Adaptive Gaussian random walk centered on the group's off-values.

    Faithful translation of Algorithm 2 of van den Bergh+ 2026 to vector
    components: ``beta_new ~ N(off_values, diag(scale^2))``. The proposal
    density is the standard product-of-univariate Gaussians, so
    ``log_density`` is closed-form. ``scale`` may be a scalar (broadcast
    to every parameter in the group) or a 1-D array of length
    ``group.ndim``.

    Centering on the off-value coincides with Algorithm 2's "centered on
    current beta" rule at the moment of an ADD move: when gamma_k = 0 the
    current beta is the off-value. The DELETE move is deterministic
    (returns to the off-value), so the forward and reverse proposals are
    symmetric in the canonical spike-and-slab sense.

    Adaptation
    ----------
    Implements van den Bergh+ 2026 Eq. D1: after each accept/reject
    decision, the (log) proposal scale is updated by

        log tau_{t+1} = log tau_t + (t + 1)^{-phi} * (1{accept} - alpha)

    with phi in (1/2, 1] (default 0.75 per the paper) so that the
    adaptation is diminishing and Roberts-Rosenthal containment holds.
    The default target acceptance is 0.44 for univariate groups and 0.234
    for vector groups (Roberts & Rosenthal 2001 optimal scaling); pass
    ``alpha_target`` to override. Pass ``warmup_steps`` to freeze the
    scale after a fixed number of accept/reject decisions (after which
    the kernel is non-adaptive and the Markov property is exactly
    preserved). The default ``warmup_steps=None`` keeps the diminishing
    adaptation running indefinitely.

    The ``NestedSampler`` calls :meth:`update` once per MoMS flip
    attempt that involves this birth proposal -- both ADD (forward draw
    uses the scale) and DELETE (reverse density uses the scale) -- so the
    adaptation tracks the joint accept rate of the full trans-dim move.
    """

    def __init__(
        self,
        scale: float | np.ndarray = 1.0,
        alpha_target: float | None = None,
        phi: float = 0.75,
        warmup_steps: int | None = None,
        scale_min: float = 1e-5,
        scale_max: float = 1e6,
    ) -> None:
        if not (0.5 < phi <= 1.0):
            raise ValueError(
                f"phi must be in (1/2, 1] for Robbins-Monro convergence, got {phi}"
            )
        if alpha_target is not None and not (0.0 < alpha_target < 1.0):
            raise ValueError(
                f"alpha_target must be in (0, 1), got {alpha_target}"
            )
        if scale_min <= 0.0 or scale_max <= scale_min:
            raise ValueError(
                f"require 0 < scale_min < scale_max, got "
                f"scale_min={scale_min}, scale_max={scale_max}"
            )
        self._init_scale = scale
        self.alpha_target = alpha_target
        self.phi = phi
        self.warmup_steps = warmup_steps
        self.scale_min = scale_min
        self.scale_max = scale_max
        self._scale_arr: np.ndarray | None = None
        self._alpha_target_resolved: float | None = None
        self._t: int = 0

    @property
    def scale(self) -> np.ndarray | float:
        """Current proposal scale. Materialised on first ``propose`` call.

        Returns the per-coordinate scale array once a group has been seen,
        otherwise the constructor argument (scalar or array). Adaptive
        updates mutate this array in place.
        """
        return self._scale_arr if self._scale_arr is not None else self._init_scale

    def _scale_array(self, group: "Group") -> np.ndarray:
        """Return the working scale array, materialising it on first use."""
        if self._scale_arr is not None:
            if self._scale_arr.size != group.ndim:
                raise ValueError(
                    f"GaussianRWBirth previously initialised for ndim="
                    f"{self._scale_arr.size} but called with group.ndim="
                    f"{group.ndim}; reuse on a different group is not supported"
                )
            return self._scale_arr
        if np.isscalar(self._init_scale):
            self._scale_arr = np.full(group.ndim, float(self._init_scale))
        else:
            scale_arr = np.asarray(self._init_scale, dtype=np.float64)
            if scale_arr.size != group.ndim:
                raise ValueError(
                    f"GaussianRWBirth scale array length {scale_arr.size} does not "
                    f"match group.ndim = {group.ndim}"
                )
            self._scale_arr = scale_arr.copy()
        if self._alpha_target_resolved is None:
            # Roberts & Rosenthal 2001 optimal scaling: 0.44 univariate,
            # 0.234 multivariate. Paper App D uses 0.44 for the per-
            # coefficient updates of Algorithm 2 (group.ndim == 1).
            self._alpha_target_resolved = (
                float(self.alpha_target) if self.alpha_target is not None
                else (0.44 if group.ndim == 1 else 0.234)
            )
        return self._scale_arr

    def propose(
        self,
        group: "Group",
        state: "State",
        rng: np.random.Generator,
    ) -> BirthProposalResult:
        scale = self._scale_array(group)
        off = group.off_values
        delta = rng.standard_normal(group.ndim) * scale
        proposed = off + delta
        # log N(proposed; off, diag(scale^2))
        log_q = float(
            -0.5 * np.sum((delta / scale) ** 2)
            - np.sum(np.log(scale))
            - 0.5 * group.ndim * float(np.log(2.0 * np.pi))
        )
        return BirthProposalResult(proposed_beta=proposed, log_q_forward=log_q)

    def log_density(
        self,
        group: "Group",
        state: "State",
        beta_query: np.ndarray,
    ) -> float:
        scale = self._scale_array(group)
        delta = beta_query - group.off_values
        return float(
            -0.5 * np.sum((delta / scale) ** 2)
            - np.sum(np.log(scale))
            - 0.5 * group.ndim * float(np.log(2.0 * np.pi))
        )

    def update(self, accepted: bool) -> None:
        """Robbins-Monro adaptive Metropolis update (van den Bergh+ 2026, Eq. D1).

        Called by NestedSampler once per accept/reject decision. No-op if
        ``propose``/``log_density`` has not been called yet (scale not
        materialised) or if the warmup budget has been exhausted.
        """
        if self._scale_arr is None or self._alpha_target_resolved is None:
            return
        if self.warmup_steps is not None and self._t >= self.warmup_steps:
            return
        gamma_t = (self._t + 1.0) ** (-self.phi)
        signal = (1.0 if accepted else 0.0) - self._alpha_target_resolved
        log_scale = np.log(self._scale_arr) + gamma_t * signal
        np.exp(log_scale, out=log_scale)
        np.clip(log_scale, self.scale_min, self.scale_max, out=self._scale_arr)
        self._t += 1


class PriorDrawBirth:
    """Independent draw from a user-supplied prior sampler.

    The proposal IS the prior, so the q_beta and pi_beta densities
    cancel in the M-H ratio. The user supplies a `sampler` callable
    that returns a draw of length `group.ndim` and a `log_density`
    callable that evaluates the same prior at an arbitrary beta. This
    is the right birth when the prior is the natural starting point
    (broad, smooth, no informed structure) but you want a cold draw
    every flip rather than a Gaussian random walk near the off-value.
    """

    def __init__(
        self,
        sampler: Callable[[np.random.Generator], np.ndarray],
        log_density_fn: Callable[[np.ndarray], float],
    ) -> None:
        self._sampler = sampler
        self._log_density = log_density_fn

    def propose(
        self,
        group: "Group",
        state: "State",
        rng: np.random.Generator,
    ) -> BirthProposalResult:
        proposed = np.asarray(self._sampler(rng), dtype=np.float64)
        if proposed.size != group.ndim:
            raise ValueError(
                f"PriorDrawBirth sampler returned {proposed.size} values, "
                f"expected {group.ndim} for group {group.name!r}"
            )
        return BirthProposalResult(
            proposed_beta=proposed, log_q_forward=float(self._log_density(proposed))
        )

    def log_density(
        self,
        group: "Group",
        state: "State",
        beta_query: np.ndarray,
    ) -> float:
        return float(self._log_density(beta_query))

    def update(self, accepted: bool) -> None:
        """No-op: independence proposal from a fixed prior has no scale to adapt."""


@dataclass
class _PeriodCDF:
    """Inverse-CDF sampler over log-period from a precomputed periodogram.

    Internal helper for :class:`BLSPeriodBirth` and :class:`GLSPeriodBirth`.
    Stores the periodogram-derived (unnormalised) log-density on a
    log-period grid, the cumulative density for inverse-CDF sampling, and
    the log normaliser.
    """

    log_periods: np.ndarray  # ascending
    log_density: np.ndarray  # log of unnormalised density at each log period
    cdf: np.ndarray          # cumulative normalised density, last entry = 1.0
    log_norm: float          # log of the integral that normalises density
    # Optional: for box-shaped (BLS) periodograms the matched-filter
    # transit_time per trial period. None when the backend doesn't
    # supply one (e.g. Lomb-Scargle). Same length as log_periods.
    transit_time_at_logp: np.ndarray | None = None
    # Optional: matched-filter best-fit depth per trial period. For BLS
    # this is the in-transit flux deficit; for a Mandel-Agol slot it
    # informs the rp/R* proposal via rr ~ sqrt(depth).
    depth_at_logp: np.ndarray | None = None

    def sample_log_period(self, rng: np.random.Generator) -> float:
        u = float(rng.uniform())
        idx = int(np.searchsorted(self.cdf, u))
        idx = min(idx, self.log_periods.size - 1)
        return float(self.log_periods[idx])

    def transit_time_at(self, log_p: float) -> float | None:
        if self.transit_time_at_logp is None:
            return None
        idx = int(np.searchsorted(self.log_periods, log_p))
        idx = min(idx, self.log_periods.size - 1)
        return float(self.transit_time_at_logp[idx])

    def depth_at(self, log_p: float) -> float | None:
        if self.depth_at_logp is None:
            return None
        idx = int(np.searchsorted(self.log_periods, log_p))
        idx = min(idx, self.log_periods.size - 1)
        return float(self.depth_at_logp[idx])

    def log_density_at(self, log_p: float) -> float:
        if log_p < self.log_periods[0] or log_p > self.log_periods[-1]:
            return -np.inf
        idx = int(np.searchsorted(self.log_periods, log_p))
        idx = min(idx, self.log_periods.size - 1)
        return float(self.log_density[idx]) - self.log_norm


def _build_period_cdf(
    log_periods: np.ndarray,
    raw_power: np.ndarray,
    sharpening: float,
    transit_time: np.ndarray | None = None,
    depth: np.ndarray | None = None,
) -> _PeriodCDF:
    """Convert a periodogram power array into a normalised log-period density.

    The mapping is ``density(log_p) ~ exp(sharpening * power_normalised)``
    with `power` rescaled to ``[0, 1]`` first. Higher ``sharpening`` puts
    more mass on the tallest peaks; sharpening=0 recovers a uniform
    proposal in log-period.

    Optional ``transit_time`` (one value per trial period) is carried
    through unchanged for box-shaped periodograms (BLS) whose
    matched-filter output also supplies a best-fit transit time. Used
    by :class:`BLSPeriodBirth` to propose ``t0`` jointly with the
    period rather than drawing it uniformly.
    """
    if log_periods.size != raw_power.size:
        raise ValueError(
            f"log_periods.size={log_periods.size} must equal raw_power.size={raw_power.size}"
        )
    if log_periods.size < 2:
        raise ValueError("need at least 2 grid points to build a CDF")
    if transit_time is not None and transit_time.size != raw_power.size:
        raise ValueError(
            f"transit_time.size={transit_time.size} must equal raw_power.size={raw_power.size}"
        )
    p_min = float(np.min(raw_power))
    p_max = float(np.max(raw_power))
    norm_power = (raw_power - p_min) / max(p_max - p_min, 1e-12)
    log_density = sharpening * norm_power
    log_max = float(np.max(log_density))
    density = np.exp(log_density - log_max)
    dx = (log_periods[-1] - log_periods[0]) / (log_periods.size - 1)
    integral = float(_trapezoid(density, dx=dx))
    log_norm = log_max + float(np.log(integral))
    cdf = np.cumsum(density) / float(np.sum(density))
    return _PeriodCDF(
        log_periods=log_periods.copy(),
        log_density=log_density,
        cdf=cdf,
        log_norm=log_norm,
        transit_time_at_logp=(
            np.asarray(transit_time, dtype=np.float64).copy()
            if transit_time is not None else None
        ),
        depth_at_logp=(
            np.asarray(depth, dtype=np.float64).copy()
            if depth is not None else None
        ),
    )


class _PeriodogramBirth:
    """Shared machinery for periodogram-informed birth proposals.

    Subclasses implement :meth:`_compute_periodogram_power` to produce
    the raw periodogram for a given residual time series. The rest of
    the kernel -- the sample/density mixture, gamma-keyed CDF caching,
    optional harmonic suppression, and the M-H ratio plumbing -- lives
    here.

    The proposal samples the log-period coordinate from a periodogram
    of the *current chain residuals* (so once one signal has been
    fitted, the next birth lights up on the next unfit periodicity)
    and the remaining slot coordinates uniformly from user-specified
    boxes. The informed sampler is mixed with a uniform fallback at
    weight ``1 - alpha`` to keep the chain ergodic on multi-modal
    posteriors.

    Parameters
    ----------
    time
        Time array of the data, passed to the periodogram backend.
    residual_fn
        Callable ``residual_fn(state) -> np.ndarray`` returning the
        residual flux/RV/spectrum after subtracting the model implied
        by ``state``. The CDF is rebuilt the first time each distinct
        ``state.gamma`` is encountered.
    prior_log_period
        ``(lo, hi)`` log-period prior bounds.
    non_period_priors
        Sequence of ``(lo, hi)`` uniform priors for the slot's
        non-period parameters, in the order they appear in
        ``Group.params`` after the period coordinate is removed. Total
        length must equal ``group.ndim - 1``.
    period_param_offset
        Index within ``Group.params`` that holds the log-period
        coordinate. Default 0.
    n_periods
        Periodogram resolution (number of trial periods).
    sharpening
        Density sharpening factor; see :func:`_build_period_cdf`.
    alpha
        Mixture weight on the informed component; ``1 - alpha`` is
        the uniform fallback.
    active_periods_fn
        Optional callable ``state -> list[float]`` returning the
        currently-fitted period of each active slot in physical units
        (days). When provided, the CDF is multiplicatively suppressed
        within ``+/- harmonic_window_log`` of every harmonic of every
        active period before sampling. When ``None``, harmonic
        suppression is off.
    harmonic_ratios, harmonic_window_log
        Suppression configuration; see :meth:`_apply_harmonic_mask`.
    active_period_cache_precision_dex
        Periodogram CDF cache key precision in dex of ``log P``. The
        cache is keyed by the tuple ``(gamma, rounded log P of every
        active candidate)`` so the residuals fed to the periodogram
        always reflect the chain's current MAP for active slots.
        Smaller values = more aggressive cache invalidation (so the
        residual subtraction tracks chain refinement more closely) at
        the cost of more periodogram recomputations. The default of
        ``1e-3`` corresponds to ~0.23 % period precision (~80 s on a
        4 d planet). Has no effect when ``active_periods_fn`` is
        ``None``, since residuals then can't depend on beta in a way
        the cache could detect.
    """

    def __init__(
        self,
        time: np.ndarray,
        residual_fn: Callable[["State"], np.ndarray],
        prior_log_period: tuple[float, float],
        non_period_priors: Sequence[tuple[float, float]],
        period_param_offset: int = 0,
        active_periods_fn: Callable[["State"], list[float]] | None = None,
        n_periods: int = 5000,
        sharpening: float = 30.0,
        alpha: float = 0.7,
        harmonic_ratios: Sequence[float] = (1.0 / 3.0, 0.5, 1.0, 2.0, 3.0),
        harmonic_window_log: float = 0.05,
        active_period_cache_precision_dex: float = 1e-3,
        shared_cdf_cache: dict | None = None,
        cdf_cache_max_entries: int = 256,
    ) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if prior_log_period[0] >= prior_log_period[1]:
            raise ValueError(
                f"prior_log_period must be (lo, hi) with lo < hi, "
                f"got {prior_log_period}"
            )
        if n_periods < 2:
            raise ValueError(f"n_periods must be >= 2, got {n_periods}")
        if period_param_offset < 0:
            raise ValueError(
                f"period_param_offset must be non-negative, got {period_param_offset}"
            )
        self.time = np.asarray(time, dtype=np.float64)
        self.residual_fn = residual_fn
        self.prior_log_period = (
            float(prior_log_period[0]),
            float(prior_log_period[1]),
        )
        self.non_period_priors = tuple(
            (float(lo), float(hi)) for lo, hi in non_period_priors
        )
        for lo, hi in self.non_period_priors:
            if lo >= hi:
                raise ValueError(
                    f"each non_period_priors entry must be (lo, hi) with lo < hi; "
                    f"got ({lo}, {hi})"
                )
        self.period_param_offset = int(period_param_offset)
        self.active_periods_fn = active_periods_fn
        self.n_periods = int(n_periods)
        self.sharpening = float(sharpening)
        self.alpha = float(alpha)
        self.harmonic_ratios = tuple(float(r) for r in harmonic_ratios)
        self.harmonic_window_log = float(harmonic_window_log)
        if active_period_cache_precision_dex <= 0:
            raise ValueError(
                f"active_period_cache_precision_dex must be > 0, got "
                f"{active_period_cache_precision_dex}"
            )
        self.active_period_cache_precision_dex = float(
            active_period_cache_precision_dex
        )
        # CDF cache keyed by (gamma, active-period-hash). Building the
        # periodogram is the expensive step (~1-3 s per call on TESS-sized
        # data), so we pay it once per distinct (gamma, active-period)
        # tuple. The active-period hash is only included when an
        # ``active_periods_fn`` is provided (otherwise residuals can't
        # depend on beta in a way the cache could detect anyway), and the
        # precision is set by ``active_period_cache_precision_dex``: when
        # an active period shifts by more than that many dex of log P
        # the cache invalidates and the periodogram is recomputed against
        # the refreshed residuals.
        # Caller may inject a pre-existing dict so that multiple
        # BirthProposal instances built on the same residual_fn (e.g. all
        # planet slots in a multi-candidate transit search) share their
        # periodogram cache. With trans-dim sampling the gamma state
        # changes frequently; without a shared cache, every slot
        # independently recomputes BLS for the same (gamma, active
        # periods) tuple. Sharing collapses that to one computation.
        # The cache is BOUNDED (LRU, insertion-ordered dict): each entry
        # holds O(n_periods) float arrays (~60 KB at n_periods = 2500),
        # and in a trans-dim run with several active wandering periods
        # the (gamma, rounded-active-periods) key space is effectively
        # unbounded, so an uncapped cache grows linearly with exploration
        # and can reach tens of GB over a long multi-slot run. Eviction
        # only costs a periodogram recomputation.
        if cdf_cache_max_entries < 1:
            raise ValueError(
                f"cdf_cache_max_entries must be >= 1, got "
                f"{cdf_cache_max_entries}"
            )
        self.cdf_cache_max_entries = int(cdf_cache_max_entries)
        self._cdf_by_gamma: dict[bytes, _PeriodCDF] = (
            shared_cdf_cache if shared_cdf_cache is not None else {}
        )

    # --- subclass extension point ----------------------------------------

    def _compute_periodogram_power(
        self,
        time: np.ndarray,
        residuals: np.ndarray,
        periods: np.ndarray,
    ) -> np.ndarray:
        """Return periodogram power on the given trial periods. Override
        in subclasses to plug in BLS, Lomb-Scargle, etc."""
        raise NotImplementedError

    def _compute_periodogram(
        self,
        time: np.ndarray,
        residuals: np.ndarray,
        periods: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (power, transit_time) on the given trial periods.

        Default implementation falls back to :meth:`_compute_periodogram_power`
        and returns ``transit_time = None``. Subclasses whose backend
        supplies a matched-filter transit time per trial period
        (currently BLS) override this to return the transit_time
        array so that :class:`BLSPeriodBirth` can use it for joint
        ``(P, t0)`` proposals.
        """
        power = self._compute_periodogram_power(time, residuals, periods)
        return np.asarray(power, dtype=np.float64), None

    # --- internals -------------------------------------------------------

    @property
    def _log_prior_box_widths(self) -> float:
        return float(sum(np.log(hi - lo) for lo, hi in self.non_period_priors))

    def _active_period_hash(self, state: "State") -> bytes:
        """Compute a low-precision hash of the active candidates' periods.

        Forms part of the periodogram cache key so that the cache is
        invalidated whenever an active candidate's period shifts by more
        than ``active_period_cache_precision_dex`` dex of log P.

        Returns an empty bytes object when no ``active_periods_fn`` was
        provided, in which case the cache reduces to the gamma-only key
        (residuals can't depend on beta in a way the cache can detect).
        """
        if self.active_periods_fn is None:
            return b""
        active_periods = self.active_periods_fn(state)
        if not active_periods:
            return b""
        log_p = np.log(np.asarray(active_periods, dtype=np.float64))
        precision = self.active_period_cache_precision_dex
        rounded = np.round(log_p / precision) * precision
        return rounded.tobytes()

    def _cdf_for_state(self, state: "State") -> _PeriodCDF:
        key = state.gamma.tobytes() + b":" + self._active_period_hash(state)
        cached = self._cdf_by_gamma.get(key)
        if cached is not None:
            # LRU refresh: re-insert so eviction removes the stalest key.
            self._cdf_by_gamma[key] = self._cdf_by_gamma.pop(key)
            return cached
        residuals = np.asarray(self.residual_fn(state), dtype=np.float64)
        log_p_lo, log_p_hi = self.prior_log_period
        log_periods = np.linspace(log_p_lo, log_p_hi, self.n_periods)
        periods = np.exp(log_periods)
        pres = self._compute_periodogram(self.time, residuals, periods)
        # Backwards-compatible: subclasses may return (power, transit_time)
        # 2-tuple or (power, transit_time, depth) 3-tuple.
        if len(pres) == 2:
            power, transit_time = pres
            depth = None
        else:
            power, transit_time, depth = pres
        cdf = _build_period_cdf(
            log_periods, power, self.sharpening,
            transit_time=transit_time, depth=depth,
        )
        while len(self._cdf_by_gamma) >= self.cdf_cache_max_entries:
            self._cdf_by_gamma.pop(next(iter(self._cdf_by_gamma)))
        self._cdf_by_gamma[key] = cdf
        return cdf

    def _apply_harmonic_mask(
        self, cdf: _PeriodCDF, active_periods: Sequence[float]
    ) -> _PeriodCDF:
        """Return a copy of ``cdf`` with density suppressed near every
        active period's harmonics. Active periods are typically retrieved
        via ``self.active_periods_fn(state)`` by the caller.
        """
        if not active_periods or self.active_periods_fn is None:
            return cdf
        log_dens = cdf.log_density.copy()
        log_periods = cdf.log_periods
        for p_active in active_periods:
            log_p_active = float(np.log(p_active))
            for ratio in self.harmonic_ratios:
                log_target = log_p_active + float(np.log(ratio))
                mask = np.abs(log_periods - log_target) < self.harmonic_window_log
                log_dens[mask] -= 100.0
        log_max = float(np.max(log_dens))
        density = np.exp(log_dens - log_max)
        s = float(np.sum(density))
        if s <= 0.0:
            return cdf  # fully suppressed; fall back to the cached version
        cdf_arr = np.cumsum(density) / s
        dx = (log_periods[-1] - log_periods[0]) / (log_periods.size - 1)
        integral = float(_trapezoid(density, dx=dx))
        log_norm = log_max + float(np.log(integral))
        return _PeriodCDF(
            log_periods=log_periods,
            log_density=log_dens,
            cdf=cdf_arr,
            log_norm=log_norm,
        )

    def _mixture_log_density(self, log_p: float, cdf: _PeriodCDF) -> float:
        lp_lo, lp_hi = self.prior_log_period
        if log_p < lp_lo or log_p > lp_hi:
            return -np.inf
        log_q_uniform = -float(np.log(lp_hi - lp_lo))
        log_q_informed = cdf.log_density_at(log_p)
        # Handle the alpha = 1 (pure informed) corner without taking log(0)
        # on the uniform component, and the symmetric alpha = 0 case.
        if self.alpha >= 1.0:
            return log_q_informed if np.isfinite(log_q_informed) else -np.inf
        log_one_minus_alpha = float(np.log(1.0 - self.alpha))
        if not np.isfinite(log_q_informed):
            return log_q_uniform + log_one_minus_alpha
        a = float(np.log(self.alpha)) + log_q_informed
        b = log_one_minus_alpha + log_q_uniform
        return float(np.logaddexp(a, b))

    def _check_layout(self, group: "Group") -> None:
        """Validate that ``group`` has the layout this proposal expects."""
        if group.ndim != 1 + len(self.non_period_priors):
            raise ValueError(
                f"{type(self).__name__} configured for {1 + len(self.non_period_priors)} "
                f"params per slot (1 period + {len(self.non_period_priors)} others), "
                f"but group {group.name!r} has ndim={group.ndim}"
            )
        if not (0 <= self.period_param_offset < group.ndim):
            raise ValueError(
                f"period_param_offset={self.period_param_offset} out of range "
                f"for group {group.name!r} of ndim={group.ndim}"
            )

    # --- BirthProposal interface -----------------------------------------

    def propose(
        self, group: "Group", state: "State", rng: np.random.Generator
    ) -> BirthProposalResult:
        self._check_layout(group)
        cdf = self._cdf_for_state(state)
        if self.active_periods_fn is not None:
            cdf = self._apply_harmonic_mask(cdf, self.active_periods_fn(state))

        if rng.uniform() >= self.alpha:
            lp_lo, lp_hi = self.prior_log_period
            log_p = float(lp_lo + (lp_hi - lp_lo) * rng.uniform())
        else:
            log_p = cdf.sample_log_period(rng)

        proposed = np.empty(group.ndim, dtype=np.float64)
        proposed[self.period_param_offset] = log_p
        non_period_idx = 0
        for j in range(group.ndim):
            if j == self.period_param_offset:
                continue
            lo, hi = self.non_period_priors[non_period_idx]
            proposed[j] = float(lo + (hi - lo) * rng.uniform())
            non_period_idx += 1

        log_q = self._mixture_log_density(log_p, cdf) - self._log_prior_box_widths
        return BirthProposalResult(proposed_beta=proposed, log_q_forward=log_q)

    def log_density(
        self, group: "Group", state: "State", beta_query: np.ndarray
    ) -> float:
        self._check_layout(group)
        cdf = self._cdf_for_state(state)
        if self.active_periods_fn is not None:
            cdf = self._apply_harmonic_mask(cdf, self.active_periods_fn(state))
        log_p = float(beta_query[self.period_param_offset])
        non_period_idx = 0
        for j in range(group.ndim):
            if j == self.period_param_offset:
                continue
            lo, hi = self.non_period_priors[non_period_idx]
            v = float(beta_query[j])
            if v < lo or v > hi:
                return -np.inf
            non_period_idx += 1
        return self._mixture_log_density(log_p, cdf) - self._log_prior_box_widths

    def update(self, accepted: bool) -> None:
        """No-op: the proposal density is fixed by the periodogram, no
        scale to adapt. Periodogram refresh is gated on gamma changes,
        not on accept/reject signals."""


class BLSPeriodBirth(_PeriodogramBirth):
    """BLS-periodogram-informed birth for transit-style signals.

    The proposal samples ``log P`` from peaks of an astropy Box Least
    Squares periodogram of the current chain residuals, mixed with a
    uniform fallback. Appropriate for transit detection in time-series
    photometry where the period of an unfit transit produces a clear
    BLS spike at the true period (and aliases at its harmonics --
    optionally suppressed via ``active_periods_fn``).

    Extra parameter beyond :class:`_PeriodogramBirth`:

    duration
        Trial transit duration (in days) passed to BoxLeastSquares. The
        BLS power scales with ``sqrt(duration)`` and is most peaked
        when the trial duration matches the true transit duration; the
        default of 0.1 day is a reasonable compromise for short-period
        TESS transits. For longer-period systems pass a larger value.

    Requires astropy at call time (not at import).
    """

    def __init__(
        self,
        time: np.ndarray,
        residual_fn: Callable[["State"], np.ndarray],
        prior_log_period: tuple[float, float],
        non_period_priors: Sequence[tuple[float, float]],
        duration: float = 0.1,
        period_param_offset: int = 0,
        active_periods_fn: Callable[["State"], list[float]] | None = None,
        n_periods: int = 5000,
        sharpening: float = 30.0,
        alpha: float = 0.7,
        harmonic_ratios: Sequence[float] = (1.0 / 3.0, 0.5, 1.0, 2.0, 3.0),
        harmonic_window_log: float = 0.05,
        active_period_cache_precision_dex: float = 1e-3,
        informed_t0_offset: int | None = 0,
        t0_phase_sigma_in_durations: float = 1.0,
        t0_alpha: float | None = None,
        informed_rr_offset: int | None = None,    # off by default; depth-informed is an opt-in enhancement
        log_rr_sigma: float = 1.0,
        rr_alpha: float | None = None,
        shared_cdf_cache: dict | None = None,
        cdf_cache_max_entries: int = 256,
    ) -> None:
        """
        ``informed_t0_offset`` (default 0): index *within* ``non_period_priors``
        of the ``t0_phase`` parameter (i.e. position after the log-period
        coordinate is removed from the group's params). When set, the
        proposal samples ``t0_phase`` from a wrapped-normal centred on the
        BLS-fitted transit time at the sampled period (with width
        ``t0_phase_sigma_in_durations`` transit durations in phase),
        mixed against a uniform fallback at weight ``t0_alpha``. Set to
        ``None`` to keep the uniform ``t0_phase`` proposal of older
        revisions.

        ``t0_phase_sigma_in_durations`` (default 1.0): wrapped-normal
        width as a multiple of (duration / period). 1 transit duration
        gives a proposal whose ``+/- 1 sigma`` brackets the in-transit
        window -- tight enough to keep b's narrow ridge findable, wide
        enough that the proposal is never an effective delta and the
        reverse density stays finite for the M-H ratio.

        ``t0_alpha``: mixture weight on the BLS-informed wrapped-normal
        component; ``1 - t0_alpha`` is the uniform fallback. Defaults
        to ``alpha`` (the period mixture weight) when not set.
        """
        super().__init__(
            time=time,
            residual_fn=residual_fn,
            prior_log_period=prior_log_period,
            non_period_priors=non_period_priors,
            period_param_offset=period_param_offset,
            active_periods_fn=active_periods_fn,
            n_periods=n_periods,
            sharpening=sharpening,
            alpha=alpha,
            harmonic_ratios=harmonic_ratios,
            harmonic_window_log=harmonic_window_log,
            active_period_cache_precision_dex=active_period_cache_precision_dex,
            shared_cdf_cache=shared_cdf_cache,
            cdf_cache_max_entries=cdf_cache_max_entries,
        )
        if duration <= 0.0:
            raise ValueError(f"duration must be positive, got {duration}")
        self.duration = float(duration)
        if informed_t0_offset is not None:
            if informed_t0_offset < 0 or informed_t0_offset >= len(non_period_priors):
                raise ValueError(
                    f"informed_t0_offset={informed_t0_offset} out of range for "
                    f"{len(non_period_priors)} non-period priors"
                )
            lo, hi = non_period_priors[informed_t0_offset]
            if not (lo == 0.0 and hi == 1.0):
                raise ValueError(
                    f"informed_t0_offset slot must have prior (0.0, 1.0) "
                    f"(t0_phase normalised to fraction of period), got ({lo}, {hi})"
                )
        if t0_phase_sigma_in_durations <= 0.0:
            raise ValueError(
                f"t0_phase_sigma_in_durations must be > 0, got "
                f"{t0_phase_sigma_in_durations}"
            )
        self.informed_t0_offset = informed_t0_offset
        self.t0_phase_sigma_in_durations = float(t0_phase_sigma_in_durations)
        self.t0_alpha = float(t0_alpha) if t0_alpha is not None else float(alpha)
        if informed_rr_offset is not None:
            if informed_rr_offset < 0 or informed_rr_offset >= len(non_period_priors):
                raise ValueError(
                    f"informed_rr_offset={informed_rr_offset} out of range for "
                    f"{len(non_period_priors)} non-period priors"
                )
        if log_rr_sigma <= 0.0:
            raise ValueError(f"log_rr_sigma must be > 0, got {log_rr_sigma}")
        self.informed_rr_offset = informed_rr_offset
        self.log_rr_sigma = float(log_rr_sigma)
        self.rr_alpha = float(rr_alpha) if rr_alpha is not None else float(alpha)

    def _compute_periodogram_power(
        self,
        time: np.ndarray,
        residuals: np.ndarray,
        periods: np.ndarray,
    ) -> np.ndarray:
        from astropy import units as u
        from astropy.timeseries import BoxLeastSquares

        bls = BoxLeastSquares(time * u.day, residuals)
        return np.asarray(bls.power(periods * u.day, self.duration * u.day).power)

    def _compute_periodogram(
        self,
        time: np.ndarray,
        residuals: np.ndarray,
        periods: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        from astropy import units as u
        from astropy.timeseries import BoxLeastSquares

        bls = BoxLeastSquares(time * u.day, residuals)
        result = bls.power(periods * u.day, self.duration * u.day)
        power = np.asarray(result.power, dtype=np.float64)
        transit_time = (
            np.asarray(result.transit_time, dtype=np.float64)
            if self.informed_t0_offset is not None else None
        )
        depth = (
            np.asarray(result.depth, dtype=np.float64)
            if self.informed_rr_offset is not None else None
        )
        return power, transit_time, depth

    # ----- joint (P, t0) proposal helpers --------------------------------

    def _t0_phase_mu_at_logp(self, log_p: float, cdf: _PeriodCDF) -> float | None:
        """BLS-implied ``t0_phase = (transit_time / P) mod 1`` at this log P.

        Returns ``None`` when the informed ``t0_phase`` proposal is
        disabled or the cached periodogram doesn't carry a
        ``transit_time`` array (e.g. cached before this feature was
        wired in, or non-BLS backend).
        """
        if self.informed_t0_offset is None or cdf.transit_time_at_logp is None:
            return None
        t_bls = cdf.transit_time_at(log_p)
        if t_bls is None or not np.isfinite(t_bls):
            return None
        P = float(np.exp(log_p))
        return float((t_bls / P) % 1.0)

    def _t0_phase_sigma_at_logp(self, log_p: float) -> float:
        P = float(np.exp(log_p))
        sigma = self.t0_phase_sigma_in_durations * (self.duration / P)
        return float(min(max(sigma, 1e-4), 0.3))

    @staticmethod
    def _circ_dist(t0: float, mu: float) -> float:
        return float(((t0 - mu + 0.5) % 1.0) - 0.5)

    def _log_rr_mu_at_logp(self, log_p: float, cdf: _PeriodCDF) -> float | None:
        """BLS-implied ``log rr ≈ 0.5 * log(BLS_depth)`` at this log P.

        For a central transit, the box-model depth maps to the
        Mandel-Agol radius ratio via ``rr ≈ sqrt(depth)`` (limb
        darkening makes this approximate but it's the right order of
        magnitude and the right place to start the chain). Returns
        ``None`` when the rr-informed proposal is off or the cached
        periodogram doesn't carry a depth array.
        """
        if self.informed_rr_offset is None or cdf.depth_at_logp is None:
            return None
        depth = cdf.depth_at(log_p)
        if depth is None or not np.isfinite(depth) or depth <= 0.0:
            return None
        return 0.5 * float(np.log(depth))

    def _t0_phase_log_density(self, t0: float, mu: float, sigma: float) -> float:
        """Log of the mixture density ``alpha * wrapped_normal + (1-alpha) * uniform``
        evaluated at ``t0`` in [0, 1)."""
        if t0 < 0.0 or t0 > 1.0:
            return -np.inf
        d = self._circ_dist(t0, mu)
        log_pdf_normal = (
            -0.5 * (d / sigma) * (d / sigma)
            - 0.5 * float(np.log(2.0 * np.pi))
            - float(np.log(sigma))
        )
        log_uniform = 0.0  # uniform on [0, 1] has density 1
        if self.t0_alpha >= 1.0:
            return log_pdf_normal
        log_alpha = float(np.log(self.t0_alpha))
        log_one_minus_alpha = float(np.log(1.0 - self.t0_alpha))
        a = log_alpha + log_pdf_normal
        b = log_one_minus_alpha + log_uniform
        return float(np.logaddexp(a, b))

    def _sample_t0_phase(
        self, mu: float, sigma: float, rng: np.random.Generator
    ) -> float:
        if rng.uniform() < self.t0_alpha:
            return float((mu + rng.normal() * sigma) % 1.0)
        return float(rng.uniform())

    # ----- BirthProposal interface overrides ----------------------------

    def propose(
        self, group: "Group", state: "State", rng: np.random.Generator
    ) -> BirthProposalResult:
        if self.informed_t0_offset is None and self.informed_rr_offset is None:
            # Falls back to the fully-uniform behaviour of the base class.
            return super().propose(group, state, rng)
        self._check_layout(group)
        cdf = self._cdf_for_state(state)
        if self.active_periods_fn is not None:
            cdf = self._apply_harmonic_mask(cdf, self.active_periods_fn(state))

        if rng.uniform() >= self.alpha:
            lp_lo, lp_hi = self.prior_log_period
            log_p = float(lp_lo + (lp_hi - lp_lo) * rng.uniform())
        else:
            log_p = cdf.sample_log_period(rng)

        proposed = np.empty(group.ndim, dtype=np.float64)
        proposed[self.period_param_offset] = log_p

        # Informed-t0 setup
        t0_mu = self._t0_phase_mu_at_logp(log_p, cdf) if self.informed_t0_offset is not None else None
        t0_sigma = self._t0_phase_sigma_at_logp(log_p) if self.informed_t0_offset is not None else 0.0
        # Informed-rr setup
        rr_mu = self._log_rr_mu_at_logp(log_p, cdf) if self.informed_rr_offset is not None else None

        log_q_extra = 0.0  # accumulates informed-density contributions
        non_period_idx = 0
        for j in range(group.ndim):
            if j == self.period_param_offset:
                continue
            lo, hi = self.non_period_priors[non_period_idx]
            if non_period_idx == self.informed_t0_offset and t0_mu is not None:
                t0 = self._sample_t0_phase(t0_mu, t0_sigma, rng)
                proposed[j] = t0
                log_q_extra += self._t0_phase_log_density(t0, t0_mu, t0_sigma)
            elif non_period_idx == self.informed_rr_offset and rr_mu is not None:
                # log-normal mixture in the slot's log-rr coordinate
                # (slot prior is uniform on log rr in [lo, hi]).
                rr_mu_clip = float(np.clip(rr_mu, lo, hi))
                if rng.uniform() < self.rr_alpha:
                    log_rr = float(rr_mu_clip + rng.normal() * self.log_rr_sigma)
                    log_rr = float(np.clip(log_rr, lo, hi))
                else:
                    log_rr = float(lo + (hi - lo) * rng.uniform())
                proposed[j] = log_rr
                log_q_extra += self._log_rr_log_density(log_rr, rr_mu_clip, lo, hi)
            else:
                proposed[j] = float(lo + (hi - lo) * rng.uniform())
            non_period_idx += 1

        # The base ``_log_prior_box_widths`` accounts for uniform proposal
        # density on every non-period slot. For each informed slot we
        # ADD log_q_informed to REPLACE the uniform contribution that's
        # being subtracted (box_width=1 -> log=0 for t0; for log_rr the
        # box width is (hi-lo) and we subtract that in _log_density too).
        log_q = (
            self._mixture_log_density(log_p, cdf)
            + log_q_extra
            - self._log_prior_box_widths
        )
        return BirthProposalResult(proposed_beta=proposed, log_q_forward=log_q)

    def _log_rr_log_density(
        self, log_rr: float, mu: float, lo: float, hi: float
    ) -> float:
        if log_rr < lo or log_rr > hi:
            return float("-inf")
        # Normal (in log-rr coord) component
        z = (log_rr - mu) / self.log_rr_sigma
        log_normal = (
            -0.5 * z * z
            - 0.5 * float(np.log(2.0 * np.pi))
            - float(np.log(self.log_rr_sigma))
        )
        # Uniform on [lo, hi] in log-rr coord
        log_uniform = -float(np.log(hi - lo))
        if self.rr_alpha >= 1.0:
            return log_normal
        a = float(np.log(self.rr_alpha)) + log_normal
        b_ = float(np.log(1.0 - self.rr_alpha)) + log_uniform
        # Adjusted so that subtracting _log_prior_box_widths (which
        # includes -log(hi-lo) for log_rr) gives the correct mixture
        # density relative to a fully-uniform reference. We return the
        # informed-mixture density WITH the +log(hi-lo) added back so
        # the caller's subtraction cancels appropriately.
        return float(np.logaddexp(a, b_)) + float(np.log(hi - lo))

    def log_density(
        self, group: "Group", state: "State", beta_query: np.ndarray
    ) -> float:
        if self.informed_t0_offset is None:
            return super().log_density(group, state, beta_query)
        self._check_layout(group)
        cdf = self._cdf_for_state(state)
        if self.active_periods_fn is not None:
            cdf = self._apply_harmonic_mask(cdf, self.active_periods_fn(state))
        log_p = float(beta_query[self.period_param_offset])

        mu = self._t0_phase_mu_at_logp(log_p, cdf)
        sigma = self._t0_phase_sigma_at_logp(log_p)
        log_q_t0 = 0.0
        non_period_idx = 0
        for j in range(group.ndim):
            if j == self.period_param_offset:
                continue
            if non_period_idx == self.informed_t0_offset and mu is not None:
                t0 = float(beta_query[j])
                d = self._t0_phase_log_density(t0, mu, sigma)
                if not np.isfinite(d):
                    return -np.inf
                log_q_t0 = d
            else:
                lo, hi = self.non_period_priors[non_period_idx]
                v = float(beta_query[j])
                if v < lo or v > hi:
                    return -np.inf
            non_period_idx += 1
        return (
            self._mixture_log_density(log_p, cdf)
            + log_q_t0
            - self._log_prior_box_widths
        )


class GLSPeriodBirth(_PeriodogramBirth):
    """Generalised-Lomb-Scargle-informed birth for sinusoidal-style signals.

    Same machinery as :class:`BLSPeriodBirth` but uses astropy's
    :class:`LombScargle` to score trial periods. Appropriate for radial-
    velocity planet detection, photometric variability, or any
    sinusoidal periodicity where BLS's box-shape assumption is wrong.

    Extra parameter beyond :class:`_PeriodogramBirth`:

    nterms
        Number of Fourier terms in the Lomb-Scargle model (default 1).
        Use ``nterms=2`` or higher when the underlying signal is non-
        sinusoidal (eccentric RV orbit, ellipsoidal variations, etc.).

    Requires astropy at call time.
    """

    def __init__(
        self,
        time: np.ndarray,
        residual_fn: Callable[["State"], np.ndarray],
        prior_log_period: tuple[float, float],
        non_period_priors: Sequence[tuple[float, float]],
        nterms: int = 1,
        period_param_offset: int = 0,
        active_periods_fn: Callable[["State"], list[float]] | None = None,
        n_periods: int = 5000,
        sharpening: float = 30.0,
        alpha: float = 0.7,
        harmonic_ratios: Sequence[float] = (1.0 / 3.0, 0.5, 1.0, 2.0, 3.0),
        harmonic_window_log: float = 0.05,
        active_period_cache_precision_dex: float = 1e-3,
        cdf_cache_max_entries: int = 256,
    ) -> None:
        super().__init__(
            time=time,
            residual_fn=residual_fn,
            prior_log_period=prior_log_period,
            non_period_priors=non_period_priors,
            period_param_offset=period_param_offset,
            active_periods_fn=active_periods_fn,
            n_periods=n_periods,
            sharpening=sharpening,
            alpha=alpha,
            harmonic_ratios=harmonic_ratios,
            harmonic_window_log=harmonic_window_log,
            active_period_cache_precision_dex=active_period_cache_precision_dex,
            cdf_cache_max_entries=cdf_cache_max_entries,
        )
        if nterms < 1:
            raise ValueError(f"nterms must be >= 1, got {nterms}")
        self.nterms = int(nterms)

    def _compute_periodogram_power(
        self,
        time: np.ndarray,
        residuals: np.ndarray,
        periods: np.ndarray,
    ) -> np.ndarray:
        from astropy.timeseries import LombScargle

        ls = LombScargle(time, residuals, nterms=self.nterms)
        frequencies = 1.0 / periods
        return np.asarray(ls.power(frequencies))
