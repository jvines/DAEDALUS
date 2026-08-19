"""Within-model constrained-MCMC samplers.

A `Sampler` advances a live point one constrained step. Subclasses expose
`step_once` (a single MCMC move) so the outer NS loop can interleave
within-model moves with trans-dim MoMS flips at frequency
`transdim_fraction`. They also expose `step`, which composes `step_once`
n_steps times and updates any adapted internal state at the end -- this is
the path used in pure continuous-only NS where no trans-dim mixing is
needed.

A `step_once` is not always one elementary MCMC sweep: the slice kernel
performs `n_slices` of them. `sweeps_per_step` reports that factor, and
`run_nested` divides the user's `n_mcmc` sweep budget by it, so `n_mcmc`
means the same amount of mixing effort whichever kernel is chosen.

Samplers operate in *unit-cube* space (`u`-coords). Under any user-supplied
`prior_transform`, the prior on u is uniform on [0, 1]^ndim, which makes
the within-model M-H acceptance ratio reduce to the indicator
1[L(prior_transform(u_*)) > L_*]. The physical parameter vector beta
returned to the user equals `prior_transform(u)` for active coordinates and
the off-values for inactive coordinates (handled in the trans-dim layer).

Built-ins:
    * `UniformSampler`     -- rejection sample from the bound until L > L_*.
                             Exact (no MCMC autocorrelation) but acceptance
                             degrades exponentially with prior volume; only
                             practical when paired with a tight bound.
    * `RandomWalkSampler`  -- adaptive Gaussian RW in u-space, akin to
                             dynesty's `rwalk`. Robust default; tunes step
                             sizes via Robbins-Monro toward 0.234.
    * `RandomSliceSampler` -- random-direction slice sampling, akin to
                             PolyChord's `rslice` and dynesty's `rslice`. No
                             tuning required; better mixing on smooth targets.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .state import State

if TYPE_CHECKING:
    from .bounds import BoundingRegion
    from .groups import Group


class SliceBracketTruncationWarning(UserWarning):
    """A slice step-out exhausted ``max_steps_out`` with the bracket end still
    inside the slice, so that sweep sampled from a bracket strictly smaller
    than its slice. See `RandomSliceSampler.__init__` (``doubling``).
    """


LogLikelihood = Callable[[np.ndarray, np.ndarray], float]
PriorTransform = Callable[[np.ndarray], np.ndarray]


def _noop_apply_off_values(beta: np.ndarray, gamma: np.ndarray) -> None:
    return


ApplyOffValues = Callable[[np.ndarray, np.ndarray], None]


try:
    from . import _kernels

    _wrap_periodic_jit = _kernels.wrap_periodic
    _outside_cube_jit = _kernels.outside_cube_nonperiodic
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - graceful degrade if numba unavailable
    _HAVE_NUMBA = False


def _wrap_periodic(u: np.ndarray, periodic: np.ndarray) -> np.ndarray:
    """Reduce periodic coordinates modulo 1 so they stay in [0, 1).

    Mutates `u` in-place and returns it for chaining. Non-periodic
    coordinates are untouched and may still fall outside the unit cube,
    in which case the caller is expected to reject. ``periodic`` may be a
    Python tuple or an int-typed numpy array; the JIT path is taken when
    numba is available.
    """
    if _HAVE_NUMBA:
        _wrap_periodic_jit(u, np.asarray(periodic, dtype=np.intp))
    else:
        for i in periodic:
            u[i] -= np.floor(u[i])
    return u


def _outside_cube_nonperiodic(u: np.ndarray, periodic_mask: np.ndarray) -> bool:
    """Check whether any non-periodic coordinate is outside [0, 1]."""
    if _HAVE_NUMBA:
        return bool(_outside_cube_jit(u, periodic_mask))
    if periodic_mask.size == 0:
        return bool(np.any(u < 0.0) or np.any(u > 1.0))
    free = ~periodic_mask
    return bool(np.any(u[free] < 0.0) or np.any(u[free] > 1.0))


class Sampler(ABC):
    """Base class for within-model constrained-MCMC kernels."""

    def bind(
        self,
        loglike: LogLikelihood,
        prior_transform: PriorTransform,
        ndim: int,
        groups: Sequence["Group"] = (),
        periodic: Sequence[int] = (),
        apply_off_values: ApplyOffValues = _noop_apply_off_values,
    ) -> None:
        self._loglike = loglike
        self._prior_transform = prior_transform
        self._ndim = ndim
        self._groups = tuple(groups)
        self._periodic = tuple(periodic)
        self._periodic_mask = np.zeros(ndim, dtype=bool)
        for i in self._periodic:
            self._periodic_mask[i] = True
        self._apply_off_values = apply_off_values

    @abstractmethod
    def step_once(
        self,
        state: State,
        L_threshold: float,
        bound: "BoundingRegion",
        rng: np.random.Generator,
    ) -> tuple[State, int]:
        """One within-model constrained MCMC move. Returns (new_state, n_calls)."""

    def step(
        self,
        state: State,
        L_threshold: float,
        bound: "BoundingRegion",
        rng: np.random.Generator,
        n_steps: int,
    ) -> tuple[State, int]:
        """Compose `step_once` n_steps times. Default impl; subclasses may
        override to amortise per-step setup or apply end-of-batch adaptation.
        """
        n_calls = 0
        for _ in range(n_steps):
            state, calls = self.step_once(state, L_threshold, bound, rng)
            n_calls += calls
        self._on_batch_complete()
        return state, n_calls

    def _on_batch_complete(self) -> None:
        """Hook called at the end of each `step` batch (e.g. for adaptation)."""

    @property
    def sweeps_per_step(self) -> int:
        """Elementary MCMC sweeps performed by a single `step_once`.

        `n_mcmc` is a per-replacement *sweep* budget, and `run_nested`
        divides it by this factor to get the number of `step_once` calls, so
        the budget buys the same amount of mixing whichever kernel is
        selected. Every kernel here does one sweep per `step_once` except
        `RandomSliceSampler`, which does `n_slices`.
        """
        return 1

    def recommended_n_mcmc(self, ndim: int) -> int:
        """Default within-model sweep budget when ``run_nested`` is called
        with ``n_mcmc=None``.

        The random-walk / differential-evolution decorrelation time grows
        ~linearly with dimension, so a fixed step count under-mixes in high
        dimension and biases the evidence high (measured: rwalk at 25 steps
        gives a +0.67 evidence bias at 20D, ~3x its analytic error; 5*ndim
        removes it). The 25 floor preserves the cheap low-dimensional
        default. Kernels with different mixing behaviour override this.
        """
        return max(25, 5 * ndim)


class UniformSampler(Sampler):
    """Rejection sample from the bound until the likelihood threshold is satisfied.

    Each call draws u from `bound.sample`, maps it to physical params via
    `prior_transform`, evaluates `loglike`, and returns the first point with
    `log_L > L_threshold`. The accepted point is independent of the input
    state -- there is no MCMC autocorrelation, so a single `step_once` call
    is sufficient per replacement (n_mcmc has no effect on mixing here).

    Acceptance rate is V_bound_inside_constraint / V_bound. Use only with
    bounds that approximate the constrained prior support (single/multi
    ellipsoid). With `NoBound`, acceptance shrinks like the prior volume X
    and the sampler degenerates after a few hundred iterations on any
    multi-dim problem.
    """

    def __init__(self, max_attempts: int = 100_000) -> None:
        self.max_attempts = max_attempts

    def step_once(
        self,
        state: State,
        L_threshold: float,
        bound: "BoundingRegion",
        rng: np.random.Generator,
    ) -> tuple[State, int]:
        n = 0
        for _ in range(self.max_attempts):
            u = bound.sample(rng)
            if self._periodic:
                _wrap_periodic(u, self._periodic)
            if _outside_cube_nonperiodic(u, self._periodic_mask):
                continue
            n += 1
            beta = np.asarray(self._prior_transform(u), dtype=np.float64)
            self._apply_off_values(beta, state.gamma)
            log_L = float(self._loglike(beta, state.gamma))
            if log_L > L_threshold:
                # In-place mutation: avoids a fresh State allocation +
                # gamma.copy() per accepted draw. The donor State is owned
                # by the replacement loop (a snapshot of the live cloud)
                # so mutation is safe.
                state.u[:] = u
                state.beta[:] = beta
                state.log_likelihood = log_L
                return state, n
        raise RuntimeError(
            f"UniformSampler exhausted {self.max_attempts} attempts without "
            f"finding a point with log_L > {L_threshold}; the bound likely no "
            f"longer overlaps the constrained support. Switch to a tighter "
            f"bound (single/multi) or a Markov sampler (rwalk/rslice)."
        )

    def step(
        self,
        state: State,
        L_threshold: float,
        bound: "BoundingRegion",
        rng: np.random.Generator,
        n_steps: int,
    ) -> tuple[State, int]:
        # Rejection samples are i.i.d. so n_steps > 1 wastes work.
        return self.step_once(state, L_threshold, bound, rng)

    def recommended_n_mcmc(self, ndim: int) -> int:
        # Rejection draws are independent; the step count has no effect.
        return 1


class RandomWalkSampler(Sampler):
    """Adaptive Gaussian random walk in the unit-cube subspace.

    Each `step_once` proposes ``u_new = u + scale * proposal_cov @ z`` where
    ``z`` is an iid standard normal vector. ``proposal_cov`` is taken from
    the bound's natural metric (the Cholesky factor for an ellipsoid),
    defaulting to the identity for `NoBound`. M-H acceptance reduces to the
    indicator ``1[u_new in [0,1]^ndim] * 1[L(prior_transform(u_new)) > L_*]``
    because the prior on u is uniform and the proposal is symmetric.

    The scale is adapted at the end of each `step` batch toward a 0.234
    acceptance rate (Roberts & Rosenthal 2001) via a multiplicative log-
    scale update; clamped to ``[scale_min, scale_max]`` to prevent runaway
    under degenerate constraints. The adaptation is perpetual rather than
    diminishing because the constrained-MCMC target distribution shifts
    every NS iteration as the likelihood threshold rises -- a Robbins-
    Monro schedule that freezes early would lose the ability to track the
    changing geometry. Each batch's kernel is fixed (n_mcmc steps with
    the same scale), so the chain producing each replacement live point
    is a proper M-H kernel; only the *between*-batch scale evolves. This
    matches dynesty's `rwalk` convention (Speagle 2020).
    """

    def __init__(
        self,
        scale: float = 1.0,
        scale_min: float = 1e-2,
        scale_max: float = 1e2,
        target_accept: float = 0.5,
        proposal: str = "ball",
    ) -> None:
        """
        ``proposal``: ``"ball"`` (default, matches dynesty's rwalk) draws
        steps uniformly on the n-ball of radius ``scale`` transformed by
        the bound axes -- bounded step length, no heavy tails.
        ``"gaussian"`` uses iid standard-normal steps (Robbins-Monro-
        compatible but tail-heavy, can jump off narrow ridges).

        ``scale_min``: floor on the adapted step scale, as a fraction of
        the bound's own extent. This is a *correctness* guard, not a
        tuning knob. On a razor-thin likelihood ridge the acceptance-rate
        controller can satisfy ``target_accept`` by shrinking the step
        toward zero instead of by mixing: arbitrarily small steps barely
        change L, so they are accepted at whatever rate is asked for
        while the chain stays put. The acceptance statistic then reads
        perfectly healthy (measured: median 0.500, zero empty batches)
        while the live points are near-clones of their donors and the
        run terminates early in whatever mode it started in.

        A floor of 1e-2 -- steps at least 1% of the bound extent --
        prevents the controller from trading step size for acceptance.
        On the WASP-47 transit benchmark (4-d, razor-thin in t0) the old
        1e-5 floor let the scale collapse to ~3e-5 for the majority of
        batches and the run missed the global mode by 3105 nats; with
        this floor the same run reaches the true global optimum found by
        independent multi-start optimisation (max log L -4624.48) in 5.7x
        fewer likelihood calls than the slice kernel, which was
        previously the only kernel that solved it at all. 1e-3 was also
        measured and is *not* sufficient (still misses the mode).

        ``target_accept``: dynesty defaults to 0.5 for rwalk. The older
        Roberts-Rosenthal 0.234 optimum is for full-posterior Gaussian RW
        MCMC; for NS's constrained MCMC dynesty's 0.5 is the empirical
        choice -- target_accept and scale_bump together are tuned for
        narrow-ridge constrained problems (Speagle 2020).
        """
        if proposal not in ("ball", "gaussian"):
            raise ValueError(f"proposal must be 'ball' or 'gaussian', got {proposal!r}")
        self.scale = scale
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.target_accept = target_accept
        self.proposal = proposal
        self._batch_attempts = 0
        self._batch_accepts = 0
        # Proposal metric for the current batch. The bound does not change
        # mid-batch and the batch kernel is fixed by design (see class
        # docstring), so the containing-ellipsoid Cholesky factor is
        # resolved once per batch in ``step`` rather than re-derived on
        # every walk. ``None`` outside a batch -> ``step_once`` computes it
        # on demand (e.g. for direct callers/tests).
        self._batch_prop: np.ndarray | None = None
        # Pre-drawn per-walk step vectors for the current batch. The whole
        # ball/gaussian draw (RNG + normalisation) is vectorised once per
        # ``step`` into ``_stepbuf`` and consumed by index, turning n_mcmc
        # scalar RNG dispatches per replacement into two batched draws.
        # ``None`` outside a batch (or on overrun) -> single-draw fallback.
        self._stepbuf: np.ndarray | None = None
        self._buf_i = 0

    def _proposal_cov(
        self,
        bound: "BoundingRegion",
        ndim: int,
        u_current: np.ndarray | None = None,
    ) -> np.ndarray:
        # SingleEllipsoid (or any bound exposing a top-level Cholesky): use it.
        L = getattr(bound, "_L", None)
        if isinstance(L, np.ndarray) and L.shape == (ndim, ndim):
            return L
        # MultiEllipsoid: find the ellipsoid containing the donor and use its
        # local Cholesky factor. Previously this code path fell through to
        # ``np.eye(ndim)`` because MultiEllipsoid stores per-cluster L matrices
        # in ``_ellipsoids[k].L`` rather than a top-level ``_L`` -- so rwalk
        # silently used isotropic steps in u-space and couldn't follow narrow
        # ridges in multimodal posteriors. The chain collapsed onto whatever
        # local mode it first encountered, with delta-function posterior width
        # in each direction.
        ells = getattr(bound, "_ellipsoids", None)
        if ells:
            if u_current is not None:
                for ell in ells:
                    if ell.contains(u_current):
                        return ell.L
            # Donor not contained in any ellipsoid (rare, e.g. just after a
            # MoMS flip remaps u). Fall back to the volume-weighted largest
            # ellipsoid -- still better than isotropic identity.
            biggest = max(ells, key=lambda e: e.log_det_L)
            return biggest.L
        return np.eye(ndim)

    def _fill_step_buffer(
        self, n: int, ndim: int, rng: np.random.Generator
    ) -> None:
        """Vectorise the whole batch of ``n`` per-walk position increments
        in one pair of RNG draws plus one matmul. For ``ball`` the Marsaglia
        direction + uniform-radius normalisation is done on the entire
        ``(n, ndim)`` block at once; then the batch-constant proposal
        rotation and scale are folded in (``scale * Z @ prop.T``), so each
        buffered walk is a single vector add ``u + buf[i]`` with no per-walk
        RNG dispatch or matmul. ``prop`` and ``scale`` are fixed for the
        batch (scale adapts only in ``_on_batch_complete``). Consumed in
        order by ``step_once``; an overrun past ``n`` (walk-until-accept)
        falls back to scalar draws via ``_draw_raw``."""
        Z = rng.standard_normal((n, ndim))
        if self.proposal == "ball":
            radii = rng.uniform(size=n) ** (1.0 / ndim)
            norms = np.sqrt(np.einsum("ij,ij->i", Z, Z)) + 1e-300
            Z *= (radii / norms)[:, None]
        # A degenerate/very wide bound can push an increment to +/-inf; that
        # is by design (an inf proposal lands outside the cube and is rejected
        # for free, see step_once), so the expected FPE is silenced here to
        # match the per-walk path's behaviour.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            self._stepbuf = self.scale * (Z @ self._batch_prop.T)
        self._buf_i = 0

    def _draw_raw(self, ndim: int, rng: np.random.Generator) -> np.ndarray:
        """Single unscaled, un-rotated step (z) for the buffer-exhausted or
        direct-call fallback; matches the batched Marsaglia math per row."""
        if self.proposal == "ball":
            direction = rng.standard_normal(ndim)
            radius = rng.uniform() ** (1.0 / ndim)
            direction *= radius / (np.sqrt(direction @ direction) + 1e-300)
            return direction
        return rng.standard_normal(ndim)

    def step_once(
        self,
        state: State,
        L_threshold: float,
        bound: "BoundingRegion",
        rng: np.random.Generator,
    ) -> tuple[State, int]:
        ndim = state.u.size
        buf = self._stepbuf
        i = self._buf_i
        if buf is not None and i < buf.shape[0]:
            self._buf_i = i + 1
            u_new = state.u + buf[i]
        else:
            prop = self._batch_prop
            if prop is None:
                prop = self._proposal_cov(bound, ndim, u_current=state.u)
            u_new = state.u + self.scale * (prop @ self._draw_raw(ndim, rng))
        if self._periodic:
            _wrap_periodic(u_new, self._periodic)
        # The bound fix keeps the proposal metric finite, so an overflowing
        # step lands as +/-inf and is caught by the cube test (inf is
        # "outside"); the likelihood is therefore only ever called on a
        # finite point. No separate per-step finiteness pass needed.
        if _outside_cube_nonperiodic(u_new, self._periodic_mask):
            self._batch_attempts += 1
            return state, 0
        beta_new = np.asarray(self._prior_transform(u_new), dtype=np.float64)
        self._apply_off_values(beta_new, state.gamma)
        log_L_new = float(self._loglike(beta_new, state.gamma))
        self._batch_attempts += 1
        if log_L_new > L_threshold:
            self._batch_accepts += 1
            # In-place mutation of the donor snapshot.
            state.u[:] = u_new
            state.beta[:] = beta_new
            state.log_likelihood = log_L_new
            return state, 1
        return state, 1

    def _on_batch_complete(self) -> None:
        if self._batch_attempts == 0:
            return
        rate = self._batch_accepts / self._batch_attempts
        ncdim = max(getattr(self, "_ndim", 1), 1)
        denom = ncdim * max(self.target_accept, 1e-3)
        bump = float(np.exp((rate - self.target_accept) / denom))
        # Cap scale at sqrt(ndim): otherwise on narrow ridges adaptation
        # can let the scale grow unbounded (towards scale_max=1e2), each
        # step jumping far enough to be rejected, and the chain returns
        # the donor as a clone. NestedSamplers.jl / dynesty both cap at
        # sqrt(n) for the same reason.
        scale_cap = min(self.scale_max, float(np.sqrt(ncdim)))
        self.scale = float(np.clip(self.scale * bump, self.scale_min, scale_cap))
        self._batch_attempts = 0
        self._batch_accepts = 0

    def step(
        self,
        state: State,
        L_threshold: float,
        bound: "BoundingRegion",
        rng: np.random.Generator,
        n_steps: int,
    ) -> tuple[State, int]:
        """RWalk batch: at least ``n_steps`` walks AND at least one accept.

        The base ``Sampler.step`` does exactly ``n_steps`` walks regardless
        of acceptance. When the scale is too large for the local ridge
        every walk gets rejected and the returned state is the donor's
        snapshot -- a literal clone. Compounded over the chain that
        collapses the posterior. Following NestedSamplers.jl, we keep
        walking past ``n_steps`` until at least one acceptance has
        occurred, with a safety cap (``50 * n_steps``) that triggers a
        scale shrink + reset on the very rare unrecoverable plateau.
        """
        n_calls = 0
        ndim = max(getattr(self, "_ndim", 1), 1)
        safety_cap = 50 * n_steps
        nc = 0
        accept_count = 0
        # Resolve the proposal metric once for this batch (the bound is
        # static mid-batch; this is the fixed M-H kernel of the class
        # docstring). step_once reads self._batch_prop instead of
        # re-deriving the containing-ellipsoid Cholesky every walk.
        self._batch_prop = self._proposal_cov(bound, state.u.size, u_current=state.u)
        # Vectorise the batch's per-walk step draws once (see _fill_step_buffer).
        self._fill_step_buffer(n_steps, state.u.size, rng)
        while nc < n_steps or accept_count == 0:
            accepts_before = self._batch_accepts
            state, calls = self.step_once(state, L_threshold, bound, rng)
            n_calls += calls
            if self._batch_accepts > accepts_before:
                accept_count += 1
            nc += 1
            if nc >= safety_cap and accept_count == 0:
                # Shrink scale and retry. NestedSamplers.jl behaviour:
                # exp(-1/ndim) shrink + reset counters.
                self.scale = float(
                    max(self.scale * float(np.exp(-1.0 / ndim)), self.scale_min)
                )
                nc = 0
                self._batch_attempts = 0
                self._batch_accepts = 0
                safety_cap = min(safety_cap * 2, 1000 * n_steps)
        self._batch_prop = None
        self._stepbuf = None
        self._on_batch_complete()
        return state, n_calls


class DifferentialEvolutionSampler(Sampler):
    """Differential-evolution constrained-MCMC kernel (Ter Braak 2006).

    For each step: pick two distinct *other* live points u_a, u_b, propose
    ``u_new = u + gamma * (u_a - u_b) + eps`` with the standard
    ``gamma = 2.38 / sqrt(2 * ndim)`` from Ter Braak (2006) and a tiny
    Gaussian jitter ``eps`` to break degeneracy. The proposal is symmetric
    in u-space, so the M-H ratio reduces to ``1[u_new in cube] *
    1[L > L_threshold]`` -- same simplification as RW under a uniform u
    prior.

    DE adapts to the live-cloud's correlation structure for free: in
    high-dim posteriors with narrow ridges (e.g. transit fits where
    period, T0, and depth are degenerate), the difference ``u_a - u_b``
    naturally points along the ridge so steps don't waste effort
    exploring orthogonal directions. This is the standard fix for the
    14+-dim multimodal posteriors that defeat plain RW or slice
    sampling.

    Requires NestedSampler to inject the live-point cloud via
    ``set_live_points`` once per outer iteration; falls back to RW when
    the cloud has fewer than 3 points (so the sampler still works on
    very small problems).
    """

    def __init__(
        self,
        gamma_factor: str | float = "auto",
        noise_scale: float = 1e-6,
        scale_min: float = 1e-2,
        scale_max: float = 1e2,
        target_accept: float = 0.234,
    ) -> None:
        # ``scale_min``: same correctness guard as RandomWalkSampler -- the
        # acceptance-rate controller must not be allowed to buy its target
        # by shrinking the step toward zero instead of by mixing. DE is
        # less exposed than rwalk because its increment is a live-point
        # difference (already carrying the cloud's scale), but it is not
        # immune: on the WASP-47 transit benchmark the old 1e-5 floor was
        # reached in 23.7% of batches. Raising it is a strict improvement
        # there -- max log L moves from 4.3 to 0.2 nats of the global
        # optimum while likelihood calls drop 7% and wall-clock 11%,
        # because a chain that actually moves needs fewer iterations to
        # reach the same evidence.
        self.gamma_factor = gamma_factor
        self.noise_scale = noise_scale
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.target_accept = target_accept
        self._live_u: np.ndarray | None = None
        self._batch_attempts = 0
        self._batch_accepts = 0
        # Multiplicative drift on the DE step length, adapted toward
        # ``target_accept`` at the end of each batch. Perpetual (not
        # diminishing) because the constrained-MCMC target shifts every
        # NS iteration; mirrors dynesty's `rwalk` convention.
        self._scale = 1.0
        # Per-batch precomputed position increments (donor-pair difference
        # + jitter). The DE increment is independent of the walker position,
        # and gamma/scale/cloud are batch-constant, so the whole batch is
        # vectorised once in ``step`` and consumed by index in ``step_once``.
        self._incbuf: np.ndarray | None = None
        self._buf_i = 0

    def set_live_points(self, live_u: np.ndarray) -> None:
        """Called by NestedSampler once per outer iteration; reference, not copy."""
        self._live_u = live_u

    def _gamma(self, ndim: int) -> float:
        if self.gamma_factor == "auto":
            return 2.38 / float(np.sqrt(2.0 * ndim))
        return float(self.gamma_factor)

    def step_once(
        self,
        state: State,
        L_threshold: float,
        bound: "BoundingRegion",
        rng: np.random.Generator,
    ) -> tuple[State, int]:
        ndim = state.u.size
        buf = self._incbuf
        if buf is not None and self._buf_i < buf.shape[0]:
            i = self._buf_i
            self._buf_i = i + 1
            u_new = state.u + buf[i]
        elif self._live_u is None or self._live_u.shape[0] < 3:
            # Pre-bind fallback or pathologically small live cloud.
            u_new = state.u + rng.standard_normal(ndim) * self._scale
        else:
            # Direct call outside a batch (e.g. tests): single donor pair.
            n_live = self._live_u.shape[0]
            idx_a, idx_b = rng.choice(n_live, size=2, replace=False)
            delta = self._gamma(ndim) * self._scale * (
                self._live_u[idx_a] - self._live_u[idx_b]
            )
            u_new = state.u + delta + self.noise_scale * rng.standard_normal(ndim)

        if self._periodic:
            _wrap_periodic(u_new, self._periodic)
        if _outside_cube_nonperiodic(u_new, self._periodic_mask):
            self._batch_attempts += 1
            return state, 0
        beta_new = np.asarray(self._prior_transform(u_new), dtype=np.float64)
        self._apply_off_values(beta_new, state.gamma)
        log_L_new = float(self._loglike(beta_new, state.gamma))
        self._batch_attempts += 1
        if log_L_new > L_threshold:
            self._batch_accepts += 1
            state.u[:] = u_new
            state.beta[:] = beta_new
            state.log_likelihood = log_L_new
            return state, 1
        return state, 1

    def step(
        self,
        state: State,
        L_threshold: float,
        bound: "BoundingRegion",
        rng: np.random.Generator,
        n_steps: int,
    ) -> tuple[State, int]:
        """DE batch: vectorise the ``n_steps`` position increments up front.

        The DE proposal (donor-pair difference plus jitter) does not depend
        on the walker's current position, and gamma, scale, and the live
        cloud are all fixed for the batch, so every increment can be built
        in two batched RNG draws plus one gather. This collapses the
        per-walk ``rng.choice`` (whose internal reductions dominate the DE
        hot loop) and the per-walk ``_gamma`` recompute to per-batch cost.
        Falls back to per-walk ``step_once`` when the cloud is too small.
        """
        n_calls = 0
        lu = self._live_u
        if lu is not None and lu.shape[0] >= 3:
            ndim = state.u.size
            nlv = lu.shape[0]
            gscale = self._gamma(ndim) * self._scale
            a = rng.integers(nlv, size=n_steps)
            b = rng.integers(nlv, size=n_steps)
            # Resample the (rare) self-collisions so each donor pair is
            # distinct, matching rng.choice(replace=False).
            coll = a == b
            while coll.any():
                b[coll] = rng.integers(nlv, size=int(coll.sum()))
                coll = a == b
            jitter = rng.standard_normal((n_steps, ndim))
            self._incbuf = gscale * (lu[a] - lu[b]) + self.noise_scale * jitter
            self._buf_i = 0
        for _ in range(n_steps):
            state, calls = self.step_once(state, L_threshold, bound, rng)
            n_calls += calls
        self._incbuf = None
        self._on_batch_complete()
        return state, n_calls

    def _on_batch_complete(self) -> None:
        if self._batch_attempts == 0:
            return
        rate = self._batch_accepts / self._batch_attempts
        bump = float(np.exp(rate - self.target_accept))
        self._scale = float(np.clip(self._scale * bump, self.scale_min, self.scale_max))
        self._batch_attempts = 0
        self._batch_accepts = 0


class RandomSliceSampler(Sampler):
    """Random-direction slice sampling in the unit-cube subspace.

    Each `step_once` runs `n_slices` slice-sampling sweeps. For each sweep:
        1. Draw a random direction z ~ N(0, I) and normalise; rotate by the
           bound's natural metric so steps follow constraint geometry.
        2. Bracket the slice {t : L(prior_transform(u + t*v)) > L_*} via
           stepping out (PolyChord scheme): grow the initial random bracket
           outward as long as the endpoints stay in-cube and L > L_*.
        3. Sample uniformly from the bracket with shrinkage on each
           rejection: candidates outside the cube or below L_* shrink the
           bracket toward 0.

    No tuning required -- the slice expands/shrinks adaptively per sweep.
    Costs ~3-4 likelihood calls per sweep on smooth targets, vs ~1 for RW;
    in exchange, mixes much faster on degenerate geometries.
    """

    def __init__(
        self,
        n_slices: int = 5,
        init_scale: float = 1.0,
        max_steps_out: int = 100,
        max_shrinkage: int = 100,
        doubling: bool | str = False,
    ) -> None:
        """
        ``init_scale``: width of the initial bracket, in units of the
        bound's extent along the sampled direction (the direction is
        ``prop @ z`` for a unit ``z``, so ``init_scale=1`` brackets roughly
        the whole bounding ellipsoid). This is *not* a cosmetic efficiency
        knob -- **it is the kernel's global search radius**, and it is the
        one parameter that decides whether the chain can leave the mode it
        started in.

        A sweep costs ~``log2(bracket / slice)`` shrinkage evaluations, so a
        narrow bracket looks cheap: matching the bracket to the local slice
        width roughly halves the likelihood calls per sweep. What it buys
        those calls with is range. Every candidate is drawn uniformly from
        the bracket, so only a bracket that spans the constrained region can
        propose the far side of a multimodal ridge; shrink it and slice
        sampling degenerates into a local random walk with a tighter
        stationary distribution but no ability to hop.

        Measured on the WASP-47 transit benchmark (4-d, razor-thin in t0,
        true global optimum max log L = -4624.48 at P = 4.1584 d confirmed by
        independent multi-start optimisation), varying only this parameter at
        5 sweeps per replacement:

            init_scale   median P     max log L
                  0.03    10.396 d       -6737
                  0.1       8.316 d      -6539
                  0.3       8.316 d      -6539
                  1.0       4.166 d      -5002
                  3.0       4.159 d      -4624   (global optimum)

        Monotone, and spanning 2100 nats. This is why the bracket scale is
        deliberately *fixed* rather than adapted: dynesty's `rslice` tunes it
        toward equal expansion and contraction counts, which on this
        likelihood drives its bracket to ~7e-3 of the bound extent (~1e-5 at
        100 sweeps) -- squarely in the top rows of that table. It converges
        on the b x 2 sub-harmonic at 8.303 d, and, because more sweeps means
        more tuning updates, gets *worse* with more effort. Porting the same
        rule here reproduced the failure exactly: ~2x cheaper per sweep, and
        the run lands on a spurious 11.25 d mode 3000 nats down. The same
        trade-off appears as `scale_min` in `RandomWalkSampler`.

        ``doubling``: how the bracket is grown when the initial one lands
        entirely inside the slice.

        ``False`` (default) uses only the fixed-step "stepping out" of Neal
        (2003) Fig. 3: grow each end by ``init_scale`` until it leaves the
        slice, at most ``max_steps_out`` times. That costs
        ``O(slice / init_scale)`` evaluations, and on exhausting the cap it
        keeps a *truncated* bracket, so the sweep explores only part of its
        slice. Truncation is no longer silent -- it raises a
        ``SliceBracketTruncationWarning`` once per sampler.

        ``True`` uses the doubling scheme of Fig. 4 instead, which reaches
        the same bracket in ``O(log2)`` evaluations, together with the
        reversibility test of Fig. 6 (doubling can build brackets the current
        point could not have generated, so the test is not optional). Choose
        this when the warning fires, or whenever ``init_scale`` must be set
        far below the slice width.

        ``"auto"`` starts fixed-step and latches permanently onto doubling
        the first time a step-out truncates. Not the default, deliberately:
        truncation can be a one-off from a degenerate cluster ellipsoid --
        measured on Rosenbrock-2D with ``bound="multi"``, exactly 1 of 31100
        step-outs truncated, and the evidence bias was unaffected (+0.016 vs
        +0.012, both far inside the run-to-run scatter of 0.056), yet
        latching raised the cost of the whole remaining run by 18%. Making
        the kernel depend on whether a rare event happened to fire also makes
        the run harder to reproduce. Prefer reacting to the warning by
        setting ``doubling=True`` explicitly.
        """
        if not (isinstance(doubling, bool) or doubling == "auto"):
            raise ValueError(
                f"doubling must be True, False, or 'auto', got {doubling!r}"
            )
        self.n_slices = n_slices
        self.init_scale = init_scale
        self.max_steps_out = max_steps_out
        self.max_shrinkage = max_shrinkage
        self.doubling = doubling
        self._doubling_active = doubling is True
        self._warned_truncation = False

    @property
    def sweeps_per_step(self) -> int:
        return int(self.n_slices)

    @property
    def doubling_active(self) -> bool:
        """Whether the doubling step-out is currently in use. Constant under
        ``doubling=True``/``False``; latches on under ``"auto"`` once a
        fixed-step step-out has been truncated by ``max_steps_out``.
        """
        return bool(self._doubling_active)

    def recommended_n_mcmc(self, ndim: int) -> int:
        """Slice sampling self-tunes its step length per direction, so it
        stays unbiased at high dimension with far fewer sweeps than a
        fixed-scale random walk needs steps -- ~2*ndim total sweeps is
        unbiased through 20D (measured), and we target ~3*ndim for margin
        on curved geometries, with a floor for cheap low-d problems. This
        avoids inheriting the random-walk 5*ndim default.

        Like every other kernel's recommendation this is a count of *sweeps*,
        not of ``step_once`` calls; ``run_nested`` divides by
        ``sweeps_per_step``. Returning sweeps here is what keeps the budget
        comparable across kernels -- see `Sampler.sweeps_per_step`.
        """
        return max(25, 3 * ndim)

    def _proposal_cov(
        self,
        bound: "BoundingRegion",
        ndim: int,
        u_current: np.ndarray | None = None,
    ) -> np.ndarray:
        # SingleEllipsoid (or any bound with a top-level Cholesky): use it.
        L = getattr(bound, "_L", None)
        if isinstance(L, np.ndarray) and L.shape == (ndim, ndim):
            return L
        # MultiEllipsoid stores per-cluster Cholesky factors in
        # ``_ellipsoids[k].L`` with NO top-level ``_L``. Without this branch
        # the slice direction defaulted to an identity metric, ignoring the
        # constraint geometry entirely -- catastrophic under-mixing on the
        # thin, correlated constrained priors that multi-ellipsoid bounds are
        # used for (the chain barely moves and the evidence is biased low).
        # Mirror RandomWalkSampler: use the donor's containing ellipsoid's
        # local metric, falling back to the volume-largest ellipsoid.
        ells = getattr(bound, "_ellipsoids", None)
        if ells:
            if u_current is not None:
                for ell in ells:
                    if ell.contains(u_current):
                        return ell.L
            return max(ells, key=lambda e: e.log_det_L).L
        return np.eye(ndim)

    def _eval(
        self,
        u_new: np.ndarray,
        gamma: np.ndarray,
        L_threshold: float,
    ) -> tuple[float, np.ndarray, bool]:
        """Compute L at u_new (with off-value override). Returns
        (log_L, beta, satisfies_threshold). beta is the overridden vector;
        satisfies_threshold is True iff log_L > L_threshold AND u_new is
        within the unit cube.
        """
        if self._periodic:
            _wrap_periodic(u_new, self._periodic)
        if _outside_cube_nonperiodic(u_new, self._periodic_mask):
            return -np.inf, u_new, False
        beta = np.asarray(self._prior_transform(u_new), dtype=np.float64)
        self._apply_off_values(beta, gamma)
        log_L = float(self._loglike(beta, gamma))
        return log_L, beta, log_L > L_threshold

    def _step_out_linear(
        self,
        u: np.ndarray,
        v: np.ndarray,
        scale: float,
        t_low: float,
        t_high: float,
        gamma: np.ndarray,
        L_threshold: float,
    ) -> tuple[float, float, int, bool]:
        """"Stepping out" of Neal (2003) Fig. 3 / PolyChord: grow each end of
        the bracket by a fixed ``scale`` while that end is still inside the
        slice.

        Returns ``(t_low, t_high, n_calls, truncated)``. ``truncated`` is
        True when ``max_steps_out`` was exhausted with the endpoint still
        inside the slice -- the returned bracket is then a strict subset of
        the true one, and sampling from it under-explores the slice.
        """
        n_calls = 0
        truncated = False
        for _ in range(self.max_steps_out):
            _, _, ok = self._eval(u + t_low * v, gamma, L_threshold)
            n_calls += 1
            if not ok:
                break
            t_low -= scale
        else:
            truncated = True

        for _ in range(self.max_steps_out):
            _, _, ok = self._eval(u + t_high * v, gamma, L_threshold)
            n_calls += 1
            if not ok:
                break
            t_high += scale
        else:
            truncated = True

        return t_low, t_high, n_calls, truncated

    def _report_truncation(self) -> None:
        """Surface a truncated step-out, once per sampler.

        A truncated bracket is a real defect in that sweep -- the slice was
        wider than ``max_steps_out * init_scale`` and the sweep sampled only
        part of it -- so it must not pass unnoticed. It is reported rather
        than repaired in place because the repair (``doubling=True``) changes
        the kernel for the whole run, and truncation is often a one-off from a
        single degenerate cluster ellipsoid; see the ``doubling`` entry in
        ``__init__`` for the measured numbers.
        """
        if self.doubling == "auto":
            self._doubling_active = True
        if self._warned_truncation:
            return
        self._warned_truncation = True
        warnings.warn(
            f"slice step-out hit max_steps_out ({self.max_steps_out}) with the "
            "bracket end still inside the slice, so that sweep sampled a "
            "truncated bracket and explored only part of its slice. The "
            f"bracket is init_scale ({self.init_scale}) times the bound's "
            "extent along the sampled direction, so this can happen even at "
            "init_scale=1 when the bound is a multi-ellipsoid with a "
            "degenerate cluster. Pass doubling=True for the Neal (2003) "
            "doubling step-out, which reaches the full bracket in log2 "
            "evaluations"
            + (
                " -- switching to it now for the rest of this run "
                "(doubling='auto')."
                if self.doubling == "auto"
                else ", or raise max_steps_out. An isolated occurrence in a "
                "long run is usually harmless; see the doubling docstring."
            ),
            SliceBracketTruncationWarning,
            stacklevel=3,
        )

    def _step_out_doubling(
        self,
        u: np.ndarray,
        v: np.ndarray,
        t_low: float,
        t_high: float,
        gamma: np.ndarray,
        L_threshold: float,
        rng: np.random.Generator,
    ) -> tuple[float, float, bool, bool, int]:
        """Doubling step-out of Neal (2003) Fig. 4: repeatedly double the
        bracket's width, extending a randomly chosen side, until both ends
        are outside the slice.

        Reaches a bracket of width W in ``log2(W / scale)`` evaluations where
        the fixed-step scheme needs ``W / scale``, which is what makes an
        over-narrow ``init_scale`` survivable. The price is that doubling can
        build brackets from which the *current* point could not have been
        reached, so reversibility needs the extra test in
        ``_doubling_accept``.

        Returns ``(t_low, t_high, ok_low, ok_high, n_calls)``; the endpoint
        flags are what the acceptance test starts its bisection from.
        """
        _, _, ok_low = self._eval(u + t_low * v, gamma, L_threshold)
        _, _, ok_high = self._eval(u + t_high * v, gamma, L_threshold)
        n_calls = 2
        for _ in range(self.max_steps_out):
            if not (ok_low or ok_high):
                break
            width = t_high - t_low
            if rng.uniform() < 0.5:
                t_low -= width
                _, _, ok_low = self._eval(u + t_low * v, gamma, L_threshold)
            else:
                t_high += width
                _, _, ok_high = self._eval(u + t_high * v, gamma, L_threshold)
            n_calls += 1
        return t_low, t_high, ok_low, ok_high, n_calls

    def _doubling_accept(
        self,
        u: np.ndarray,
        v: np.ndarray,
        scale: float,
        t_cand: float,
        t_low: float,
        t_high: float,
        ok_low: bool,
        ok_high: bool,
        gamma: np.ndarray,
        L_threshold: float,
    ) -> tuple[bool, int]:
        """Reversibility test of Neal (2003) Fig. 6, required whenever the
        bracket came from ``_step_out_doubling``.

        Bisects the doubling bracket back down toward the initial width and
        rejects if any intermediate interval that *separates* the current
        point from the candidate has both ends outside the slice: started
        from the candidate, the doubling procedure would have stopped at that
        interval and could never have produced the bracket actually used, so
        accepting would break detailed balance. Without this test the
        doubling scheme samples a subtly wrong distribution.

        Works in the sweep's ``t`` coordinate with the current point at
        ``t = 0``. Returns ``(accepted, n_calls)``.
        """
        lo, hi = t_low, t_high
        ok_lo, ok_hi = ok_low, ok_high
        n_calls = 0
        separated = False
        # 1.1 * scale, not scale, per Neal: guards the comparison against the
        # floating-point error accumulated by the halvings.
        while hi - lo > 1.1 * scale:
            mid = 0.5 * (lo + hi)
            if (0.0 < mid <= t_cand) or (t_cand < mid <= 0.0):
                separated = True
            if t_cand < mid:
                hi = mid
                _, _, ok_hi = self._eval(u + hi * v, gamma, L_threshold)
            else:
                lo = mid
                _, _, ok_lo = self._eval(u + lo * v, gamma, L_threshold)
            n_calls += 1
            if separated and not ok_lo and not ok_hi:
                return False, n_calls
        return True, n_calls

    def step_once(
        self,
        state: State,
        L_threshold: float,
        bound: "BoundingRegion",
        rng: np.random.Generator,
    ) -> tuple[State, int]:
        ndim = state.u.size
        prop = self._proposal_cov(bound, ndim, u_current=state.u)
        # Slice mutates state in place; `u`, `beta`, `log_L` shadow the
        # current accepted state so the inner loop avoids re-reading
        # state.<attr> on every shrinkage candidate.
        u = state.u
        beta = state.beta
        log_L = state.log_likelihood
        n_calls = 0

        for _ in range(self.n_slices):
            z = rng.standard_normal(ndim)
            z /= float(np.linalg.norm(z))
            v = prop @ z
            scale = self.init_scale

            # Initial random bracket containing 0.
            t_low = -scale * float(rng.uniform())
            t_high = t_low + scale

            # Step out until both ends are outside the slice.
            doubled = self._doubling_active
            if doubled:
                t_low, t_high, ok_low, ok_high, calls = self._step_out_doubling(
                    u, v, t_low, t_high, state.gamma, L_threshold, rng
                )
                n_calls += calls
            else:
                t_low, t_high, calls, truncated = self._step_out_linear(
                    u, v, scale, t_low, t_high, state.gamma, L_threshold
                )
                n_calls += calls
                if truncated:
                    self._report_truncation()
                ok_low = ok_high = False
            # The reversibility test bisects the bracket as it came out of
            # the step-out, so keep it before shrinkage overwrites the ends.
            d_low, d_high = t_low, t_high

            # Sample with shrinkage.
            accepted = False
            for _ in range(self.max_shrinkage):
                t = float(rng.uniform(t_low, t_high))
                u_cand = u + t * v
                log_L_cand, beta_cand, ok = self._eval(u_cand, state.gamma, L_threshold)
                n_calls += 1
                if ok and doubled:
                    ok, extra = self._doubling_accept(
                        u, v, scale, t, d_low, d_high, ok_low, ok_high,
                        state.gamma, L_threshold,
                    )
                    n_calls += extra
                if ok:
                    u[:] = u_cand
                    beta[:] = beta_cand
                    log_L = log_L_cand
                    accepted = True
                    break
                # Shrink toward 0.
                if t < 0.0:
                    t_low = t
                else:
                    t_high = t
            if not accepted:
                # Pathological: bracket collapsed without a valid sample.
                # Keep current state for this sweep and move on.
                continue

        state.log_likelihood = log_L
        return state, n_calls


def make_sampler(name: str) -> Sampler:
    """Construct a built-in sampler by name."""
    table: dict[str, type[Sampler]] = {
        "unif": UniformSampler,
        "rwalk": RandomWalkSampler,
        "rslice": RandomSliceSampler,
        "de": DifferentialEvolutionSampler,
    }
    if name not in table:
        raise ValueError(f"unknown sampler {name!r}, expected one of {sorted(table)}")
    return table[name]()
