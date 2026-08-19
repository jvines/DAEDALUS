"""Bounding regions for the active continuous subspace.

A bounding region is a tractable superset of the prior support restricted to
L > L_*. Constrained-MCMC samplers (and the uniform rejection sampler) draw
proposals from inside the bound rather than from the prior at large, which
substantially improves efficiency once the prior volume X_i becomes small.

Bounds operate on the u-space coordinates (unit cube). Trans-dim flips are
handled separately by the MoMS kernel in `daedalus.proposals`, not by the
bound.

Built-ins:
    * `NoBound`         -- the unit cube; no acceleration; correct on any
                           problem but acceptance shrinks with prior volume.
    * `SingleEllipsoid` -- one enlarged ellipsoid around the live points
                           (Mukherjee+ 2006); the workhorse for unimodal
                           targets.
    * `MultiEllipsoid`  -- recursive splitting into a union of ellipsoids
                           (Feroz+ 2009); handles multimodal targets.

Performance notes:
    * Live points are passed in as a single (n_live, ndim) NumPy view; the
      fit code never iterates point-by-point in Python.
    * `SingleEllipsoid.sample` pre-computes the Cholesky factor at fit time
      and reuses it for every draw (one O(d^2) matvec per sample).
    * Containment uses a single dot product against the inverse Cholesky
      factor (no explicit C^-1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class BoundingRegion(ABC):
    @abstractmethod
    def fit(self, points: np.ndarray, log_volume_prior: float = 0.0) -> None:
        """Update the bound from the current live-point cloud.

        Args:
            points: (n_live, n_active) array of live-point u-coords.
            log_volume_prior: log of the remaining prior volume X; available
                              for bounds that adapt their enlargement to the
                              shrinkage rate.
        """

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one point uniformly from the bound."""

    @abstractmethod
    def contains(self, point: np.ndarray) -> bool:
        """Test whether a point lies inside the bound."""

    @abstractmethod
    def log_volume(self) -> float:
        """Log of the bound's volume in the active subspace."""


class NoBound(BoundingRegion):
    """The unit cube. Sampling draws uniformly from [0, 1]^ndim."""

    def __init__(self, ndim: int) -> None:
        self.ndim = ndim

    def fit(self, points: np.ndarray, log_volume_prior: float = 0.0) -> None:
        return

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(size=self.ndim)

    def contains(self, point: np.ndarray) -> bool:
        return bool(np.all((point >= 0.0) & (point <= 1.0)))

    def log_volume(self) -> float:
        return 0.0


def _log_unit_ball_volume(ndim: int) -> float:
    """log of the volume of the d-dimensional unit ball: pi^(d/2) / Gamma(d/2 + 1)."""
    from math import lgamma  # local import to avoid scipy dep here

    return 0.5 * ndim * float(np.log(np.pi)) - lgamma(0.5 * ndim + 1.0)


class SingleEllipsoid(BoundingRegion):
    """Single enlarged ellipsoid bounding the live points (Mukherjee+ 2006).

    Fit step:
        1. mean = mean of live points
        2. cov  = sample covariance of live points (with `n` not `n-1` to
                  match Mukherjee+; minor difference for n_live >> 1)
        3. Compute Cholesky factor L of cov.
        4. Find the largest Mahalanobis distance among the points,
           r_max^2 = max_i (x_i - mean)^T cov^-1 (x_i - mean), and rescale
           L by r_max so all points are exactly enclosed at the bound's
           surface.
        5. Apply user `enlargement` factor as a final multiplicative scale
           on L (volume scales as enlargement^ndim).
    """

    def __init__(self, ndim: int, enlargement: float = 1.25) -> None:
        if enlargement <= 0.0:
            raise ValueError(f"enlargement must be positive, got {enlargement}")
        self.ndim = ndim
        self.enlargement = enlargement
        self._mean: np.ndarray | None = None
        self._L: np.ndarray | None = None  # Cholesky factor of fitted cov
        self._L_inv: np.ndarray | None = None
        self._log_det_L: float = 0.0
        self._log_unit_ball: float = _log_unit_ball_volume(ndim)

    def fit(self, points: np.ndarray, log_volume_prior: float = 0.0) -> None:
        if points.ndim != 2 or points.shape[1] != self.ndim:
            raise ValueError(
                f"expected (n_live, {self.ndim}) array, got shape {points.shape}"
            )
        n_live = points.shape[0]
        if n_live <= self.ndim:
            raise ValueError(
                f"SingleEllipsoid needs n_live > ndim, got n_live={n_live}, "
                f"ndim={self.ndim}"
            )

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mean = points.mean(axis=0)
            centred = points - mean
            # Use 1/n (population covariance) to match Mukherjee+ convention.
            cov = (centred.T @ centred) / n_live
            # Regularise to avoid singular covariance from degenerate live clouds.
            cov.flat[:: self.ndim + 1] += 1e-12

            L = np.linalg.cholesky(cov)
            # Mahalanobis distances: solve L y = centred^T columnwise.
            y = np.linalg.solve(L, centred.T)  # (ndim, n_live)
            r_sq = np.sum(y * y, axis=0)
            r_max = float(np.sqrt(np.max(r_sq)))

            scale = r_max * self.enlargement
            self._mean = mean
            self._L = L * scale
            self._L_inv = np.linalg.inv(self._L)
            self._log_det_L = float(np.sum(np.log(np.diag(self._L))))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        if self._mean is None or self._L is None:
            raise RuntimeError("SingleEllipsoid.sample called before fit")
        # Uniform sample in unit ball -> push through L and shift.
        z = rng.standard_normal(self.ndim)
        z /= float(np.linalg.norm(z))
        r = float(rng.uniform()) ** (1.0 / self.ndim)
        return self._mean + self._L @ (r * z)

    # Tolerance to absorb the round-off when the boundary fit point is
    # tested for inclusion: r_max * L^-1 cancellation is exact in real
    # arithmetic but accumulates ~ULP-scale error in float64.
    _CONTAINS_EPS = 1e-10

    def contains(self, point: np.ndarray) -> bool:
        if self._mean is None or self._L_inv is None:
            raise RuntimeError("SingleEllipsoid.contains called before fit")
        y = self._L_inv @ (point - self._mean)
        return bool(np.dot(y, y) <= 1.0 + self._CONTAINS_EPS)

    def log_volume(self) -> float:
        # log V = log V_unit_ball + log |det(L)| where L is the scaled Cholesky.
        if self._L is None:
            raise RuntimeError("SingleEllipsoid.log_volume called before fit")
        return self._log_unit_ball + self._log_det_L


class GammaScopedEllipsoid(SingleEllipsoid):
    """A `SingleEllipsoid` whose covariance is fit only on a subset of
    u-coordinates ("active dims"), with the remaining ("inactive") dims
    given an axis-aligned proposal width derived from the live cloud's
    spread.

    Use case: gamma-adaptive MoMS-NS. For each gamma-configuration, the
    u-coordinates of *active* groups carry the constrained likelihood
    geometry; the u-coordinates of inactive groups float freely (they
    don't enter the likelihood once off-values are applied). Fitting a
    full-ndim ellipsoid on a small per-gamma cluster gives a noisy
    covariance that mis-shapes the within-model proposal direction;
    fitting only on active dims is rank-stable with far fewer cluster
    points, and the inactive dims get a sensible diagonal scale from
    the cluster's own spread along those dims.

    Falls back to an axis-aligned bounding box on the active subspace
    when the cluster has too few points to fit even the subspace
    ellipsoid (``len(cluster) <= len(active_dims)``).

    .. warning::

        EXPERIMENTAL / UNUSED. This class is not used to produce any
        published result, is reachable only via the unsupported
        ``NestedSampler(gamma_adaptive=True)`` code path, and is not
        exported from the :mod:`daedalus` namespace.

        Its ``contains()`` is INEXACT for the region the class name
        implies. Despite fitting the active subspace and the inactive
        dims separately, ``contains()`` is inherited from
        :class:`SingleEllipsoid` and applies a single *joint*
        Mahalanobis test over the full block-diagonal ``L_inv``:
        ``y = L_inv @ (x - mean); accept iff y·y <= 1``. That tests
        one ndim-ellipsoid, NOT the product region
        ``{active-subspace ellipsoid} x {inactive-dim box}`` that the
        separate fits suggest. The active Mahalanobis contribution and
        the inactive-box contribution share a single ``<= 1`` budget,
        so a fit point on the active ellipsoid's surface fails
        ``contains()`` as soon as it has any inactive-dim offset. With
        ``enlargement=1.0`` a large fraction of the fit points fail
        their own ``contains()`` test. ``sample()`` and
        ``log_volume()`` are likewise the inherited single-ellipsoid
        forms, not the product-region versions. Do not rely on these
        for correctness-critical work.
    """

    def __init__(
        self,
        ndim: int,
        active_dims: Sequence[int] | np.ndarray,
        enlargement: float = 1.25,
    ) -> None:
        super().__init__(ndim, enlargement)
        active = np.asarray(sorted(set(int(d) for d in active_dims)), dtype=np.intp)
        if active.size and (active.min() < 0 or active.max() >= ndim):
            raise ValueError(
                f"active_dims out of [0, {ndim}): {active.tolist()}"
            )
        self._active_dims = active
        self._inactive_dims = np.array(
            [d for d in range(ndim) if d not in set(active.tolist())],
            dtype=np.intp,
        )

    def fit(
        self, points: np.ndarray, log_volume_prior: float = 0.0
    ) -> None:
        if points.ndim != 2 or points.shape[1] != self.ndim:
            raise ValueError(
                f"expected (n_live, {self.ndim}) array, got shape {points.shape}"
            )
        n_pts = points.shape[0]
        n_active = self._active_dims.size
        n_inactive = self._inactive_dims.size

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            # ---- active subspace ----
            if n_active > 0:
                active_pts = points[:, self._active_dims]
                mean_active = active_pts.mean(axis=0)
                if n_pts > n_active:
                    centred = active_pts - mean_active
                    cov = (centred.T @ centred) / n_pts
                    cov.flat[:: n_active + 1] += 1e-12
                    L_a = np.linalg.cholesky(cov)
                    y = np.linalg.solve(L_a, centred.T)
                    r_max = float(np.sqrt(np.max(np.sum(y * y, axis=0))))
                    L_a = L_a * (r_max * self.enlargement)
                else:
                    # Underdetermined: axis-aligned box on the active subspace.
                    spread = np.maximum(
                        active_pts.max(axis=0) - active_pts.min(axis=0), 1e-12
                    )
                    L_a = np.diag(spread / 2.0) * self.enlargement
            else:
                mean_active = np.zeros(0)
                L_a = np.zeros((0, 0))

            # ---- inactive subspace: axis-aligned spread ----
            if n_inactive > 0:
                inactive_pts = points[:, self._inactive_dims]
                mean_inactive = inactive_pts.mean(axis=0)
                inactive_spread = np.maximum(
                    inactive_pts.max(axis=0) - inactive_pts.min(axis=0),
                    1e-12,
                )
                inactive_diag = (inactive_spread / 2.0) * self.enlargement
            else:
                mean_inactive = np.zeros(0)
                inactive_diag = np.zeros(0)

            # ---- embed both into full ndim x ndim Cholesky ----
            L_full = np.zeros((self.ndim, self.ndim), dtype=np.float64)
            if n_active > 0:
                idx = np.ix_(self._active_dims, self._active_dims)
                L_full[idx] = L_a
            for j in range(n_inactive):
                d = int(self._inactive_dims[j])
                L_full[d, d] = inactive_diag[j]
            mean_full = np.zeros(self.ndim, dtype=np.float64)
            if n_active > 0:
                mean_full[self._active_dims] = mean_active
            if n_inactive > 0:
                mean_full[self._inactive_dims] = mean_inactive

            self._mean = mean_full
            self._L = L_full
            # Block-structured inverse: active block uses inv(L_a),
            # inactive diagonal flips entry-wise.
            L_inv_full = np.zeros((self.ndim, self.ndim), dtype=np.float64)
            if n_active > 0:
                idx = np.ix_(self._active_dims, self._active_dims)
                L_inv_full[idx] = np.linalg.inv(L_a)
            for j in range(n_inactive):
                d = int(self._inactive_dims[j])
                L_inv_full[d, d] = 1.0 / max(float(inactive_diag[j]), 1e-12)
            self._L_inv = L_inv_full
            diag_entries = np.diag(L_full)
            self._log_det_L = float(
                np.sum(np.log(np.where(diag_entries > 0, diag_entries, 1.0)))
            )


class GammaScopedMultiEllipsoid(BoundingRegion):
    """Multi-ellipsoid bound scoped to a gamma-configuration's active dims.

    Like :class:`GammaScopedEllipsoid`, but fits a clustered
    :class:`MultiEllipsoid` on the active subspace instead of a single
    ellipsoid. This is the bound to use when a single γ-tuple has
    multiple permutation-equivalent posterior modes (e.g. exchangeable
    Keplerian slots: an N-active config has N! equivalent modes).

    Inactive dims (u-coords of off-groups) are given an axis-aligned
    spread from the live cloud, just as in :class:`GammaScopedEllipsoid`.

    The bound exposes ``_ellipsoids`` (list of full-ndim ellipsoids,
    each combining one active-subspace sub-ellipsoid with the inactive
    axis-aligned box) so the within-model rwalk kernel can locate the
    donor's containing ellipsoid and use its local Cholesky for the
    proposal step — the same mechanism :class:`MultiEllipsoid` uses.

    .. warning::

        EXPERIMENTAL / UNUSED. This class is not used to produce any
        published result and is not exported from the :mod:`daedalus`
        namespace.

        Each per-cluster ``_Ellipsoid`` it builds combines the active
        block and the inactive diagonal into one full-ndim ``L_inv``,
        and ``_Ellipsoid.contains`` applies a single *joint*
        Mahalanobis test (``y·y <= 1``) over that block-diagonal
        ``L_inv``. As with :class:`GammaScopedEllipsoid`, this tests a
        single ndim-ellipsoid rather than the product region
        ``{active ellipsoid} x {inactive box}`` the construction
        implies, so fit points can fail their own ``contains()``.
        ``sample()`` and ``log_volume()`` inherit the same joint
        single-ellipsoid geometry. Do not rely on these for
        correctness-critical work.
    """

    def __init__(
        self,
        ndim: int,
        active_dims: Sequence[int] | np.ndarray,
        enlargement: float = 1.25,
        vol_dec_factor: float = 0.85,
        max_clusters: int = 256,
        min_cluster_size: int | None = None,
    ) -> None:
        self.ndim = ndim
        self.enlargement = enlargement
        self.vol_dec_factor = vol_dec_factor
        self.max_clusters = int(max_clusters)
        active = np.asarray(sorted(set(int(d) for d in active_dims)), dtype=np.intp)
        if active.size and (active.min() < 0 or active.max() >= ndim):
            raise ValueError(
                f"active_dims out of [0, {ndim}): {active.tolist()}"
            )
        self._active_dims = active
        self._inactive_dims = np.array(
            [d for d in range(ndim) if d not in set(active.tolist())],
            dtype=np.intp,
        )
        n_active = int(active.size)
        if min_cluster_size is None:
            # Permutation-symmetric problems need a relaxed minimum:
            # for an N-active gamma-config with n_live = M, each
            # permutation mode has M/N! points. Setting
            # min_cluster_size = max(2, n_active) lets us discover
            # clusters down to a couple of points per mode.
            min_cluster_size = max(2, n_active)
        self.min_cluster_size = int(min_cluster_size)
        self._active_multi: MultiEllipsoid | None = None
        self._inactive_mean: np.ndarray = np.zeros(0)
        self._inactive_diag: np.ndarray = np.zeros(0)
        self._ellipsoids: list[_Ellipsoid] = []
        self._log_total_volume: float = 0.0
        self._log_unit_ball: float = _log_unit_ball_volume(ndim)

    def fit(self, points: np.ndarray, log_volume_prior: float = 0.0) -> None:
        if points.ndim != 2 or points.shape[1] != self.ndim:
            raise ValueError(
                f"expected (n_live, {self.ndim}) array, got shape {points.shape}"
            )
        n_pts = points.shape[0]
        n_active = self._active_dims.size
        n_inactive = self._inactive_dims.size

        # Inactive-dim axis-aligned spread (same as GammaScopedEllipsoid).
        if n_inactive > 0:
            inactive_pts = points[:, self._inactive_dims]
            self._inactive_mean = inactive_pts.mean(axis=0)
            inactive_spread = np.maximum(
                inactive_pts.max(axis=0) - inactive_pts.min(axis=0), 1e-12
            )
            self._inactive_diag = (inactive_spread / 2.0) * self.enlargement
        else:
            self._inactive_mean = np.zeros(0)
            self._inactive_diag = np.zeros(0)

        # Active-subspace MultiEllipsoid fit. If the active subspace is
        # too narrow (n_pts <= n_active) we fall back to a single
        # axis-aligned box in the active subspace.
        if n_active == 0:
            # All dims inactive: bound is just the inactive box.
            self._active_multi = None
            self._ellipsoids = [self._build_inactive_only_ellipsoid()]
        elif n_pts <= n_active:
            # Underdetermined: single axis-aligned active box.
            self._active_multi = None
            self._ellipsoids = [self._build_axis_aligned_active_ellipsoid(points)]
        else:
            active_pts = points[:, self._active_dims]
            self._active_multi = MultiEllipsoid(
                ndim=n_active,
                enlargement=self.enlargement,
                vol_dec_factor=self.vol_dec_factor,
                max_clusters=self.max_clusters,
                min_cluster_size=self.min_cluster_size,
            )
            self._active_multi.fit(active_pts)
            # Lift each active sub-ellipsoid to a synthetic full-ndim
            # _Ellipsoid so the rwalk kernel can use it.
            self._ellipsoids = [
                self._lift_active_subellipsoid(sub)
                for sub in self._active_multi._ellipsoids
            ]

        log_vols = np.array(
            [self._log_unit_ball + e.log_det_L for e in self._ellipsoids]
        )
        self._log_total_volume = float(np.logaddexp.reduce(log_vols))

    def _build_inactive_box_block(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (L_inactive_block, L_inv_inactive_block, log_det_L_inactive)
        for the axis-aligned inactive-dims box."""
        n_inactive = self._inactive_dims.size
        if n_inactive == 0:
            return np.zeros((0, 0)), np.zeros((0, 0)), 0.0
        diag = self._inactive_diag
        L_block = np.diag(diag)
        L_inv_block = np.diag(1.0 / np.maximum(diag, 1e-12))
        log_det = float(np.sum(np.log(np.maximum(diag, 1e-12))))
        return L_block, L_inv_block, log_det

    def _build_inactive_only_ellipsoid(self) -> "_Ellipsoid":
        """Synthetic ellipsoid when n_active = 0."""
        L_in, L_in_inv, log_det_in = self._build_inactive_box_block()
        L_full = np.zeros((self.ndim, self.ndim))
        L_inv_full = np.zeros((self.ndim, self.ndim))
        for j, d in enumerate(self._inactive_dims):
            L_full[d, d] = L_in[j, j]
            L_inv_full[d, d] = L_in_inv[j, j]
        mean_full = np.zeros(self.ndim)
        if self._inactive_dims.size > 0:
            mean_full[self._inactive_dims] = self._inactive_mean
        return _Ellipsoid(
            mean_full, L_full, L_inv_full, log_det_in,
            np.arange(0, dtype=np.intp),
        )

    def _build_axis_aligned_active_ellipsoid(
        self, points: np.ndarray
    ) -> "_Ellipsoid":
        """Synthetic ellipsoid: axis-aligned in active dims + box in inactive."""
        active_pts = points[:, self._active_dims]
        mean_active = active_pts.mean(axis=0)
        spread_active = np.maximum(
            active_pts.max(axis=0) - active_pts.min(axis=0), 1e-12
        )
        diag_active = (spread_active / 2.0) * self.enlargement
        return self._assemble_full_ellipsoid(
            mean_active, np.diag(diag_active),
            np.diag(1.0 / diag_active),
            float(np.sum(np.log(diag_active))),
            np.arange(0, dtype=np.intp),
        )

    def _lift_active_subellipsoid(self, sub: "_Ellipsoid") -> "_Ellipsoid":
        """Lift one active-subspace sub-ellipsoid to a full-ndim record."""
        return self._assemble_full_ellipsoid(
            sub.mean, sub.L, sub.L_inv, sub.log_det_L, sub.indices,
        )

    def _assemble_full_ellipsoid(
        self,
        mean_active: np.ndarray,
        L_active: np.ndarray,
        L_inv_active: np.ndarray,
        log_det_L_active: float,
        indices: np.ndarray,
    ) -> "_Ellipsoid":
        """Build a full-ndim _Ellipsoid from an active-subspace ellipsoid
        and the cached inactive-dims axis-aligned box."""
        n_active = self._active_dims.size
        n_inactive = self._inactive_dims.size
        L_in, L_in_inv, log_det_in = self._build_inactive_box_block()

        L_full = np.zeros((self.ndim, self.ndim))
        L_inv_full = np.zeros((self.ndim, self.ndim))
        if n_active > 0:
            idx_a = np.ix_(self._active_dims, self._active_dims)
            L_full[idx_a] = L_active
            L_inv_full[idx_a] = L_inv_active
        for j in range(n_inactive):
            d = int(self._inactive_dims[j])
            L_full[d, d] = L_in[j, j]
            L_inv_full[d, d] = L_in_inv[j, j]
        mean_full = np.zeros(self.ndim)
        if n_active > 0:
            mean_full[self._active_dims] = mean_active
        if n_inactive > 0:
            mean_full[self._inactive_dims] = self._inactive_mean

        return _Ellipsoid(
            mean_full, L_full, L_inv_full,
            log_det_L_active + log_det_in,
            indices,
        )

    # --- BoundingRegion interface ---

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        if not self._ellipsoids:
            raise RuntimeError(
                "GammaScopedMultiEllipsoid.sample called before fit"
            )
        # Sample like MultiEllipsoid: pick by volume, then accept by
        # 1/n_overlap. Overlap is computed in the FULL ndim space,
        # which is the right invariant because the rwalk kernel
        # operates on full ndim u-vectors.
        log_vols = np.array(
            [self._log_unit_ball + e.log_det_L for e in self._ellipsoids]
        )
        probs = np.exp(log_vols - self._log_total_volume)
        probs /= probs.sum()
        while True:
            k = int(rng.choice(len(self._ellipsoids), p=probs))
            x = self._ellipsoids[k].sample(rng, self.ndim)
            n_overlap = sum(1 for e in self._ellipsoids if e.contains(x))
            if rng.uniform() < 1.0 / max(n_overlap, 1):
                return x

    def contains(self, point: np.ndarray) -> bool:
        if not self._ellipsoids:
            raise RuntimeError(
                "GammaScopedMultiEllipsoid.contains called before fit"
            )
        return any(e.contains(point) for e in self._ellipsoids)

    def log_volume(self) -> float:
        if not self._ellipsoids:
            raise RuntimeError(
                "GammaScopedMultiEllipsoid.log_volume called before fit"
            )
        return self._log_total_volume


class _Ellipsoid:
    """Lightweight ellipsoid record used by `MultiEllipsoid`."""

    __slots__ = ("mean", "L", "L_inv", "log_det_L", "indices")

    def __init__(
        self,
        mean: np.ndarray,
        L: np.ndarray,
        L_inv: np.ndarray,
        log_det_L: float,
        indices: np.ndarray,
    ) -> None:
        self.mean = mean
        self.L = L
        self.L_inv = L_inv
        self.log_det_L = log_det_L
        self.indices = indices  # which live-point rows feed this ellipsoid

    def contains(self, point: np.ndarray, tol: float = 1e-10) -> bool:
        y = self.L_inv @ (point - self.mean)
        return bool(np.dot(y, y) <= 1.0 + tol)

    def sample(self, rng: np.random.Generator, ndim: int) -> np.ndarray:
        z = rng.standard_normal(ndim)
        z /= float(np.linalg.norm(z))
        r = float(rng.uniform()) ** (1.0 / ndim)
        return self.mean + self.L @ (r * z)


def _fit_one_ellipsoid(
    points: np.ndarray, indices: np.ndarray, enlargement: float, ndim: int
) -> _Ellipsoid:
    """Fit a snug ellipsoid around `points`, scaled by `enlargement`.

    Both branches bound the metric's condition number with a RELATIVE floor
    (the smallest axis/variance is clamped to a fixed fraction of the
    largest). Without it a near-collapsed direction --- common in a small
    multi-ellipsoid cluster on a high-dimensional, tightly-constrained
    likelihood --- yields a near-singular Cholesky factor whose inverse
    overflows to inf, corrupting the proposal metric and stalling the
    chain. The floor is negligible for well-conditioned clusters, so it
    leaves existing results unchanged.
    """
    if points.shape[0] <= ndim:
        # Underdetermined cluster: fall back to an axis-aligned bounding box.
        mean = points.mean(axis=0)
        spread = points.max(axis=0) - points.min(axis=0)
        sfloor = max(1e-12, 1e-4 * float(spread.max()))
        half = np.maximum(spread, sfloor) / 2.0 * enlargement
        L = np.diag(half)
        L_inv = np.diag(1.0 / half)  # diagonal: exact inverse, no overflow
        log_det_L = float(np.sum(np.log(half)))
        return _Ellipsoid(mean, L, L_inv, log_det_L, indices)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = points.mean(axis=0)
        centred = points - mean
        cov = (centred.T @ centred) / points.shape[0]
        # Relative diagonal regularisation bounds the condition number to
        # ~1e8 (the old absolute 1e-12 left it unbounded once a direction's
        # variance fell far below the others, overflowing L_inv).
        max_var = float(np.max(np.diag(cov)))
        # Absolute + relative diagonal floor. The absolute term keeps the
        # covariance non-singular even when a whole cluster collapses to a
        # point (max_var ~ 0); the relative term bounds the condition number
        # for merely ill-conditioned clusters. Both are negligible for
        # well-conditioned clusters.
        cov.flat[:: ndim + 1] += 1e-12 + 1e-8 * max_var
        L = np.linalg.cholesky(cov)
        y = np.linalg.solve(L, centred.T)
        # Floor r_max so a cluster collapsed to a point still yields a
        # finite-volume ellipsoid rather than a zero-scale (singular) one
        # whose inverse overflows to inf.
        r_max = max(float(np.sqrt(np.max(np.sum(y * y, axis=0)))), 1e-3)
        L_scaled = L * (r_max * enlargement)
        L_inv = np.linalg.inv(L_scaled)
        log_det_L = float(np.sum(np.log(np.diag(L_scaled))))
    return _Ellipsoid(mean, L_scaled, L_inv, log_det_L, indices)


def _major_axis_endpoints(ell: "_Ellipsoid") -> tuple[np.ndarray, np.ndarray]:
    """Endpoints of the parent ellipsoid's longest semi-axis.

    The Cholesky factor ``L`` of the scaled ellipsoid satisfies
    ``L L^T = scale^2 * cov``. SVD gives ``L = U diag(s) V^T``; the
    column of ``U`` paired with the largest singular value is the
    major axis direction, and the corresponding singular value is its
    length. The endpoints are ``mean +/- major_dir`` -- the two
    extremes along the bound's elongation.
    """
    U, s, _ = np.linalg.svd(ell.L)
    major_dir = U[:, 0] * s[0]
    return (ell.mean - major_dir, ell.mean + major_dir)


def _kmeans_2way(
    points: np.ndarray,
    rng: np.random.Generator,
    n_iter: int = 20,
    init_centres: np.ndarray | None = None,
    feature_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Lloyd's algorithm with k=2. Returns cluster labels (0 or 1).

    ``init_centres``: optional (2, ndim) array of initial centres. When
    provided, replaces the k-means++ randomised init. The MultiEllipsoid
    splitter passes the parent ellipsoid's major-axis endpoints here,
    so the first split direction is aligned with the live cloud's
    elongation -- the right heuristic for separating modes that have
    stretched the parent bound.

    ``feature_scale``: optional (ndim,) per-dimension scaling applied
    before computing distances. Standardising by ``points.std(0)`` makes
    the clustering isotropic across dimensions of disparate natural
    scale (e.g. log_period vs phase vs impact parameter).
    """
    n = points.shape[0]
    if feature_scale is None:
        feature_scale = points.std(axis=0) + 1e-12
    pts = points / feature_scale
    if init_centres is not None:
        centres = np.asarray(init_centres, dtype=np.float64) / feature_scale
    else:
        # k-means++ fallback.
        idx0 = int(rng.integers(n))
        c0 = pts[idx0]
        d2 = np.sum((pts - c0) ** 2, axis=1)
        s = float(d2.sum())
        if s <= 0.0:
            idx1 = (idx0 + 1) % n
        else:
            idx1 = int(np.searchsorted(np.cumsum(d2 / s), rng.uniform()))
            idx1 = min(idx1, n - 1)
        centres = np.vstack([pts[idx0], pts[idx1]])
    labels = np.zeros(n, dtype=np.intp)
    for _ in range(n_iter):
        d0 = np.sum((pts - centres[0]) ** 2, axis=1)
        d1 = np.sum((pts - centres[1]) ** 2, axis=1)
        new_labels = (d1 < d0).astype(np.intp)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in (0, 1):
            mask = labels == k
            if mask.any():
                centres[k] = pts[mask].mean(axis=0)
    return labels


class MultiEllipsoid(BoundingRegion):
    """Union of ellipsoids fitted by recursive k-means splitting (Feroz+ 2009).

    Fit step:
        1. Fit a single snug ellipsoid to all live points.
        2. Try a 2-way k-means split. Fit a snug ellipsoid to each cluster.
        3. If the sum of child volumes is less than `vol_dec_factor` times
           the parent volume, accept the split and recurse on each cluster;
           else keep the parent un-split.
        4. Stop when no more clusters can be split or `max_clusters` is
           reached.

    Sample step:
        1. Pick an ellipsoid weighted by its volume.
        2. Draw a uniform sample from it.
        3. Compute n_overlap = number of ellipsoids that contain the sample;
           accept with probability 1 / n_overlap. This corrects for
           overlapping ellipsoids so the marginal distribution is uniform
           over the union.

    For unimodal targets the recursion terminates with a single ellipsoid
    and the bound reduces to `SingleEllipsoid`. For multimodal targets the
    bound tracks each mode separately.
    """

    def __init__(
        self,
        ndim: int,
        enlargement: float = 1.25,
        vol_dec_factor: float = 0.85,
        max_clusters: int = 256,
        min_cluster_size: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        max_clusters
            Hard cap on the number of ellipsoids. Default 256 (was 64).
            With many shallow modes (e.g. transit-period aliases on
            multi-cycle data) the previous cap silently merged modes
            together and starved unsampled modes of new live points.
        min_cluster_size
            Minimum number of points per ellipsoid. When ``None``
            (default), the splitter accepts clusters as small as
            ``max(4, ndim // 4)`` -- much smaller than the previous
            ``ndim`` requirement. Small clusters fall back to an
            axis-aligned bounding box (already handled by
            ``_fit_one_ellipsoid``); the relaxation lets the splitter
            keep finding modes when n_live is comparable to (or only
            modestly larger than) the number of modes in the target.
        """
        if enlargement <= 0.0:
            raise ValueError(f"enlargement must be positive, got {enlargement}")
        if not 0.0 < vol_dec_factor <= 1.0:
            raise ValueError(
                f"vol_dec_factor must be in (0, 1], got {vol_dec_factor}"
            )
        self.ndim = ndim
        self.enlargement = enlargement
        self.vol_dec_factor = vol_dec_factor
        self.max_clusters = int(max_clusters)
        if min_cluster_size is None:
            min_cluster_size = max(4, ndim // 4)
        if min_cluster_size < 2:
            raise ValueError(
                f"min_cluster_size must be >= 2, got {min_cluster_size}"
            )
        self.min_cluster_size = int(min_cluster_size)
        self._ellipsoids: list[_Ellipsoid] = []
        self._log_volumes: np.ndarray = np.empty(0)
        self._log_total_volume: float = 0.0
        self._log_unit_ball: float = _log_unit_ball_volume(ndim)
        self._sample_rng_state: tuple = ()  # placeholder for future caching

    def fit(self, points: np.ndarray, log_volume_prior: float = 0.0) -> None:
        if points.ndim != 2 or points.shape[1] != self.ndim:
            raise ValueError(
                f"expected (n_live, {self.ndim}) array, got shape {points.shape}"
            )
        if points.shape[0] <= self.ndim:
            raise ValueError(
                f"MultiEllipsoid needs n_live > ndim, got n_live={points.shape[0]}, "
                f"ndim={self.ndim}"
            )
        rng = np.random.default_rng(0)  # deterministic split decisions
        all_idx = np.arange(points.shape[0])
        root = _fit_one_ellipsoid(points, all_idx, self.enlargement, self.ndim)
        self._ellipsoids = []
        self._recurse_split(points, root, rng, depth=0)
        self._log_volumes = np.array(
            [self._log_unit_ball + e.log_det_L for e in self._ellipsoids]
        )
        self._log_total_volume = float(np.logaddexp.reduce(self._log_volumes))

    def _recurse_split(
        self,
        points: np.ndarray,
        parent: _Ellipsoid,
        rng: np.random.Generator,
        depth: int,
    ) -> None:
        if (
            len(self._ellipsoids) >= self.max_clusters
            or parent.indices.size < 2 * self.min_cluster_size
        ):
            self._ellipsoids.append(parent)
            return
        sub_pts = points[parent.indices]
        # Init the 2-way split at the parent ellipsoid's major-axis
        # endpoints so the split direction is aligned with the live
        # cloud's elongation (Pelleg 2000 x-means heuristic).
        p1, p2 = _major_axis_endpoints(parent)
        labels = _kmeans_2way(sub_pts, rng, init_centres=np.vstack([p1, p2]))
        if labels.min() == labels.max():
            # All points landed in one cluster -- can't split.
            self._ellipsoids.append(parent)
            return
        idx_a = parent.indices[labels == 0]
        idx_b = parent.indices[labels == 1]
        if idx_a.size < self.min_cluster_size or idx_b.size < self.min_cluster_size:
            self._ellipsoids.append(parent)
            return
        child_a = _fit_one_ellipsoid(
            points[idx_a], idx_a, self.enlargement, self.ndim
        )
        child_b = _fit_one_ellipsoid(
            points[idx_b], idx_b, self.enlargement, self.ndim
        )
        log_v_a = self._log_unit_ball + child_a.log_det_L
        log_v_b = self._log_unit_ball + child_b.log_det_L
        log_v_parent = self._log_unit_ball + parent.log_det_L
        log_sum_children = float(np.logaddexp(log_v_a, log_v_b))
        if log_sum_children < log_v_parent + float(np.log(self.vol_dec_factor)):
            self._recurse_split(points, child_a, rng, depth + 1)
            self._recurse_split(points, child_b, rng, depth + 1)
        else:
            self._ellipsoids.append(parent)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        if not self._ellipsoids:
            raise RuntimeError("MultiEllipsoid.sample called before fit")
        # Pick an ellipsoid in proportion to its volume.
        probs = np.exp(self._log_volumes - self._log_total_volume)
        probs /= probs.sum()  # numerical hygiene
        while True:
            k = int(rng.choice(len(self._ellipsoids), p=probs))
            x = self._ellipsoids[k].sample(rng, self.ndim)
            n_overlap = sum(
                1 for e in self._ellipsoids if e.contains(x)
            )
            # n_overlap >= 1 because the chosen ellipsoid contains the sample
            # (modulo float roundoff, hence the tol in `contains`).
            if rng.uniform() < 1.0 / n_overlap:
                return x

    def contains(self, point: np.ndarray) -> bool:
        if not self._ellipsoids:
            raise RuntimeError("MultiEllipsoid.contains called before fit")
        return any(e.contains(point) for e in self._ellipsoids)

    def log_volume(self) -> float:
        if not self._ellipsoids:
            raise RuntimeError("MultiEllipsoid.log_volume called before fit")
        return self._log_total_volume


def make_bound(name: str, ndim: int) -> BoundingRegion:
    """Construct a built-in bound by name."""
    table: dict[str, type[BoundingRegion]] = {
        "none": NoBound,
        "single": SingleEllipsoid,
        "multi": MultiEllipsoid,
    }
    if name not in table:
        raise ValueError(f"unknown bound {name!r}, expected one of {sorted(table)}")
    return table[name](ndim=ndim)
