"""Tests for bounding regions.

The bounds carry no NS state -- they're pure geometric objects -- so we can
pin behaviour against synthetic point clouds with known properties:
analytical volumes, exact containment for points used in the fit, and
uniform sampling diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest

from daedalus.bounds import (
    MultiEllipsoid,
    NoBound,
    SingleEllipsoid,
    _log_unit_ball_volume,
    make_bound,
)


# ---------------- NoBound ----------------------------------------------------


def test_no_bound_samples_inside_unit_cube() -> None:
    bound = NoBound(ndim=3)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = bound.sample(rng)
        assert np.all((u >= 0.0) & (u <= 1.0))


def test_no_bound_log_volume_is_zero() -> None:
    assert NoBound(ndim=4).log_volume() == 0.0


# ---------------- SingleEllipsoid ------------------------------------------


def test_single_ellipsoid_rejects_premature_sample() -> None:
    bound = SingleEllipsoid(ndim=2)
    with pytest.raises(RuntimeError, match="before fit"):
        bound.sample(np.random.default_rng(0))


def test_single_ellipsoid_rejects_premature_volume_query() -> None:
    bound = SingleEllipsoid(ndim=2)
    with pytest.raises(RuntimeError, match="before fit"):
        bound.log_volume()


def test_single_ellipsoid_requires_n_live_above_ndim() -> None:
    bound = SingleEllipsoid(ndim=3)
    with pytest.raises(ValueError, match="n_live > ndim"):
        bound.fit(np.eye(3))


def test_single_ellipsoid_contains_all_fit_points() -> None:
    """All live points used in the fit must lie inside the bound."""
    rng = np.random.default_rng(123)
    n, d = 50, 3
    points = rng.standard_normal(size=(n, d))
    bound = SingleEllipsoid(ndim=d, enlargement=1.0)
    bound.fit(points)
    for x in points:
        assert bound.contains(x)


def test_single_ellipsoid_volume_matches_unit_ball_for_unit_basis_cloud() -> None:
    """For points at +/- e_i (i = 1..d), the snug bound is exactly the unit
    ball and log_volume = log V_d.

    Setup: 2d points at +/- standard basis vectors. Mean = 0, population
    covariance = I/d. Mahalanobis distance from mean to each point is
    sqrt(d), so r_max = sqrt(d). The fitted Cholesky factor is I/sqrt(d)
    scaled by r_max = sqrt(d), giving the identity. Volume = unit ball.
    """
    d = 4
    points = np.vstack([np.eye(d), -np.eye(d)])
    bound = SingleEllipsoid(ndim=d, enlargement=1.0)
    bound.fit(points)
    expected = _log_unit_ball_volume(d)
    assert np.isclose(bound.log_volume(), expected, atol=1e-10)


def test_single_ellipsoid_enlargement_increases_log_volume_by_d_log_factor() -> None:
    rng = np.random.default_rng(7)
    d = 3
    points = rng.standard_normal(size=(60, d))

    snug = SingleEllipsoid(ndim=d, enlargement=1.0)
    snug.fit(points)
    big = SingleEllipsoid(ndim=d, enlargement=2.0)
    big.fit(points)

    delta = big.log_volume() - snug.log_volume()
    assert np.isclose(delta, d * np.log(2.0), atol=1e-12)


def test_single_ellipsoid_samples_lie_inside_bound() -> None:
    rng = np.random.default_rng(9)
    d = 4
    points = rng.standard_normal(size=(80, d))
    bound = SingleEllipsoid(ndim=d, enlargement=1.0)
    bound.fit(points)
    for _ in range(200):
        u = bound.sample(rng)
        assert bound.contains(u)


def test_single_ellipsoid_uniform_sampling_distribution() -> None:
    """Sample mean and second-moment trace should be close to the analytic
    expectations for a uniform distribution on the bound: E[x] = mean, and
    E[(x - mean)(x - mean)^T] = (cov_factor / (d + 2)).

    For our scaled Cholesky L with r_max enlargement, the bound has shape
    L L^T, so sample covariance trace converges to trace(L L^T) / (d + 2).
    """
    rng = np.random.default_rng(2026)
    d = 3
    points = rng.standard_normal(size=(200, d))
    bound = SingleEllipsoid(ndim=d, enlargement=1.0)
    bound.fit(points)

    n_samples = 4000
    samples = np.array([bound.sample(rng) for _ in range(n_samples)])
    sample_mean = samples.mean(axis=0)
    assert np.allclose(sample_mean, bound._mean, atol=0.1)  # type: ignore[attr-defined]

    centred = samples - bound._mean  # type: ignore[attr-defined]
    sample_cov = (centred.T @ centred) / n_samples
    expected_cov = (bound._L @ bound._L.T) / (d + 2)  # type: ignore[attr-defined]
    rel = np.linalg.norm(sample_cov - expected_cov) / np.linalg.norm(expected_cov)
    assert rel < 0.15, f"sample covariance off by {rel:.2%}"


def test_make_bound_factory_dispatches_correctly() -> None:
    assert isinstance(make_bound("none", ndim=2), NoBound)
    assert isinstance(make_bound("single", ndim=2), SingleEllipsoid)
    assert isinstance(make_bound("multi", ndim=2), MultiEllipsoid)


# ---------------- MultiEllipsoid ------------------------------------------


def test_multi_ellipsoid_unimodal_does_not_oversplit() -> None:
    """For a unimodal Gaussian cluster the bound should converge on a small
    number of ellipsoids (typically one) -- splitting would just add
    volume without capturing real structure.
    """
    rng = np.random.default_rng(11)
    d = 3
    points = rng.standard_normal(size=(80, d))
    bound = MultiEllipsoid(ndim=d, enlargement=1.0)
    bound.fit(points)
    assert len(bound._ellipsoids) <= 2, (
        f"expected <=2 ellipsoids on a unimodal cloud, got "
        f"{len(bound._ellipsoids)}"
    )
    for x in points:
        assert bound.contains(x)


def test_multi_ellipsoid_two_well_separated_clusters_split() -> None:
    """Two well-separated Gaussian clusters should yield two ellipsoids."""
    rng = np.random.default_rng(13)
    d = 2
    a = rng.standard_normal(size=(50, d)) + np.array([5.0, 5.0])
    b = rng.standard_normal(size=(50, d)) + np.array([-5.0, -5.0])
    points = np.vstack([a, b])
    bound = MultiEllipsoid(ndim=d, enlargement=1.0)
    bound.fit(points)
    assert len(bound._ellipsoids) >= 2
    for x in points:
        assert bound.contains(x), f"point {x} not contained"


def test_multi_ellipsoid_samples_lie_in_union() -> None:
    rng = np.random.default_rng(17)
    d = 2
    a = rng.standard_normal(size=(50, d)) + np.array([5.0, 0.0])
    b = rng.standard_normal(size=(50, d)) + np.array([-5.0, 0.0])
    points = np.vstack([a, b])
    bound = MultiEllipsoid(ndim=d, enlargement=1.0)
    bound.fit(points)
    for _ in range(200):
        x = bound.sample(rng)
        assert bound.contains(x)


def test_multi_ellipsoid_log_volume_matches_single_for_unimodal() -> None:
    """For a unimodal point cloud both bounds should land on essentially
    the same volume (modulo enlargement and split criterion epsilons).
    """
    rng = np.random.default_rng(23)
    d = 3
    points = rng.standard_normal(size=(80, d))

    multi = MultiEllipsoid(ndim=d, enlargement=1.0)
    multi.fit(points)
    single = SingleEllipsoid(ndim=d, enlargement=1.0)
    single.fit(points)
    assert np.isclose(multi.log_volume(), single.log_volume(), atol=1e-9)


def test_multi_ellipsoid_rejects_premature_sample() -> None:
    bound = MultiEllipsoid(ndim=2)
    with pytest.raises(RuntimeError, match="before fit"):
        bound.sample(np.random.default_rng(0))


def test_log_unit_ball_volume_matches_known_values() -> None:
    """V_1 = 2, V_2 = pi, V_3 = 4 pi / 3, V_4 = pi^2 / 2."""
    assert np.isclose(_log_unit_ball_volume(1), float(np.log(2.0)))
    assert np.isclose(_log_unit_ball_volume(2), float(np.log(np.pi)))
    assert np.isclose(_log_unit_ball_volume(3), float(np.log(4.0 * np.pi / 3.0)))
    assert np.isclose(_log_unit_ball_volume(4), float(np.log(np.pi**2 / 2.0)))
