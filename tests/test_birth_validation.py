"""Tests for ``daedalus.validate_birth_consistency`` and the
``BirthConsistencyWarning`` emitted by ``NestedSampler.__init__``.

The validator's job is to catch the silent failure mode where a user's
``Group.log_prior_continuous`` doesn't match the pushforward of
``prior_transform`` on the group's coords. We exercise:

  * a *consistent* setup -- uniform pushforward + matching uniform
    log_prior_continuous -- and require ``passed=True``;
  * a *consistent* Gaussian-pushforward setup (norm.ppf in
    prior_transform + matching log-N density) -- ``passed=True``;
  * three deliberately-broken setups (wrong constant, wrong support,
    wrong shape) -- ``passed=False`` with informative messages;
  * the coordinate-coupling detection branch -- ``passed=False`` with
    the coupling coords reported;
  * the NestedSampler integration -- a broken Group triggers
    ``BirthConsistencyWarning`` at construction; passing
    ``validate_births=False`` suppresses it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import norm

from daedalus import (
    BirthConsistencyWarning,
    GaussianRWBirth,
    Group,
    NestedSampler,
    validate_birth_consistency,
)


def _uniform_log_prior_factory(lo: float, hi: float):
    log_density = -float(np.log(hi - lo))

    def log_prior(beta: np.ndarray) -> float:
        if np.any(beta < lo) or np.any(beta > hi):
            return -np.inf
        return float(beta.size) * log_density

    return log_prior


def test_validator_passes_on_consistent_uniform_setup() -> None:
    """Uniform prior_transform on [-W, W] + matching uniform density."""
    W = 5.0

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return 2.0 * W * u - W

    group = Group(
        name="x",
        params=[0],
        off_values=np.array([0.0]),
        birth_proposal=GaussianRWBirth(scale=1.0),
        log_prior_continuous=_uniform_log_prior_factory(-W, W),
    )
    report = validate_birth_consistency(
        group, prior_transform, ndim=1, n_samples=100, seed=0
    )
    assert report["passed"], (
        f"consistent uniform setup must pass: max_residual="
        f"{report['max_residual']}, msg={report['message']}"
    )
    assert report["max_residual"] < 1e-3
    assert report["n_support_violations"] == 0
    assert not report["coupling_coords"]


def test_validator_passes_on_consistent_gaussian_setup() -> None:
    """norm.ppf prior_transform + matching N(0, sigma^2) density."""
    sigma = 2.5

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return norm.ppf(np.clip(u, 1e-12, 1 - 1e-12)) * sigma

    log_const = -0.5 * float(np.log(2.0 * np.pi * sigma ** 2))
    inv_2sigma2 = 0.5 / sigma ** 2

    def log_prior(beta: np.ndarray) -> float:
        b = float(beta[0])
        return log_const - inv_2sigma2 * b * b

    group = Group(
        name="x",
        params=[0],
        off_values=np.array([0.0]),
        birth_proposal=GaussianRWBirth(scale=1.0),
        log_prior_continuous=log_prior,
    )
    report = validate_birth_consistency(
        group, prior_transform, ndim=1, n_samples=200, seed=0,
        # FD on a Gaussian inverse-CDF can be a touch noisier.
        fd_step=1e-6,
        tol=5e-3,
    )
    assert report["passed"], (
        f"consistent Gaussian setup must pass: max_residual="
        f"{report['max_residual']}, mean_residual={report['mean_residual']}, "
        f"msg={report['message']}"
    )


def test_validator_catches_wrong_constant() -> None:
    """User multiplied log_prior by a constant offset -- mean_residual
    should equal the offset and the validator must fail."""
    W = 5.0

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return 2.0 * W * u - W

    correct = -float(np.log(2.0 * W))
    offset = 1.234

    def wrong_log_prior(beta: np.ndarray) -> float:
        if np.any(np.abs(beta) > W):
            return -np.inf
        return correct + offset  # wrong by a constant

    group = Group(
        name="x",
        params=[0],
        off_values=np.array([0.0]),
        birth_proposal=GaussianRWBirth(scale=1.0),
        log_prior_continuous=wrong_log_prior,
    )
    report = validate_birth_consistency(
        group, prior_transform, ndim=1, n_samples=100, seed=0
    )
    assert not report["passed"]
    assert np.isclose(report["mean_residual"], offset, atol=1e-3)
    assert "log-normalisation constant" in report["message"]


def test_validator_catches_wrong_support() -> None:
    """User claimed support is narrower than what prior_transform produces.
    Half the samples should fall outside, triggering the support check."""
    W = 5.0

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return 2.0 * W * u - W  # samples uniform on [-5, 5]

    group = Group(
        name="x",
        params=[0],
        off_values=np.array([0.0]),
        birth_proposal=GaussianRWBirth(scale=1.0),
        log_prior_continuous=_uniform_log_prior_factory(-1.0, 1.0),  # too narrow
    )
    report = validate_birth_consistency(
        group, prior_transform, ndim=1, n_samples=200, seed=0
    )
    assert not report["passed"]
    assert report["n_support_violations"] > 100, (
        f"~80% of samples should fall outside [-1, 1] given uniform draws "
        f"on [-5, 5]; got n_violations={report['n_support_violations']}"
    )
    assert "support" in report["message"]


def test_validator_catches_wrong_shape() -> None:
    """User gave a Gaussian density when prior_transform is uniform.
    The residuals should be position-dependent (not a constant offset)."""
    W = 5.0

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return 2.0 * W * u - W

    sigma = 1.0
    log_const = -0.5 * float(np.log(2.0 * np.pi * sigma ** 2))

    def gaussian_log_prior(beta: np.ndarray) -> float:
        if np.any(np.abs(beta) > W):
            return -np.inf
        b = float(beta[0])
        return log_const - 0.5 * (b / sigma) ** 2

    group = Group(
        name="x",
        params=[0],
        off_values=np.array([0.0]),
        birth_proposal=GaussianRWBirth(scale=1.0),
        log_prior_continuous=gaussian_log_prior,
    )
    report = validate_birth_consistency(
        group, prior_transform, ndim=1, n_samples=200, seed=0
    )
    assert not report["passed"]
    # Gaussian on [-5, 5] vs uniform: residuals are position-dependent,
    # so mean_residual is NOT close to max_residual (would be the
    # diagnostic for a constant offset).
    assert report["max_residual"] > 1.0
    assert "shape disagrees" in report["message"] or "log-normalisation" in report["message"]


def test_validator_detects_coordinate_coupling() -> None:
    """Coupled prior_transform (in-group beta depends on an out-of-group
    u-coord) must be flagged: the change-of-variables check assumes
    coordinate-separability on the group's params."""
    # 3-coord prior_transform; group covers only beta[0]. The transform
    # makes beta[0] depend on BOTH u[0] (in-group) AND u[2] (out-of-
    # group), so the validator's separability assumption is violated.
    W = 3.0

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return np.array(
            [
                (2.0 * W * u[0] - W) + 0.5 * u[2],   # beta[0] couples to u[2]
                2.0 * W * u[1] - W,
                2.0 * W * u[2] - W,
            ]
        )

    group = Group(
        name="coupled",
        params=[0],
        off_values=np.array([0.0]),
        birth_proposal=GaussianRWBirth(scale=1.0),
        log_prior_continuous=_uniform_log_prior_factory(-W, W),
    )
    report = validate_birth_consistency(
        group, prior_transform, ndim=3, n_samples=20, seed=0
    )
    assert not report["passed"]
    assert report["coupling_coords"], (
        "coupling of out-of-group u-coords into in-group beta must be detected"
    )
    assert 2 in report["coupling_coords"]
    assert "couples" in report["message"]


def test_nested_sampler_warns_on_broken_birth_by_default() -> None:
    """A Group with mismatched log_prior_continuous must trigger
    BirthConsistencyWarning when NestedSampler is constructed with
    the default ``validate_births=True``."""
    W = 5.0

    def loglike(theta: np.ndarray) -> float:
        return -0.5 * float(np.dot(theta, theta))

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return 2.0 * W * u - W

    bad_group = Group(
        name="bad",
        params=[0],
        off_values=np.array([0.0]),
        birth_proposal=GaussianRWBirth(scale=1.0),
        log_prior_continuous=lambda b: 999.0,  # absurd constant
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        NestedSampler(
            loglike=loglike,
            prior_transform=prior_transform,
            ndim=1,
            groups=[bad_group],
            n_live=10,
            seed=0,
        )
    consistency_warnings = [
        w for w in caught if issubclass(w.category, BirthConsistencyWarning)
    ]
    assert consistency_warnings, "expected BirthConsistencyWarning to be raised"
    assert "bad" in str(consistency_warnings[0].message)


def test_nested_sampler_validate_births_false_suppresses_check() -> None:
    """``validate_births=False`` must skip the consistency check entirely
    (no warnings, no exception)."""
    W = 5.0

    def loglike(theta: np.ndarray) -> float:
        return 0.0

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return 2.0 * W * u - W

    bad_group = Group(
        name="bad",
        params=[0],
        off_values=np.array([0.0]),
        birth_proposal=GaussianRWBirth(scale=1.0),
        log_prior_continuous=lambda b: 999.0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        NestedSampler(
            loglike=loglike,
            prior_transform=prior_transform,
            ndim=1,
            groups=[bad_group],
            n_live=10,
            seed=0,
            validate_births=False,
        )
    consistency_warnings = [
        w for w in caught if issubclass(w.category, BirthConsistencyWarning)
    ]
    assert not consistency_warnings, (
        "validate_births=False must suppress all consistency warnings"
    )


def test_nested_sampler_skips_validation_when_no_custom_birth() -> None:
    """Default uniform-u path doesn't use log_prior_continuous and needs
    no validation. Construction must be silent."""
    W = 5.0

    def loglike(theta: np.ndarray) -> float:
        return 0.0

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return 2.0 * W * u - W

    plain_group = Group(
        name="plain",
        params=[0],
        off_values=np.array([0.0]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        NestedSampler(
            loglike=loglike,
            prior_transform=prior_transform,
            ndim=1,
            groups=[plain_group],
            n_live=10,
            seed=0,
        )
    consistency_warnings = [
        w for w in caught if issubclass(w.category, BirthConsistencyWarning)
    ]
    assert not consistency_warnings


def test_validator_raises_when_log_prior_continuous_missing() -> None:
    """If a group has no log_prior_continuous (e.g., default uniform-u),
    explicit calls to validate_birth_consistency are user errors."""
    group = Group(name="x", params=[0], off_values=np.array([0.0]))

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return u

    with pytest.raises(ValueError, match="no log_prior_continuous"):
        validate_birth_consistency(group, prior_transform, ndim=1)
