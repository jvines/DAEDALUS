"""Smoke tests for the public API shape.

These tests verify that the package imports cleanly, that the constructor
signatures accept the documented arguments, and that input validation rejects
mis-declared groups. They do not exercise any actual sampling — that comes
once the implementations are filled in.
"""

from __future__ import annotations

import numpy as np
import pytest

import daedalus


def _trivial_loglike(theta: np.ndarray) -> float:
    return -0.5 * float(np.sum(theta * theta))


def _trivial_prior_transform(u: np.ndarray) -> np.ndarray:
    return 10.0 * u - 5.0


def test_package_exposes_public_api() -> None:
    assert hasattr(daedalus, "NestedSampler")
    assert hasattr(daedalus, "Group")
    assert hasattr(daedalus, "Results")
    assert hasattr(daedalus, "State")


def test_continuous_only_construction_accepts_no_groups() -> None:
    sampler = daedalus.NestedSampler(
        loglike=_trivial_loglike,
        prior_transform=_trivial_prior_transform,
        ndim=3,
        bound="none",
        sample="rwalk",
        n_live=50,
        seed=0,
    )
    assert sampler.ndim == 3
    assert sampler.groups == ()


def test_group_construction_validates_off_value_length() -> None:
    with pytest.raises(ValueError, match="off_values"):
        daedalus.Group(
            name="bad",
            params=[0, 1, 2],
            off_values=np.array([0.0, 0.0]),  # wrong length
        )


def test_group_construction_validates_inclusion_prior_range() -> None:
    with pytest.raises(ValueError, match="inclusion_prior"):
        daedalus.Group(
            name="bad",
            params=[0],
            off_values=np.array([0.0]),
            inclusion_prior=0.0,
        )


def test_sampler_rejects_overlapping_group_indices() -> None:
    g1 = daedalus.Group(name="a", params=[0, 1], off_values=np.zeros(2))
    g2 = daedalus.Group(name="b", params=[1, 2], off_values=np.zeros(2))
    with pytest.raises(ValueError, match="more than one group"):
        daedalus.NestedSampler(
            loglike=_trivial_loglike,
            prior_transform=_trivial_prior_transform,
            ndim=3,
            groups=[g1, g2],
            bound="none",
            n_live=50,
            seed=0,
        )


def test_sampler_rejects_out_of_range_group_index() -> None:
    g = daedalus.Group(name="a", params=[5], off_values=np.zeros(1))
    with pytest.raises(ValueError, match="outside"):
        daedalus.NestedSampler(
            loglike=_trivial_loglike,
            prior_transform=_trivial_prior_transform,
            ndim=3,
            groups=[g],
            bound="none",
            n_live=50,
            seed=0,
        )


def test_unknown_bound_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown bound"):
        daedalus.NestedSampler(
            loglike=_trivial_loglike,
            prior_transform=_trivial_prior_transform,
            ndim=3,
            bound="not_a_bound",
            n_live=50,
            seed=0,
        )


def test_unknown_sampler_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown sampler"):
        daedalus.NestedSampler(
            loglike=_trivial_loglike,
            prior_transform=_trivial_prior_transform,
            ndim=3,
            sample="not_a_sampler",
            n_live=50,
            seed=0,
        )
