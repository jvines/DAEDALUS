"""Smoke tests for ``daedalus.plotting``.

Each test runs a short NS chain on a small benchmark, then calls one
plotting entry point and verifies it returns a matplotlib Figure
without raising. We don't validate visual content -- the goal is
"importing and calling the plotters does not crash on the kind of
Results objects users actually produce" -- which catches structural
regressions (missing fields, wrong dtypes, indexing errors).

We force the matplotlib Agg backend before importing pyplot so the
tests don't try to open windows on headless CI.
"""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("corner")

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

import daedalus
from daedalus import plotting
from daedalus.benchmarks import gaussians


@pytest.fixture(scope="module")
def fixed_dim_results() -> daedalus.Results:
    """A short continuous-only NS run on a 3D Gaussian — enough samples
    to make every plot well-defined, fast enough for a fixture."""
    problem = gaussians.make_problem(ndim=3, prior_half_width=10.0)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="single",
        sample="rwalk",
        n_live=120,
        seed=0,
    )
    return sampler.run_nested(dlogz=1.0, n_mcmc=10, show_progress=False)


@pytest.fixture(scope="module")
def trans_dim_results() -> daedalus.Results:
    """Trans-dim run with two declared groups, enough samples that every
    inclusion configuration is visited at least once."""
    problem = gaussians.make_problem(ndim=2, prior_half_width=10.0)
    g0 = daedalus.Group(name="g0", params=[0], off_values=np.array([0.0]),
                        inclusion_prior=0.5)
    g1 = daedalus.Group(name="g1", params=[1], off_values=np.array([0.0]),
                        inclusion_prior=0.5)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=[g0, g1],
        bound="none",
        sample="rwalk",
        n_live=120,
        seed=0,
    )
    return sampler.run_nested(
        dlogz=1.0, n_mcmc=15, transdim_fraction=0.5, show_progress=False
    )


# ---- cornerplot -------------------------------------------------------------


def test_cornerplot_returns_figure(fixed_dim_results) -> None:
    fig = plotting.cornerplot(fixed_dim_results)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_cornerplot_with_labels_and_truths(fixed_dim_results) -> None:
    fig = plotting.cornerplot(
        fixed_dim_results,
        labels=["x", "y", "z"],
        truths=[0.0, 0.0, 0.0],
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_cornerplot_active_only_on_trans_dim(trans_dim_results) -> None:
    fig = plotting.cornerplot(
        trans_dim_results,
        active_only=True,
        group_index=0,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_cornerplot_active_only_requires_group_index(fixed_dim_results) -> None:
    with pytest.raises(ValueError, match="requires group_index"):
        plotting.cornerplot(fixed_dim_results, active_only=True)


def test_cornerplot_active_only_requires_groups(fixed_dim_results) -> None:
    with pytest.raises(ValueError, match="trans-dim Results"):
        plotting.cornerplot(
            fixed_dim_results, active_only=True, group_index=0
        )


# ---- traceplot --------------------------------------------------------------


def test_traceplot_returns_figure(fixed_dim_results) -> None:
    fig = plotting.traceplot(fixed_dim_results)
    assert isinstance(fig, plt.Figure)
    # n coords -> n rows x 2 cols
    assert len(fig.axes) == 2 * fixed_dim_results.samples.shape[1]
    plt.close(fig)


def test_traceplot_subset_of_coords(fixed_dim_results) -> None:
    fig = plotting.traceplot(
        fixed_dim_results,
        coords=[0, 2],
        labels=[r"$x_0$", r"$x_2$"],
        truths=[0.0, 0.0, 0.0],
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4  # 2 coords * 2 cols
    plt.close(fig)


def test_traceplot_default_cmap_is_cool(fixed_dim_results) -> None:
    """Per project convention, weight-colored plots default to the
    'cool' colormap. Pin this so a future style change doesn't flip
    it back to viridis or jet silently."""
    fig = plotting.traceplot(fixed_dim_results, coords=[0])
    # The trace (left column) is the weight-colored scatter.
    ax_trace = fig.axes[0]
    scatters = ax_trace.collections
    assert scatters, "expected a scatter collection on the trace axis"
    assert scatters[0].get_cmap().name == "cool"
    plt.close(fig)


def test_traceplot_cmap_override(fixed_dim_results) -> None:
    fig = plotting.traceplot(fixed_dim_results, coords=[0], cmap="cool_r")
    scatters = fig.axes[0].collections
    assert scatters[0].get_cmap().name == "cool_r"
    plt.close(fig)


# ---- runplot ----------------------------------------------------------------


def test_runplot_returns_figure(fixed_dim_results) -> None:
    fig = plotting.runplot(fixed_dim_results)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3  # log L, log Z, importance weight
    plt.close(fig)


# ---- model_probability_plot -------------------------------------------------


def test_model_probability_plot(trans_dim_results) -> None:
    fig = plotting.model_probability_plot(trans_dim_results)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_model_probability_plot_top_n(trans_dim_results) -> None:
    fig = plotting.model_probability_plot(trans_dim_results, top_n=2)
    assert isinstance(fig, plt.Figure)
    # x-tick count should not exceed top_n
    ax = fig.axes[0]
    assert len(ax.get_xticks()) <= 2
    plt.close(fig)


def test_model_probability_plot_rejects_continuous_only(fixed_dim_results) -> None:
    with pytest.raises(ValueError, match="trans-dim Results"):
        plotting.model_probability_plot(fixed_dim_results)


# ---- inclusion_probability_plot ---------------------------------------------


def test_inclusion_probability_plot(trans_dim_results) -> None:
    fig = plotting.inclusion_probability_plot(trans_dim_results)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_inclusion_probability_plot_unsorted(trans_dim_results) -> None:
    fig = plotting.inclusion_probability_plot(trans_dim_results, sort=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_inclusion_probability_plot_rejects_continuous_only(fixed_dim_results) -> None:
    with pytest.raises(ValueError, match="trans-dim Results"):
        plotting.inclusion_probability_plot(fixed_dim_results)
