"""Post-processing plots for daedalus :class:`Results` objects.

Provides five visualisation entry points:

* :func:`cornerplot` -- corner plot of the equally-weighted posterior
  beta samples. Wrapper around ``corner.corner`` with sensible defaults
  (smoothed contours, true-value markers, optional gamma masking).
* :func:`traceplot` -- per-coordinate sample value as a function of NS
  iteration, coloured by importance weight; the dynesty-equivalent
  diagnostic for posterior structure across the run.
* :func:`runplot` -- log-Z accumulation, prior volume, and live-point
  log-likelihood vs. NS iteration. Quick visual sanity check that the
  run terminated cleanly.
* :func:`model_probability_plot` -- bar chart of the top-N visited
  inclusion configurations by posterior probability. Trans-dim only.
* :func:`inclusion_probability_plot` -- bar chart of the per-group
  marginal inclusion probability. Trans-dim only.

All five accept an optional ``ax`` (or ``fig``) argument so callers can
embed them in a larger figure layout, and all return the matplotlib
``Figure`` so the caller can further customise. Matplotlib is imported
lazily; ``corner`` is imported lazily inside :func:`cornerplot` only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure
    from .results import Results


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "daedalus.plotting requires matplotlib. "
            "Install with: pip install 'daedalus[plotting]'"
        ) from exc
    return plt


def _require_corner():
    try:
        import corner
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "daedalus.plotting.cornerplot requires the corner package. "
            "Install with: pip install 'daedalus[plotting]'"
        ) from exc
    return corner


def cornerplot(
    results: "Results",
    *,
    truths: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    active_only: bool = False,
    group_index: int | None = None,
    smooth: float = 0.8,
    show_titles: bool = True,
    title_fmt: str = ".3f",
    fig: "matplotlib.figure.Figure | None" = None,
    **corner_kwargs: Any,
) -> "matplotlib.figure.Figure":
    """Corner plot of the equally-weighted posterior samples.

    Parameters
    ----------
    results
        The :class:`Results` object returned by ``NestedSampler.run_nested``.
    truths
        Optional ground-truth values for each beta coordinate (length
        ``ndim``). Drawn as orthogonal lines if supplied.
    labels
        Optional axis labels for each beta coordinate. Default uses
        ``r"$\\beta_{i}$"``.
    active_only
        If True, restrict the plot to posterior samples in which a
        specific group is active (requires ``group_index``). Useful for
        plotting only the slab posterior of one slot, ignoring the
        spike contribution.
    group_index
        Group index used by ``active_only``. The slot's parameters are
        the only coordinates plotted in this case.
    smooth
        Gaussian smoothing in 1D and 2D, in units of bin widths
        (``corner.corner`` default convention).
    show_titles, title_fmt
        Forwarded to ``corner.corner``.
    fig
        Existing matplotlib figure to draw onto. If None a new one is
        created.
    **corner_kwargs
        Any additional kwargs are forwarded to ``corner.corner`` (e.g.
        ``range``, ``color``, ``hist_kwargs``).

    Returns
    -------
    matplotlib.figure.Figure
        The figure (also accessible via ``corner.corner``'s return).
    """
    plt = _require_matplotlib()
    corner = _require_corner()

    samples = results.samples
    if active_only:
        if group_index is None:
            raise ValueError("active_only=True requires group_index to be set")
        if results.gamma.shape[1] == 0:
            raise ValueError(
                "active_only=True requires a trans-dim Results (groups declared)"
            )
        if not (0 <= group_index < results.gamma.shape[1]):
            raise ValueError(
                f"group_index={group_index} out of range "
                f"[0, {results.gamma.shape[1]})"
            )
        mask = results.gamma[:, group_index].astype(bool)
        if mask.sum() < 2:
            raise ValueError(
                f"group {group_index} has fewer than 2 active samples; "
                f"corner plot is undefined"
            )
        samples = samples[mask]

    ndim = samples.shape[1]
    if labels is None:
        labels = [rf"$\beta_{{{i}}}$" for i in range(ndim)]

    return corner.corner(
        samples,
        labels=list(labels),
        truths=list(truths) if truths is not None else None,
        smooth=smooth,
        show_titles=show_titles,
        title_fmt=title_fmt,
        fig=fig,
        **corner_kwargs,
    )


def traceplot(
    results: "Results",
    *,
    coords: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    truths: Sequence[float] | None = None,
    cmap: str = "cool",
    figsize: tuple[float, float] | None = None,
    fig: "matplotlib.figure.Figure | None" = None,
) -> "matplotlib.figure.Figure":
    """Per-coordinate sample-value-vs-iteration trace plot.

    For each beta coordinate, plots ``beta[i]`` against the NS dead-point
    iteration index (left column) and a weighted KDE histogram of the
    posterior marginal (right column), with points coloured by
    importance weight ``exp(log_weights + log_likelihoods - log_Z)``.

    Parameters
    ----------
    results
        :class:`Results` object.
    coords
        Coordinate indices to plot. Default: all.
    labels
        Optional axis labels.
    truths
        Optional ground-truth values, drawn as horizontal/vertical lines.
    cmap
        Matplotlib colormap name for the per-sample importance-weight
        colouring. Default ``"cool"``; pass ``"cool_r"`` to invert.
    figsize
        Figure size. Defaults to ``(10, 1.6 * len(coords))``.
    fig
        Existing matplotlib figure to draw onto. If None a new one is
        created.
    """
    plt = _require_matplotlib()
    if coords is None:
        coords = list(range(results.samples.shape[1]))
    n = len(coords)
    if labels is None:
        labels = [rf"$\beta_{{{i}}}$" for i in coords]

    log_weights = results.log_weights + results.log_likelihoods - results.log_Z
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= weights.sum()

    if figsize is None:
        figsize = (10.0, 1.6 * n)
    if fig is None:
        fig, axes = plt.subplots(n, 2, figsize=figsize, squeeze=False)
    else:
        axes = np.asarray(fig.axes).reshape(n, 2)

    # Dead-point sequence is what's stored in dead_log_likelihoods /
    # dead_log_weights; results.samples is the resampled chain. Use
    # the resampled chain on the iteration axis -- they're already
    # equally-weighted and in order.
    iters = np.arange(results.samples.shape[0])

    for row, (i, label) in enumerate(zip(coords, labels)):
        ax_trace, ax_hist = axes[row, 0], axes[row, 1]
        ax_trace.scatter(
            iters,
            results.samples[:, i],
            c=weights,
            cmap=cmap,
            s=2,
            alpha=0.5,
            rasterized=True,
        )
        ax_trace.set_ylabel(label)
        if row == n - 1:
            ax_trace.set_xlabel("posterior sample index")
        else:
            ax_trace.set_xticklabels([])

        # Right column: weighted histogram.
        ax_hist.hist(
            results.samples[:, i],
            bins=40,
            histtype="step",
            color="C0",
            density=True,
        )
        ax_hist.set_yticks([])
        if row == n - 1:
            ax_hist.set_xlabel(label)
        else:
            ax_hist.set_xticklabels([])

        if truths is not None:
            ax_trace.axhline(truths[i], color="C3", lw=0.8, ls="--")
            ax_hist.axvline(truths[i], color="C3", lw=0.8, ls="--")

    fig.tight_layout()
    return fig


def runplot(
    results: "Results",
    *,
    figsize: tuple[float, float] = (8.0, 6.0),
    fig: "matplotlib.figure.Figure | None" = None,
) -> "matplotlib.figure.Figure":
    """Diagnostic plot of the NS run.

    Three panels:
      1. Dead-point log-likelihood vs. iteration.
      2. Cumulative log-Z vs. iteration with the final value annotated.
      3. Per-iteration importance weight ``log L + log dX - log Z``,
         showing where the posterior mass concentrates.

    A clean run shows monotonically rising log-likelihood, a log-Z
    curve that asymptotes well before termination, and an importance-
    weight peak well clear of the run's start and end.
    """
    plt = _require_matplotlib()
    if fig is None:
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    else:
        axes = np.asarray(fig.axes)

    iters = np.arange(results.log_likelihoods.size)
    log_L = results.log_likelihoods
    log_dX = results.log_weights
    cum_log_Z = np.logaddexp.accumulate(log_L + log_dX)
    log_w = log_L + log_dX - results.log_Z

    axes[0].plot(iters, log_L, color="C0", lw=1.0)
    axes[0].set_ylabel(r"$\log\mathcal{L}$")

    axes[1].plot(iters, cum_log_Z, color="C2", lw=1.2)
    axes[1].axhline(results.log_Z, color="C2", lw=0.6, ls="--")
    axes[1].set_ylabel(r"$\log\mathcal{Z}$ (cumulative)")
    axes[1].annotate(
        f"final $\\log\\mathcal{{Z}} = {results.log_Z:.3f} \\pm "
        f"{results.log_Z_err:.3f}$",
        xy=(0.98, 0.05),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=9,
    )

    axes[2].plot(iters, log_w, color="C3", lw=1.0)
    axes[2].set_ylabel(r"$\log L + \log\,\Delta X - \log\mathcal{Z}$")
    axes[2].set_xlabel("dead-point iteration")

    fig.tight_layout()
    return fig


def model_probability_plot(
    results: "Results",
    *,
    top_n: int | None = 10,
    ax: "matplotlib.axes.Axes | None" = None,
    figsize: tuple[float, float] = (7.0, 4.0),
) -> "matplotlib.figure.Figure":
    """Bar chart of the top-N posterior model probabilities P(gamma|y).

    For trans-dim runs only. Each bar represents one visited inclusion
    configuration, sorted by posterior probability. The x-axis labels
    encode the gamma vectors as binary strings; large group counts are
    truncated to the active indices for readability.
    """
    plt = _require_matplotlib()
    if results.gamma.shape[1] == 0:
        raise ValueError(
            "model_probability_plot requires a trans-dim Results "
            "(no toggleable groups declared)"
        )
    probs = results.model_probabilities()
    if not probs:
        raise ValueError(
            "no visited gamma configurations; nothing to plot"
        )

    items = sorted(probs.items(), key=lambda kv: -kv[1])
    if top_n is not None:
        items = items[:top_n]

    g = results.gamma.shape[1]
    if g <= 8:
        # Render as full binary string e.g. "(1,0,1,0,...)"
        labels = [
            "(" + ",".join("1" if x else "0" for x in key) + ")"
            for key, _ in items
        ]
    else:
        # Show only active-slot indices: "{2,5,7}" or "{}" for the empty model.
        labels = []
        for key, _ in items:
            active = [str(i) for i, x in enumerate(key) if x]
            labels.append("{" + ",".join(active) + "}" if active else r"$\emptyset$")
    values = [p for _, p in items]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.bar(np.arange(len(values)), values, color="C0", edgecolor="black", lw=0.5)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(r"$P(\gamma \mid y)$")
    ax.set_xlabel("inclusion configuration")
    if top_n is not None and len(values) >= top_n:
        ax.set_title(f"top-{top_n} configurations by posterior probability")
    fig.tight_layout()
    return fig


def inclusion_probability_plot(
    results: "Results",
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    sort: bool = True,
) -> "matplotlib.figure.Figure":
    """Bar chart of per-group marginal P(gamma_k = 1 | y).

    For trans-dim runs only. Provides the at-a-glance which-component-is-
    supported summary for a multi-component fit.
    """
    plt = _require_matplotlib()
    if results.gamma.shape[1] == 0:
        raise ValueError(
            "inclusion_probability_plot requires a trans-dim Results"
        )
    inc = results.inclusion_probabilities()
    names = list(inc.keys())
    values = np.array(list(inc.values()), dtype=np.float64)
    if sort:
        order = np.argsort(values)[::-1]
        names = [names[i] for i in order]
        values = values[order]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.bar(np.arange(len(values)), values, color="C2", edgecolor="black", lw=0.5)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(r"$P(\gamma_k = 1 \mid y)$")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.5, color="grey", lw=0.6, ls="--")
    fig.tight_layout()
    return fig
