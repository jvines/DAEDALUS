"""Nested-sampling diagnostics: insertion-index test and evidence-error helpers.

This module promotes the working parts of ``scripts/insertion_index.py`` into
the daedalus public API.

Insertion-index test (Fowlie, Handley & Su 2020)
------------------------------------------------
Fowlie, Handley & Su (2020), "Nested sampling cross-checks using order
statistics" (MNRAS 497, 5256; arXiv:2006.03371). When a new live point is born
at likelihood L under a *correct* NS kernel (an i.i.d. draw from the prior
restricted to {L > L*}), its rank among the current ``n_live`` live-point
likelihoods is Uniform{0, 1, ..., n_live - 1}. A kernel that fails to
decorrelate the newborn from the donor it walked from biases the rank away from
uniform: the newborn lands systematically high or low because it never moved
far from a point that was itself near (or far from) threshold. A significant
departure from uniformity is the signature of within-model under-mixing.

The per-iteration insertion index for a daedalus run is recorded in-loop by
passing ``insertion_recorder=[]`` to ``NestedSampler.run_nested``; feed the
resulting list to :func:`insertion_index_test`.

Evidence error
--------------
The analytic Skilling (2006) ``log Z`` error from the information ``H`` is a
LOWER bound on hard/correlated problems: it assumes the NS prior-volume
shrinkage is exactly the textbook ``log t`` schedule and that the replacement
draws are i.i.d. from the constrained prior. When the kernel under-mixes both
assumptions break, and the true run-to-run scatter exceeds the analytic error
(Higson et al. 2018; Fowlie et al. 2020). The honest checks are therefore (a)
the multi-run ``log Z`` scatter, computed by :func:`multirun_logZ_error`, and
(b) the insertion-index test above.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------
# Insertion-index uniformity test (Fowlie, Handley & Su 2020)
# --------------------------------------------------------------------------

@dataclass
class InsertionTestResult:
    n_iter: int
    n_live: int
    mean_fraction: float            # mean(index / (n_live - 1)); ~0.5 if correct
    mean_fraction_se: float         # analytic SE of the mean under U
    ks_stat: float                  # KS statistic of index/n_live vs U(0,1)
    ks_pvalue: float                # global KS p-value
    rolling_min_pvalue: float       # min p over rolling-window mean z-tests
    rolling_window: int
    rolling_n_windows: int
    rolling_frac_windows_low: float # frac of windows with mean frac < 0.5 at 3sigma
    z_mean: float                   # global z-score of mean fraction vs 0.5
    verdict: str

    def as_dict(self) -> dict:
        return asdict(self)


def insertion_index_test(
    indices: np.ndarray,
    n_live: int,
    rolling_window: int | None = None,
    alpha: float = 0.05,
) -> InsertionTestResult:
    """Test a sequence of insertion indices for U{0,...,n_live-1}.

    Two complementary tests:

    1. Global KS test of ``(index + 0.5) / n_live`` against U(0,1). The
       +0.5 continuity correction maps the discrete uniform to the
       continuous one without edge bias.

    2. Rolling-window mean-fraction z-test. Under U, ``index/(n_live-1)``
       has mean 0.5 and the window-mean has SE = sqrt(Var / W) with
       Var = ((n_live+1)/(n_live-1)) / 12 -> ~1/12 for large n_live. We
       z-test each window's mean against 0.5 and report the minimum
       two-sided p-value (Bonferroni-naive but reported alongside the
       count of low windows). A single global KS can wash out a slow drift
       in mixing quality across the run; the rolling test localises it.

    The headline ``mean_fraction`` (mean of index/(n_live-1)) is ~0.5 for
    correct sampling. Under-mixing pushes it AWAY from 0.5; the sign depends
    on the donor-selection scheme. daedalus picks the donor uniformly from
    the live set (``_pick_donor``), so a stalled newborn stays near a
    typical (high-L) donor and inserts HIGH (mean_fraction > 0.5). A scheme
    that seeds the kernel from the just-killed near-threshold point would
    instead skew LOW. Either way the diagnostic flags the departure; do not
    assume the sign a priori (measured HD 10180: mean ~0.505-0.510, z up to
    +8.1, KS p down to 2e-15 -- high-skew under-mixing).
    """
    indices = np.asarray(indices, dtype=float)
    n = indices.size
    if n == 0:
        raise ValueError("empty insertion-index sequence")

    # Fraction in [0, 1] via (n_live-1) so the discrete range {0..n_live-1}
    # maps to {0, ..., 1}. Mean is 0.5 under U.
    frac = indices / (n_live - 1)
    mean_fraction = float(frac.mean())
    # Variance of discrete U{0..m} with m=n_live-1 is m(m+2)/12 in index
    # units; as a fraction of (n_live-1) it's (n_live+1)/((n_live-1)*12).
    var_frac = (n_live + 1.0) / ((n_live - 1.0) * 12.0)
    mean_fraction_se = float(np.sqrt(var_frac / n))
    z_mean = float((mean_fraction - 0.5) / mean_fraction_se)

    # Global KS against U(0,1) with continuity correction.
    u = (indices + 0.5) / n_live
    ks_stat, ks_p = stats.kstest(u, "uniform")

    # Rolling-window mean test.
    if rolling_window is None:
        # ~ default to a window giving >= 20 windows, but at least 200 wide.
        rolling_window = max(200, n // 20)
    rolling_window = int(min(rolling_window, n))
    n_windows = n // rolling_window
    rolling_min_p = 1.0
    n_low = 0
    if n_windows >= 1:
        win_se = float(np.sqrt(var_frac / rolling_window))
        for w in range(n_windows):
            seg = frac[w * rolling_window:(w + 1) * rolling_window]
            zw = (seg.mean() - 0.5) / win_se
            pw = 2.0 * stats.norm.sf(abs(zw))
            rolling_min_p = min(rolling_min_p, float(pw))
            if zw < -3.0:
                n_low += 1
    frac_windows_low = float(n_low / n_windows) if n_windows else 0.0

    # Verdict.
    #
    # Fowlie+2020's null is U{0..n_live-1}; ANY significant departure
    # (KS reject and/or |z_mean| large) means the replacement draws are not
    # i.i.d. from the constrained prior, i.e. the kernel is not
    # decorrelating the newborn from its donor. The DIRECTION of the mean
    # skew depends on the donor-selection scheme:
    #   - donor drawn near threshold  -> stalled newborn inserts LOW.
    #   - donor drawn uniformly from the live set (daedalus _pick_donor)
    #     -> stalled newborn stays near a typical high-L donor and inserts
    #     HIGH.
    # So we flag non-uniformity in EITHER direction as under-mixing and
    # report the sign separately rather than assuming low-skew a priori.
    ks_reject = ks_p < alpha
    mean_reject = abs(z_mean) > 3.0
    if ks_reject or mean_reject:
        if z_mean < -3.0:
            skew = "low (newborns insert near bottom)"
        elif z_mean > 3.0:
            skew = "high (newborns insert near top)"
        else:
            skew = "shape (KS reject, mean ~0.5)"
        verdict = (
            f"UNDER-MIXING: insertion indices NON-uniform; skew {skew}; "
            f"kernel not decorrelating newborn from donor"
        )
    else:
        verdict = "consistent with uniform (no under-mixing detected)"

    return InsertionTestResult(
        n_iter=n,
        n_live=int(n_live),
        mean_fraction=mean_fraction,
        mean_fraction_se=mean_fraction_se,
        ks_stat=float(ks_stat),
        ks_pvalue=float(ks_p),
        rolling_min_pvalue=float(rolling_min_p),
        rolling_window=int(rolling_window),
        rolling_n_windows=int(n_windows),
        rolling_frac_windows_low=frac_windows_low,
        z_mean=z_mean,
        verdict=verdict,
    )


# --------------------------------------------------------------------------
# Multi-run evidence error
# --------------------------------------------------------------------------

def multirun_logZ_error(logZ_values) -> tuple[float, float]:
    """Honest evidence error from the scatter of independent NS runs.

    Parameters
    ----------
    logZ_values
        1-D sequence of ``log Z`` point estimates from independent nested
        sampling runs of the SAME problem with DIFFERENT seeds. At least two
        values are required.

    Returns
    -------
    (mean, std)
        ``mean`` is the sample mean of ``log Z`` across runs; ``std`` is the
        sample standard deviation (``ddof=1``, the unbiased estimator) of the
        per-run ``log Z``. ``std`` is the honest, distribution-free estimate
        of the ``log Z`` uncertainty for a single run.

    Notes
    -----
    The per-run analytic Skilling (2006) error ``sqrt(H / n_live)`` is a LOWER
    bound on hard/correlated problems: it assumes the prior-volume shrinkage
    follows the exact ``log t`` schedule and that replacement draws are i.i.d.
    from the constrained prior. When the within-model kernel under-mixes, both
    assumptions fail and the true run-to-run scatter EXCEEDS the analytic error
    (Higson et al. 2018, "Sampling Errors in Nested Sampling Parameter
    Estimation", Bayesian Anal. 13, 873, arXiv:1703.09701; Fowlie, Handley & Su
    2020, MNRAS 497, 5256, arXiv:2006.03371). The multi-run scatter returned
    here -- or, for a single run, the insertion-index test
    (:func:`insertion_index_test`) -- is therefore the honest check on the
    reported evidence uncertainty.

    This deliberately does NOT implement a single-run bootstrap of the analytic
    error: a within-run bootstrap of the textbook ``log t`` recursion just
    reproduces the analytic Skilling error and would hide exactly the
    correlation-driven excess scatter this helper is meant to expose. The only
    honest way to capture that excess is to run the sampler several times with
    different seeds and measure the spread directly.
    """
    logZ = np.asarray(logZ_values, dtype=float).ravel()
    if logZ.size < 2:
        raise ValueError(
            "multirun_logZ_error needs >= 2 independent-run logZ values; "
            f"got {logZ.size}"
        )
    mean = float(logZ.mean())
    std = float(logZ.std(ddof=1))
    return mean, std
