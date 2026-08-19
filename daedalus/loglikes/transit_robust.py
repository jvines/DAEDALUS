"""Per-cycle robust loglike for transit data.

Standard chi-squared loglikes can be "fooled" by candidates whose
predicted transit windows coincidentally overlap with another
candidate's real transits. The per-cycle robust loglike addresses
this by decomposing chi-squared contributions cycle by cycle and
aggregating via a robust statistic, so a candidate only gains chi
squared when its signal is detectable across MANY independent cycles.

Usage:
    from daedalus.loglikes import per_cycle_robust_transit_loglike
    loglike = per_cycle_robust_transit_loglike(
        time, flux, flux_err,
        evaluate_planet,
        n_planets,
        cycle_window_T14_factor=1.5,
        aggregator="trimmed_mean",
        trim_fraction=0.10,
    )

.. warning::

    EXPERIMENTAL. This module is not used to produce any published
    result, has no test coverage, and is not part of the supported
    public API of :mod:`daedalus`.

    Note also that with any aggregator other than ``"sum"`` the return
    value is a robustified *detection score*, not a normalised
    log-likelihood: a log-evidence accumulated from it is not a
    calibrated marginal likelihood and must not be read as one.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def per_cycle_robust_transit_loglike(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    evaluate_planet: Callable[[np.ndarray, int, dict], None],
    n_planets: int,
    cycle_window_T14_factor: float = 1.5,
    aggregator: str = "trimmed_mean",
    trim_fraction: float = 0.10,
    rho_star_kg_m3: float | None = None,
    M_star_msun: float = 1.0,
    R_star_rsun: float = 1.0,
):
    """Build a per-cycle robust loglike.

    Parameters
    ----------
    time, flux, flux_err
        Light curve arrays.
    evaluate_planet
        Callable ``evaluate_planet(out, k, params)`` writing the
        per-planet light curve (1 outside transit, < 1 inside) into
        ``out``. ``params`` is a dict with keys ``log_period``,
        ``t0_phase``, ``log_rr``, ``b``, ``u1``, ``u2``.
    n_planets
        Number of candidate slots.
    cycle_window_T14_factor
        Per-cycle integration window expressed in multiples of T14.
        Default 1.5 (each cycle window is 3 T14 wide).
    aggregator
        ``"trimmed_mean"`` (default) or ``"median"`` or ``"sum"`` (
        which reduces to standard chi-squared).
    trim_fraction
        For trimmed-mean: fraction of cycles to drop from each end of
        the per-cycle Δchi-squared distribution.

    Returns
    -------
    loglike
        Callable ``loglike(beta, gamma) -> float``.
    """
    inv_var = 1.0 / (flux_err * flux_err)

    # Pre-allocate buffers (kept in a list-of-1 so the closure can mutate
    # them via index without a nonlocal declaration)
    planet_buf_per_k = np.empty((n_planets, time.size), dtype=np.float64)
    full_model_buf = np.empty_like(time)

    # Stellar density used to derive a/Rs from period (Kepler 3rd law)
    if rho_star_kg_m3 is None:
        rho_sun = 1411.0
        rho_star_kg_m3 = rho_sun * (M_star_msun / R_star_rsun ** 3)

    def _t14_d(P_d, rr, b):
        """Approximate T14 in days from (P, rr, b) using rho-based a/Rs."""
        # a/R* = (G * rho * P^2 / (3 pi))^(1/3) (in per-orbit days)
        # (Seager & Mallen-Ornelas 2003 eq 5)
        # Use the simpler form: a^3 / R*^3 = G * M / (4 pi^2) * P^2 / R*^3
        #                                = (rho * 4/3 pi) * G / (4 pi^2) * P^2
        #                                = G * rho * P^2 / (3 pi)
        G = 6.674e-11  # m^3 kg^-1 s^-2
        P_s = P_d * 86400.0
        a_over_Rs = (G * rho_star_kg_m3 * P_s ** 2 / (3.0 * np.pi)) ** (1.0 / 3.0)
        if a_over_Rs <= 0:
            return 0.0
        arg = ((1.0 + rr) ** 2 - b ** 2) / (a_over_Rs ** 2)
        if arg <= 0:
            return 0.0
        return (P_d / np.pi) * float(np.arcsin(np.sqrt(arg)))

    def _aggregate(values, w):
        if values.size == 0:
            return 0.0
        if w == "sum":
            return float(values.sum())
        if w == "median":
            return float(np.median(values)) * values.size
        if w == "trimmed_mean":
            n = values.size
            n_trim = int(round(trim_fraction * n))
            if n_trim * 2 >= n:
                # Edge case: too few cycles to trim, fall back to median
                return float(np.median(values)) * n
            sorted_vals = np.sort(values)
            kept = sorted_vals[n_trim : n - n_trim]
            return float(np.mean(kept)) * n
        raise ValueError(f"unknown aggregator {w!r}")

    def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
        # 1. Build the per-planet light curves and the full model.
        full_model = full_model_buf
        full_model[:] = 1.0
        active_indices = []
        for k in range(n_planets):
            if not gamma[k]:
                continue
            params = dict(
                log_period=float(beta[4 * k]),
                t0_phase=float(beta[4 * k + 1]),
                log_rr=float(beta[4 * k + 2]),
                b=float(beta[4 * k + 3]),
                u1=None, u2=None,  # filled in by caller via closure
            )
            try:
                evaluate_planet(planet_buf_per_k[k], k, params)
            except Exception:
                return float("-inf")
            full_model += planet_buf_per_k[k] - 1.0
            active_indices.append(k)

        if not active_indices:
            # Pure noise vs flat baseline -- standard chi-squared
            resid = flux - full_model
            return float(-0.5 * np.dot(resid * resid, inv_var))

        # 2. Standard residual chi-squared for the BASELINE
        #    (model with no planets active = constant baseline).
        baseline_resid_sq = (flux - 1.0) ** 2
        baseline_chi2 = float(np.dot(baseline_resid_sq, inv_var))

        # 3. Per-candidate per-cycle Δchi-squared aggregation.
        total_score = 0.0
        for k in active_indices:
            P = float(np.exp(beta[4 * k]))
            T0 = float(beta[4 * k + 1]) * P
            rr = float(np.exp(beta[4 * k + 2]))
            b_imp = float(beta[4 * k + 3])
            T14 = _t14_d(P, rr, b_imp)
            if T14 <= 0:
                continue
            half_window = cycle_window_T14_factor * T14

            n_first = int(np.ceil((time.min() - T0) / P))
            n_last = int(np.floor((time.max() - T0) / P))
            if n_last < n_first:
                continue

            # m_with_k = m, m_no_k = m - delta_k where delta_k = planet_k - 1.
            # chi2_with = (f - m)^2 / sigma^2
            # chi2_no   = (f - m + delta_k)^2 / sigma^2
            # Δchi2 = chi2_no - chi2_with = (2 (f - m) delta_k + delta_k^2) / sigma^2
            # If Δchi2 > 0 within a cycle window, "with k" beats "no k" -- k helps.
            full_model_resid = flux - full_model  # data - m_with_k
            planet_dip = planet_buf_per_k[k] - 1.0  # delta_k = planet_k - 1

            cycle_centers = T0 + np.arange(n_first, n_last + 1) * P
            cycle_chi2_gains = []
            for t_n in cycle_centers:
                mask = (time >= t_n - half_window) & (time <= t_n + half_window)
                if not mask.any():
                    continue
                inv_var_n = inv_var[mask]
                resid_n = full_model_resid[mask]
                dip_n = planet_dip[mask]
                gain = float(np.sum(
                    (2.0 * resid_n * dip_n + dip_n * dip_n) * inv_var_n
                ))
                cycle_chi2_gains.append(gain)

            if not cycle_chi2_gains:
                continue
            score_k = _aggregate(np.asarray(cycle_chi2_gains), aggregator)
            total_score += score_k

        # 4. Total log L = -0.5 * (baseline_chi2 - total_score)
        #    Equivalent to: -0.5 * baseline_chi2 + 0.5 * total_score
        #    where total_score is the (robust) chi-squared improvement
        #    by including the active candidates relative to flat baseline.
        return -0.5 * baseline_chi2 + 0.5 * total_score

    return loglike
