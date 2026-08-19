"""Stellar flare counting on a real TESS light curve.

A flaring star's light curve has a smoothly varying baseline (rotational
modulation, granulation) plus discrete flare events: rapid rise + slow
exponential decay, each with peak time, peak amplitude, and decay
timescale. The science question -- "how many *real* flares are in this
light curve, vs noise spikes?" -- is exactly the variable-selection
problem MoMS-NS handles.

Implementation:
  - Bundled data: TESS Sector 1 SPOC PDCSAP for AU Mic, a young flaring
    M dwarf with multiple ~1-5% amplitude events per day.
  - Baseline detrending: rolling-median (window picked to preserve
    flares but kill rotational variability).
  - Candidate flare peaks: pre-detected via threshold (>3 sigma above
    detrended baseline, with minimum time spacing) -- drives the slot
    structure.
  - Per-candidate parameters: log peak amplitude, log decay timescale.
    Peak time is fixed at the detected center (no within-flare slack);
    the toggle decides whether the candidate is real signal or noise.
  - Likelihood: Gaussian chi^2 on detrended flux with per-pixel
    errors.

This is a confirmatory variant: pre-detection identifies candidates,
trans-dim NS does the model selection. Default uniform-u birth is
sufficient because each candidate's peak time is already pinned.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from ..groups import Group


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_LC = DATA_DIR / "au_mic_tess_s1.npz"


def load_lightcurve(path: Path | str = DEFAULT_LC) -> dict:
    arch = np.load(path, allow_pickle=False)
    return {
        "time": np.asarray(arch["time"], dtype=np.float64),
        "flux": np.asarray(arch["flux"], dtype=np.float64),
        "flux_err": np.asarray(arch["flux_err"], dtype=np.float64),
    }


def detrend_lc(
    flux: np.ndarray, baseline_window: int = 121
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract a rolling-median baseline. Returns (baseline, residual)."""
    baseline = median_filter(flux, size=baseline_window, mode="nearest")
    return baseline, flux - baseline


def find_flare_peaks(
    time: np.ndarray,
    residual: np.ndarray,
    flux_err: np.ndarray,
    snr_threshold: float = 4.0,
    min_spacing_days: float = 0.05,
    n_peaks_max: int = 30,
) -> list[float]:
    """Pre-detect candidate flare peaks: above-threshold residual maxima
    with minimum time spacing.
    """
    # Sigma per cadence -- use the median photon-noise scale.
    sigma = float(np.median(flux_err))
    threshold = snr_threshold * sigma
    is_local_max = (
        (residual[1:-1] > residual[:-2])
        & (residual[1:-1] > residual[2:])
        & (residual[1:-1] > threshold)
    )
    peak_idx = np.where(is_local_max)[0] + 1
    if peak_idx.size == 0:
        return []
    # Greedy pick by amplitude with min spacing.
    order = np.argsort(residual[peak_idx])[::-1]
    chosen: list[int] = []
    for j in order:
        i = int(peak_idx[j])
        t_i = time[i]
        if any(abs(t_i - time[c]) < min_spacing_days for c in chosen):
            continue
        chosen.append(i)
        if len(chosen) >= n_peaks_max:
            break
    return [float(time[i]) for i in chosen]


@dataclass
class FlaresProblem:
    loglike: Callable[[np.ndarray, np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    groups: list[Group]
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    baseline: np.ndarray
    residual: np.ndarray
    candidate_peaks: list[float]


def make_problem(
    candidate_peaks: list[float] | None = None,
    n_candidates: int = 20,
    inclusion_prior: float = 0.5,
    log_amp_range: tuple[float, float] = (np.log(1e-4), np.log(0.1)),
    log_tau_range: tuple[float, float] = (np.log(0.005), np.log(0.5)),
    snr_threshold: float = 4.0,
    baseline_window: int = 121,
    lc_path: Path | str = DEFAULT_LC,
) -> FlaresProblem:
    """Build a flare-counting trans-dim NS problem.

    Each candidate slot carries two free parameters: log peak amplitude
    (relative to baseline) and log e-folding decay timescale (days). The
    peak time is fixed at the detected candidate center -- the toggle
    decides whether the candidate flare is supported by the data.

    Args:
        candidate_peaks: list of peak times (BTJD) to use as flare
            candidates. If None, runs `find_flare_peaks` to pick the
            top ``n_candidates`` empirical peaks above ``snr_threshold``.
        n_candidates: how many peaks to keep (only used when
            ``candidate_peaks`` is None).
        inclusion_prior: P(gamma_k = 1) per candidate.
        log_amp_range: log-uniform prior on each flare's peak amplitude.
        log_tau_range: log-uniform prior on each flare's decay
            timescale (days). Defaults cover ~7 minutes to 12 hours.
        snr_threshold: detection threshold for candidate-finding (only
            used when ``candidate_peaks`` is None).
        baseline_window: rolling-median window size (cadences) for the
            baseline-subtraction step.
        lc_path: bundled TESS NPZ.
    """
    lc = load_lightcurve(lc_path)
    time = lc["time"]
    flux = lc["flux"]
    flux_err = lc["flux_err"]

    baseline, residual = detrend_lc(flux, baseline_window=baseline_window)

    if candidate_peaks is None:
        candidate_peaks = find_flare_peaks(
            time, residual, flux_err,
            snr_threshold=snr_threshold,
            n_peaks_max=n_candidates,
        )
    n_flares = len(candidate_peaks)
    peak_times = np.asarray(candidate_peaks, dtype=np.float64)

    inv_var = 1.0 / (flux_err * flux_err)
    la_lo, la_hi = log_amp_range
    lt_lo, lt_hi = log_tau_range
    model_buf = np.empty_like(residual)

    def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
        model_buf[:] = 0.0
        for k in range(n_flares):
            if not gamma[k]:
                continue
            log_amp = beta[2 * k]
            log_tau = beta[2 * k + 1]
            amp = float(np.exp(log_amp))
            tau = float(np.exp(log_tau))
            t_peak = peak_times[k]
            # Exponential decay starting at peak time. Pre-peak: zero.
            after = time >= t_peak
            model_buf[after] += amp * np.exp(-(time[after] - t_peak) / tau)
        resid = residual - model_buf
        return float(-0.5 * np.dot(resid * resid, inv_var))

    def prior_transform(u: np.ndarray) -> np.ndarray:
        v = np.empty_like(u)
        for k in range(n_flares):
            v[2 * k] = la_lo + (la_hi - la_lo) * u[2 * k]
            v[2 * k + 1] = lt_lo + (lt_hi - lt_lo) * u[2 * k + 1]
        return v

    groups = []
    for k, t_peak in enumerate(candidate_peaks):
        params = [2 * k, 2 * k + 1]
        off_values = np.array(
            [
                0.5 * (la_lo + la_hi),
                0.5 * (lt_lo + lt_hi),
            ]
        )
        groups.append(
            Group(
                name=f"flare_{k}_t{t_peak:.3f}",
                params=params,
                off_values=off_values,
                inclusion_prior=inclusion_prior,
            )
        )

    return FlaresProblem(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=2 * n_flares,
        groups=groups,
        time=time,
        flux=flux,
        flux_err=flux_err,
        baseline=baseline,
        residual=residual,
        candidate_peaks=candidate_peaks,
    )
