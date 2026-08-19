"""Asteroseismic peak bagging: trans-dim NS over Lorentzian modes.

The power density spectrum (PDS) of a solar-like oscillator has a noise
background plus a comb of Lorentzian peaks corresponding to p-modes
(pressure-driven standing waves). Each mode has three free parameters:
central frequency, height (amplitude), and width (linewidth). The
science question -- "how many of N candidate modes are statistically
supported by the data?" -- is exactly the variable-selection problem
MoMS-NS is built for.

Likelihood: each PDS bin follows an exponential distribution
(chi^2 with 2 DOF) under the hypothesis ``observed = M_true * x_i``
with ``x_i ~ Exp(1)``. The corresponding log-likelihood is

    log L = sum_i [ -log M_i  -  P_i / M_i ]

where M_i = background_i + sum_k gamma_k * lorentzian_k(freq_i).

Background model: a constant white-noise floor plus a single
Harvey/granulation profile

    B(nu) = W + a / (1 + (nu / nu_g)^c)

with c = 4 fixed (Harvey 1985 standard). White noise W and
granulation amplitude a + characteristic frequency nu_g are always-on.

Bundled data: 4 stitched Kepler short-cadence quarters of KIC 6603624
("Saxo", a solar-like sub-giant in the Lund+ 2017 LEGACY sample),
rebinned to 0.4 uHz/bin in the [1500, 3500] uHz oscillation band.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..groups import Group


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PDS = DATA_DIR / "kic6603624_pds.npz"


def load_pds(path: Path | str = DEFAULT_PDS) -> dict:
    arch = np.load(path, allow_pickle=False)
    return {
        "freq": np.asarray(arch["freq"], dtype=np.float64),
        "power": np.asarray(arch["power"], dtype=np.float64),
        "kic": int(arch["kic"]),
    }


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smoothing (reflective boundaries)."""
    if window <= 1:
        return x.copy()
    pad = window // 2
    xp = np.concatenate([x[pad:0:-1], x, x[-2 : -pad - 2 : -1]])
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(xp, kernel, mode="valid")[: x.size]


def find_pds_peaks(
    freq: np.ndarray,
    power: np.ndarray,
    n_peaks: int = 12,
    smooth_window: int = 5,
    min_freq_spacing: float = 8.0,
    snr_floor: float = 2.0,
) -> list[float]:
    """Empirical peak detection in the PDS.

    Smooths the spectrum, identifies local maxima above ``snr_floor``
    times the global median, then greedy-picks the top ``n_peaks``
    keeping a minimum frequency spacing. Returns the list of peak
    centers in uHz, sorted by power (descending).
    """
    smoothed = _smooth(power, smooth_window)
    median = float(np.median(smoothed))
    is_local_max = (
        (smoothed[1:-1] > smoothed[:-2])
        & (smoothed[1:-1] > smoothed[2:])
        & (smoothed[1:-1] > snr_floor * median)
    )
    peak_idx = np.where(is_local_max)[0] + 1
    if peak_idx.size == 0:
        return []
    peak_powers = smoothed[peak_idx]
    order = np.argsort(peak_powers)[::-1]

    chosen: list[int] = []
    for j in order:
        i = int(peak_idx[j])
        if any(abs(freq[i] - freq[c]) < min_freq_spacing for c in chosen):
            continue
        chosen.append(i)
        if len(chosen) >= n_peaks:
            break
    return [float(freq[i]) for i in chosen]


@dataclass
class AsteroseismicProblem:
    loglike: Callable[[np.ndarray, np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    groups: list[Group]
    freq: np.ndarray
    power: np.ndarray
    candidate_frequencies: list[float]


def make_problem(
    candidate_frequencies: list[float] | None = None,
    n_candidates: int = 10,
    inclusion_prior: float = 0.5,
    freq_window: float = 1.5,
    log_height_range: tuple[float, float] = (np.log(1e-9), np.log(1e-4)),
    log_width_range: tuple[float, float] = (np.log(0.1), np.log(10.0)),
    log_white_range: tuple[float, float] = (np.log(1e-8), np.log(1e-5)),
    log_gran_amp_range: tuple[float, float] = (np.log(1e-9), np.log(1e-4)),
    log_gran_freq_range: tuple[float, float] = (np.log(50.0), np.log(2000.0)),
    pds_path: Path | str = DEFAULT_PDS,
) -> AsteroseismicProblem:
    """Build a confirmatory peak-bagging trans-dim problem.

    Each candidate slot is prior'd on a single mode frequency in a
    narrow ``+/- freq_window`` window. The chain confirms which
    candidates are statistically supported and returns physical
    (height, width) fits for the active modes plus the joint N_modes
    posterior.

    If ``candidate_frequencies`` is None, the function runs
    `find_pds_peaks` on the bundled PDS to seed the top
    ``n_candidates`` peaks empirically (data-driven discovery).
    Otherwise ``n_candidates`` is ignored.

    Layout: per-mode params (log_freq, log_height, log_width) for each
    candidate, plus three always-on background params (log_white,
    log_gran_amp, log_gran_freq). Total ndim = 3 N + 3.
    """
    pds = load_pds(pds_path)
    freq = pds["freq"]
    power = pds["power"]

    if candidate_frequencies is None:
        candidate_frequencies = find_pds_peaks(freq, power, n_peaks=n_candidates)

    n_modes = len(candidate_frequencies)
    log_freq_centers = np.log(candidate_frequencies)
    log_freq_los = log_freq_centers - np.log1p(freq_window / np.array(candidate_frequencies))
    log_freq_his = log_freq_centers + np.log1p(freq_window / np.array(candidate_frequencies))

    lh_lo, lh_hi = log_height_range
    lw_lo, lw_hi = log_width_range
    lwn_lo, lwn_hi = log_white_range
    lga_lo, lga_hi = log_gran_amp_range
    lgf_lo, lgf_hi = log_gran_freq_range

    model_buf = np.empty_like(freq)

    def background_inplace(out: np.ndarray, white: float, a: float, nu_g: float) -> None:
        out[:] = white + a / (1.0 + (freq / nu_g) ** 4)

    def add_lorentzian_inplace(out: np.ndarray, nu: float, h: float, gamma: float) -> None:
        out[:] += h / (1.0 + 4.0 * ((freq - nu) / gamma) ** 2)

    def loglike(beta: np.ndarray, gamma_arr: np.ndarray) -> float:
        # Background params: last three slots
        log_white = beta[3 * n_modes]
        log_gran_amp = beta[3 * n_modes + 1]
        log_gran_freq = beta[3 * n_modes + 2]
        background_inplace(
            model_buf,
            float(np.exp(log_white)),
            float(np.exp(log_gran_amp)),
            float(np.exp(log_gran_freq)),
        )
        for k in range(n_modes):
            if not gamma_arr[k]:
                continue
            log_nu = beta[3 * k]
            log_h = beta[3 * k + 1]
            log_g = beta[3 * k + 2]
            add_lorentzian_inplace(
                model_buf,
                float(np.exp(log_nu)),
                float(np.exp(log_h)),
                float(np.exp(log_g)),
            )
        # Exponential-data log-likelihood
        return float(-np.sum(np.log(model_buf) + power / model_buf))

    def prior_transform(u: np.ndarray) -> np.ndarray:
        v = np.empty_like(u)
        for k in range(n_modes):
            v[3 * k] = log_freq_los[k] + (log_freq_his[k] - log_freq_los[k]) * u[3 * k]
            v[3 * k + 1] = lh_lo + (lh_hi - lh_lo) * u[3 * k + 1]
            v[3 * k + 2] = lw_lo + (lw_hi - lw_lo) * u[3 * k + 2]
        v[3 * n_modes] = lwn_lo + (lwn_hi - lwn_lo) * u[3 * n_modes]
        v[3 * n_modes + 1] = lga_lo + (lga_hi - lga_lo) * u[3 * n_modes + 1]
        v[3 * n_modes + 2] = lgf_lo + (lgf_hi - lgf_lo) * u[3 * n_modes + 2]
        return v

    groups = []
    for k, nu in enumerate(candidate_frequencies):
        params = [3 * k, 3 * k + 1, 3 * k + 2]
        off_values = np.array(
            [
                0.5 * (log_freq_los[k] + log_freq_his[k]),
                0.5 * (lh_lo + lh_hi),
                0.5 * (lw_lo + lw_hi),
            ]
        )
        groups.append(
            Group(
                name=f"mode_{int(round(nu))}uHz",
                params=params,
                off_values=off_values,
                inclusion_prior=inclusion_prior,
            )
        )

    return AsteroseismicProblem(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=3 * n_modes + 3,
        groups=groups,
        freq=freq,
        power=power,
        candidate_frequencies=candidate_frequencies,
    )
