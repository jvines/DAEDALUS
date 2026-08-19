"""Multi-line spectroscopy benchmark.

Synthesises a noisy emission spectrum on a fixed wavelength grid where K
candidate Gaussian lines sit at known centers; for each candidate the
amplitude and width are unknown and toggleable -- gamma_k = 1 means the
line is present, gamma_k = 0 collapses its amplitude and width to the
off-values (zero amplitude effectively removes the line). The continuous
prior on amplitude and width is uniform on a user-specified box.

This is the smallest realistic astrophysical demo for trans-dim NS:
the chain delivers per-line posterior inclusion probabilities and a
calibrated log-evidence that integrates over which subset of candidate
lines is supported by the data. Friendly to the default uniform-u
birth (the line's center is fixed, so we don't need a periodogram-
informed proposal as we would for unknown periods).

The benchmark generator places `n_true` of the `n_candidates` lines at
nontrivial amplitudes and synthesises Gaussian-noise data; the chain
should recover those `n_true` lines as `P(gamma=1) ~ 1` while keeping
`P(gamma=1) <~ 0.5` for the absent lines.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class SpectroscopyProblem:
    loglike: Callable[[np.ndarray, np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    groups_kwargs: list[dict]
    wavelengths: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    line_centers: np.ndarray
    true_active: np.ndarray  # boolean (n_candidates,) -- which lines were truly there
    true_amplitudes: np.ndarray
    true_widths: np.ndarray


def make_problem(
    n_candidates: int = 5,
    n_true: int = 3,
    n_wavelengths: int = 200,
    wavelength_range: tuple[float, float] = (4000.0, 7000.0),
    noise_sigma: float = 0.1,
    amplitude_range: tuple[float, float] = (0.5, 5.0),
    width_range: tuple[float, float] = (5.0, 30.0),
    inclusion_prior: float = 0.5,
    seed: int = 0,
) -> SpectroscopyProblem:
    """Generate a synthetic emission spectrum + a daedalus-ready problem.

    Args:
        n_candidates: total number of toggleable line slots.
        n_true: how many of those slots actually carry signal in the data.
        n_wavelengths: spectral grid size.
        wavelength_range: (lambda_min, lambda_max) for the spectrum.
        noise_sigma: 1-sigma Gaussian noise per spectral bin.
        amplitude_range: uniform prior bounds on each line amplitude.
        width_range: uniform prior bounds on each line Gaussian width (Å).
        inclusion_prior: P(gamma_k = 1) per line.
        seed: RNG seed for the synthetic data + true line parameters.
    """
    if n_true > n_candidates:
        raise ValueError(f"n_true={n_true} > n_candidates={n_candidates}")
    rng = np.random.default_rng(seed)

    lam_min, lam_max = wavelength_range
    wavelengths = np.linspace(lam_min, lam_max, n_wavelengths)
    # Evenly-spaced candidate centers, slightly inset from the edges.
    pad = 0.1 * (lam_max - lam_min)
    line_centers = np.linspace(
        lam_min + pad, lam_max - pad, n_candidates
    )

    # Synthesise truth.
    true_active = np.zeros(n_candidates, dtype=bool)
    true_active[rng.choice(n_candidates, size=n_true, replace=False)] = True
    a_lo, a_hi = amplitude_range
    w_lo, w_hi = width_range
    true_amplitudes = np.where(
        true_active, rng.uniform(a_lo + 0.5, a_hi, size=n_candidates), 0.0
    )
    true_widths = np.where(
        true_active, rng.uniform(w_lo, w_hi, size=n_candidates), 0.5 * (w_lo + w_hi)
    )

    def model(amps: np.ndarray, widths: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.zeros_like(wavelengths)
        for k in range(n_candidates):
            if not mask[k]:
                continue
            out += amps[k] * np.exp(
                -0.5 * ((wavelengths - line_centers[k]) / widths[k]) ** 2
            )
        return out

    truth = model(true_amplitudes, true_widths, true_active)
    flux = truth + noise_sigma * rng.standard_normal(n_wavelengths)
    flux_err = np.full(n_wavelengths, noise_sigma)

    # Each candidate has 2 continuous parameters: amplitude (param 2k),
    # width (param 2k+1). Off-values: amplitude = 0 (line vanishes),
    # width = midpoint of width range (placeholder, doesn't enter L when
    # amplitude is 0).
    ndim = 2 * n_candidates
    width_mid = 0.5 * (w_lo + w_hi)

    def prior_transform(u: np.ndarray) -> np.ndarray:
        v = np.empty_like(u)
        for k in range(n_candidates):
            v[2 * k] = a_lo + (a_hi - a_lo) * u[2 * k]
            v[2 * k + 1] = w_lo + (w_hi - w_lo) * u[2 * k + 1]
        return v

    inv_var = 1.0 / (flux_err * flux_err)

    def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
        amps = beta[0::2]
        widths = beta[1::2]
        modeled = model(amps, widths, gamma)
        resid = flux - modeled
        return float(-0.5 * np.dot(resid * resid, inv_var))

    groups_kwargs = [
        dict(
            name=f"line_{k}_at_{int(line_centers[k])}",
            params=[2 * k, 2 * k + 1],
            off_values=np.array([0.0, width_mid]),
            inclusion_prior=inclusion_prior,
        )
        for k in range(n_candidates)
    ]

    return SpectroscopyProblem(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=ndim,
        groups_kwargs=groups_kwargs,
        wavelengths=wavelengths,
        flux=flux,
        flux_err=flux_err,
        line_centers=line_centers,
        true_active=true_active,
        true_amplitudes=true_amplitudes,
        true_widths=true_widths,
    )
