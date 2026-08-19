"""Real-data spectroscopy demo: SDSS star-forming galaxy emission lines.

Loads a bundled SDSS DR17 BOSS spectrum (plate=11341, MJD=58428,
fiberID=136, classified as a STARFORMING galaxy at z=0.038) and sets
up a trans-dim NS problem to determine, from the data alone, which of
a panel of classic optical emission lines are statistically supported.

Each candidate line is parameterised by:
  - amplitude  (peak above continuum, uniform prior on a wide box)
  - width      (Gaussian sigma in observed Angstroms, uniform prior)

Lines that drive star-forming emission (Halpha, Hbeta, [OIII] 4959/5007,
[NII] 6548/6584, [SII] 6716/6731, [OII] 3727) are expected to come out
as P(gamma = 1) ~ 1; AGN/LINER-tracer lines (He II 4686, [O I] 6300)
are expected to be much weaker or absent in this starburst spectrum, so
their inclusion probabilities are diagnostic of the algorithm.

Continuum is removed via a running median + Gaussian smooth before the
fit; only the residual flux feeds the chi-squared likelihood. Per-pixel
errors are taken from the SDSS inverse variance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_SPECTRUM = DATA_DIR / "sdss_galaxy.npz"


# Rest-frame wavelengths in Angstroms (vacuum / NIST values).
# Mix of strong star-forming tracers and AGN/LINER tracers for a
# discriminative trans-dim test.
CANDIDATE_LINES_REST: dict[str, float] = {
    "[OII]3727":   3727.5,   # SF tracer (doublet, treated as single)
    "Hbeta":       4861.3,   # Balmer
    "[OIII]4959":  4958.9,   # SF + AGN
    "[OIII]5007":  5006.8,   # SF + AGN
    "HeII4686":    4685.7,   # AGN-only (should NOT appear in pure SF)
    "[OI]6300":    6300.3,   # LINER tracer (weak in SF)
    "Halpha":      6562.8,   # Balmer (strongest SF line)
    "[NII]6548":   6548.1,   # SF
    "[NII]6584":   6583.4,   # SF
    "[SII]6717":   6716.4,   # SF
    "[SII]6731":   6730.8,   # SF
}

EXPECTED_PRESENT: tuple[str, ...] = (
    "[OII]3727", "Hbeta", "[OIII]4959", "[OIII]5007",
    "Halpha", "[NII]6548", "[NII]6584", "[SII]6717", "[SII]6731",
)
EXPECTED_ABSENT_OR_WEAK: tuple[str, ...] = ("HeII4686", "[OI]6300")


@dataclass
class RealSpectroscopyProblem:
    loglike: Callable[[np.ndarray, np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    groups_kwargs: list[dict]
    wavelengths: np.ndarray            # observed Angstroms
    wavelengths_rest: np.ndarray       # rest-frame Angstroms
    flux: np.ndarray                   # original observed flux
    flux_residual: np.ndarray          # continuum-subtracted
    flux_err: np.ndarray
    line_centers_observed: np.ndarray  # observed Angstroms, ordered like CANDIDATE_LINES_REST
    line_names: tuple[str, ...]
    z: float
    spectrum_id: str


def load_spectrum(path: Path | str = DEFAULT_SPECTRUM) -> dict:
    """Load the bundled (or user-supplied) compact SDSS spectrum NPZ."""
    arch = np.load(path, allow_pickle=False)
    out = {
        "loglam": np.asarray(arch["loglam"], dtype=np.float64),
        "flux": np.asarray(arch["flux"], dtype=np.float64),
        "ivar": np.asarray(arch["ivar"], dtype=np.float64),
        "mask": np.asarray(arch["mask"], dtype=np.int64),
        "z": float(arch["z"]),
        "plate": int(arch["plate"]),
        "mjd": int(arch["mjd"]),
        "fiberid": int(arch["fiberid"]),
    }
    return out


def estimate_continuum(
    flux: np.ndarray,
    median_window: int = 51,
    smooth_sigma: float = 5.0,
) -> np.ndarray:
    """Crude continuum: running median followed by a Gaussian smooth.

    The median is robust to narrow emission/absorption peaks, and the
    smooth fills in the residual sawtooth pattern.
    """
    med = median_filter(flux, size=median_window, mode="nearest")
    return gaussian_filter1d(med, sigma=smooth_sigma, mode="nearest")


def make_problem(
    inclusion_prior: float = 0.5,
    amplitude_range: tuple[float, float] = (0.0, 30.0),
    width_range_obs: tuple[float, float] = (1.0, 12.0),
    spectrum_path: Path | str = DEFAULT_SPECTRUM,
) -> RealSpectroscopyProblem:
    """Set up the SDSS trans-dim emission-line problem.

    Args:
        inclusion_prior: P(gamma_k = 1) for each candidate line.
        amplitude_range: uniform prior on each amplitude.
        width_range_obs: uniform prior on each Gaussian width (observed
                         Angstroms; rest-frame width = obs / (1 + z)).
        spectrum_path: NPZ with `loglam`, `flux`, `ivar`, `z`, etc.
    """
    spec = load_spectrum(spectrum_path)
    loglam = spec["loglam"]
    flux = spec["flux"]
    ivar = spec["ivar"]
    z = spec["z"]

    wavelengths = 10.0**loglam
    wavelengths_rest = wavelengths / (1.0 + z)
    flux_err = np.where(ivar > 0, 1.0 / np.sqrt(np.where(ivar > 0, ivar, 1.0)), np.inf)

    continuum = estimate_continuum(flux)
    flux_residual = flux - continuum

    line_names = tuple(CANDIDATE_LINES_REST.keys())
    line_centers_observed = np.array(
        [CANDIDATE_LINES_REST[n] * (1.0 + z) for n in line_names]
    )
    n_candidates = len(line_names)

    # Drop pixels with bad ivar (inf err) from the chi-squared so
    # masked/sky-subtracted regions don't dominate the likelihood.
    good = np.isfinite(flux_err) & (flux_err > 0)
    inv_var = np.where(good, 1.0 / (flux_err * flux_err), 0.0)

    # Restrict the model evaluation to the line region of each candidate
    # (a narrow window around the observed center), so the loglike doesn't
    # waste time on flat continuum stretches. Window: ±5 sigma of the
    # widest possible Gaussian.
    window_half = 5.0 * width_range_obs[1]
    line_pixel_ranges: list[tuple[int, int]] = []
    for c in line_centers_observed:
        lo = int(np.searchsorted(wavelengths, c - window_half))
        hi = int(np.searchsorted(wavelengths, c + window_half))
        line_pixel_ranges.append((lo, hi))

    a_lo, a_hi = amplitude_range
    w_lo, w_hi = width_range_obs
    width_mid = 0.5 * (w_lo + w_hi)

    def prior_transform(u: np.ndarray) -> np.ndarray:
        v = np.empty_like(u)
        for k in range(n_candidates):
            v[2 * k] = a_lo + (a_hi - a_lo) * u[2 * k]
            v[2 * k + 1] = w_lo + (w_hi - w_lo) * u[2 * k + 1]
        return v

    # Pre-allocate a single buffer for the modeled spectrum and reuse it
    # across calls; zeros_like inside loglike was a per-call allocation
    # hot-spot in the profile.
    modeled_buf = np.zeros_like(flux_residual)

    def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
        modeled_buf[:] = 0.0
        for k in range(n_candidates):
            if not gamma[k]:
                continue
            amp = beta[2 * k]
            sigma = beta[2 * k + 1]
            lo, hi = line_pixel_ranges[k]
            x = wavelengths[lo:hi] - line_centers_observed[k]
            modeled_buf[lo:hi] += amp * np.exp(-0.5 * (x / sigma) ** 2)
        resid = flux_residual - modeled_buf
        return float(-0.5 * np.dot(resid * resid, inv_var))

    groups_kwargs = [
        dict(
            name=name,
            params=[2 * k, 2 * k + 1],
            off_values=np.array([0.0, width_mid]),
            inclusion_prior=inclusion_prior,
        )
        for k, name in enumerate(line_names)
    ]

    spectrum_id = f"SDSS p{spec['plate']:04d}-m{spec['mjd']:05d}-f{spec['fiberid']:04d}"

    return RealSpectroscopyProblem(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=2 * n_candidates,
        groups_kwargs=groups_kwargs,
        wavelengths=wavelengths,
        wavelengths_rest=wavelengths_rest,
        flux=flux,
        flux_residual=flux_residual,
        flux_err=flux_err,
        line_centers_observed=line_centers_observed,
        line_names=line_names,
        z=z,
        spectrum_id=spectrum_id,
    )
