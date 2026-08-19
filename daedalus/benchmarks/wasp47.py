"""WASP-47 trans-dim transit recovery demo (TESS Sector 42).

Real-data astrophysical demo on a TESS short-cadence light curve of
the WASP-47 multi-planet system. Two operating modes are supported,
each headlined by a different scientific question:

1. **Tight-prior mode (`make_problem_tight`)** -- confirmatory pattern.
   Three candidate slots, each prior'd on the literature period of one
   known planet (b: 4.16 d, d: 9.03 d, e: 0.79 d) with a narrow window.
   The chain confirms or rejects each candidate via P(gamma_k = 1 | data);
   the headline output is the N_p posterior P(N_planets = 0, 1, 2, 3).
   Default uniform-u birth (== PriorBirth) is sufficient because the
   prior is already concentrated on the right answer.

2. **Search mode (`make_problem`)** -- N candidate slots with a wide
   shared log-period prior; the chain has to find the periods from
   the data alone. Comes with a custom `BLSPeriodBirth` (an
   InformedBirth specialised to BLS for transits): iterative-residual
   Box Least Squares periodogram + uniform-mixture proposal + cache
   by gamma vector.

Each candidate has parameters ``(log P, T0_phase, log depth,
log duration)`` modelled as a box transit. A future revision will
swap the box for a Mandel-Agol model with limb darkening to bring
the depth fits in line with literature radii.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..birth_proposals import (
    BLSPeriodBirth,
    _build_period_cdf,
    _PeriodCDF,
)
from ..groups import Group


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_LC = DATA_DIR / "wasp47_tess_s42.npz"
MULTISECTOR_LC = DATA_DIR / "wasp47_tess_multisector.npz"


KNOWN_PLANETS_DAYS: dict[str, float] = {
    "WASP-47b": 4.1592,
    "WASP-47d": 9.0306,
    "WASP-47e": 0.7896,
}

# WASP-47 host star parameters (Almenara+ 2016, Vanderburg+ 2017).
WASP47_M_SUN = 1.04   # solar masses
WASP47_R_SUN = 1.137  # solar radii

# Gaussian gravitational constant^2: G * M_sun in AU^3 / day^2.
_G_AU3_PER_MSUN_DAY2 = 2.9591220828559115e-4
_R_SUN_IN_AU = 4.6505e-3  # 6.957e8 m / 1.496e11 m


def _a_over_rs(period_days: float, M_msun: float, R_rsun: float) -> float:
    """Scaled semi-major axis a/Rs from Kepler's third law (circular).

    Uses ``a^3 = G M_star P^2 / (4 pi^2)`` with G * M_sun expressed in
    AU^3 / (M_sun day^2). The ``1 / (4 pi^2)`` factor is essential --
    omitting it makes a/Rs (4 pi^2)^(1/3) ~ 3.4x too large, which
    propagates to transits that are ~3.4x too narrow.
    """
    a_au_cubed = (
        _G_AU3_PER_MSUN_DAY2 * M_msun * period_days * period_days
        / (4.0 * np.pi ** 2)
    )
    a_au = a_au_cubed ** (1.0 / 3.0)
    return a_au / (R_rsun * _R_SUN_IN_AU)


def _q_to_u(q1: float, q2: float) -> tuple[float, float]:
    """Kipping (2013) reparametrisation of quadratic limb-darkening:
    q1, q2 in [0, 1] -> u1, u2 in the physically valid wedge.
    """
    sqrt_q1 = float(np.sqrt(max(q1, 0.0)))
    u1 = 2.0 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1.0 - 2.0 * q2)
    return u1, u2


def load_lightcurve(path: Path | str = DEFAULT_LC) -> dict:
    """Load the bundled (or user-supplied) compact TESS light curve NPZ.

    Returns a dict with `time` (BTJD days), `flux` (PDCSAP, normalised
    to median = 1 per sector), and `flux_err` (same units). Optional
    extras (sector indices, original median for the single-sector LC)
    are surfaced when present in the archive.
    """
    arch = np.load(path, allow_pickle=False)
    out: dict[str, object] = {
        "time": np.asarray(arch["time"], dtype=np.float64),
        "flux": np.asarray(arch["flux"], dtype=np.float64),
        "flux_err": np.asarray(arch["flux_err"], dtype=np.float64),
    }
    for optional in ("median", "sector"):
        if optional in arch.files:
            out[optional] = arch[optional]
    return out


# ---------- Transit models ---------------------------------------------------


def box_transit_model(
    t: np.ndarray,
    period: float,
    t0_phase: float,
    depth: float,
    duration: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Single-planet box transit (rectangular dip).

    Approximation, not physics: the depth fit is biased low because real
    transits are U-shaped with limb darkening. Useful only when speed
    matters more than fidelity. Use ``mandel_agol_transit_model`` for
    science.
    """
    if out is None:
        out = np.ones_like(t)
    else:
        out.fill(1.0)
    t0 = t0_phase * period
    half = 0.5 * duration
    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    in_transit = np.abs(phase) < half
    out -= depth * in_transit
    return out


class MandelAgolModel:
    """Vectorised Mandel-Agol transit model wrapped around batman-package.

    Builds one ``batman.TransitModel`` per planet against a fixed time
    grid; each ``light_curve(...)`` call is then ~50us per evaluation
    on a 25k-cadence TESS light curve. The host-star mass and radius
    are needed to convert orbital period into the scaled semi-major
    axis ``a/Rs`` via Kepler's third law, so the user supplies them
    once at construction.

    Limb darkening is shared across all planets (a property of the
    star) and parametrised with the Kipping (2013) ``(q1, q2) in
    [0, 1]^2`` square so a uniform prior covers the physically valid
    wedge.
    """

    def __init__(
        self,
        time: np.ndarray,
        n_planets: int,
        M_msun: float,
        R_rsun: float,
        nominal_periods: tuple[float, ...] | None = None,
        exp_time: float = 0.0,
        supersample_factor: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        exp_time
            Effective exposure time per data point in days. Pass the
            cadence of the binned light curve being modelled (e.g.
            5/60/24 = 0.00347 d for 5-min binned TESS data) so batman
            integrates the transit shape over each exposure rather than
            evaluating instantaneously. Important for short transits
            (USPs) and/or long baselines: an instantaneous model has
            sharper in/egress than the data, and the chi-squared
            minimum can shift to slightly-off periods to compensate.
        supersample_factor
            Integer number of sub-cadence samples used by batman for
            the exposure-time integration. Setting ``exp_time > 0`` and
            ``supersample_factor >= 5`` is the recommended pair for
            5-min binned data; lower supersample saves compute but
            biases in/egress.
        """
        import batman

        self._batman = batman
        self.time = time
        self.n_planets = n_planets
        self.M_msun = M_msun
        self.R_rsun = R_rsun
        self.exp_time = float(exp_time)
        self.supersample_factor = int(supersample_factor)

        if nominal_periods is None:
            nominal_periods = tuple([1.0] * n_planets)

        self._models: list = []
        self._params_template: list = []
        for k in range(n_planets):
            p = batman.TransitParams()
            p.t0 = 0.0
            p.per = float(nominal_periods[k])
            p.rp = 0.05
            p.a = float(_a_over_rs(p.per, M_msun, R_rsun))
            p.inc = 90.0
            p.ecc = 0.0
            p.w = 90.0
            p.u = [0.4, 0.3]
            p.limb_dark = "quadratic"
            self._params_template.append(p)
            self._models.append(batman.TransitModel(
                p, time,
                exp_time=self.exp_time,
                supersample_factor=self.supersample_factor,
            ))

    def evaluate(
        self,
        out: np.ndarray,
        log_periods: np.ndarray,
        t0_phases: np.ndarray,
        log_rrs: np.ndarray,
        impact_params: np.ndarray,
        gamma: np.ndarray,
        u1: float,
        u2: float,
    ) -> None:
        """Compute the multi-planet transit model in place into ``out``.

        For each planet k with ``gamma[k] = True``, build a Mandel-Agol
        contribution and subtract from ``out``. ``out`` is initialised
        to 1.0 here.
        """
        out[:] = 1.0
        for k in range(self.n_planets):
            if not gamma[k]:
                continue
            P = float(np.exp(log_periods[k]))
            t0 = float(t0_phases[k]) * P
            rr = float(np.exp(log_rrs[k]))
            b = float(impact_params[k])
            a_rs = _a_over_rs(P, self.M_msun, self.R_rsun)
            # Convert impact parameter to inclination (degrees).
            cos_i = b / a_rs
            cos_i = float(np.clip(cos_i, -1.0, 1.0))
            inc_deg = float(np.degrees(np.arccos(cos_i)))

            p = self._params_template[k]
            p.t0 = t0
            p.per = P
            p.rp = rr
            p.a = a_rs
            p.inc = inc_deg
            p.u = [u1, u2]

            flux_k = self._models[k].light_curve(p)
            # batman returns the per-planet light curve normalised to 1
            # outside transit; the multi-planet model is the sum of dips.
            out[:] += flux_k - 1.0

    def is_physical(self, log_period: float, log_rr: float, b: float) -> bool:
        """Return False if (P, R_p/R_*, b) describe a non-physical /
        non-transiting configuration:
            - planet inside the star (a/R_* < 1 + R_p/R_*)
            - orbit does not transit (b > 1 + R_p/R_*)
            - impact parameter exceeds orbital radius (b > a/R_*)
        Standard transit-fitting tools (juliet, exoplanet, etc.) reject
        these by returning -inf log L; we expose an explicit check so
        the per-benchmark loglike functions can do the same instead of
        letting batman evaluate at clipped inclination.
        """
        P = float(np.exp(log_period))
        rr = float(np.exp(log_rr))
        a_rs = _a_over_rs(P, self.M_msun, self.R_rsun)
        if a_rs < 1.0 + rr:
            return False
        if b > 1.0 + rr:
            return False
        if b > a_rs:
            return False
        return True

    def evaluate_single(
        self,
        out: np.ndarray,
        k: int,
        log_period: float,
        t0_phase: float,
        log_rr: float,
        b: float,
        u1: float,
        u2: float,
    ) -> None:
        """Single-planet evaluation (used by the residual-subtraction path).

        Callers that need physicality enforcement should call
        :meth:`is_physical` first and treat ``False`` as a -inf log L.
        This method itself silently clips ``b/a_rs`` for numerical
        robustness; the clipping does NOT make a non-physical
        configuration physical, but it lets batman evaluate without
        crashing for chains that happen to propose such states.
        """
        out[:] = 1.0
        P = float(np.exp(log_period))
        t0 = float(t0_phase) * P
        rr = float(np.exp(log_rr))
        a_rs = _a_over_rs(P, self.M_msun, self.R_rsun)
        cos_i = float(np.clip(b / a_rs, -1.0, 1.0))
        inc_deg = float(np.degrees(np.arccos(cos_i)))
        p = self._params_template[k]
        p.t0 = t0
        p.per = P
        p.rp = rr
        p.a = a_rs
        p.inc = inc_deg
        p.u = [u1, u2]
        out[:] = self._models[k].light_curve(p)


# ---------- BLS diagnostic helper --------------------------------------------
#
# The BLS-informed birth proposal itself now lives in
# ``daedalus.birth_proposals`` as the package-level ``BLSPeriodBirth``. This
# module retains a thin ``build_bls_period_cdf`` wrapper purely to populate
# the diagnostic ``WASP47Problem.bls_cdf`` field with a periodogram of the
# raw flux (no residual subtraction; for plotting / sanity checks only).


def build_bls_period_cdf(
    time: np.ndarray,
    flux: np.ndarray,
    period_range: tuple[float, float],
    n_periods: int = 5000,
    duration: float = 0.1,
    sharpening: float = 50.0,
) -> _PeriodCDF:
    """Compute a BLS periodogram and convert to a normalised log-period density.

    Convenience wrapper used to populate :attr:`WASP47Problem.bls_cdf` for
    diagnostic plots; for the actual chain proposal use the package-level
    :class:`daedalus.BLSPeriodBirth` (which evaluates the periodogram on the
    chain residuals, not the raw flux). ``sharpening`` controls how peaked
    the density is around BLS peaks; see
    :func:`daedalus.birth_proposals._build_period_cdf`.
    """
    from astropy import units as u
    from astropy.timeseries import BoxLeastSquares

    bls = BoxLeastSquares(time * u.day, flux)
    log_p_lo = float(np.log(period_range[0]))
    log_p_hi = float(np.log(period_range[1]))
    log_periods = np.linspace(log_p_lo, log_p_hi, n_periods)
    periods = np.exp(log_periods)
    power = np.asarray(bls.power(periods * u.day, duration * u.day).power)
    return _build_period_cdf(log_periods, power, sharpening)


def _planet_active_periods_fn(state) -> list[float]:
    """Active periods extractor for the WASP-47 4-param-per-planet layout
    (log P, T0_phase, log depth/log rr, log duration/impact). The period
    sits at index ``4 * k`` of ``state.beta`` for slot ``k``.

    Used as the ``active_periods_fn`` argument to ``BLSPeriodBirth`` when
    harmonic suppression is enabled. Layout-specific to this benchmark;
    that's why it lives here rather than in the package."""
    out: list[float] = []
    for k in range(state.gamma.size):
        if state.gamma[k]:
            out.append(float(np.exp(state.beta[4 * k])))
    return out


# ---------- Problem builder --------------------------------------------------


@dataclass
class WASP47Problem:
    loglike: Callable[[np.ndarray, np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    groups: list[Group]
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    n_planets: int
    log_period_range: tuple[float, float]
    log_depth_range: tuple[float, float]
    log_duration_range: tuple[float, float]
    bls_cdf: _PeriodCDF


def make_problem(
    n_planets: int = 1,
    inclusion_prior: float = 0.5,
    log_period_range: tuple[float, float] | None = None,
    log_depth_range: tuple[float, float] | None = None,
    log_duration_range: tuple[float, float] | None = None,
    use_bls_birth: bool = True,
    bls_n_periods: int = 5000,
    bls_sharpening: float = 50.0,
    bls_harmonic_suppression: bool = False,
    model: str = "box",
    log_rr_range: tuple[float, float] | None = None,
    impact_param_range: tuple[float, float] = (0.0, 1.0),
    M_msun: float = WASP47_M_SUN,
    R_rsun: float = WASP47_R_SUN,
    lc_path: Path | str = DEFAULT_LC,
) -> WASP47Problem:
    """Build a 1-to-N planet trans-dim problem for the WASP-47 light curve.

    Args:
        n_planets: number of toggleable planet candidates.
        inclusion_prior: P(gamma_k = 1) per planet candidate.
        log_period_range: log-uniform prior on each planet's period (days).
            Default: log of (0.5, 12.0) so all three known WASP-47 transits
            sit inside the prior.
        log_depth_range: log-uniform prior on each planet's transit depth.
        log_duration_range: log-uniform prior on each planet's transit duration.
        use_bls_birth: if True, attach BLSPeriodBirth to every candidate.
            If False, default uniform-u birth (slow but unbiased).
        bls_n_periods: BLS periodogram resolution for the birth proposal.
        bls_sharpening: peakedness of the BLS-informed proposal density.
        lc_path: path to the bundled light-curve NPZ.
    """
    lc = load_lightcurve(lc_path)
    time = lc["time"]
    flux = lc["flux"]
    flux_err = lc["flux_err"]

    if log_period_range is None:
        log_period_range = (float(np.log(0.5)), float(np.log(12.0)))
    if log_depth_range is None:
        log_depth_range = (float(np.log(1e-5)), float(np.log(0.05)))
    if log_duration_range is None:
        log_duration_range = (float(np.log(0.01)), float(np.log(0.5)))

    inv_var = 1.0 / (flux_err * flux_err)
    span = float(time.max() - time.min())

    if model not in ("box", "mandel_agol"):
        raise ValueError(f"model must be 'box' or 'mandel_agol', got {model!r}")
    use_ma = model == "mandel_agol"

    if log_rr_range is None:
        log_rr_range = (float(np.log(0.005)), float(np.log(0.2)))

    lp_lo, lp_hi = log_period_range
    ld_lo, ld_hi = log_depth_range
    ldur_lo, ldur_hi = log_duration_range
    lrr_lo, lrr_hi = log_rr_range
    b_lo, b_hi = impact_param_range

    # Pre-allocated buffers reused across loglike / residual_fn calls.
    model_buf = np.empty_like(flux)
    res_buf = np.empty_like(flux)
    planet_buf = np.empty_like(flux)

    if use_ma:
        # Build one batman TransitModel per planet against this time grid.
        # Nominal periods: spread across the prior range so each model
        # has a sensible supersampling default for short-period planets.
        nominal_periods = tuple(
            float(np.exp(lp_lo + (lp_hi - lp_lo) * (k + 0.5) / n_planets))
            for k in range(n_planets)
        )
        ma_model = MandelAgolModel(
            time=time,
            n_planets=n_planets,
            M_msun=M_msun,
            R_rsun=R_rsun,
            nominal_periods=nominal_periods,
        )

        def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
            model_buf[:] = 1.0
            q1 = float(beta[4 * n_planets])
            q2 = float(beta[4 * n_planets + 1])
            u1, u2 = _q_to_u(q1, q2)
            for k in range(n_planets):
                if not gamma[k]:
                    continue
                lp = float(beta[4 * k])
                lrr = float(beta[4 * k + 2])
                bk = float(beta[4 * k + 3])
                if not ma_model.is_physical(lp, lrr, bk):
                    return float("-inf")
                ma_model.evaluate_single(
                    planet_buf, k,
                    log_period=lp,
                    t0_phase=float(beta[4 * k + 1]),
                    log_rr=lrr,
                    b=bk,
                    u1=u1, u2=u2,
                )
                model_buf[:] += planet_buf - 1.0
            resid = flux - model_buf
            return float(-0.5 * np.dot(resid * resid, inv_var))

        def prior_transform(u: np.ndarray) -> np.ndarray:
            v = np.empty_like(u)
            for k in range(n_planets):
                v[4 * k] = lp_lo + (lp_hi - lp_lo) * u[4 * k]
                v[4 * k + 1] = u[4 * k + 1]
                v[4 * k + 2] = lrr_lo + (lrr_hi - lrr_lo) * u[4 * k + 2]
                v[4 * k + 3] = b_lo + (b_hi - b_lo) * u[4 * k + 3]
            v[4 * n_planets] = u[4 * n_planets]      # q1
            v[4 * n_planets + 1] = u[4 * n_planets + 1]  # q2
            return v

        log_pi_beta_const = (
            -float(np.log(lp_hi - lp_lo))
            - float(np.log(1.0))
            - float(np.log(lrr_hi - lrr_lo))
            - float(np.log(b_hi - b_lo))
        )

        def log_prior_continuous(b_grp: np.ndarray) -> float:
            if (
                b_grp[0] < lp_lo or b_grp[0] > lp_hi
                or b_grp[1] < 0.0 or b_grp[1] > 1.0
                or b_grp[2] < lrr_lo or b_grp[2] > lrr_hi
                or b_grp[3] < b_lo or b_grp[3] > b_hi
            ):
                return -np.inf
            return log_pi_beta_const

        def compute_residuals(state) -> np.ndarray:
            res_buf[:] = flux
            beta = state.beta
            gamma = state.gamma
            q1 = float(beta[4 * n_planets])
            q2 = float(beta[4 * n_planets + 1])
            u1, u2 = _q_to_u(q1, q2)
            for k in range(n_planets):
                if not gamma[k]:
                    continue
                ma_model.evaluate_single(
                    planet_buf, k,
                    log_period=float(beta[4 * k]),
                    t0_phase=float(beta[4 * k + 1]),
                    log_rr=float(beta[4 * k + 2]),
                    b=float(beta[4 * k + 3]),
                    u1=u1, u2=u2,
                )
                # planet_buf is the per-planet flux (1 outside, < 1 inside);
                # add back the dip = subtract the planet from the data.
                res_buf[:] += 1.0 - planet_buf
            return res_buf

        # Per-slot off-values + indices for groups (4 per planet);
        # q1, q2 are always-on so they don't belong to any group.
        slot_off = lambda k: np.array(
            [
                0.5 * (lp_lo + lp_hi),
                0.0,
                0.5 * (lrr_lo + lrr_hi),
                0.5 * (b_lo + b_hi),
            ]
        )
    else:
        def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
            model_buf[:] = 1.0
            for k in range(n_planets):
                if not gamma[k]:
                    continue
                log_p = beta[4 * k]
                t0_phase = beta[4 * k + 1]
                log_d = beta[4 * k + 2]
                log_dur = beta[4 * k + 3]
                P = float(np.exp(log_p))
                depth = float(np.exp(log_d))
                duration = float(np.exp(log_dur))
                if duration >= 0.9 * P:
                    return float("-inf")
                t0 = t0_phase * P
                half = 0.5 * duration
                phase = ((time - t0 + 0.5 * P) % P) - 0.5 * P
                in_transit = np.abs(phase) < half
                model_buf[:] -= depth * in_transit
            resid = flux - model_buf
            return float(-0.5 * np.dot(resid * resid, inv_var))

        def prior_transform(u: np.ndarray) -> np.ndarray:
            v = np.empty_like(u)
            for k in range(n_planets):
                v[4 * k] = lp_lo + (lp_hi - lp_lo) * u[4 * k]
                v[4 * k + 1] = u[4 * k + 1]
                v[4 * k + 2] = ld_lo + (ld_hi - ld_lo) * u[4 * k + 2]
                v[4 * k + 3] = ldur_lo + (ldur_hi - ldur_lo) * u[4 * k + 3]
            return v

        log_pi_beta_const = (
            -float(np.log(lp_hi - lp_lo))
            - float(np.log(1.0))
            - float(np.log(ld_hi - ld_lo))
            - float(np.log(ldur_hi - ldur_lo))
        )

        def log_prior_continuous(b_grp: np.ndarray) -> float:
            if (
                b_grp[0] < lp_lo or b_grp[0] > lp_hi
                or b_grp[1] < 0.0 or b_grp[1] > 1.0
                or b_grp[2] < ld_lo or b_grp[2] > ld_hi
                or b_grp[3] < ldur_lo or b_grp[3] > ldur_hi
            ):
                return -np.inf
            return log_pi_beta_const

        def compute_residuals(state) -> np.ndarray:
            res_buf[:] = flux
            beta = state.beta
            gamma = state.gamma
            for k in range(n_planets):
                if not gamma[k]:
                    continue
                log_p = beta[4 * k]
                t0_phase = beta[4 * k + 1]
                log_d = beta[4 * k + 2]
                log_dur = beta[4 * k + 3]
                P = float(np.exp(log_p))
                depth = float(np.exp(log_d))
                duration = float(np.exp(log_dur))
                if duration >= 0.9 * P:
                    continue
                t0 = t0_phase * P
                half = 0.5 * duration
                phase = ((time - t0 + 0.5 * P) % P) - 0.5 * P
                in_transit = np.abs(phase) < half
                res_buf[:] += depth * in_transit
            return res_buf

        slot_off = lambda k: np.array(
            [
                0.5 * (lp_lo + lp_hi),
                0.0,
                0.5 * (ld_lo + ld_hi),
                0.5 * (ldur_lo + ldur_hi),
            ]
        )

    groups = []
    for k in range(n_planets):
        params = [4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3]
        off_values = slot_off(k)
        if use_bls_birth:
            if use_ma:
                non_period_priors = (
                    (0.0, 1.0),                     # T0 phase
                    log_rr_range,                   # log radius ratio
                    impact_param_range,             # impact parameter
                )
            else:
                non_period_priors = (
                    (0.0, 1.0),                     # T0 phase
                    log_depth_range,                # log depth
                    log_duration_range,             # log duration
                )
            birth = BLSPeriodBirth(
                time=time,
                residual_fn=compute_residuals,
                prior_log_period=log_period_range,
                non_period_priors=non_period_priors,
                n_periods=bls_n_periods,
                sharpening=bls_sharpening,
                # Harmonic suppression needs to know the active slots'
                # periods; the WASP-47 4-param layout puts log P at
                # ``state.beta[4*k]`` for each planet slot. Wired only
                # when the user opts in via ``bls_harmonic_suppression``.
                active_periods_fn=(
                    _planet_active_periods_fn
                    if bls_harmonic_suppression
                    else None
                ),
                # Relaxed cache invalidation: 1e-2 dex of log P (~2.3 %
                # period). Tight enough to invalidate on real period
                # convergence (uniform draw -> data-pulled period), loose
                # enough that the chain doesn't recompute the BLS
                # periodogram on every chain step within a converged
                # mode. Default 1e-3 was found to make the chain run
                # ~10x slower without any benefit on this problem.
                active_period_cache_precision_dex=1e-2,
            )
            groups.append(
                Group(
                    name=f"planet_{k}",
                    params=params,
                    off_values=off_values,
                    inclusion_prior=inclusion_prior,
                    birth_proposal=birth,
                    log_prior_continuous=log_prior_continuous,
                )
            )
        else:
            groups.append(
                Group(
                    name=f"planet_{k}",
                    params=params,
                    off_values=off_values,
                    inclusion_prior=inclusion_prior,
                )
            )

    # Build a snapshot CDF for diagnostics / the test that reads it.
    period_range_phys = (float(np.exp(lp_lo)), float(np.exp(lp_hi)))
    bls_cdf = build_bls_period_cdf(
        time, flux, period_range_phys, n_periods=bls_n_periods,
        sharpening=bls_sharpening,
    )

    return WASP47Problem(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=4 * n_planets + (2 if use_ma else 0),
        groups=groups,
        time=time,
        flux=flux,
        flux_err=flux_err,
        n_planets=n_planets,
        log_period_range=log_period_range,
        log_depth_range=log_depth_range,
        log_duration_range=log_duration_range,
        bls_cdf=bls_cdf,
    )


# ---------- Tight-prior confirmatory setup -----------------------------------


@dataclass
class WASP47TightProblem:
    loglike: Callable[[np.ndarray, np.ndarray], float]
    prior_transform: Callable[[np.ndarray], np.ndarray]
    ndim: int
    groups: list[Group]
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    candidate_names: tuple[str, ...]
    candidate_periods: tuple[float, ...]


# Literature periods + tolerated half-width windows for the tight prior.
TIGHT_PRIOR_WINDOWS: tuple[tuple[str, float, float], ...] = (
    # (planet name, log P width [days], P uncertainty within which the
    # narrow Gaussian-equivalent uniform prior runs)
    ("WASP-47b", KNOWN_PLANETS_DAYS["WASP-47b"], 0.06),
    ("WASP-47d", KNOWN_PLANETS_DAYS["WASP-47d"], 0.10),
    ("WASP-47e", KNOWN_PLANETS_DAYS["WASP-47e"], 0.012),
)


def _bin_lc(
    time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, bin_minutes: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a TESS light curve to ``bin_minutes`` cadence (mean per bin).

    5-min binning brings the chi-squared cost down by ~4x while preserving
    every transit because the shortest transit (WASP-47e) is ~30 min long.
    """
    bin_d = bin_minutes / (60.0 * 24.0)
    bin_id = np.floor((time - time.min()) / bin_d).astype(np.int64)
    ub, idx = np.unique(bin_id, return_inverse=True)
    n_bin = ub.size
    counts = np.bincount(idx, minlength=n_bin)
    t_b = np.bincount(idx, weights=time, minlength=n_bin) / counts
    f_b = np.bincount(idx, weights=flux, minlength=n_bin) / counts
    e_b = np.bincount(idx, weights=flux_err, minlength=n_bin) / (counts * np.sqrt(counts))
    return t_b, f_b, e_b


def make_problem_tight(
    candidates: tuple[tuple[str, float, float], ...] = TIGHT_PRIOR_WINDOWS,
    inclusion_prior: float = 0.5,
    log_depth_range: tuple[float, float] | None = None,
    log_duration_range: tuple[float, float] | None = None,
    log_rr_range: tuple[float, float] | None = None,
    impact_param_range: tuple[float, float] = (0.0, 1.0),
    q1_range: tuple[float, float] = (0.0, 1.0),
    q2_range: tuple[float, float] = (0.0, 1.0),
    M_msun: float = WASP47_M_SUN,
    R_rsun: float = WASP47_R_SUN,
    model: str = "box",
    bin_minutes: float = 5.0,
    fit_jitter: bool = False,
    log_jitter_range: tuple[float, float] | None = None,
    lc_path: Path | str = DEFAULT_LC,
) -> WASP47TightProblem:
    """Confirmatory trans-dim setup for WASP-47.

    Each candidate slot is prior'd on a single known planet:
    ``(name, P_lit, half_width)`` -> uniform period prior on
    ``[P_lit - half_width, P_lit + half_width]`` days. The chain
    confirms or rejects each candidate via P(gamma = 1) and
    delivers the joint N_p posterior.

    Default uniform-u birth (i.e. PriorBirth) is sufficient when the
    prior is already tight on the right answer, so no
    ``birth_proposal`` is attached.

    Args:
        candidates: tuple of (name, period_days, half_width_days) for each
            candidate slot. Defaults to the three known WASP-47 planets.
        inclusion_prior: P(gamma_k = 1) per slot.
        log_depth_range: log-uniform depth prior; defaults to
            (log 1e-5, log 0.05) covering ultra-shallow USP transits.
        log_duration_range: log-uniform duration prior in days.
        q1_range, q2_range: bounds for the Kipping (2013) limb-darkening
            reparametrisation; only used when ``model == 'mandel_agol'``.
            Default (0, 1) is uniform-uninformative; pass narrower
            ranges to inject stellar-physics knowledge (e.g. for a
            Sun-like star in TESS bandpass, q1 ~ 0.35, q2 ~ 0.30).
        bin_minutes: bin the TESS light curve to this cadence (default 5).
        fit_jitter: if True, add a global ``log_jitter`` parameter so the
            effective per-point variance becomes ``flux_err**2 +
            jitter**2`` and the likelihood includes the log-determinant
            term ``sum(log(sigma_eff**2))``. Required whenever the
            published flux_err underestimates the true scatter -- TESS
            PDCSAP errors typically miss systematics, and on multi-sector
            data we measured chi^2/n ~ 2.85 at the joint best-fit
            without jitter, which causes alias-control slots to fire at
            P(gamma = 1) = 1 by absorbing what the model treats as
            statistically significant residual structure.
        log_jitter_range: log-uniform jitter prior; only used when
            ``fit_jitter=True``. Defaults to ``(log 1e-6, log 1e-2)``,
            spanning sub-ppm jitter through ~1% additional scatter.
        lc_path: bundled light curve path.
    """
    q1_lo, q1_hi = q1_range
    q2_lo, q2_hi = q2_range
    if not (0.0 <= q1_lo < q1_hi <= 1.0):
        raise ValueError(f"q1_range must satisfy 0 <= lo < hi <= 1, got {q1_range}")
    if not (0.0 <= q2_lo < q2_hi <= 1.0):
        raise ValueError(f"q2_range must satisfy 0 <= lo < hi <= 1, got {q2_range}")
    if log_depth_range is None:
        log_depth_range = (float(np.log(1e-5)), float(np.log(0.05)))
    if log_duration_range is None:
        log_duration_range = (float(np.log(0.01)), float(np.log(0.5)))
    if log_rr_range is None:
        # Floor at 1e-6 (1 ppm transit equivalent) so an active slot whose
        # candidate period has no real signal can drive its depth to
        # effectively zero. The previous 0.005 floor forced every gamma=1
        # slot to contribute at least a 25 ppm dip per transit, which the
        # joint fit on multi-sector WASP-47 exploited by fragmenting b's
        # transit signature across harmonic-alias slots; with the floor
        # near zero, alias slots collapse to non-contributing and the
        # Bayes factor for their inclusion drops by the prior-volume
        # Occam factor, matching juliet's fixed-dim depth-near-zero
        # rejection behaviour for unsupported candidates.
        log_rr_range = (float(np.log(1e-6)), float(np.log(0.2)))
    if log_jitter_range is None:
        log_jitter_range = (float(np.log(1e-6)), float(np.log(1e-2)))
    ljit_lo, ljit_hi = log_jitter_range

    if model not in ("box", "mandel_agol"):
        raise ValueError(f"model must be 'box' or 'mandel_agol', got {model!r}")
    use_ma = model == "mandel_agol"

    lc = load_lightcurve(lc_path)
    if bin_minutes > 0.0:
        time, flux, flux_err = _bin_lc(lc["time"], lc["flux"], lc["flux_err"], bin_minutes)
    else:
        time, flux, flux_err = lc["time"], lc["flux"], lc["flux_err"]

    inv_var = 1.0 / (flux_err * flux_err)
    var_data = flux_err * flux_err
    n_planets = len(candidates)

    log_p_los = np.array(
        [float(np.log(c[1] - c[2])) for c in candidates], dtype=np.float64
    )
    log_p_his = np.array(
        [float(np.log(c[1] + c[2])) for c in candidates], dtype=np.float64
    )
    ld_lo, ld_hi = log_depth_range
    ldur_lo, ldur_hi = log_duration_range
    lrr_lo, lrr_hi = log_rr_range
    b_lo, b_hi = impact_param_range
    # Per-candidate log_rr bounds. Each candidate may optionally carry a
    # 4th element giving its own (rr_lo, rr_hi) tuple (in linear, not
    # log, units). Tightening per-planet depth priors breaks the
    # (P, depth) ridge that lets the chain settle on wrong-period
    # configurations by reducing the recovered depth.
    lrr_los = np.empty(n_planets, dtype=np.float64)
    lrr_his = np.empty(n_planets, dtype=np.float64)
    for k, c in enumerate(candidates):
        if len(c) >= 4 and c[3] is not None:
            rr_lo, rr_hi = c[3]
            lrr_los[k] = float(np.log(rr_lo))
            lrr_his[k] = float(np.log(rr_hi))
        else:
            lrr_los[k] = lrr_lo
            lrr_his[k] = lrr_hi

    model_buf = np.empty_like(flux)
    planet_buf = np.empty_like(flux)

    if use_ma:
        nominal_periods = tuple(c[1] for c in candidates)
        # When binning the LC, set batman's exposure-time integration
        # to match. Otherwise the model is evaluated instantaneously
        # while the data is averaged over each bin, biasing in/egress
        # and pulling the chi-squared minimum off the true period.
        ma_exp_time = (bin_minutes / (60.0 * 24.0)) if bin_minutes > 0 else 0.0
        ma_supersample = 7 if bin_minutes > 0 else 1
        ma_model = MandelAgolModel(
            time=time, n_planets=n_planets,
            M_msun=M_msun, R_rsun=R_rsun,
            nominal_periods=nominal_periods,
            exp_time=ma_exp_time,
            supersample_factor=ma_supersample,
        )

        if fit_jitter:
            jitter_idx = 4 * n_planets + 2

            def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
                model_buf[:] = 1.0
                q1 = float(beta[4 * n_planets])
                q2 = float(beta[4 * n_planets + 1])
                u1, u2 = _q_to_u(q1, q2)
                for k in range(n_planets):
                    if not gamma[k]:
                        continue
                    lp = float(beta[4 * k])
                    lrr = float(beta[4 * k + 2])
                    bk = float(beta[4 * k + 3])
                    if not ma_model.is_physical(lp, lrr, bk):
                        return float("-inf")
                    ma_model.evaluate_single(
                        planet_buf, k,
                        log_period=lp,
                        t0_phase=float(beta[4 * k + 1]),
                        log_rr=lrr,
                        b=bk,
                        u1=u1, u2=u2,
                    )
                    model_buf[:] += planet_buf - 1.0
                resid = flux - model_buf
                jit2 = float(np.exp(2.0 * float(beta[jitter_idx])))
                sigma2_eff = var_data + jit2
                return float(
                    -0.5 * np.sum(resid * resid / sigma2_eff + np.log(sigma2_eff))
                )

            def prior_transform(u: np.ndarray) -> np.ndarray:
                v = np.empty_like(u)
                for k in range(n_planets):
                    v[4 * k] = log_p_los[k] + (log_p_his[k] - log_p_los[k]) * u[4 * k]
                    v[4 * k + 1] = u[4 * k + 1]
                    v[4 * k + 2] = lrr_los[k] + (lrr_his[k] - lrr_los[k]) * u[4 * k + 2]
                    v[4 * k + 3] = b_lo + (b_hi - b_lo) * u[4 * k + 3]
                v[4 * n_planets] = q1_lo + (q1_hi - q1_lo) * u[4 * n_planets]
                v[4 * n_planets + 1] = q2_lo + (q2_hi - q2_lo) * u[4 * n_planets + 1]
                v[jitter_idx] = ljit_lo + (ljit_hi - ljit_lo) * u[jitter_idx]
                return v
        else:
            def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
                model_buf[:] = 1.0
                q1 = float(beta[4 * n_planets])
                q2 = float(beta[4 * n_planets + 1])
                u1, u2 = _q_to_u(q1, q2)
                for k in range(n_planets):
                    if not gamma[k]:
                        continue
                    lp = float(beta[4 * k])
                    lrr = float(beta[4 * k + 2])
                    bk = float(beta[4 * k + 3])
                    if not ma_model.is_physical(lp, lrr, bk):
                        return float("-inf")
                    ma_model.evaluate_single(
                        planet_buf, k,
                        log_period=lp,
                        t0_phase=float(beta[4 * k + 1]),
                        log_rr=lrr,
                        b=bk,
                        u1=u1, u2=u2,
                    )
                    model_buf[:] += planet_buf - 1.0
                resid = flux - model_buf
                return float(-0.5 * np.dot(resid * resid, inv_var))

            def prior_transform(u: np.ndarray) -> np.ndarray:
                v = np.empty_like(u)
                for k in range(n_planets):
                    v[4 * k] = log_p_los[k] + (log_p_his[k] - log_p_los[k]) * u[4 * k]
                    v[4 * k + 1] = u[4 * k + 1]
                    v[4 * k + 2] = lrr_los[k] + (lrr_his[k] - lrr_los[k]) * u[4 * k + 2]
                    v[4 * k + 3] = b_lo + (b_hi - b_lo) * u[4 * k + 3]
                v[4 * n_planets] = q1_lo + (q1_hi - q1_lo) * u[4 * n_planets]
                v[4 * n_planets + 1] = q2_lo + (q2_hi - q2_lo) * u[4 * n_planets + 1]
                return v
    else:
        if fit_jitter:
            jitter_idx_box = 4 * n_planets

            def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
                model_buf[:] = 1.0
                for k in range(n_planets):
                    if not gamma[k]:
                        continue
                    log_p = beta[4 * k]
                    t0_phase = beta[4 * k + 1]
                    log_d = beta[4 * k + 2]
                    log_dur = beta[4 * k + 3]
                    P = float(np.exp(log_p))
                    depth = float(np.exp(log_d))
                    duration = float(np.exp(log_dur))
                    if duration >= 0.9 * P:
                        return float("-inf")
                    t0 = t0_phase * P
                    half = 0.5 * duration
                    phase = ((time - t0 + 0.5 * P) % P) - 0.5 * P
                    in_transit = np.abs(phase) < half
                    model_buf[:] -= depth * in_transit
                resid = flux - model_buf
                jit2 = float(np.exp(2.0 * float(beta[jitter_idx_box])))
                sigma2_eff = var_data + jit2
                return float(
                    -0.5 * np.sum(resid * resid / sigma2_eff + np.log(sigma2_eff))
                )

            def prior_transform(u: np.ndarray) -> np.ndarray:
                v = np.empty_like(u)
                for k in range(n_planets):
                    v[4 * k] = log_p_los[k] + (log_p_his[k] - log_p_los[k]) * u[4 * k]
                    v[4 * k + 1] = u[4 * k + 1]
                    v[4 * k + 2] = ld_lo + (ld_hi - ld_lo) * u[4 * k + 2]
                    v[4 * k + 3] = ldur_lo + (ldur_hi - ldur_lo) * u[4 * k + 3]
                v[jitter_idx_box] = ljit_lo + (ljit_hi - ljit_lo) * u[jitter_idx_box]
                return v
        else:
            def loglike(beta: np.ndarray, gamma: np.ndarray) -> float:
                model_buf[:] = 1.0
                for k in range(n_planets):
                    if not gamma[k]:
                        continue
                    log_p = beta[4 * k]
                    t0_phase = beta[4 * k + 1]
                    log_d = beta[4 * k + 2]
                    log_dur = beta[4 * k + 3]
                    P = float(np.exp(log_p))
                    depth = float(np.exp(log_d))
                    duration = float(np.exp(log_dur))
                    if duration >= 0.9 * P:
                        return float("-inf")
                    t0 = t0_phase * P
                    half = 0.5 * duration
                    phase = ((time - t0 + 0.5 * P) % P) - 0.5 * P
                    in_transit = np.abs(phase) < half
                    model_buf[:] -= depth * in_transit
                resid = flux - model_buf
                return float(-0.5 * np.dot(resid * resid, inv_var))

            def prior_transform(u: np.ndarray) -> np.ndarray:
                v = np.empty_like(u)
                for k in range(n_planets):
                    v[4 * k] = log_p_los[k] + (log_p_his[k] - log_p_los[k]) * u[4 * k]
                    v[4 * k + 1] = u[4 * k + 1]
                    v[4 * k + 2] = ld_lo + (ld_hi - ld_lo) * u[4 * k + 2]
                    v[4 * k + 3] = ldur_lo + (ldur_hi - ldur_lo) * u[4 * k + 3]
                return v

    groups = []
    for k, c in enumerate(candidates):
        name, P_lit = c[0], c[1]
        params = [4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3]
        if use_ma:
            off_values = np.array(
                [
                    0.5 * (log_p_los[k] + log_p_his[k]),
                    0.0,
                    0.5 * (lrr_los[k] + lrr_his[k]),
                    0.5 * (b_lo + b_hi),
                ]
            )
        else:
            off_values = np.array(
                [
                    0.5 * (log_p_los[k] + log_p_his[k]),
                    0.0,
                    0.5 * (ld_lo + ld_hi),
                    0.5 * (ldur_lo + ldur_hi),
                ]
            )
        groups.append(
            Group(
                name=name,
                params=params,
                off_values=off_values,
                inclusion_prior=inclusion_prior,
            )
        )

    base_ndim = 4 * n_planets + (2 if use_ma else 0)
    return WASP47TightProblem(
        loglike=loglike,
        prior_transform=prior_transform,
        ndim=base_ndim + (1 if fit_jitter else 0),
        groups=groups,
        time=time,
        flux=flux,
        flux_err=flux_err,
        candidate_names=tuple(c[0] for c in candidates),
        candidate_periods=tuple(c[1] for c in candidates),
    )


@dataclass
class IterativeDiscoveryResult:
    """One pass of iterative discovery."""

    iteration: int
    detected: bool
    period: float | None
    rr: float | None
    impact_param: float | None
    t0_phase: float | None
    inclusion_prob: float
    log_Z: float
    n_likelihood_calls: int
    log_Z_no_planet: float | None  # for Bayes-factor model selection
    bayes_factor: float | None     # exp(log_Z - log_Z_no_planet)


def iterative_discovery(
    n_max_planets: int = 4,
    inclusion_threshold: float = 0.5,
    log_period_range: tuple[float, float] | None = None,
    log_rr_range: tuple[float, float] | None = None,
    impact_param_range: tuple[float, float] = (0.0, 1.0),
    M_msun: float = WASP47_M_SUN,
    R_rsun: float = WASP47_R_SUN,
    n_live: int = 300,
    n_mcmc: int = 25,
    transdim_fraction: float = 0.4,
    dlogz: float = 0.5,
    bls_n_periods: int = 5000,
    bls_sharpening: float = 30.0,
    seed: int = 42,
    lc_path: Path | str = DEFAULT_LC,
    verbose: bool = True,
) -> list[IterativeDiscoveryResult]:
    """Iterative single-planet discovery with full residual subtraction.

    Runs up to ``n_max_planets`` rounds of single-planet trans-dim NS,
    subtracting the MAP fit of each detected planet from the flux before
    the next round. This is the "BLS+TLS" pipeline pattern lifted into
    the MoMS-NS framework: each inner search is a well-posed 1-planet
    problem that the chain converges on reliably, even when the joint
    multi-planet posterior has the local-mode trapping that defeats
    one-shot discovery.

    Stops when the latest round's posterior inclusion probability falls
    below ``inclusion_threshold`` (no further planet supported).

    Returns a list of `IterativeDiscoveryResult` (one per round). Each
    has the recovered (P, rr, b, T0_phase) MAP estimate plus the round's
    log_Z and the Bayes factor against a no-planet model on the same
    residual flux.
    """
    import daedalus

    lc = load_lightcurve(lc_path)
    time_arr = lc["time"]
    flux_orig = lc["flux"].copy()
    flux_err = lc["flux_err"]
    flux_residual = flux_orig.copy()  # mutated each round

    if log_rr_range is None:
        log_rr_range = (float(np.log(0.005)), float(np.log(0.2)))
    if log_period_range is None:
        log_period_range = (float(np.log(0.5)), float(np.log(12.0)))

    # We need to evaluate Mandel-Agol for the subtraction step too.
    ma_subtractor = MandelAgolModel(
        time=time_arr, n_planets=1, M_msun=M_msun, R_rsun=R_rsun,
        nominal_periods=(1.0,),
    )
    subtract_buf = np.empty_like(flux_orig)

    def write_residual_to_disk(residual: np.ndarray) -> Path:
        """Build a tempfile NPZ that make_problem can load as the working LC."""
        from tempfile import NamedTemporaryFile

        tmp = NamedTemporaryFile(
            prefix="wasp47_iter_residual_", suffix=".npz", delete=False
        )
        tmp.close()
        np.savez_compressed(
            tmp.name,
            time=time_arr,
            flux=residual,
            flux_err=flux_err,
        )
        return Path(tmp.name)

    out: list[IterativeDiscoveryResult] = []
    for k in range(n_max_planets):
        residual_path = write_residual_to_disk(flux_residual)
        problem = make_problem(
            n_planets=1,
            use_bls_birth=True,
            model="mandel_agol",
            log_period_range=log_period_range,
            log_rr_range=log_rr_range,
            impact_param_range=impact_param_range,
            M_msun=M_msun,
            R_rsun=R_rsun,
            bls_n_periods=bls_n_periods,
            bls_sharpening=bls_sharpening,
            lc_path=residual_path,
        )
        sampler = daedalus.NestedSampler(
            loglike=problem.loglike,
            prior_transform=problem.prior_transform,
            ndim=problem.ndim,
            groups=problem.groups,
            bound="single",
            sample="de",
            n_live=n_live,
            seed=seed + k,
        )
        result = sampler.run_nested(
            dlogz=dlogz,
            n_mcmc=n_mcmc,
            transdim_fraction=transdim_fraction,
            bound_update_interval=10,
        )
        residual_path.unlink(missing_ok=True)

        p_inc = result.inclusion_probabilities()["planet_0"]
        if verbose:
            print(
                f"  iter {k}: log_Z = {result.log_Z:.2f}, "
                f"P(planet) = {p_inc:.3f}, n_calls = {result.n_calls}"
            )

        if p_inc < inclusion_threshold:
            out.append(
                IterativeDiscoveryResult(
                    iteration=k, detected=False,
                    period=None, rr=None, impact_param=None, t0_phase=None,
                    inclusion_prob=p_inc, log_Z=result.log_Z,
                    n_likelihood_calls=result.n_calls,
                    log_Z_no_planet=None, bayes_factor=None,
                )
            )
            break

        # MAP-style estimate from the active samples.
        mask = result.gamma[:, 0]
        log_p_post = np.median(result.samples[mask, 0])
        t0_phase_post = float(np.median(result.samples[mask, 1]))
        log_rr_post = float(np.median(result.samples[mask, 2]))
        b_post = float(np.median(result.samples[mask, 3]))
        # q1, q2 are the always-on slots after the planet block.
        q1_post = float(np.median(result.samples[:, 4]))
        q2_post = float(np.median(result.samples[:, 5]))

        period = float(np.exp(log_p_post))
        rr = float(np.exp(log_rr_post))
        if verbose:
            print(
                f"    -> P = {period:.5f}d, rr = {rr:.4f}, b = {b_post:.3f}"
            )

        # Subtract this planet's MAP-estimated transit from the residual.
        u1, u2 = _q_to_u(q1_post, q2_post)
        ma_subtractor.evaluate_single(
            subtract_buf, 0,
            log_period=log_p_post,
            t0_phase=t0_phase_post,
            log_rr=log_rr_post,
            b=b_post,
            u1=u1, u2=u2,
        )
        flux_residual = flux_residual + (1.0 - subtract_buf)

        out.append(
            IterativeDiscoveryResult(
                iteration=k, detected=True,
                period=period, rr=rr, impact_param=b_post,
                t0_phase=t0_phase_post,
                inclusion_prob=p_inc, log_Z=result.log_Z,
                n_likelihood_calls=result.n_calls,
                log_Z_no_planet=None, bayes_factor=None,
            )
        )

    return out


@dataclass
class BLSCandidate:
    period: float
    bls_power: float


def find_bls_candidates(
    time: np.ndarray,
    flux: np.ndarray,
    period_range: tuple[float, float],
    n_periods: int = 50000,
    duration: float = 0.08,
    n_peaks: int = 10,
    min_log_spacing: float = 0.05,
    suppress_harmonics_of: tuple[float, ...] = (),
    harmonic_ratios: tuple[float, ...] = (1.0 / 3.0, 0.5, 1.0, 2.0, 3.0),
    harmonic_window_log: float = 0.05,
) -> list[BLSCandidate]:
    """Top-N BLS peaks with harmonic suppression of already-found periods.

    Used by ``bls_then_confirmatory_discovery`` to identify candidate
    transit periods up front, before any NS chain runs. Skips peaks
    too close to each other in log-period (``min_log_spacing``) and
    masks periods that are integer-ratio harmonics of any period in
    ``suppress_harmonics_of`` (so iterative discovery doesn't refind
    the same planet's harmonics).
    """
    from astropy import units as u
    from astropy.timeseries import BoxLeastSquares

    bls = BoxLeastSquares(time * u.day, flux)
    log_p_lo = float(np.log(period_range[0]))
    log_p_hi = float(np.log(period_range[1]))
    log_periods = np.linspace(log_p_lo, log_p_hi, n_periods)
    periods = np.exp(log_periods)
    power = np.asarray(bls.power(periods * u.day, duration * u.day).power)

    # Suppress harmonics of already-found planets.
    for p_active in suppress_harmonics_of:
        log_p_active = float(np.log(p_active))
        for ratio in harmonic_ratios:
            log_target = log_p_active + float(np.log(ratio))
            mask = np.abs(log_periods - log_target) < harmonic_window_log
            power[mask] = -np.inf

    # Greedy peak picking with min spacing.
    order = np.argsort(power)[::-1]
    chosen: list[int] = []
    for idx in order:
        if power[idx] == -np.inf:
            continue
        if any(abs(log_periods[idx] - log_periods[j]) < min_log_spacing for j in chosen):
            continue
        chosen.append(int(idx))
        if len(chosen) >= n_peaks:
            break
    return [
        BLSCandidate(period=float(periods[i]), bls_power=float(power[i]))
        for i in chosen
    ]


def find_control_candidates(
    time: np.ndarray,
    flux: np.ndarray,
    real_periods: tuple[float, ...],
    n_controls: int,
    period_range: tuple[float, float] = (0.5, 12.0),
    n_periods: int = 50000,
    duration: float = 0.08,
    real_window_log: float = 0.02,
    min_log_spacing: float = 0.05,
) -> list[BLSCandidate]:
    """Top-N BLS peaks with the immediate neighborhoods of ``real_periods`` masked.

    Builds the rejection arm of a multi-component selection test:
    periods at which the BLS periodogram of the *raw* light curve has
    a local maximum but which are not literature transit signals. No
    flux subtraction is performed -- the periodogram is computed once
    on the raw flux and the immediate vicinity (``real_window_log`` in
    log P) of every period in ``real_periods`` is masked before greedy
    peak picking.

    Aliases of the real signals (e.g. P_b/2, 2 P_b, 3 P_b/2) survive
    the mask and become the test's hardest controls: they look like
    detections in the periodogram but should reject in the joint fit
    because their would-be transits are already explained by the
    corresponding real planet's contribution. A multi-component
    sampler that activates an alias slot is overfitting the noise.

    Avoids iterative-subtraction pipelines, which on multi-planet
    systems with shallow transits leak residuals at the 100 ppm level
    and corrupt downstream rejection tests.
    """
    from astropy import units as u
    from astropy.timeseries import BoxLeastSquares

    bls = BoxLeastSquares(time * u.day, flux)
    log_p_lo = float(np.log(period_range[0]))
    log_p_hi = float(np.log(period_range[1]))
    log_periods = np.linspace(log_p_lo, log_p_hi, n_periods)
    periods = np.exp(log_periods)
    power = np.asarray(bls.power(periods * u.day, duration * u.day).power).copy()

    for p_real in real_periods:
        mask = np.abs(log_periods - float(np.log(p_real))) < real_window_log
        power[mask] = -np.inf

    found: list[BLSCandidate] = []
    while len(found) < n_controls:
        idx = int(np.argmax(power))
        if not np.isfinite(power[idx]):
            break
        peak_p = float(periods[idx])
        peak_pwr = float(power[idx])
        found.append(BLSCandidate(period=peak_p, bls_power=peak_pwr))
        log_p_peak = float(np.log(peak_p))
        mask = np.abs(log_periods - log_p_peak) < min_log_spacing
        power[mask] = -np.inf
    return found


def bls_then_confirmatory_discovery(
    n_max_planets: int = 4,
    inclusion_threshold: float = 0.5,
    period_range: tuple[float, float] = (0.5, 12.0),
    period_window_frac: float = 0.02,
    log_rr_range: tuple[float, float] | None = None,
    impact_param_range: tuple[float, float] = (0.0, 1.0),
    M_msun: float = WASP47_M_SUN,
    R_rsun: float = WASP47_R_SUN,
    n_live: int = 300,
    n_mcmc: int = 25,
    transdim_fraction: float = 0.3,
    dlogz: float = 0.5,
    bls_n_periods: int = 50000,
    bin_minutes: float = 0.0,
    seed: int = 42,
    lc_path: Path | str = DEFAULT_LC,
    verbose: bool = True,
) -> list[IterativeDiscoveryResult]:
    """Two-stage discovery: BLS pre-search + per-candidate confirmatory NS.

    Sidesteps the NS chain-dynamics failure on narrow likelihood needles
    that defeats blind multi-planet discovery on long-baseline TESS
    data. The two stages each do what they're good at:

      Stage 1 -- BLS pre-search: identify the top N candidate periods
        from a high-resolution Box Least Squares periodogram of the
        residual flux. After each detection the periodogram is
        re-evaluated on the residual *and* harmonic-suppressed around
        already-found periods, so the next candidate is always a
        physically distinct signal.

      Stage 2 -- per-candidate confirmatory NS: for each BLS-picked
        period, run a tight-prior MoMS-NS (period restricted to
        period x [1 - period_window_frac, 1 + period_window_frac])
        with the Mandel-Agol model. The chain converges cleanly on
        the narrow likelihood mode and returns the physical (rr, b,
        T0_phase) fit plus the per-planet inclusion probability.

    Stops when the latest round's inclusion probability falls below
    ``inclusion_threshold`` -- i.e. the data doesn't support another
    planet at the proposed period. Returns one
    ``IterativeDiscoveryResult`` per round.
    """
    import daedalus

    lc = load_lightcurve(lc_path)
    time_arr = lc["time"]
    flux_orig = lc["flux"].copy()
    flux_err = lc["flux_err"]
    flux_residual = flux_orig.copy()

    if bin_minutes > 0.0:
        time_arr, flux_residual, flux_err = _bin_lc(
            time_arr, flux_residual, flux_err, bin_minutes
        )
        flux_orig = flux_residual.copy()

    if log_rr_range is None:
        log_rr_range = (float(np.log(0.005)), float(np.log(0.2)))

    ma_subtractor = MandelAgolModel(
        time=time_arr, n_planets=1, M_msun=M_msun, R_rsun=R_rsun,
        nominal_periods=(1.0,),
    )
    subtract_buf = np.empty_like(flux_orig)

    found_periods: list[float] = []
    out: list[IterativeDiscoveryResult] = []

    for k in range(n_max_planets):
        candidates = find_bls_candidates(
            time_arr, flux_residual, period_range,
            n_periods=bls_n_periods,
            n_peaks=1,
            suppress_harmonics_of=tuple(found_periods),
        )
        if not candidates:
            if verbose:
                print(f"  iter {k}: no BLS candidate left after harmonic suppression")
            break
        cand = candidates[0]
        if verbose:
            print(f"  iter {k}: BLS top peak P = {cand.period:.5f}d, power = {cand.bls_power:.4f}")

        # Build a tight-prior 1-planet MA problem at this candidate period.
        P_lo = cand.period * (1.0 - period_window_frac)
        P_hi = cand.period * (1.0 + period_window_frac)
        log_period_range_tight = (float(np.log(P_lo)), float(np.log(P_hi)))

        problem = make_problem(
            n_planets=1,
            use_bls_birth=False,        # uniform-u birth on a tight prior == PriorBirth
            model="mandel_agol",
            log_period_range=log_period_range_tight,
            log_rr_range=log_rr_range,
            impact_param_range=impact_param_range,
            M_msun=M_msun, R_rsun=R_rsun,
            lc_path=lc_path,             # not used; we override flux in residual
        )
        # Override the bundled flux with the current residual.
        # The simplest way without rewriting make_problem: write a
        # tempfile NPZ each round.
        from tempfile import NamedTemporaryFile

        tmp = NamedTemporaryFile(prefix="wasp47_bls_then_conf_", suffix=".npz", delete=False)
        tmp.close()
        np.savez_compressed(tmp.name, time=time_arr, flux=flux_residual, flux_err=flux_err)
        problem = make_problem(
            n_planets=1, use_bls_birth=False, model="mandel_agol",
            log_period_range=log_period_range_tight,
            log_rr_range=log_rr_range,
            impact_param_range=impact_param_range,
            M_msun=M_msun, R_rsun=R_rsun,
            lc_path=Path(tmp.name),
        )

        sampler = daedalus.NestedSampler(
            loglike=problem.loglike,
            prior_transform=problem.prior_transform,
            ndim=problem.ndim,
            groups=problem.groups,
            bound="single", sample="rwalk",
            n_live=n_live, seed=seed + k,
        )
        result = sampler.run_nested(
            dlogz=dlogz, n_mcmc=n_mcmc,
            transdim_fraction=transdim_fraction,
            bound_update_interval=10,
        )
        Path(tmp.name).unlink(missing_ok=True)

        p_inc = result.inclusion_probabilities()["planet_0"]
        if verbose:
            print(f"    NS at P in [{P_lo:.4f}, {P_hi:.4f}]: P(planet) = {p_inc:.3f}")

        if p_inc < inclusion_threshold:
            out.append(
                IterativeDiscoveryResult(
                    iteration=k, detected=False,
                    period=cand.period, rr=None, impact_param=None, t0_phase=None,
                    inclusion_prob=p_inc, log_Z=result.log_Z,
                    n_likelihood_calls=result.n_calls,
                    log_Z_no_planet=None, bayes_factor=None,
                )
            )
            break

        mask = result.gamma[:, 0]
        log_p_post = float(np.median(result.samples[mask, 0]))
        t0_phase_post = float(np.median(result.samples[mask, 1]))
        log_rr_post = float(np.median(result.samples[mask, 2]))
        b_post = float(np.median(result.samples[mask, 3]))
        q1_post = float(np.median(result.samples[:, 4]))
        q2_post = float(np.median(result.samples[:, 5]))

        period = float(np.exp(log_p_post))
        rr = float(np.exp(log_rr_post))
        if verbose:
            print(f"    -> P={period:.5f}d, rr={rr:.4f}, b={b_post:.3f}")

        u1, u2 = _q_to_u(q1_post, q2_post)
        ma_subtractor.evaluate_single(
            subtract_buf, 0,
            log_period=log_p_post,
            t0_phase=t0_phase_post,
            log_rr=log_rr_post,
            b=b_post,
            u1=u1, u2=u2,
        )
        flux_residual = flux_residual + (1.0 - subtract_buf)
        found_periods.append(period)

        out.append(
            IterativeDiscoveryResult(
                iteration=k, detected=True,
                period=period, rr=rr, impact_param=b_post,
                t0_phase=t0_phase_post,
                inclusion_prob=p_inc, log_Z=result.log_Z,
                n_likelihood_calls=result.n_calls,
                log_Z_no_planet=None, bayes_factor=None,
            )
        )

    return out


def n_planets_posterior(results) -> dict[int, float]:
    """Extract P(N_planets = k | data) from a Results' resampled gamma.

    The N_p posterior is the headline trans-dim output for
    confirmatory variable-selection problems (van den Bergh+ 2026 §1
    inclusion probabilities, but at the model-level rather than the
    predictor-level).
    """
    if results.gamma.shape[1] == 0:
        return {}
    np_counts = results.gamma.sum(axis=1)
    n_total = np_counts.size
    out: dict[int, float] = {}
    for k in range(results.gamma.shape[1] + 1):
        out[k] = float(np.sum(np_counts == k)) / n_total
    return out
