"""Real-data MoMS-NS demo: SDSS star-forming galaxy emission lines.

Loads a bundled SDSS DR17 BOSS spectrum (a STARFORMING galaxy at
z=0.038) and runs a trans-dim NS over 11 candidate optical emission
lines. Posterior inclusion probabilities should:

  - mark the strong star-forming lines (Halpha, Hbeta, [OIII] 5007,
    [NII] 6548/6584, [SII] 6717/6731, [OII] 3727) as P > 0.8
  - reject the AGN-only HeII 4686 as P < 0.2

The [OIII] 4959 line and [OI] 6300 are diagnostic of degenerate fits /
weak lines and are not pinned.
"""

from __future__ import annotations

import pytest

import daedalus
from daedalus.benchmarks import spectroscopy_real


@pytest.mark.benchmark
@pytest.mark.slow
def test_sdss_galaxy_recovers_starforming_lines_and_rejects_agn() -> None:
    problem = spectroscopy_real.make_problem()
    groups = [daedalus.Group(**kwargs) for kwargs in problem.groups_kwargs]
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=groups,
        bound="single",
        sample="rwalk",
        n_live=200,
        seed=42,
    )
    results = sampler.run_nested(
        dlogz=0.5, n_mcmc=25, transdim_fraction=0.5, bound_update_interval=5
    )

    inc = results.inclusion_probabilities()
    failures = []

    expected_strong = (
        "Halpha", "Hbeta", "[OIII]5007", "[NII]6548", "[NII]6584",
        "[SII]6717", "[SII]6731", "[OII]3727",
    )
    for name in expected_strong:
        p = inc[name]
        if p < 0.8:
            failures.append(
                f"strong SF line {name}: P(gamma=1) = {p:.3f}, expected > 0.8"
            )

    # AGN-only line should not be detected in a STARFORMING spectrum.
    p_he = inc["HeII4686"]
    if p_he > 0.2:
        failures.append(f"AGN-only HeII4686: P(gamma=1) = {p_he:.3f}, expected < 0.2")

    assert not failures, "SDSS emission line recovery failures:\n" + "\n".join(failures)
