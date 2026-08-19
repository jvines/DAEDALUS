"""daedalus likelihood functions for trans-dim inference.

The standard chi-squared sum likelihood lives in the per-benchmark
``loglike`` constructions (e.g. ``wasp47.make_problem_tight``). This
package provides alternative likelihood machinery that's robust to
specific failure modes -- in particular the
``per_cycle_robust_transit`` likelihood, which decomposes chi-squared
contributions cycle-by-cycle and aggregates via a robust statistic
(median / trimmed mean) so that a candidate gains chi-squared only
when the signal is consistent across cycles, not just from a few
cycles that happen to coincide with another candidate's real transits.

See the module docstring of :mod:`daedalus.loglikes.transit_robust` for
the design rationale.
"""

from .transit_robust import per_cycle_robust_transit_loglike  # noqa: F401
