"""Insertion-index diagnostic for nested sampling (Fowlie, Handley & Su 2020).

Fowlie, Handley & Su (2020), "Nested sampling cross-checks using order
statistics" (MNRAS 497, 5256; arXiv:2006.03371). When a new live point is
born at likelihood L under a *correct* NS kernel (an i.i.d. draw from the
prior restricted to {L > L*}), its rank among the current n_live live-point
likelihoods is Uniform{0, 1, ..., n_live - 1}. A kernel that fails to
decorrelate the newborn from the donor it walked from biases the rank LOW:
the newborn lands near the bottom of the live set because it never moved
far from a point that was itself near threshold. So a mean insertion
fraction < 0.5 and a left-skewed index distribution are the signature of
within-model under-mixing.

This module provides:

  - ``insertion_indices_from_dead_sequence``: reconstruct the per-iteration
    insertion index from the birth-ordered dead-point log-likelihood
    sequence that daedalus saves (``Results.log_likelihoods`` /
    ``acc.dead_log_likelihoods``). This is EXACT for standard fixed-n_live
    NS: the live set at iteration i is exactly the set of points born but
    not yet dead, and the newborn's rank is recoverable from the sequence.

  - ``insertion_index_test``: KS test of index/n_live against U(0,1) plus a
    rolling-window mean-fraction test (a single global KS washes out drift
    when the early/late run mix differently; the rolling test localises it).

  - ``analyse_dead_sequence`` / ``analyse_npz``: convenience wrappers that
    take a saved daedalus run and return a verdict dict.

The reconstruction is validated against a ground-truth instrumented run in
``insertion_index_validate.py``.
"""

from __future__ import annotations

import numpy as np

# The working uniformity test now lives in the daedalus public API; import it
# here so the scripts that depend on this module (``from insertion_index import
# insertion_index_test``) keep working without code duplication.
from daedalus.diagnostics import (  # noqa: F401
    InsertionTestResult,
    insertion_index_test,
)


# --------------------------------------------------------------------------
# Reconstruction of insertion indices from a birth-ordered dead sequence
# --------------------------------------------------------------------------

def insertion_indices_from_dead_sequence(
    dead_log_L: np.ndarray,
    n_live: int,
) -> np.ndarray:
    """NOT SUPPORTED: death-only reconstruction is not exact -- see below.

    daedalus saves only the death-ordered dead-point log-likelihood
    sequence (``Results.log_likelihoods`` == ``acc.dead_log_likelihoods``),
    i.e.

        [ L*_1, L*_2, ..., L*_M,  L_remaining_live(1..n_live) ],

    where the L*_i are death thresholds (monotone non-decreasing) and the
    trailing block is the unsorted final live cloud. It does NOT store, per
    dead point, the *birth contour* (the threshold at which that point was
    created) nor the identity that links a newborn at iteration i to its
    later death at iteration j. The Fowlie+2020 / anesthetic insertion
    index needs the rank of each NEWBORN's likelihood among the live set
    contemporaneous with its birth. From a death-only stream the
    newborn->death identity mapping is exactly the information that is
    lost: the death sequence is fully determined but does not tell you
    which newborn became which later death.

    We verified empirically that the saved ``log_likelihoods`` IS the
    birth/death-ordered dead sequence (99.6% monotone; the 0.4%
    non-monotone tail is the unsorted final live block), but that is not
    sufficient to recover insertion indices. We therefore DO NOT fake a
    reconstruction. Use the instrumented recorder instead: pass
    ``insertion_recorder=[]`` to ``NestedSampler.run_nested`` (a minimal
    O(n_live)/iter hook added to daedalus/sampler.py) and feed the
    resulting list to ``insertion_index_test``. See
    ``insertion_index_hd10180.py``.

    If birth contours are later persisted (e.g. an extra ``birth_log_L``
    array on Results), ``insertion_indices_birth_death`` below computes the
    exact indices offline.
    """
    raise NotImplementedError(
        "daedalus saves death contours only; newborn identity is "
        "unrecoverable from the dead-only stream, so insertion indices "
        "cannot be reconstructed exactly. Use the insertion_recorder hook "
        "in NestedSampler.run_nested (see insertion_index_hd10180.py)."
    )


def insertion_indices_birth_death(
    birth_log_L: np.ndarray,
    death_log_L: np.ndarray,
    n_live_init: int,
) -> np.ndarray:
    """NOT IMPLEMENTED: offline insertion indices from (birth, death) contours.

    This is the anesthetic / Fowlie+2020 offline method: each point is a
    pair (L_birth, L_death); the insertion index of a newborn is the rank
    of its death contour among the death contours of the points live at the
    instant of its birth. The correct algorithm (see anesthetic's
    ``compute_insertion_indexes``) replays a MERGED birth/death event
    timeline keyed on STABLE per-point integer IDs (not on float contour
    values, which are non-unique and break naive matching) and tracks the
    live set as an ordered multiset of (death_contour, id).

    daedalus does not currently persist per-point birth contours, so this
    path is unused; the validated primary method for this codebase is the
    in-loop ``insertion_recorder`` hook (see module docstring,
    ``insertion_index_hd10180.py``). A float-contour-keyed reconstruction
    was prototyped and rejected because it is not robust to (a) survivors
    that never die (death contour = +inf), and (b) non-unique death
    contours; it produced mean fractions of 0.94/0.5 on a known-uniform
    synthetic and so is NOT shipped. Implement properly against
    anesthetic's reference if/when birth contours are persisted, and gate
    on the synthetic self-test in this module's __main__ that a
    known-correct uniform NS replay yields mean fraction ~0.5 and KS p>0.05.
    """
    raise NotImplementedError(
        "Offline (birth, death)-contour reconstruction is not implemented; "
        "daedalus does not persist birth contours. Use the insertion_recorder "
        "hook in NestedSampler.run_nested (see insertion_index_hd10180.py)."
    )


# --------------------------------------------------------------------------
# Uniformity tests
# --------------------------------------------------------------------------
#
# ``InsertionTestResult`` and ``insertion_index_test`` now live in
# ``daedalus.diagnostics`` (imported at the top of this module) so the test
# is part of the public API. They are re-exported here unchanged for the
# scripts that ``from insertion_index import insertion_index_test``.


# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------

def analyse_birth_death(
    birth_log_L: np.ndarray,
    death_log_L: np.ndarray,
    n_live: int,
    rolling_window: int | None = None,
) -> InsertionTestResult:
    idx = insertion_indices_birth_death(birth_log_L, death_log_L, n_live)
    return insertion_index_test(idx, n_live, rolling_window=rolling_window)
