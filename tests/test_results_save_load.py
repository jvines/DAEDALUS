"""Round-trip persistence tests for `Results`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import daedalus
from daedalus.benchmarks import gaussians


def test_results_save_load_roundtrip(tmp_path: Path) -> None:
    problem = gaussians.make_problem(ndim=2, prior_half_width=10.0)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="single",
        sample="rwalk",
        n_live=100,
        seed=0,
    )
    original = sampler.run_nested(dlogz=1.0, n_mcmc=10)

    out = tmp_path / "results.npz"
    original.save(str(out))
    restored = daedalus.load_results(str(out))

    assert np.array_equal(restored.samples, original.samples)
    assert np.array_equal(restored.gamma, original.gamma)
    assert np.array_equal(restored.log_likelihoods, original.log_likelihoods)
    assert np.array_equal(restored.log_weights, original.log_weights)
    assert restored.log_Z == original.log_Z
    assert restored.log_Z_err == original.log_Z_err
    assert restored.H == original.H
    assert restored.n_iter == original.n_iter
    assert restored.n_calls == original.n_calls
    assert restored.group_names == list(original.group_names)


def test_results_save_load_preserves_trans_dim_outputs(tmp_path: Path) -> None:
    problem = gaussians.make_problem(ndim=2, prior_half_width=10.0)
    g0 = daedalus.Group(name="g0", params=[0], off_values=np.array([0.0]))
    g1 = daedalus.Group(name="g1", params=[1], off_values=np.array([0.0]))
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=[g0, g1],
        bound="none",
        sample="rwalk",
        n_live=100,
        seed=0,
    )
    original = sampler.run_nested(dlogz=1.0, n_mcmc=20, transdim_fraction=0.5)

    out = tmp_path / "trans_results.npz"
    original.save(str(out))
    restored = daedalus.load_results(str(out))

    assert restored.gamma.shape == original.gamma.shape
    assert np.array_equal(restored.gamma, original.gamma)
    assert restored.inclusion_probabilities() == original.inclusion_probabilities()
    # inclusion_priors must round-trip too -- model_evidences() needs them.
    assert np.array_equal(restored.inclusion_priors, original.inclusion_priors)
    assert restored.model_evidences() == original.model_evidences()


def test_model_evidences_recovers_per_model_log_Z(tmp_path: Path) -> None:
    """Sanity check on the per-model evidence formula:
        log Z_gamma = log P(gamma | y) + log Z - log P(gamma)
    The chain's per-model evidences must satisfy that identity exactly,
    and the resulting log Z_gamma values must combine back to the joint
    log Z when prior-weighted (Bayes' theorem internal consistency)."""
    problem = gaussians.make_problem(ndim=2, prior_half_width=10.0)
    g0 = daedalus.Group(
        name="g0", params=[0], off_values=np.array([0.0]), inclusion_prior=0.4
    )
    g1 = daedalus.Group(
        name="g1", params=[1], off_values=np.array([0.0]), inclusion_prior=0.7
    )
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        groups=[g0, g1],
        bound="none",
        sample="rwalk",
        n_live=200,
        seed=42,
    )
    results = sampler.run_nested(dlogz=0.5, n_mcmc=20, transdim_fraction=0.5)

    # Identity reconstruction: log Z_gamma + log P(gamma) - log P(gamma|y) = log Z
    # for every visited gamma. Within float tolerance (the formula is exact).
    log_inc = np.log(results.inclusion_priors)
    log_exc = np.log1p(-results.inclusion_priors)
    evidences = results.model_evidences()
    probs = results.model_probabilities()
    for gamma_key, (log_Z_gamma, _err) in evidences.items():
        gamma_arr = np.asarray(gamma_key, dtype=bool)
        log_prior_gamma = float(np.where(gamma_arr, log_inc, log_exc).sum())
        log_p_post = float(np.log(probs[gamma_key]))
        reconstructed = log_Z_gamma + log_prior_gamma - log_p_post
        assert np.isclose(reconstructed, results.log_Z, atol=1e-10), (
            f"per-model evidence identity broken for gamma={gamma_key}: "
            f"recovered log_Z = {reconstructed} vs results.log_Z = {results.log_Z}"
        )

    # Inverse identity: sum_gamma P(gamma) * Z_gamma = Z (joint).
    # Equivalently: logsumexp_gamma (log_prior_gamma + log_Z_gamma) == log_Z,
    # restricted to *visited* gammas; unvisited gammas have negligible
    # posterior weight at the chain's resolution.
    terms = []
    for gamma_key, (log_Z_gamma, _) in evidences.items():
        gamma_arr = np.asarray(gamma_key, dtype=bool)
        log_prior_gamma = float(np.where(gamma_arr, log_inc, log_exc).sum())
        terms.append(log_prior_gamma + log_Z_gamma)
    joint_log_Z_recovered = float(np.logaddexp.reduce(terms))
    # Loose tolerance because unvisited gammas contribute negligibly but
    # not exactly zero. With n_live=200 the chain at this scale should
    # have visited all 2^2 = 4 configurations, so the identity holds
    # very tight.
    assert np.isclose(joint_log_Z_recovered, results.log_Z, atol=1e-6)


def test_model_evidences_empty_for_continuous_only() -> None:
    """No groups -> empty model_evidences dict. Mirrors model_probabilities."""
    problem = gaussians.make_problem(ndim=2, prior_half_width=10.0)
    sampler = daedalus.NestedSampler(
        loglike=problem.loglike,
        prior_transform=problem.prior_transform,
        ndim=problem.ndim,
        bound="single",
        sample="rwalk",
        n_live=100,
        seed=0,
    )
    results = sampler.run_nested(dlogz=1.0, n_mcmc=10)
    assert results.model_evidences() == {}
    assert results.model_probabilities() == {}


def test_model_evidences_raises_when_priors_missing(tmp_path: Path) -> None:
    """Construct a Results manually with mismatched inclusion_priors;
    model_evidences() must raise rather than silently use zeros."""
    bad = daedalus.Results(
        samples=np.zeros((10, 1)),
        gamma=np.zeros((10, 2), dtype=bool),
        log_likelihoods=np.zeros(10),
        log_weights=np.zeros(10),
        log_Z=0.0,
        log_Z_err=0.1,
        H=1.0,
        n_iter=10,
        n_calls=10,
        group_names=["a", "b"],
        inclusion_priors=np.empty(0),  # missing
    )
    import pytest as _pytest

    with _pytest.raises(RuntimeError, match="inclusion_priors length"):
        bad.model_evidences()
