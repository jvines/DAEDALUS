"""Importance-sampled Rao-Blackwell inclusion for the HD 10180 labelled run.

Runs the labelled candidate-confirmation chain (DE kernel, n_live=600,
seed 42 -- the same settings as the campaign run the paper reports), then
computes the Rao-Blackwellised inclusion
marginal of Eq. (rb-inclusion) for every slot by importance sampling:

    logit_k(beta, gamma) = log p_k/(1-p_k) + log Z_on,k - log L_off,k
    Z_on,k  ~ mean over n_s prior draws of L(beta_-k, beta_k, gamma_k=1)
    L_off,k = L(beta with gamma_k=0)

The slot priors are uniform boxes (log-P window, phase, log-K), so drawing
from the prior makes the IS weights proportional to the likelihood alone.
Writes scripts/hd10180_rb_results.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import daedalus  # noqa: E402
from daedalus.samplers import DifferentialEvolutionSampler  # noqa: E402
import hd10180_lovis_labelled as lab  # noqa: E402
from insertion_index import insertion_index_test  # noqa: E402

SEED = 42
N_LIVE = 600
N_IS = 64           # IS draws per dead point per slot
N_SUB = 8000        # posterior rows used for the RB average
RB_SEED = 7


def main() -> None:
    d = _load_harps_npz()
    t, rv, err = d["t"], d["rv"], d["err"]
    loglike, prior_transform, groups, periodic = lab.build_problem(t, rv, err)

    sampler = daedalus.NestedSampler(
        loglike=loglike, prior_transform=prior_transform, ndim=lab.NDIM,
        groups=groups, bound="multi",
        sample=DifferentialEvolutionSampler(target_accept=0.234),
        n_live=N_LIVE, seed=SEED, periodic=periodic)
    rec: list[int] = []
    t0 = time.time()
    res = sampler.run_nested(dlogz=0.3, n_mcmc=40, transdim_fraction=0.35,
                             show_progress=False, insertion_recorder=rec)
    el = time.time() - t0
    it = insertion_index_test(np.asarray(rec, float), N_LIVE)
    emp = res.inclusion_probabilities()
    print(f"chain done in {el:.0f}s  logZ={res.log_Z:.3f}+/-{res.log_Z_err:.3f}"
          f"  ks_p={it.ks_pvalue:.3g} z={it.z_mean:+.2f}", flush=True)
    print("empirical inclusions:", {k: round(v, 4) for k, v in emp.items()},
          flush=True)

    # --- Rao-Blackwell over a posterior subsample --------------------------
    # Z_on,k = int pi_k(beta_k) L dbeta_k is sharply peaked in the period
    # direction (the HARPS baseline resolves ~2000 cycles of b), so
    # uniform prior draws essentially never hit the peak. We therefore use
    # a DEFENSIVE MIXTURE importance proposal per slot:
    #     q = 0.5 * prior-box + 0.5 * N(mode_k, sigma_k)
    # with (mode_k, sigma_k) estimated from the chain's own active
    # (gamma_k = 1) posterior samples and sigma inflated 3x. The prior
    # component guarantees q > 0 on the full support (unbiased); the mode
    # component resolves the peak. Weights w = pi / q.
    rng = np.random.default_rng(RB_SEED)
    n = res.samples.shape[0]
    idx = rng.choice(n, size=min(N_SUB, n), replace=False)
    slot_priors = lab.make_slot_period_priors()
    log_k_lo, log_k_hi = lab.LOG_K_LO, lab.LOG_K_HI

    t1 = time.time()
    p_rb = np.zeros(lab.N_SLOTS)
    for k in range(lab.N_SLOTS):
        lp_lo, lp_hi = slot_priors[k]
        cols = [3 * k, 3 * k + 1, 3 * k + 2]
        box = np.array([lp_hi - lp_lo, 1.0, log_k_hi - log_k_lo])
        log_pi = float(-np.sum(np.log(box)))
        lo = np.array([lp_lo, 0.0, log_k_lo])
        hi = np.array([lp_hi, 1.0, log_k_hi])
        act = res.gamma[:, k].astype(bool)
        on = res.samples[act][:, cols]
        mode = np.median(on, axis=0)
        sig = 3.0 * np.maximum(on.std(axis=0), 1e-6 * box)
        s = 0.0
        for i in idx:
            beta = res.samples[i].copy()
            gamma = res.gamma[i].copy()
            gamma[k] = False
            log_L_off = loglike(beta, gamma)
            gamma[k] = True
            half = N_IS // 2
            d_prior = rng.uniform(lo, hi, size=(half, 3))
            d_mode = rng.normal(mode, sig, size=(N_IS - half, 3))
            draws = np.vstack([d_prior, d_mode])
            inside = np.all((draws >= lo) & (draws <= hi), axis=1)
            # log q at each draw: mixture of box-uniform and the Gaussian
            log_g = np.sum(
                -0.5 * ((draws - mode) / sig) ** 2
                - np.log(sig) - 0.5 * np.log(2.0 * np.pi),
                axis=1,
            )
            log_q = logsumexp(
                np.stack([
                    np.where(inside, log_pi, -np.inf),
                    log_g,
                ]) + np.log(0.5),
                axis=0,
            )
            lls = np.full(N_IS, -np.inf)
            for j in range(N_IS):
                if not inside[j]:
                    continue  # pi = 0 outside the box: zero weight
                beta[cols] = draws[j]
                lls[j] = loglike(beta, gamma)
            log_w = np.where(inside, log_pi - log_q + lls, -np.inf)
            log_Z_on = float(logsumexp(log_w) - np.log(N_IS))
            logit = log_Z_on - log_L_off  # p_k = 0.5 -> no prior-odds term
            s += 0.5 * (1.0 + np.tanh(0.5 * logit))
        p_rb[k] = s / idx.size
        print(f"  RB slot_{lab.SLOT_NAMES[k]}: {p_rb[k]:.4f} "
              f"(empirical {emp[f'slot_{lab.SLOT_NAMES[k]}']:.4f})", flush=True)
    print(f"RB pass in {time.time() - t1:.0f}s", flush=True)

    out = {
        "seed": SEED, "n_live": N_LIVE, "n_is": N_IS, "n_sub": int(idx.size),
        "log_Z": float(res.log_Z), "log_Z_err": float(res.log_Z_err),
        "ks_p": float(it.ks_pvalue), "z_mean": float(it.z_mean),
        "empirical": {k: float(v) for k, v in emp.items()},
        "rao_blackwell": {f"slot_{lab.SLOT_NAMES[k]}": float(p_rb[k])
                          for k in range(lab.N_SLOTS)},
    }
    with open(os.path.join(HERE, "hd10180_rb_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote hd10180_rb_results.json", flush=True)


def _load_harps_npz():
    """HARPS RVs as a dict with keys t/rv/err; fetches and caches on first use."""
    t, rv, err = lab.load_harps()
    return {"t": t, "rv": rv, "err": err}


if __name__ == "__main__":
    main()
