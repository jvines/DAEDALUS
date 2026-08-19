"""Phase-1 spike: vectorized nested sampling with a batched (GEMM-friendly)
likelihood, via a vectorized differential-evolution kernel.

Design (correctness-first):
  - Standard SINGLE-point NS removal loop -> reuse daedalus's tested
    SkillingAccumulator for the evidence recursion (zero recursion risk).
  - Replacements come from a QUEUE of candidates that are batch-generated K
    at a time: K lockstep DE chains, each of n_mcmc steps, evaluating all K
    proposals in ONE loglike_vec call per step (the GEMM). Candidates that
    fall below the (rising) threshold before use are discarded by rejection
    -- valid by the NS restriction property (prior|L>a conditioned on L>b is
    prior|L>b).

Validates:
  1. Unbiased evidence: logZ matches analytic truth (Gaussian) across seeds,
     and matches serial daedalus-DE.
  2. The win: on a design-matrix likelihood, the SAME algorithm
     with a batched-GEMM loglike vs a serial-looped loglike -> wall speedup.
"""
from __future__ import annotations
import time
from collections import deque
import numpy as np
from daedalus.recursion import SkillingAccumulator


# ---------------------------------------------------------------- kernel ----
def _batch_generate_de(U, L, L_min, K, n_mcmc, loglike_vec, ptform, rng, scale):
    """K lockstep DE chains constrained to L > L_min. One loglike_vec call
    per step (evaluated on the in-cube subset). Returns (Uc (K,ndim), Lc (K,),
    n_calls) -- n_calls counts POINT evaluations (for accounting)."""
    n_live, ndim = U.shape
    gamma = 2.38 / np.sqrt(2.0 * ndim)
    elig = np.flatnonzero(L > L_min)          # start strictly above threshold
    if elig.size == 0:
        elig = np.arange(n_live)
    Uc = U[rng.choice(elig, size=K)].copy()   # (K, ndim)
    Lc = loglike_vec(ptform(Uc))
    n_calls = K
    acc_total = 0
    for _ in range(n_mcmc):
        a = rng.integers(n_live, size=K)
        b = rng.integers(n_live, size=K)
        coll = a == b
        while coll.any():
            b[coll] = rng.integers(n_live, size=int(coll.sum()))
            coll = a == b
        prop = Uc + gamma * scale * (U[a] - U[b]) + 1e-6 * rng.standard_normal((K, ndim))
        in_cube = ((prop >= 0.0) & (prop <= 1.0)).all(axis=1)
        Lp = np.full(K, -np.inf)
        if in_cube.any():
            Lp[in_cube] = loglike_vec(ptform(prop[in_cube]))   # ONE batched call
            n_calls += int(in_cube.sum())
        take = in_cube & (Lp > L_min)
        Uc[take] = prop[take]
        Lc[take] = Lp[take]
        acc_total += int(take.sum())
    accept_rate = acc_total / (n_mcmc * K)
    return Uc, Lc, n_calls, accept_rate


# ------------------------------------------------------------------- loop ----
def vectorized_ns(loglike_vec, ptform, ndim, n_live=500, K=64, n_mcmc=None,
                  dlogz=0.1, seed=0, scale=1.0):
    if n_mcmc is None:
        n_mcmc = max(25, 5 * ndim)
    rng = np.random.default_rng(seed)
    U = rng.random((n_live, ndim))
    L = loglike_vec(ptform(U))
    n_calls = n_live
    acc = SkillingAccumulator(n_live=n_live)
    queue: deque = deque()   # (L_cand, u_cand), FIFO; discard stale from front
    log_dlogz = np.log(dlogz)
    n_stale = 0
    accs = []
    dead_u = []                       # unit-cube coords of dead points, aligned 1:1 with acc weights
    while True:
        i_min = int(np.argmin(L))
        L_min = float(L[i_min])
        if acc.termination_gap(float(L.max())) < log_dlogz and acc.n_iter > n_live:
            break
        acc.add_dead(L_min)
        dead_u.append(U[i_min].copy())   # record the dying point BEFORE it is overwritten
        # get a replacement above L_min: discard stale from front, refill if empty
        while True:
            while queue and queue[0][0] <= L_min:
                queue.popleft(); n_stale += 1
            if queue:
                break
            Uc, Lc, nc, ar = _batch_generate_de(
                U, L, L_min, K, n_mcmc, loglike_vec, ptform, rng, scale)
            n_calls += nc
            accs.append(ar)
            for k in range(K):
                queue.append((float(Lc[k]), Uc[k]))
        Lr, Ur = queue.popleft()
        L[i_min] = Lr
        U[i_min] = Ur
    acc.add_remaining_live(L)
    dead_u.extend(u.copy() for u in U)   # final live set are dead points too (L-array order)

    # Importance-weighted posterior over the dead points. The unnormalised
    # log-weight of dead point i is log(dX_i) + log L_i -- exactly the term the
    # SkillingAccumulator sums into Z -- so samples and weights are aligned 1:1
    # (add_dead appends one entry per iteration; add_remaining_live appends one
    # per final live point in L-array order, matching how dead_u was built).
    du = np.asarray(dead_u)
    lw = np.asarray(acc.dead_log_weights) + np.asarray(acc.dead_log_likelihoods)
    assert du.shape[0] == lw.shape[0], (du.shape, lw.shape)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    samples = ptform(du)                 # unit-cube -> parameter space (batched ptform)

    return dict(logZ=acc.log_Z, logZ_err=acc.log_Z_err, n_calls=n_calls,
                n_iter=acc.n_iter, n_stale=n_stale,
                accept_rate=float(np.mean(accs)) if accs else 0.0,
                samples=samples,          # (n_dead, ndim)
                weights=w,                # (n_dead,) normalised importance weights AT the dead points
                log_weights=lw)           # (n_dead,) unnormalised log-weights (for PSIS-LOO/stacking)


# ---------------------------------------------------------------- problems ---
def gaussian(ndim, H=10.0):
    NORM = -0.5 * ndim * np.log(2.0 * np.pi)
    def ll(Theta):        # (B, ndim) -> (B,)
        return NORM - 0.5 * np.sum(Theta * Theta, axis=1)
    def pt(U):            # (B, ndim) -> (B, ndim)
        return 2.0 * H * U - H
    return ll, pt, -ndim * np.log(2.0 * H)   # analytic logZ


def design_matrix(ndim, n_data=2000, H=5.0, mode="gemm", seed=0):
    """Design-matrix likelihood: chi2 = ||y - X theta||^2. mode='gemm' does one GEMM per
    batch; mode='loop' does B serial matvecs (same math, no BLAS batching)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_data, ndim))
    y = rng.standard_normal(n_data)
    def ll_gemm(Theta):                 # (B, ndim)
        R = y[:, None] - X @ Theta.T    # (n_data, B) one GEMM
        return -0.5 * np.sum(R * R, axis=0)
    def ll_loop(Theta):
        out = np.empty(Theta.shape[0])
        for i in range(Theta.shape[0]):
            r = y - X @ Theta[i]
            out[i] = -0.5 * float(r @ r)
        return out
    def pt(U):
        return 2.0 * H * U - H
    return (ll_gemm if mode == "gemm" else ll_loop), pt


# ------------------------------------------------------------------- main ----
if __name__ == "__main__":
    print("=== 1. UNBIASEDNESS: vectorized-NS logZ vs analytic (Gaussian) ===")
    for ndim in (5, 10):
        ll, pt, truth = gaussian(ndim)
        errs = []
        for seed in range(6):
            r = vectorized_ns(ll, pt, ndim, n_live=500, K=64, dlogz=0.1, seed=seed)
            errs.append(r["logZ"] - truth)
        errs = np.array(errs)
        ex = vectorized_ns(ll, pt, ndim, n_live=500, K=64, dlogz=0.1, seed=0)
        print(f"  {ndim}D: truth={truth:.3f}  bias={errs.mean():+.3f}±{errs.std():.3f} "
              f"(6 seeds)  |  seed0: logZ={ex['logZ']:.3f}±{ex['logZ_err']:.3f} "
              f"calls={ex['n_calls']} stale={ex['n_stale']} accept={ex['accept_rate']:.2f}")

    print("\n=== 2. THE WIN: design-matrix likelihood, same algo, GEMM vs loop ===")
    for ndim, n_data in ((20, 2000), (50, 20000)):
        for K in (32, 128):
            ll_g, pt = design_matrix(ndim, n_data, mode="gemm", seed=1)
            ll_l, _ = design_matrix(ndim, n_data, mode="loop", seed=1)
            t = time.perf_counter()
            rg = vectorized_ns(ll_g, pt, ndim, n_live=400, K=K, dlogz=0.5, seed=0)
            wg = time.perf_counter() - t
            t = time.perf_counter()
            rl = vectorized_ns(ll_l, pt, ndim, n_live=400, K=K, dlogz=0.5, seed=0)
            wl = time.perf_counter() - t
            print(f"  ndim={ndim} n_data={n_data} K={K}: "
                  f"loop={wl:.2f}s  gemm={wg:.2f}s  speedup={wl/wg:.1f}x  "
                  f"(logZ gemm={rg['logZ']:.1f} loop={rl['logZ']:.1f}, calls={rg['n_calls']})")
