# Running DAEDALUS

Fixed-dimensional samplers require the model up front. You state how many
components there are, and they return the posterior over that model's
parameters. DAEDALUS is for the case where the count is itself uncertain. It
runs nested sampling over a state space in which every candidate component
carries a binary indicator, so a single run returns the posterior over which
components are present, the continuous parameters conditional on that, and a
log evidence comparable across dimensionalities.

This tutorial takes one problem from start to finish: a spectrum with three
candidate emission lines, one genuinely absent and one marginal. The problem is
small enough to solve exactly by enumeration, so at every stage the sampler's
answer can be checked against the answer it should give. Later sections cover
the sampler settings, the validity check worth running on every result, and the
failure modes that are known and documented.

Written for `daedalus` 0.0.1.dev0. Status is pre-alpha and the API will change.
Every code block below was executed and its printed output pasted verbatim. The
whole tutorial is a little over a minute of compute on a laptop.

## Install

```bash
git clone https://github.com/jvines/DAEDALUS
cd DAEDALUS
pip install -e ".[all]"
```

Python 3.10 or newer. The core package needs only `numpy>=2.0` and
`scipy>=1.10`. The `[all]` extra adds `tqdm` for the progress bar, `matplotlib`
and `corner` for the plotting helpers, `numba` for the JIT likelihood kernels,
`scikit-learn` and `astropy` for the data-bearing benchmarks, and `batman` for
transit shapes. Everything in this tutorial runs on the core install except the
diabetes section, which loads its dataset through scikit-learn and so needs
`[benchmarks]`.

The numbers below come from numpy 2.5.2 and scipy 1.18.0. Seeds are fixed, so
the same environment reproduces them exactly; a different scipy or BLAS can move
the last digit.

## The problem

A 140-pixel spectrum around H-alpha with three candidate lines at known rest
wavelengths and known instrumental width. What is unknown is which of them the
data require, and how strong they are. The continuum is a free constant that is
always present.

Save this as `lines.py`. The rest of the tutorial imports from it.

```python
"""A synthetic emission-line spectrum with three candidate lines."""
import numpy as np
from scipy.stats import norm

WAVE = np.linspace(6530.0, 6600.0, 140)
CENTRES = np.array([6548.0, 6562.8, 6583.5])   # [NII], H-alpha, [NII]
LINE_SIGMA = 1.8      # instrumental width in Angstrom, known
NOISE = 0.15          # per-pixel noise, known
PRIOR_SD = 1.0        # Gaussian prior width on the continuum and every amplitude

AMP_TRUE = np.array([0.0, 1.20, 0.13])   # [NII] 6548 is genuinely absent
CONT_TRUE = 0.30

def design_matrix(wave):
    """Columns: constant continuum, then one fixed-shape profile per line."""
    cols = [np.ones_like(wave)]
    for c in CENTRES:
        cols.append(np.exp(-0.5 * ((wave - c) / LINE_SIGMA) ** 2))
    return np.column_stack(cols)

M = design_matrix(WAVE)
BETA_TRUE = np.concatenate([[CONT_TRUE], AMP_TRUE])
FLUX = M @ BETA_TRUE + NOISE * np.random.default_rng(7).standard_normal(WAVE.size)

_LOG_NORM = -0.5 * WAVE.size * np.log(2.0 * np.pi * NOISE ** 2)

def loglike(beta):
    r = FLUX - M @ beta
    return _LOG_NORM - 0.5 * float(r @ r) / NOISE ** 2

def prior_transform(u):
    return PRIOR_SD * norm.ppf(u)
```

Those two functions are what every DAEDALUS run needs, so it is worth being
explicit about their contracts.

`loglike(beta)` takes the physical parameter vector and returns a scalar log
likelihood. It may also be written `loglike(beta, gamma)` when the likelihood
needs to know which components are active; DAEDALUS inspects the signature and
passes whichever it finds. Include the normalisation constant if you want the
log evidence on an absolute scale, which is what makes it comparable to an
independent calculation.

`prior_transform(u)` takes a vector in the unit cube and returns the physical
parameters, following the usual nested-sampling convention that the prior is
whatever the transform pushes a uniform cube draw through. Here every parameter
gets a zero-mean Gaussian of unit width via `norm.ppf`. Gaussian amplitude
priors are not the most physical choice for emission lines, but they keep the
model linear-Gaussian, which is what allows the exact answer to be enumerated
later.

The amplitudes are ordered so that [NII] 6548 is absent, H-alpha is strong at
1.20, and [NII] 6584 sits at 0.13, about one and a half times the per-pixel
noise spread over a handful of pixels. That last one is the interesting case: it
is neither a detection nor a non-detection.

## 1. The continuous model on its own

Before declaring any toggleable structure, run the fixed-dimensional model with
all three lines forced on. With `groups=()` DAEDALUS is an ordinary nested
sampler, and this is the cheapest way to find out whether the likelihood and
prior transform are wired up correctly.

```python
import numpy as np
import daedalus
from lines import loglike, prior_transform

sampler = daedalus.NestedSampler(
    loglike=loglike,
    prior_transform=prior_transform,
    ndim=4,
    bound="single",
    sample="rwalk",
    n_live=250,
    seed=1,
)
res = sampler.run_nested(dlogz=0.5, show_progress=False)

print(f"log Z      = {res.log_Z:.3f} +/- {res.log_Z_err:.3f}")
print(f"iterations = {res.n_iter}, likelihood calls = {res.n_calls}")
print("posterior mean beta =", np.round(res.samples.mean(axis=0), 3))
```

```
log Z      = 68.212 +/- 0.215
iterations = 3617, likelihood calls = 83482
posterior mean beta = [0.271 0.005 1.198 0.176]
```

The exact log evidence of this four-parameter model is 68.325, computed in
step 3, so the run is 0.5 sigma low. Agreement within one or two sigma is what a
correctly configured run at this `n_live` should give. If the discrepancy is
several sigma, go to step 6 before trusting anything else.

`n_live=250` and `dlogz=0.5` are deliberately loose so that the tutorial runs
quickly. For published numbers, raise `n_live` until the answer stops moving and
tighten `dlogz` to 0.1 or below.

The posterior means already show what motivates the rest of this document. The
absent line comes back at 0.005, consistent with zero. The marginal line comes
back at 0.176, larger than its true 0.13 and carrying no indication of whether
the line is there at all. A fixed-dimensional fit cannot say "absent"; it can
only shrink an amplitude toward zero and leave you to decide.

## 2. Declaring the toggleable components

A `Group` is one component that can be switched off. It names a subset of `beta`
by index, gives the values those parameters take when the component is off, and
sets the prior probability that it is on.

```python
import numpy as np
import daedalus
from lines import loglike, prior_transform

groups = [
    daedalus.Group(name="[NII] 6548", params=[1], off_values=[0.0], inclusion_prior=0.5),
    daedalus.Group(name="H-alpha",    params=[2], off_values=[0.0], inclusion_prior=0.5),
    daedalus.Group(name="[NII] 6584", params=[3], off_values=[0.0], inclusion_prior=0.5),
]

sampler = daedalus.NestedSampler(
    loglike=loglike,
    prior_transform=prior_transform,
    ndim=4,
    groups=groups,
    bound="single",
    sample="rwalk",
    n_live=250,
    seed=1,
)
res = sampler.run_nested(dlogz=0.5, transdim_fraction=0.3, show_progress=False)

print(f"log Z = {res.log_Z:.3f} +/- {res.log_Z_err:.3f}")
print("P(line present):")
for name, p in res.inclusion_probabilities().items():
    print(f"  {name:12s} {p:.3f}")
print("model probabilities:")
for g, p in sorted(res.model_probabilities().items(), key=lambda kv: -kv[1]):
    if p < 0.005:
        continue
    on = [n for n, flag in zip(res.group_names, g) if flag]
    print(f"  {p:.3f}  {', '.join(on) if on else '(continuum only)'}")

res.save("lines_run.npz")
```

```
log Z = 69.215 +/- 0.202
P(line present):
  [NII] 6548   0.061
  H-alpha      1.000
  [NII] 6584   0.737
model probabilities:
  0.695  H-alpha, [NII] 6584
  0.245  H-alpha
  0.042  [NII] 6548, H-alpha, [NII] 6584
  0.019  [NII] 6548, H-alpha
```

The continuum, parameter index 0, belongs to no group, so it is always active.
Every parameter outside a group stays on for the whole run, and no parameter may
appear in more than one group.

`off_values` is the value each of the group's parameters takes when its
indicator is zero, which for a line amplitude is 0.0. The whole construction
rests on that off-state being a genuine "component not present" configuration;
pin a parameter at a value where the component still contributes and the model
probabilities are meaningless.

`log_Z` is the joint evidence over the union of all eight configurations,
weighted by their inclusion priors. It is not the evidence of any one model.
Step 4 gives those.

`res.save(path)` writes a `.npz` archive and `daedalus.load_results(path)` reads
it back. The next steps load `lines_run.npz` rather than re-running.

## 3. Checking against the exact answer

The model is linear in the amplitudes with Gaussian priors and Gaussian noise,
so every one of the eight configurations has a closed-form evidence: the data
are marginally normal with covariance `sigma^2 I + s^2 M_g M_g^T`, where `M_g`
holds the active columns. Save this as `exact.py`.

```python
"""Exact evidences by enumeration (the model is linear-Gaussian)."""
import itertools
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from lines import M, FLUX, NOISE, PRIOR_SD

def log_Z_model(gamma):
    cols = [0] + [1 + k for k, on in enumerate(gamma) if on]
    Mg = M[:, cols]
    cov = NOISE**2 * np.eye(FLUX.size) + PRIOR_SD**2 * (Mg @ Mg.T)
    return float(multivariate_normal.logpdf(FLUX, mean=np.zeros(FLUX.size), cov=cov))

def enumerate_models(inclusion_prior=0.5):
    rows = {}
    for gamma in itertools.product([False, True], repeat=3):
        lp = sum(np.log(inclusion_prior if g else 1 - inclusion_prior) for g in gamma)
        rows[gamma] = (log_Z_model(gamma), lp)
    log_terms = np.array([lz + lp for lz, lp in rows.values()])
    log_Z = float(logsumexp(log_terms))
    post = {g: float(np.exp(lz + lp - log_Z)) for g, (lz, lp) in rows.items()}
    incl = [sum(p for g, p in post.items() if g[k]) for k in range(3)]
    return log_Z, post, incl
```

```python
import daedalus
from exact import enumerate_models

res = daedalus.load_results("lines_run.npz")
log_Z_exact, post_exact, incl_exact = enumerate_models()

print(f"log Z   run {res.log_Z:8.3f} +/- {res.log_Z_err:.3f}   exact {log_Z_exact:8.3f}")
print()
print(f"{'group':12s} {'run':>7s} {'exact':>7s}")
for (name, p), q in zip(res.inclusion_probabilities().items(), incl_exact):
    print(f"{name:12s} {p:7.3f} {q:7.3f}")
print()
run_post = res.model_probabilities()
print(f"{'model':32s} {'run':>7s} {'exact':>7s}")
for g, q in sorted(post_exact.items(), key=lambda kv: -kv[1])[:4]:
    on = [n for n, flag in zip(res.group_names, g) if flag]
    label = ", ".join(on) if on else "(continuum only)"
    print(f"{label:32s} {run_post.get(g, 0.0):7.3f} {q:7.3f}")
```

```
log Z   run   69.215 +/- 0.202   exact   69.367

group            run   exact
[NII] 6548     0.061   0.060
H-alpha        1.000   1.000
[NII] 6584     0.737   0.743

model                                run   exact
H-alpha, [NII] 6584                0.695   0.698
H-alpha                            0.245   0.242
[NII] 6548, H-alpha, [NII] 6584    0.042   0.044
[NII] 6548, H-alpha                0.019   0.016
```

The evidence is 0.75 sigma low and every probability agrees to about 0.005,
which is around the Monte Carlo scatter of a 3600-point resample. That is what a
healthy trans-dimensional run looks like on a problem whose answer is known.

Read the result as it stands. The absent line is rejected, but not at
overwhelming odds, because a weak absent line and a weak present line are hard to
tell apart. H-alpha is certain. [NII] 6584 has a 74% chance of being real, and
the honest summary of "how many lines" is 70% two lines, 25% one line, 5% three.
That distribution is the output; collapsing it to a count throws away what the
run computed.

## 4. Per-model evidence and Bayes factors

To compare specific configurations, use `model_evidences()`. It recovers each
`Z_gamma` from the chain frequency, the joint evidence and the inclusion prior,
and returns it with an error combining the Skilling error with the binomial Monte
Carlo error on the frequency.

```python
import daedalus
from exact import log_Z_model

res = daedalus.load_results("lines_run.npz")
ev = res.model_evidences()

two = (False, True, True)     # H-alpha + [NII] 6584
one = (False, True, False)    # H-alpha only

for g in (two, one):
    lz, err = ev[g]
    print(f"{str(g):24s} log Z = {lz:7.3f} +/- {err:.3f}   exact {log_Z_model(g):7.3f}")

log_bf = ev[two][0] - ev[one][0]
print(f"\nlog BF(two lines : one line) = {log_bf:.3f}"
      f"   exact {log_Z_model(two) - log_Z_model(one):.3f}")
```

```
(False, True, True)      log Z =  70.930 +/- 0.202   exact  71.088
(False, True, False)     log Z =  69.886 +/- 0.204   exact  70.027

log BF(two lines : one line) = 1.044   exact 1.061
```

Both per-model evidences are low by about the same 0.15, because they share the
joint `log_Z`, and the Bayes factor between them is accurate to 0.02. That
cancellation is general: ratios between configurations depend only on the chain
frequencies and the priors, so they are considerably better determined than the
absolute evidences.

Configurations the chain never visited do not appear in the dictionary. Their
evidence is below the chain's resolution rather than zero. To bound one, raise
`n_live` and re-run.

## 5. Rao-Blackwellised inclusion probabilities

`inclusion_probabilities()` counts how often each indicator was on in the
posterior resample. That estimator is exposed to an asymmetry in the
trans-dimensional kernel: at high likelihood thresholds an add move only needs a
fresh continuous draw that clears the threshold, while a delete move needs the
off-state itself to clear it, which is strictly harder. The chain therefore leans
toward active configurations. The documented size of the effect is +0.03 to
+0.04, and it is the reason `tests/test_e2e_wasp47.py` carries an `xfail` on its
real-versus-control inclusion test.

`rao_blackwell_inclusion()` removes the asymmetry by marginalising the indicator
analytically at each posterior sample instead of counting it. You supply a
function returning the conditional log-odds that each component is present given
the other parameters. For this model that is a one-dimensional Gaussian
integral: with residual `r` computed with component `k` removed,
`A = ||phi_k||^2 / sigma^2 + 1 / s^2` and `B = phi_k . r / sigma^2`, the log-odds
is `log(pi / (1 - pi)) - 0.5 log(s^2 A) + B^2 / (2A)`.

```python
import numpy as np
import daedalus
from lines import M, FLUX, NOISE, PRIOR_SD
from exact import enumerate_models

PHI = M[:, 1:]                      # one column per candidate line
A = (PHI ** 2).sum(axis=0) / NOISE ** 2 + 1.0 / PRIOR_SD ** 2
LOG_ODDS_PRIOR = np.log(0.5 / 0.5)

def gamma_conditional_logits(beta, gamma):
    """log-odds that line k is present, given every other amplitude."""
    logits = np.empty(3)
    for k in range(3):
        b = beta.copy()
        b[k + 1] = 0.0                      # drop line k's own contribution
        r = FLUX - M @ b
        B = float(PHI[:, k] @ r) / NOISE ** 2
        logits[k] = (LOG_ODDS_PRIOR
                     - 0.5 * np.log(PRIOR_SD ** 2 * A[k])
                     + B ** 2 / (2.0 * A[k]))
    return logits

res = daedalus.load_results("lines_run.npz")
emp = res.inclusion_probabilities()
rb = res.rao_blackwell_inclusion(gamma_conditional_logits)
_, _, exact = enumerate_models()

print(f"{'group':12s} {'empirical':>10s} {'Rao-Black':>10s} {'exact':>8s}")
for (name, p), q, t in zip(emp.items(), rb.values(), exact):
    print(f"{name:12s} {p:10.3f} {q:10.3f} {t:8.3f}")
```

```
group         empirical  Rao-Black    exact
[NII] 6548        0.061      0.060    0.060
H-alpha           1.000      1.000    1.000
[NII] 6584        0.737      0.733    0.743
```

On this problem the two estimators agree, because the likelihood is mild enough
that deletes keep firing for most of the run and the asymmetry never builds up.
That is the expected outcome here, and it is also the point: the bias is a
property of hard problems with sharply peaked components, not a constant offset
to subtract everywhere.

Reach for the Rao-Blackwellised estimator when the components are well detected,
when the run reaches high thresholds, or when an inclusion probability saturates
at 1.0 somewhere an injected control should have been rejected. The cost is that
you must be able to write the conditional, which is analytic for conjugate models
and needs the group's within-model parameters integrated out, or
importance-sampled, otherwise. For the non-conjugate case,
`daedalus.rao_blackwell.make_is_logit_fn` builds the same callable by importance
sampling from each group's birth proposal.

## 6. Was the run valid?

Every result above is worth exactly as much as the mixing behind it, and the
quoted `log_Z_err` will not tell you when that mixing failed. Skilling's error
assumes the replacement draws are independent samples from the constrained prior.
When the within-model kernel fails to decorrelate a new live point from the one
it walked away from, the error stays small and the evidence moves.

The check is the insertion-index test of Fowlie, Handley & Su (2020). Under a
correct kernel the rank of each newborn among the current live points is uniform.
Pass `insertion_recorder=[]` to `run_nested`, then hand the list to
`daedalus.insertion_index_test`.

```python
import numpy as np
import daedalus
from lines import loglike, prior_transform

groups = [
    daedalus.Group("[NII] 6548", [1], [0.0]),
    daedalus.Group("H-alpha",    [2], [0.0]),
    daedalus.Group("[NII] 6584", [3], [0.0]),
]
recorder = []
sampler = daedalus.NestedSampler(loglike, prior_transform, ndim=4, groups=groups,
                                 bound="single", sample="rwalk", n_live=250, seed=1)
res = sampler.run_nested(dlogz=0.5, insertion_recorder=recorder, show_progress=False)

test = daedalus.insertion_index_test(np.asarray(recorder), n_live=250)
print(f"mean insertion fraction = {test.mean_fraction:.4f} "
      f"(expect 0.5 +/- {test.mean_fraction_se:.4f})")
print(f"z = {test.z_mean:+.2f}   KS p = {test.ks_pvalue:.3f}   "
      f"rolling min p = {test.rolling_min_pvalue:.3f}")
print(f"rolling windows = {test.rolling_n_windows}")
print(test.verdict)
```

```
mean insertion fraction = 0.5013 (expect 0.5 +/- 0.0050)
z = +0.26   KS p = 0.647   rolling min p = 0.050
rolling windows = 16
consistent with uniform (no under-mixing detected)
```

That is what a clean run looks like: the mean fraction sits within a fraction of
a sigma of 0.5 and the KS test is unremarkable. The rolling minimum p-value of
0.05 is the smallest of sixteen window tests, close to what sixteen draws would
give under perfect uniformity. The rolling test localises a drift in mixing
quality across a run; it is not to be read as a single p-value.

### What a bad run looks like

An eight-dimensional Gaussian with correlation 0.95 between every pair, over a
box prior wide enough that the analytic evidence is just the prior volume. The
random walk is given five sweeps per replacement instead of the default forty.

```python
import numpy as np
import daedalus

ndim, rho, W = 8, 0.95, 15.0
C = np.full((ndim, ndim), rho)
np.fill_diagonal(C, 1.0)
C_inv = np.linalg.inv(C)
_, log_det = np.linalg.slogdet(C)
NORM = -0.5 * (ndim * np.log(2.0 * np.pi) + log_det)

def loglike(x):
    return NORM - 0.5 * float(x @ C_inv @ x)

def prior_transform(u):
    return 2.0 * W * u - W

log_Z_true = -ndim * np.log(2.0 * W)   # the Gaussian is well inside the box
print(f"analytic log Z = {log_Z_true:.3f}\n")

for sample, n_mcmc in (("rwalk", 5), ("rwalk", None), ("de", None)):
    recorder = []
    s = daedalus.NestedSampler(loglike, prior_transform, ndim,
                               bound="single", sample=sample, n_live=250, seed=2)
    r = s.run_nested(dlogz=0.5, n_mcmc=n_mcmc,
                     insertion_recorder=recorder, show_progress=False)
    t = daedalus.insertion_index_test(np.asarray(recorder), n_live=250)
    print(f"sample={sample:5s} n_mcmc={str(n_mcmc):4s}  "
          f"log Z = {r.log_Z:7.3f} +/- {r.log_Z_err:.3f}  "
          f"(error {r.log_Z - log_Z_true:+.3f})  calls = {r.n_calls:6d}")
    print(f"    insertion mean {t.mean_fraction:.4f}  z {t.z_mean:+5.2f}  "
          f"KS p {t.ks_pvalue:.1e}")
    print(f"    {t.verdict}")
```

```
analytic log Z = -27.210

sample=rwalk n_mcmc=5     log Z = -24.561 +/- 0.304  (error +2.648)  calls =  33162
    insertion mean 0.5154  z +4.38  KS p 6.8e-06
    UNDER-MIXING: insertion indices NON-uniform; skew high (newborns insert near top); kernel not decorrelating newborn from donor
sample=rwalk n_mcmc=None  log Z = -27.495 +/- 0.321  (error -0.286)  calls = 278000
    insertion mean 0.5065  z +1.93  KS p 6.0e-02
    consistent with uniform (no under-mixing detected)
sample=de    n_mcmc=None  log Z = -27.162 +/- 0.318  (error +0.048)  calls = 253319
    insertion mean 0.5083  z +2.46  KS p 1.8e-02
    UNDER-MIXING: insertion indices NON-uniform; skew shape (KS reject, mean ~0.5); kernel not decorrelating newborn from donor
```

The first row is the failure. The evidence is 2.65 nats high, nearly nine times
its own quoted error, and high in the direction under-mixing always pushes: an
under-mixed chain does not shrink the prior volume as fast as the recursion
assumes. The diagnostic catches it cleanly, at z = +4.4 and a KS p-value of
7e-6. Nothing in the run's own output would have told you.

The third row shows how to read the verdict rather than obey it. The DE kernel
returns the most accurate evidence of the three, 0.05 nats from truth, and still
trips the verdict because its KS p-value is 0.018 against a default alpha of
0.05. With four thousand iterations the KS test is sensitive, and a marginal
p-value alongside a mean fraction two sigma from 0.5 is a reason to look again,
not a result to discard. The first row is what an actual failure looks like, and
the two are not close.

Record insertion indices on any run you intend to publish; the overhead is one
comparison loop per iteration. When the test fails, raise `n_mcmc` or change
kernel before touching anything else, because no number of live points fixes a
kernel that is not mixing.

### Run-to-run scatter

The distribution-free complement is to run the same problem under several seeds
and look at the spread. `daedalus.multirun_logZ_error` returns the mean and the
sample standard deviation.

```python
import numpy as np
import daedalus
from lines import loglike, prior_transform
from exact import enumerate_models

names = ("[NII] 6548", "H-alpha", "[NII] 6584")
log_Zs, marginal = [], []
for seed in range(5):
    groups = [daedalus.Group(n, [k + 1], [0.0]) for k, n in enumerate(names)]
    s = daedalus.NestedSampler(loglike, prior_transform, 4, groups=groups,
                               bound="single", sample="rwalk", n_live=250,
                               seed=seed)
    r = s.run_nested(dlogz=0.5, show_progress=False)
    log_Zs.append(r.log_Z)
    marginal.append(r.inclusion_probabilities()["[NII] 6584"])
    print(f"seed {seed}: log Z = {r.log_Z:.3f} +/- {r.log_Z_err:.3f}"
          f"   P([NII] 6584) = {marginal[-1]:.3f}")

mean, scatter = daedalus.multirun_logZ_error(log_Zs)
exact_log_Z, _, exact_incl = enumerate_models()
print(f"\nmean log Z over 5 runs   {mean:.3f}   (exact {exact_log_Z:.3f})")
print(f"run-to-run scatter       {scatter:.3f}   (quoted per-run ~0.20)")
print(f"P([NII] 6584) spread     {np.min(marginal):.3f} to {np.max(marginal):.3f}"
      f"   (exact {exact_incl[2]:.3f})")
```

```
seed 0: log Z = 69.761 +/- 0.196   P([NII] 6584) = 0.728
seed 1: log Z = 69.215 +/- 0.202   P([NII] 6584) = 0.737
seed 2: log Z = 69.580 +/- 0.197   P([NII] 6584) = 0.732
seed 3: log Z = 69.533 +/- 0.200   P([NII] 6584) = 0.781
seed 4: log Z = 69.319 +/- 0.199   P([NII] 6584) = 0.690

mean log Z over 5 runs   69.482   (exact 69.367)
run-to-run scatter       0.217   (quoted per-run ~0.20)
P([NII] 6584) spread     0.690 to 0.781   (exact 0.743)
```

Here the analytic error is honest: the measured scatter of 0.217 is consistent
with the 0.20 each run quotes for itself. Where it is not honest the scatter
exceeds it, sometimes by a large factor, and that ratio is the number to report.

The second lesson is in the last line. The marginal line's inclusion probability
wanders over 0.09 between seeds at this `n_live`, so quoting 0.737 to three
digits from step 2 would be overstating it. Inclusion probabilities carry Monte
Carlo error like any other estimate, and unlike `log_Z` the sampler does not
quote one for them.

Use the seed scatter to audit a pipeline, not as the uncertainty attached to a
published result. A published run should be a single run whose validity is
established by the insertion test.

## Choosing the settings

### n_mcmc

`n_mcmc` is the number of elementary within-model MCMC sweeps spent on each
replacement, and it is the single most important accuracy setting. The demo
above is why: at five sweeps the eight-dimensional correlated Gaussian is off by
2.65 nats, and at forty it is correct.

The default, `n_mcmc=None`, asks the kernel for a dimension-aware recommendation
rather than using a fixed number. Random-walk and differential-evolution
decorrelation times grow roughly linearly with dimension, so a constant step
count silently under-mixes as `ndim` rises and biases the evidence high. Measured
on the package's own benchmarks, `rwalk` at 25 steps carries a +0.67 evidence
bias at 20D, about three times its analytic error, and `5 * ndim` removes it.

```python
from daedalus.samplers import make_sampler

print(f"{'kernel':8s} {'sweeps/step':>12s} {'n_mcmc(4D)':>11s} {'n_mcmc(20D)':>12s}")
for name in ("unif", "rwalk", "rslice", "de"):
    k = make_sampler(name)
    print(f"{name:8s} {k.sweeps_per_step:12d} "
          f"{k.recommended_n_mcmc(4):11d} {k.recommended_n_mcmc(20):12d}")
```

```
kernel    sweeps/step  n_mcmc(4D)  n_mcmc(20D)
unif                1           1            1
rwalk               1          25          100
rslice              5          25           60
de                  1          25          100
```

The budget is quoted in sweeps, not in kernel calls. One `rslice` call performs
five sweeps, so `run_nested` divides `n_mcmc` by `sweeps_per_step` before the
loop. That conversion is what makes `n_mcmc=50` mean the same mixing effort under
every kernel, and its absence is the origin of an older claim that the slice
kernel needed 32 times dynesty's likelihood calls on the WASP-47 benchmark. Like
for like it is about 2.8 times.

Set it explicitly when there is a reason to. Doubling it and seeing the evidence
stay put is a reasonable convergence check; doubling it and seeing the evidence
drop means the shorter run was under-mixed.

### bound

The bounding region decides where proposals are drawn from. `'none'` uses the
whole unit cube, `'single'` fits one enlarged ellipsoid to the live points, and
`'multi'` clusters the live points first and fits an ellipsoid per cluster, which
is what handles separated modes.

```python
import daedalus
from daedalus.benchmarks import gaussian_shells

p = gaussian_shells.make_problem()
print(f"reference log Z = {p.log_Z_true:.3f}   ndim = {p.ndim}")
for sample in ("unif", "rwalk"):
    for bound in ("none", "single", "multi"):
        s = daedalus.NestedSampler(p.loglike, p.prior_transform, p.ndim,
                                   bound=bound, sample=sample, n_live=400, seed=5)
        r = s.run_nested(dlogz=0.5, show_progress=False)
        print(f"sample={sample:5s} bound={bound:6s} "
              f"log Z = {r.log_Z:6.3f} +/- {r.log_Z_err:.3f}   "
              f"calls = {r.n_calls:6d}")
```

```
reference log Z = -1.750   ndim = 2
sample=unif  bound=none   log Z = -1.686 +/- 0.080   calls =  26280
sample=unif  bound=single log Z = -1.686 +/- 0.081   calls =  14879
sample=unif  bound=multi  log Z = -1.774 +/- 0.081   calls =  10001
sample=rwalk bound=none   log Z = -1.816 +/- 0.082   calls =  39145
sample=rwalk bound=single log Z = -1.738 +/- 0.082   calls =  37626
sample=rwalk bound=multi  log Z = -1.717 +/- 0.081   calls =  37952
```

Under `unif` the bound is the proposal, and tightening it from the whole cube to
clustered ellipsoids cuts the likelihood calls by a factor of 2.6. Under `rwalk`
the bound only supplies the metric for the walk, so the cost barely moves. All
six agree with the reference on this problem, which is small and gentle.

Use `'multi'` as the default, which is what the constructor does. Drop to
`'single'` when the posterior is unimodal and the clustering is only overhead, as
in the line problem above. Use `'none'` only with a Markov kernel, and only when
the constrained region is expected to fill the cube; under `'unif'` an unbounded
proposal degenerates as soon as the prior volume shrinks. It is also the right
choice when there is no continuous geometry worth bounding, as in the diabetes
run below.

The bound is refit every `n_live / 20` iterations by default rather than every
iteration, which is a 2.5 to 3.5 times wall-clock speedup with no change in
likelihood-call count or evidence, since the bound only sets the proposal region.
Pass `bound_update_interval=1` to restore per-iteration refits.

### sample

`'unif'` draws from the bound and rejects until the threshold is cleared. Its
draws are independent, so `n_mcmc` does nothing, and its acceptance rate is the
fraction of the bound inside the constraint, which collapses as dimension rises
and as the bound loosens. It is viable with a tight bound on a low-dimensional
problem and nowhere else.

`'rwalk'` is an adaptive Gaussian random walk in the unit cube, using the bound's
Cholesky factor as its proposal metric. It costs about one likelihood call per
sweep and is the default.

`'rslice'` is slice sampling along random directions. It tunes its own step
length, needs no scale set by hand, and mixes well on smooth targets at three to
four likelihood calls per sweep. Its bracket width is the slice sampler's
`init_scale`, which is a global search radius rather than an efficiency knob: the
default of 1 brackets roughly the whole bounding ellipsoid, and shrinking it
turns slice sampling into a local walk that cannot hop between modes.

`'de'` proposes along the difference between two other live points. That
difference already points along whatever ridge the live cloud occupies, so DE
reads the correlation structure off the cloud rather than off the bound and
adapts to curvature for free.

The honest note on `'rwalk'` is that it is the least reliable of the Markov
kernels on strongly correlated or curved continuous subspaces, which is what
radial-velocity and transit fits look like once period, phase and eccentricity
start trading off. There it fails the insertion-index test even at the
recommended budget, and `'de'` is the fix. If a run flags under-mixing and raising
`n_mcmc` does not clear it, switch to `'de'` before doing anything else.

One measurement that is not a diagnostic: the acceptance rate. Both `'rwalk'` and
`'de'` adapt their step scale toward a target acceptance, and because the nested
sampling constraint is a step function, that controller can meet its target by
shrinking the step toward zero instead of by mixing. Measured on the WASP-47
transit benchmark under the old `scale_min` floor of 1e-5, the scale collapsed to
about 3e-5 for the majority of batches while median acceptance read exactly 0.500
with zero empty batches, and the run terminated 3105 nats below the global
optimum. The floor is now 1e-2, which prevents that particular collapse, but the
lesson stands: use the insertion-index test, not the acceptance rate.

### transdim_fraction

The probability that any given within-model step is replaced by an attempted
indicator flip. The default is 0.3.

```python
import daedalus
from lines import loglike, prior_transform

names = ("[NII] 6548", "H-alpha", "[NII] 6584")
for tf in (0.05, 0.3, 0.9):
    groups = [daedalus.Group(n, [k + 1], [0.0]) for k, n in enumerate(names)]
    s = daedalus.NestedSampler(loglike, prior_transform, 4, groups=groups,
                               bound="single", sample="rwalk", n_live=250, seed=1)
    r = s.run_nested(dlogz=0.5, transdim_fraction=tf, show_progress=False)
    p = r.inclusion_probabilities()
    print(f"transdim_fraction={tf:<5} log Z={r.log_Z:7.3f}  calls={r.n_calls:6d}  "
          f"P=[{p[names[0]]:.3f} {p[names[1]]:.3f} {p[names[2]]:.3f}]  "
          f"models visited={len(r.model_probabilities())}")
```

```
transdim_fraction=0.05  log Z= 69.681  calls= 65376  P=[0.048 1.000 0.759]  models visited=4
transdim_fraction=0.3   log Z= 69.215  calls= 73036  P=[0.061 1.000 0.737]  models visited=4
transdim_fraction=0.9   log Z= 69.525  calls= 81984  P=[0.069 1.000 0.748]  models visited=4
```

On a problem this easy the setting barely matters. All three sit within about 1.5
sigma of the exact 69.367, reach the same conclusions, and the spread between
them is the seed-to-seed scatter measured above rather than a systematic trend.

It starts to matter when there are many groups, since each flip attempt touches
one group and the chain needs enough attempts to move around the configuration
space before the likelihood threshold locks it in. Raising it costs within-model
mixing, which is the other thing the budget buys, so 0.3 is a compromise rather
than an optimum. If the chain visits only one or two configurations out of many,
raise it; if the continuous parameters look poorly mixed, lower it and raise
`n_mcmc`.

### inclusion_prior

The prior probability that a group is on, set per group, defaulting to 0.5. It is
a real prior and it moves the answer.

```python
import numpy as np
import daedalus
from lines import loglike, prior_transform
from exact import enumerate_models

for pi in (0.5, 0.1):
    groups = [daedalus.Group(n, [k + 1], [0.0], inclusion_prior=pi)
              for k, n in enumerate(("[NII] 6548", "H-alpha", "[NII] 6584"))]
    s = daedalus.NestedSampler(loglike, prior_transform, 4, groups=groups,
                               bound="single", sample="rwalk", n_live=250, seed=1)
    r = s.run_nested(dlogz=0.5, show_progress=False)
    _, _, exact = enumerate_models(inclusion_prior=pi)
    run = list(r.inclusion_probabilities().values())
    print(f"inclusion_prior = {pi}")
    print("   run   ", [f"{x:.3f}" for x in run])
    print("   exact ", [f"{x:.3f}" for x in exact])
```

```
inclusion_prior = 0.5
   run    ['0.061', '1.000', '0.737']
   exact  ['0.060', '1.000', '0.743']
inclusion_prior = 0.1
   run    ['0.010', '1.000', '0.263']
   exact  ['0.007', '1.000', '0.243']
```

Dropping the prior to 0.1 takes the marginal line from 74% to 26%, and the
sampler tracks the exact answer in both cases. A component whose posterior
probability is close to its prior is one the data have said nothing about.

Setting 0.5 everywhere is not the same as being uninformative about model size.
With `g` groups it puts uniform prior mass on all `2^g` indicator vectors, which
concentrates the prior over the number of active components at `g/2`. Where you
expect only a few of many candidates to be real, encode that with a prior below
0.5 rather than reading 50/50 as neutrality.

### n_live and dlogz

`n_live` controls resolution: the evidence error scales as `sqrt(H / n_live)`,
and the number of configurations the chain can hold simultaneously scales with
it. `dlogz` is the stopping criterion on the evidence remaining in the live
cloud. The tutorial uses 250 and 0.5 for speed. For a real trans-dimensional
problem start at 500 to 1000 live points and `dlogz=0.1`, and check that the
answer does not move when the live points are doubled.

## Groups with more than one parameter

A component is rarely a single number. A planet is a period, a semi-amplitude, a
phase and an eccentricity, and they switch on and off together. A `Group` takes
as many parameter indices as it needs, with one off-value each.

Here each line gets a free velocity shift alongside its amplitude, so each group
carries two parameters and `ndim` rises to 7. The model is no longer linear in
its parameters, so there is no closed-form check; the check is that the shifts
come back consistent with the zero they were simulated at.

```python
import numpy as np
from scipy.stats import norm
import daedalus
from lines import WAVE, FLUX, CENTRES, LINE_SIGMA, NOISE

NAMES = ("[NII] 6548", "H-alpha", "[NII] 6584")
LOG_NORM = -0.5 * WAVE.size * np.log(2.0 * np.pi * NOISE ** 2)

def model(beta):
    y = np.full_like(WAVE, beta[0])
    for k, c in enumerate(CENTRES):
        amp, shift = beta[1 + 2 * k], beta[2 + 2 * k]
        y += amp * np.exp(-0.5 * ((WAVE - c - shift) / LINE_SIGMA) ** 2)
    return y

def loglike(beta):
    r = FLUX - model(beta)
    return LOG_NORM - 0.5 * float(r @ r) / NOISE ** 2

def prior_transform(u):
    beta = np.empty(7)
    beta[0] = norm.ppf(u[0])                      # continuum, sd 1
    beta[1::2] = norm.ppf(u[1::2])                # amplitudes, sd 1
    beta[2::2] = 1.5 * norm.ppf(u[2::2])          # velocity shifts, sd 1.5 A
    return beta

groups = [daedalus.Group(name=n, params=[1 + 2 * k, 2 + 2 * k],
                         off_values=[0.0, 0.0])
          for k, n in enumerate(NAMES)]

s = daedalus.NestedSampler(loglike, prior_transform, ndim=7, groups=groups,
                           bound="multi", sample="de", n_live=300, seed=4)
res = s.run_nested(dlogz=0.5, show_progress=False)
print(f"log Z = {res.log_Z:.3f} +/- {res.log_Z_err:.3f}   calls = {res.n_calls}")
for name, p in res.inclusion_probabilities().items():
    print(f"  {name:12s} P = {p:.3f}")
on = res.gamma[:, 1]
print("H-alpha shift (A), posterior mean +/- sd = "
      f"{res.samples[on, 4].mean():+.3f} +/- {res.samples[on, 4].std():.3f}")
```

```
log Z = 66.275 +/- 0.199   calls = 111548
  [NII] 6548   P = 0.077
  H-alpha      P = 1.000
  [NII] 6584   P = 0.564
H-alpha shift (A), posterior mean +/- sd = -0.027 +/- 0.127
```

The H-alpha shift comes back at -0.03 +/- 0.13 Angstrom, consistent with the zero
it was simulated at, which is the check that the extra parameters are wired to
the right indices.

Two things moved, and both should have. The joint evidence dropped from 69.2 to
66.3, because every active line now integrates over a shift parameter it does not
need. The marginal line fell from 0.74 to 0.56 for the same reason: the Occam
factor is charged per component, so giving a component more freedom makes it
harder to justify. This is the intended behaviour of an evidence-based model
comparison, and it means a group's parameter set is a modelling decision with
consequences rather than a bookkeeping detail.

Note the indexing idiom in the last two lines. `res.gamma[:, 1]` is a boolean
mask selecting the posterior samples in which H-alpha was active, and the shift
is only meaningful in those. A parameter belonging to an inactive group sits at
its off-value, so averaging over all samples mixes the posterior with a spike.

## Interchangeable slots

The line problem so far attaches each group to a known rest wavelength. The
harder and more common case is a set of identical candidate slots, each free to
land anywhere: N possible planets, N possible peaks. This is where
trans-dimensional samplers are most useful and where they most easily mislead.

```python
import numpy as np
from scipy.stats import norm
import daedalus
from lines import WAVE, FLUX, NOISE, LINE_SIGMA

LOG_NORM = -0.5 * WAVE.size * np.log(2.0 * np.pi * NOISE ** 2)
LO, HI = 6530.0, 6600.0            # the line may sit anywhere in the window

def prior_transform(u):
    beta = np.empty(5)
    beta[0] = norm.ppf(u[0])                       # continuum
    beta[1], beta[3] = LO + (HI - LO) * u[[1, 3]]  # slot centres
    beta[2], beta[4] = norm.ppf(u[[2, 4]])         # slot amplitudes
    return beta

def loglike(beta):
    y = np.full_like(WAVE, beta[0])
    for k in range(2):
        y += beta[2 + 2 * k] * np.exp(
            -0.5 * ((WAVE - beta[1 + 2 * k]) / LINE_SIGMA) ** 2)
    r = FLUX - y
    return LOG_NORM - 0.5 * float(r @ r) / NOISE ** 2

groups = [daedalus.Group(f"slot{k}", [1 + 2 * k, 2 + 2 * k], [LO, 0.0])
          for k in range(2)]
s = daedalus.NestedSampler(loglike, prior_transform, 5, groups=groups,
                           bound="multi", sample="de", n_live=300, seed=6)
res = s.run_nested(dlogz=0.5, show_progress=False)
print(f"log Z = {res.log_Z:.3f} +/- {res.log_Z_err:.3f}")

print("per-slot P(active)  ", {k: round(v, 3)
                               for k, v in res.inclusion_probabilities().items()})
counts = res.gamma.sum(axis=1)
print("P(number of lines): ",
      {n: round(float((counts == n).mean()), 3) for n in range(3)})
centres = np.concatenate([res.samples[res.gamma[:, 0], 1],
                          res.samples[res.gamma[:, 1], 3]])
print(f"active-slot centres: median {np.median(centres):.1f} A, "
      f"16-84% [{np.percentile(centres, 16):.1f}, {np.percentile(centres, 84):.1f}]")
```

```
log Z = 64.320 +/- 0.203
per-slot P(active)   {'slot0': 0.515, 'slot1': 0.66}
P(number of lines):  {0: 0.0, 1: 0.825, 2: 0.175}
active-slot centres: median 6562.8 A, 16-84% [6562.6, 6562.9]
```

Read the per-slot line and then discard it. Slot 0 and slot 1 are identical by
construction, so the only thing separating 0.515 from 0.660 is which slot the
chain happened to use, and the difference measures label switching rather than
anything about the spectrum. Per-slot inclusion probabilities are meaningful only
when the slots are distinguishable, as they are when each is prior'd on a
different known wavelength or a different literature period.

What is meaningful is the distribution over the number of active slots, obtained
by summing `res.gamma` along its group axis, and the pooled posterior of the
active components. Both are invariant to relabelling. Here they say one line at
6562.8 Angstrom with an 84% posterior, the correct wavelength for H-alpha to the
precision quoted, and a 17% chance of a second.

The second line has become much less probable than it was in step 2, 0.175
against 0.74. Nothing about the data changed. What changed is that its centre is
now free over 70 Angstrom rather than fixed at 6583.5, so the model pays an Occam
factor of roughly the prior range over the line width for the privilege of
finding it. Where a component can be prior'd on a known position, do so; blind
searches are more expensive statistically as well as computationally.

An ordering constraint, sorting the slot centres inside `prior_transform`, is the
usual remedy for label switching in fixed-dimensional problems. It helps the
continuous mixing here too, but it does not by itself make the slot labels
interpretable, since a slot that is off sits at its off-value and falls outside
the ordering. Summarise by count and by pooled component posterior regardless.

## Ten groups at once

The line problem has three groups and eight configurations. The diabetes
benchmark of Efron et al. (2004), as set up in van den Bergh et al. (2026,
Table 1), has ten: 442 patients, 10 candidate predictors, a Bayesian linear
regression under the Jeffreys-Zellner-Siow prior. All 2^10 models are enumerable,
so the inclusion probabilities are known exactly, and this is the package's
load-bearing trans-dimensional validation gate. It needs the `[benchmarks]`
extra.

Two features of the setup transfer to other problems. First, `beta` is integrated
out analytically under the conjugate prior, so the likelihood depends only on
which predictors are active. This is the two-argument `loglike(beta, gamma)`
signature in use. Second, with nothing continuous left to sample, each `Group` is
declared with `params=[]` and an empty `off_values`; the chain then carries a
single dummy continuous coordinate, which is the minimum DAEDALUS supports, and
does all its real work on `gamma`.

```python
import daedalus
from daedalus.benchmarks import diabetes

problem = diabetes.make_problem(prior_inclusion=0.5)
groups = [daedalus.Group(**kwargs) for kwargs in problem.groups_kwargs]

sampler = daedalus.NestedSampler(
    loglike=problem.loglike,
    prior_transform=problem.prior_transform,
    ndim=problem.ndim,
    groups=groups,
    bound="none",
    sample="rwalk",
    n_live=2000,
    seed=2026,
)
res = sampler.run_nested(dlogz=0.1, n_mcmc=200, transdim_fraction=1.0,
                         show_progress=False)

inc = res.inclusion_probabilities()
print(f"log Z = {res.log_Z:.3f} +/- {res.log_Z_err:.3f}   ndim = {problem.ndim}")
print(f"{'predictor':<10}{'run':>8}{'exact':>8}{'diff':>8}")
for name in diabetes.PREDICTOR_NAMES:
    truth = problem.inclusion_prob_true[name]
    print(f"{name:<10}{inc[name]:>8.3f}{truth:>8.3f}{inc[name] - truth:>+8.3f}")

print("\nfive most probable models:")
for key, p in sorted(res.model_probabilities().items(), key=lambda kv: -kv[1])[:5]:
    active = [n for n, on in zip(diabetes.PREDICTOR_NAMES, key) if on]
    print(f"  P = {p:.3f}  {' '.join(active)}")
```

```
log Z = 136.019 +/- 0.046   ndim = 1
predictor      run   exact    diff
AGE          0.074   0.079  -0.004
SEX          0.986   0.987  -0.001
BMI          1.000   1.000  +0.000
BP           1.000   1.000  +0.000
S1           0.643   0.661  -0.018
S2           0.441   0.453  -0.012
S3           0.517   0.515  +0.002
S4           0.248   0.257  -0.009
S5           1.000   1.000  -0.000
S6           0.116   0.125  -0.009

five most probable models:
  P = 0.228  SEX BMI BP S1 S2 S5
  P = 0.206  SEX BMI BP S3 S5
  P = 0.117  SEX BMI BP S1 S4 S5
  P = 0.097  SEX BMI BP S1 S3 S5
  P = 0.057  SEX BMI BP S2 S3 S5
```

Every predictor is within 0.02 of the enumerated truth, which is the tolerance
`tests/test_e2e_diabetes.py` enforces. The largest deviations sit on the
predictors whose truth is near 0.5, where the binomial variance of the estimator
peaks; the four pinned at 0 or 1 are exact. Live points are what buy that
resolution, and this run uses 2000 of them.

Three settings differ from the runs above, each for a reason. `bound="none"`
because there is no meaningful continuous geometry to bound, only a dummy
coordinate. `transdim_fraction=1.0` because every informative move here is a
`gamma` flip. `n_mcmc=200` because with ten indicators the chain needs many flip
attempts per replacement to travel between well-separated configurations.

The five-model list is the output a marginal inclusion table hides. The posterior
is not concentrated on one model: the top two differ over whether the S1/S2 pair
or S3 carries the same information, which is exactly the correlation structure
that puts those three inclusion probabilities near 0.5.

## Reading the Results object

| Field | Contents |
|---|---|
| `samples` | `(n, ndim)` equal-weight posterior draws of `beta` |
| `gamma` | `(n, n_groups)` boolean indicators, aligned with `samples` |
| `log_likelihoods`, `log_weights` | the raw dead-point sequence, in birth order |
| `log_Z`, `log_Z_err` | joint log evidence and Skilling's error |
| `H` | information, in nats |
| `n_iter`, `n_calls`, `eff` | iterations, likelihood calls, and their ratio |
| `group_names`, `inclusion_priors` | what was declared, carried through |

`samples` and `gamma` are an importance resample of the dead points, while
`log_likelihoods` and `log_weights` are the dead points themselves. They have the
same length but they are not row-aligned. To reweight the chain yourself, work
from `log_weights + log_likelihoods - log_Z` and the dead-point arrays, not from
`samples`.

The methods are `inclusion_probabilities()`, `model_probabilities()`,
`model_evidences()`, `rao_blackwell_inclusion(fn)` and `save(path)`, with
`daedalus.load_results(path)` as the inverse. The archive is a plain `.npz` and is
portable across versions that share the field set; archives written before
`inclusion_priors` existed still load, and `model_evidences()` then raises a clear
error rather than returning wrong numbers. The likelihood and prior transform are
not serialised, so a reloaded `Results` cannot be resumed or re-run.

`daedalus.plotting` provides `cornerplot`, `traceplot`, `runplot`,
`model_probability_plot` and `inclusion_probability_plot`, all taking a `Results`.
It needs the `[plotting]` extra.

## Progress reporting

`show_progress=True` is the default and gives a tqdm bar with the running log
evidence, its error, the current likelihood threshold, the efficiency and an
estimate of the iterations remaining. Without tqdm installed it degrades silently
to no bar. For programmatic monitoring, pass `on_progress=callback` and receive an
`NSProgress` record each iteration carrying `n_iter`, `log_Z`, `log_Z_err`, `gap`,
`log_L_star`, `n_calls`, `log_dlogz` and `n_live`. The callback draws no random
numbers and evaluates no likelihoods, so a run is bit-identical with and without
one attached.

## Troubleshooting

### The evidence disagrees with an independent calculation and the error bar does not cover it

Run the insertion-index test. If it fails, the kernel is under-mixing and the
error bar is meaningless, since Skilling's error assumes independent constrained
draws. Raise `n_mcmc`, then switch to `'de'`. If it passes, the discrepancy is
more likely in the likelihood normalisation or the prior transform than in the
sampler.

### The insertion-index test fails with a high skew

DAEDALUS picks the donor uniformly from the live set, so a newborn that has not
moved far from its donor stays near a typical live point and inserts high. That is
the sign to expect here. A left skew, which is what a sampler seeding from the
killed point would produce, means the same thing. Either way the fix is more
sweeps, or a kernel that follows the geometry.

### A component is confidently detected but its parameters are wrong

This is the most serious known failure mode and it does not announce itself. A
birth proposal only fires on an off-to-on flip. The reverse move pins the group to
its off-values, so it needs the off-state to clear the current threshold, and once
the threshold climbs above what the off-state can reach, the delete move can never
be accepted again. That is correct for the evidence, since at high thresholds the
constrained posterior really does have the component on. The side effect is that
with deletes dead, births can no longer fire either, and the component is frozen
in whichever mode its one accepted birth happened to land in. Measured on WASP-47
single-planet recovery, the last accepted delete is flip 1104 of 44011, 2.5% into
the run, and the chain settles on a spurious mode about 2100 nats below the global
optimum with `P(gamma = 1) > 0.95` throughout.

The remedy is `run_nested(rejuvenate_fraction=0.3)`, which re-proposes an active
group's continuous block from its birth proposal without touching the indicator,
restoring a mode-hopping move at any threshold. Over six seeds on WASP-47 with the
DE kernel it raises global-mode recovery from 2/6 to 5/6 at 3.6% fewer likelihood
calls, though one of the six seeds ended 125 nats worse than without it. It is a
discovery aid, not a setting for calibrated inference: on the analytic
spike-and-slab, 0.3 carries a small but statistically real bias, `log Z` low by
0.058 at 4.3 sigma and `P(gamma = 1)` high by 0.022, while 0.0 is unbiased and
fractions at or below 0.1 show no detectable bias but do not fix the search. Use
it to find the mode, then re-run at zero, or under a tight confirmatory prior, to
quote the evidence.

### An inclusion probability saturates at 1.0 where it should not

The add and delete moves are not symmetric under a hard likelihood constraint, and
the empirical frequency estimator inherits that asymmetry with a documented +0.03
to +0.04 upward bias. `rao_blackwell_inclusion()` is the instrument for this. The
package's own WASP-47 end-to-end test carries a strict `xfail` on exactly this
point: the literature planets do reach `P(gamma = 1) > 0.95`, and the injected
control periods saturate rather than being rejected.

### A blind wide-prior search over several slots returns nonsense

Multi-slot searches under wide priors hit slot-permutation mode trapping, and this
is a known open limitation rather than a tuning problem. The working pattern is
two stage: pre-search with a cheap classical method to generate candidates, then
confirm each candidate under a tight prior with a trans-dimensional run. The
WASP-47 benchmark ships this as `bls_then_confirmatory_discovery`, using a
box-least-squares pre-search, and the period recovery of the single-stage default
configuration is a strict `xfail`.

### The chain only ever visits one or two configurations

Raise `transdim_fraction`, raise `n_live` so the cloud can hold more
configurations at once, and check that the off-values really do give a likelihood
that is competitive early in the run. If a component is strongly detected, an
absorbing indicator is the correct answer rather than a bug; see the frozen-mode
entry above for when it stops being harmless.

### The default birth proposal is not firing

The default is a uniform draw in the unit cube pushed through your
`prior_transform`, which is an independence proposal, so its acceptance rate is
roughly the prior volume restricted to that group. Deep into a run with an
informative likelihood that goes to zero and the trans-dimensional kernel quietly
stops working. `GaussianRWBirth` replaces it with a random walk around the
off-vector, which holds a steady acceptance rate but mixes diffusively and cannot
span the distance to a mode far from the off-vector. Neither is adequate for
narrow modes at unknown positions, such as transit or radial-velocity periods; for
those, DAEDALUS ships periodogram-informed births, `BLSPeriodBirth` and
`GLSPeriodBirth`, which place their density on periodogram peaks. A custom
`BirthProposal` must be paired with `log_prior_continuous` on the same `Group`,
because a non-uniform proposal density no longer cancels against the continuous
prior in the acceptance ratio. The constructor checks custom births for that
consistency and raises a `BirthConsistencyWarning` when the two disagree; pass
`validate_births=False` to skip the check once you are sure.

## Known limitations

The package is pre-alpha and the API will change.

Blind multi-slot searches under wide priors are limited by slot-permutation mode
trapping, and the two-stage pattern above is the current answer rather than a
workaround that will disappear.

The empirical inclusion estimator carries the +0.03 to +0.04 bias described above.
`rao_blackwell_inclusion()` removes it but requires a conditional you have to
derive.

Skilling's `log_Z_err` is a lower bound on hard or correlated problems. Where an
honest error is needed, use `daedalus.multirun_logZ_error` on the scatter of
independent runs, and check the insertion indices on each.

## Where to go next

`daedalus.benchmarks` holds every test problem the package validates against, each
with a reference answer: `gaussians`, `eggbox`, `gaussian_shells`, `rosenbrock`,
`spike_slab`, `multi_spike_slab`, `polynomial_regression`, `diabetes`,
`spectroscopy`, `spectroscopy_real`, `asteroseismic`, `flares`, `wasp47`,
`sbc_toy`. They are the fastest way to see how a real problem is set up.

```python
from daedalus.benchmarks import spike_slab

problem = spike_slab.make_problem(prior_half_width=10.0, inclusion_prior=0.5)
print(problem.log_Z_true, problem.inclusion_prob_true)
```

```
-4.489740405061735 0.11137289175635513
```

[`algorithm.md`](algorithm.md) documents the MoMS state space, the
trans-dimensional kernel and the birth-proposal choices in full.
[`../scripts/README.md`](../scripts/README.md) indexes the code behind every
result in the methods paper by section. `tests/` holds the validation gates, which
double as worked setups for the spectroscopy, asteroseismology, flare and transit
problems.