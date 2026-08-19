# DAEDALUS

You have a radial velocity time series and you do not know how many planets are
in it. The usual route is to fit one Keplerian, then two, then three, running a
separate sampler each time and comparing the evidences afterwards. DAEDALUS
moves the question inside the sampler. Whether a component is present becomes a
sampled quantity, so one run returns the joint posterior over which components
are active and over their parameters, together with a single log-evidence
covering the whole model space.

The same holds for any model with an uncertain number of parts, whether those
are emission lines in a spectrum or predictors in a regression.

The method is Skilling's nested sampling run on the Mixture of Mutually
Singular (MoMS) parameterisation of van den Bergh et al. (2026). MoMS replaces
reversible jump's dimension matching with a fixed-dimension state space in
which excluded parameters are pinned to user-supplied off-values, so a
trans-dimensional move is an ordinary Metropolis flip of a binary indicator and
there is no Jacobian to carry. [`docs/algorithm.md`](docs/algorithm.md) derives
the kernel and the estimators.

Declare no toggleable structure and you get a plain fixed-dimensional nested
sampler, which is the configuration used for the regression tests against
analytic evidences.

Status: pre-alpha. The API will change, and there is no PyPI release yet.

## Installation

```bash
git clone https://github.com/jvines/DAEDALUS
cd DAEDALUS
pip install -e ".[all]"
```

Python 3.10 or later. The sampler itself needs only `numpy>=2.0` and
`scipy>=1.10`; everything else is optional.

| Extra | Pulls in | For |
|---|---|---|
| `fast` | numba | JIT-compiled likelihood kernels |
| `progress` | tqdm | progress bars |
| `plotting` | matplotlib, corner | the `daedalus.plotting` helpers |
| `benchmarks` | scikit-learn, astropy | the data-bearing benchmark problems |
| `transit` | batman | Mandel-Agol transit shapes |
| `reproduce` | astroquery, lightkurve, and the above | the scripts in `scripts/` |
| `dev` | pytest, mypy, ruff, black, isort | running the test suite |
| `all` | every runtime extra, not `dev` | |

Use `pip install -e ".[all,dev]"` if you want to run the tests. The display
name is capitalised, the package is not:

```python
import daedalus
```

## Running a fit

### One candidate signal

Take 120 nights of synthetic radial velocities with a single 12.3 d signal
injected, and ask whether the data support a signal at all. Copy this into a
file and run it; it takes a few seconds.

```python
import numpy as np
import daedalus

rng = np.random.default_rng(0)
t = np.sort(rng.uniform(0.0, 200.0, 120))
sigma = 2.0                                    # m/s, known measurement error
rv = 8.0 * np.sin(2 * np.pi * t / 12.3 + 1.1) + rng.normal(0.0, sigma, t.size)


def loglike(theta):
    v0, K, logP, phi = theta
    model = v0 + K * np.sin(2 * np.pi * t / np.exp(logP) + phi)
    return -0.5 * float(np.sum(((rv - model) / sigma) ** 2))


def prior_transform(u):
    return np.array([
        20.0 * u[0] - 10.0,                          # systemic velocity, m/s
        50.0 * u[1],                                 # semi-amplitude, m/s
        np.log(2.0) + u[2] * np.log(50.0),           # log period, 2 to 100 d
        2.0 * np.pi * u[3],                          # phase
    ])


# The signal is one toggleable block. When its indicator is off, the three
# parameters are pinned to the off-values; K = 0 is what removes the signal,
# the other two are inert at that point.
planet = daedalus.Group(
    name="planet",
    params=[1, 2, 3],
    off_values=np.array([0.0, 0.0, 0.0]),
    inclusion_prior=0.5,
)

sampler = daedalus.NestedSampler(
    loglike=loglike,
    prior_transform=prior_transform,
    ndim=4,
    groups=[planet],
    bound="single",
    sample="rwalk",
    n_live=400,
    periodic=[3],          # phase wraps
    seed=42,
)
results = sampler.run_nested(dlogz=0.5, n_mcmc=40, transdim_fraction=0.3,
                             show_progress=False)

on = results.gamma[:, 0]
print(f"log Z     = {results.log_Z:.2f} +/- {results.log_Z_err:.2f}")
print(f"P(planet) = {results.inclusion_probabilities()['planet']:.3f}")
print(f"period    = {np.median(np.exp(results.samples[on, 2])):.3f} d")
print(f"calls     = {results.n_calls}")
```

which prints

```
log Z     = -82.46 +/- 0.22
P(planet) = 1.000
period    = 12.281 d
calls     = 336803
```

The posterior median period is 12.281 d against an injected 12.3 d, and the
signal is present in every posterior sample. Set the injected amplitude to zero
and rerun the identical script: the same chain returns `P(planet) = 0.017`, so
the indicator is doing real work rather than saturating on whatever is in the
candidate list. Running that null case is the cheapest sanity check available
on a new problem, and it is worth doing before trusting a detection.

Two arguments carry most of the trans-dimensional configuration. `params` gives
the indices into the continuous vector that switch together, so a Keplerian
with five parameters is one `Group` with five indices rather than five groups.
`transdim_fraction` is the share of constrained-MCMC steps spent on indicator
flips rather than continuous moves; 0.3 is a reasonable starting point, and
problems with many candidate slots and a cheap likelihood tolerate much higher
values.

### Several candidate slots

The interesting question is not whether one signal is there but how many are.
Adding slots is a matter of adding groups. Here two signals are injected and
two slots are offered:

```python
import numpy as np
import daedalus

rng = np.random.default_rng(0)
t = np.sort(rng.uniform(0.0, 200.0, 120))
sigma = 2.0
rv = (8.0 * np.sin(2 * np.pi * t / 12.3 + 1.1)
      + 5.0 * np.sin(2 * np.pi * t / 31.0 - 0.4)
      + rng.normal(0.0, sigma, t.size))

n_slots = 2


def signal(theta, k):
    K, logP, phi = theta[1 + 3 * k: 4 + 3 * k]
    return K * np.sin(2 * np.pi * t / np.exp(logP) + phi)


def loglike(theta):
    model = theta[0] + sum(signal(theta, k) for k in range(n_slots))
    return -0.5 * float(np.sum(((rv - model) / sigma) ** 2))


def prior_transform(u):
    x = np.empty(u.size)
    x[0] = 20.0 * u[0] - 10.0
    for k in range(n_slots):
        x[1 + 3 * k] = 50.0 * u[1 + 3 * k]
        x[2 + 3 * k] = np.log(2.0) + u[2 + 3 * k] * np.log(50.0)
        x[3 + 3 * k] = 2.0 * np.pi * u[3 + 3 * k]
    return x


groups = [
    daedalus.Group(name=f"planet_{k}",
                   params=[1 + 3 * k, 2 + 3 * k, 3 + 3 * k],
                   off_values=np.zeros(3), inclusion_prior=0.5)
    for k in range(n_slots)
]

sampler = daedalus.NestedSampler(
    loglike=loglike, prior_transform=prior_transform, ndim=1 + 3 * n_slots,
    groups=groups, bound="single", sample="rwalk", n_live=500,
    periodic=[3, 6], seed=42,
)
results = sampler.run_nested(dlogz=0.5, n_mcmc=50, transdim_fraction=0.3,
                             show_progress=False)

print(f"log Z = {results.log_Z:.2f} +/- {results.log_Z_err:.2f}")
for gamma, p in results.model_probabilities().items():
    print(f"  {gamma}  P = {p:.3f}")
for k in range(n_slots):
    on = results.gamma[:, k]
    period = np.median(np.exp(results.samples[on, 2 + 3 * k]))
    print(f"  planet_{k}  period = {period:.2f} d")
```

```
log Z = -97.84 +/- 0.26
  (True, True)  P = 1.000
  planet_0  period = 31.05 d
  planet_1  period = 12.28 d
```

`model_probabilities()` is the posterior over whole indicator vectors, and the
two-signal configuration takes all of it. Which slot picks up which signal is
arbitrary: slot 0 landed on the 31 d signal here. Identical slots are
exchangeable, so read periods off the slots rather than assuming an ordering,
and see the note on slot-permutation trapping under Limitations before running
a blind search over many slots.

### What comes back

`run_nested` returns a `Results` object holding `samples`, `gamma`,
`log_likelihoods`, `log_weights`, `log_Z`, `log_Z_err`, `H`, `n_iter` and
`n_calls`. `inclusion_probabilities()` gives the marginal per group,
`model_probabilities()` the posterior over indicator vectors,
`model_evidences()` the per-configuration log-evidence and its error, and
`rao_blackwell_inclusion(fn)` a lower-variance inclusion estimate when an
analytic gamma-conditional is available. `save()` writes a run to disk and
`daedalus.load_results` reads it back. `daedalus.plotting` has `cornerplot`,
`traceplot`, `runplot`, `inclusion_probability_plot` and
`model_probability_plot`.

### Bounds and kernels

`bound` selects the sampling region: `'none'`, `'single'` for one ellipsoid, or
`'multi'` for a decomposition into several, which is what multimodal problems
need. The bound is refit every `n_live // 20` iterations by default; raise
`bound_update_interval` if refitting dominates a cheap likelihood.

`sample` selects the within-model kernel. Rejection sampling (`'unif'`) is only
viable at low dimension. `'rwalk'` is the general-purpose choice. `'rslice'` is
tuning-free and handles curved degeneracies at a higher cost per point.
`'de'` proposes by differential evolution from the live cloud, and is the
cheapest kernel that copes with strongly correlated parameters. You can also
pass a `Sampler` instance of your own.

`n_mcmc` is a per-replacement budget of within-model sweeps, quoted in sweeps
rather than in kernel calls so that the same number buys the same mixing effort
from every kernel. Leave it at `None` to take each kernel's dimension-aware
recommendation.

### Birth proposals

A birth proposal decides where a newly activated group starts. The default
draws from the prior, which is fine for regression-style problems and poor for
periodic ones, where the likelihood is a comb of narrow peaks in period.
`GLSPeriodBirth` and `BLSPeriodBirth` place births on periodogram structure
instead, with harmonic suppression and conditioning on the periods already
active elsewhere in the state. Custom proposals subclass `BirthProposal`.
`validate_birth_consistency` checks a proposal's density against its sampler
and runs by default at construction.

### Without groups

Leave `groups` empty and the run is ordinary nested sampling. This is also the
cheapest way to check that a likelihood and prior transform behave before any
indicators are introduced. The same run can record insertion indices, which is
the first diagnostic to reach for when a result looks wrong:

```python
import numpy as np
import daedalus

ndim, W = 5, 10.0

def loglike(theta):
    return -0.5 * float(np.dot(theta, theta)) - 0.5 * ndim * np.log(2 * np.pi)

def prior_transform(u):
    return 2.0 * W * u - W

sampler = daedalus.NestedSampler(loglike, prior_transform, ndim, bound="single",
                                 sample="rslice", n_live=400, seed=42)
indices = []
results = sampler.run_nested(dlogz=0.5, n_mcmc=25, show_progress=False,
                             insertion_recorder=indices)
print(f"log Z = {results.log_Z:.3f} +/- {results.log_Z_err:.3f}  "
      f"(analytic {-ndim * np.log(2 * W):.3f})")
test = daedalus.insertion_index_test(np.asarray(indices), n_live=400)
print(f"KS p = {test.ks_pvalue:.3f}  {test.verdict}")
```

```
log Z = -15.143 +/- 0.141  (analytic -14.979)
KS p = 0.458  consistent with uniform (no under-mixing detected)
```

A low p-value means the constrained kernel is not replacing dead points from
the true constrained prior, and the evidence should not be trusted until that
is fixed, usually by raising `n_mcmc` or changing `sample`. The test follows
Fowlie, Handley & Su (2020) and works on a single run.

## Going further

[`docs/tutorial.md`](docs/tutorial.md) works one problem from start to finish, a
spectrum with three candidate emission lines, small enough that the exact answer
is available by enumeration at every stage. It covers choosing `n_mcmc`, the
bound and the kernel, checking a run with the insertion-index test, and the
failure modes that are known. [`docs/algorithm.md`](docs/algorithm.md) derives
the kernel and the estimators.

## Benchmark problems

`daedalus.benchmarks` ships the problems the test suite runs against, each with
a `make_problem` entry point. They are the fastest way to see a working setup
for a given shape of problem:

```python
from daedalus.benchmarks import spike_slab
problem = spike_slab.make_problem(prior_half_width=10.0, inclusion_prior=0.5)
```

Available: `gaussians`, `eggbox`, `gaussian_shells`, `rosenbrock`,
`spike_slab`, `multi_spike_slab`, `polynomial_regression`, `diabetes`,
`spectroscopy`, `spectroscopy_real`, `asteroseismic`, `flares`, `wasp47`,
`sbc_toy`.

## Validation

The test suite is the specification. `pytest -m benchmark` runs everything
below; `pytest -m "benchmark and not slow"` skips the long ones. Each row
states what the test asserts, not what a good run tends to produce.

| Problem | Assertion |
|---|---|
| Gaussian, 2D and 5D | log Z within 0.5 of analytic, across the `none` and `single` bounds and the `unif`, `rwalk` and `rslice` kernels |
| Eggbox | log Z within 0.5 of analytic under `single` and under `multi` |
| Gaussian shells | log Z within 0.5 of analytic under `single` and under `multi` |
| Rosenbrock | log Z within 0.5 of the quadrature reference, under `single`/`rwalk` and `multi`/`rslice` |
| Spike and slab, one group | inclusion probability within 0.05 and log Z within 0.5 of the closed form |
| Spike and slab, two groups | over five seeds: log Z within 0.30, inclusions within 0.05, posterior means and SDs within `0.2 SD + 0.1`, and every visited configuration's log Z within 0.5 |
| Polynomial degree selection | over three seeds: active degrees 0, 1 and 3 above 0.7 and inactive degrees below 0.5; mean inclusion within 0.1 and mean E[beta] within 0.5 of enumeration |
| Diabetes, marginal JZS | all ten inclusion probabilities within 0.02 of the enumerated 1024-model reference, which is itself checked against van den Bergh et al. (2026) Table 1 |
| Diabetes, full joint | over five seeds and both birth kernels: inclusions within 0.07, E[beta] and SD[beta] within a quarter of the true posterior SD |
| Simulation-based calibration | per-coordinate fractional ranks of the true beta uniform under a KS test at p > 0.01, and the mean posterior inclusion within 3 sigma of the prior |
| Synthetic emission spectrum | injected lines above 0.9, absent candidates below 0.5 |
| SDSS DR17 star-forming galaxy | eight strong star-forming lines above 0.8 and the AGN-only He II 4686 below 0.2; [O III] 4959 and [O I] 6300 are not pinned |
| KIC 6603624 peak bagging | at least one mode above 0.9, and at least three modes in the 2700 to 3100 uHz envelope above 0.5 |
| AU Mic flares, TESS Sector 1 | all five 4-sigma candidates confirmed above 0.95 |
| WASP-47, TESS | three-slot BLS-birth run: at least two slots above 0.95, one recovering WASP-47b's period to 1 per cent and one landing on a distinct period. The single-planet run with the `de` kernel and `rejuvenate_fraction=0.3` recovers b's period to 1 per cent. Two strict xfails, described below |

The diabetes benchmark is the load-bearing trans-dimensional gate: ten
predictors, 1024 models, exact ground truth by enumeration and no tuning
freedom. The SDSS, KIC 6603624, AU Mic and WASP-47 tests are real-data
demonstrations rather than proofs of correctness.

## Limitations

The trans-dimensional kernel is constrained by the nested sampling likelihood
threshold, and this produces an ADD/DELETE asymmetry. Once `L*` rises above the
best likelihood an off-state can reach, a DELETE move can never satisfy
`L > L*` again, so an active slot cannot be switched off and `gamma = 1`
becomes absorbing. Two consequences follow.

The empirical inclusion frequency is biased upward by a few hundredths, and on
hard problems it saturates outright. On the diabetes full-joint benchmark the
worst per-predictor deviation is 0.04 against a 0.07 tolerance, which is the
benign end; the first WASP-47 xfail is the other end. There the three
literature candidates pass above 0.95, but four control periods drawn from BLS
peaks of the light curve with the literature windows masked also reach 1.000
instead of being rejected below 0.20. `Results.rao_blackwell_inclusion()`
removes the bias, but it needs an analytic gamma-conditional, which a transit
or Keplerian likelihood does not provide. `rejuvenate_fraction` does not help
here, since it re-proposes within the active model and never restores DELETE
acceptance.

A birth proposal that only fires on an OFF to ON flip also stops firing at that
point, so a slot's continuous parameters freeze wherever the last accepted
birth left them. That is the second WASP-47 xfail: in the default
single-planet configuration the planet is detected, but the period drifts to a
spurious mode roughly 2100 nats below the global optimum. `rejuvenate_fraction`
re-proposes the active block without touching the indicator and restores mode
hopping at any threshold, which is what makes the companion recovery test pass.

Blind wide-prior searches over several identical slots suffer slot-permutation
mode trapping, where the chain settles into one labelling of signals to slots
and does not leave it. The practical route is two-stage: locate candidates with
a periodogram, then confirm each under a tight prior on its own period.

## Reproducing the paper

`scripts/` holds the code behind every result in the methods paper, indexed by
section in [`scripts/README.md`](scripts/README.md). Install `.[reproduce]`
first; the scripts fetch from VizieR, MAST and SDSS on first run and write
their outputs alongside themselves.

## Citing

The methods paper has been submitted to A&A and is not yet published. Until it
appears, cite this repository together with the MoMS paper and Skilling's
nested sampling paper.

```bibtex
@article{vandenbergh2026moms,
  title   = {Reversible Jump MCMC With No Regrets: Bayesian Variable Selection
             Using Mixtures of Mutually Singular Distributions},
  author  = {van den Bergh, D. and Clyde, M.~A. and Raftery, A.~E. and
             Marsman, M.},
  journal = {arXiv preprint arXiv:2604.27791},
  year    = {2026}
}

@article{skilling2006,
  title   = {Nested Sampling for General Bayesian Computation},
  author  = {Skilling, John},
  journal = {Bayesian Analysis},
  volume  = {1},
  number  = {4},
  pages   = {833--859},
  year    = {2006},
  doi     = {10.1214/06-BA127}
}
```

## License

MIT. See `LICENSE`.
