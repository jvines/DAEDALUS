# The daedalus algorithm: MoMS-NS in detail

This note derives the math underlying `daedalus`. It serves both as package
documentation and as the methods derivation for the companion paper. The
exposition follows van den Bergh, Clyde, Raftery, & Marsman (2026,
arXiv:2604.27791; hereafter VdB+26) for the MoMS construction and Skilling
(2006) for nested sampling, then bridges the two by deriving the kernel that
`daedalus` actually runs.

We assume the reader is familiar with Metropolis–Hastings, with reversible
jump (RJMCMC; Green 1995), and with the basic Skilling NS recursion
$X_i \approx \exp(-i/n_{\text{live}})$.


## 1. Problem setup

We are given:

- A continuous parameter vector $\boldsymbol{\beta} \in \mathbb{R}^{n}$.
- A grouping of the indices $\{1,\dots,n\}$ into
  - **toggleable groups** $G_1, \dots, G_g$, each declared with a
    user-supplied off-vector
    $\boldsymbol{\beta}_k^{\mathrm{off}} \in \mathbb{R}^{|G_k|}$ and an
    inclusion prior $p_k \in (0,1)$;
  - a residual set of **always-on** indices that have no toggle.
- A binary indicator vector $\boldsymbol{\gamma} \in \{0,1\}^g$. When
  $\gamma_k = 0$, the coordinates $\boldsymbol{\beta}_{G_k}$ are pinned
  to $\boldsymbol{\beta}_k^{\mathrm{off}}$; when $\gamma_k = 1$, they
  are sampled from the user's continuous prior.
- A user-supplied prior transform $T : [0,1)^n \to \mathbb{R}^n$ whose
  pushforward of the uniform measure on the cube is the user's
  continuous prior $\pi_\beta$ on $\boldsymbol{\beta}$.
- A log-likelihood $\mathcal{L}(\boldsymbol{\beta},\boldsymbol{\gamma})$.

We want, in a single run, posterior samples of
$(\boldsymbol{\beta},\boldsymbol{\gamma})$ and the marginal evidence

$$
Z = \int \mathcal{L}(\boldsymbol{\beta},\boldsymbol{\gamma})\, d\pi(\boldsymbol{\beta},\boldsymbol{\gamma}).
$$


## 2. State space

The state space and the prior measure on it can be written in two equivalent
ways. Both appear in the codebase: `samplers.py` works in u-space; the
paper's M-H derivation lives in $\beta$-space.

### 2.1 u-space formulation

Let $\mathcal{X}_u = [0,1)^n \times \{0,1\}^g$. The prior on
$(\boldsymbol{u},\boldsymbol{\gamma})$ factorises as

$$
\pi_u(d\boldsymbol{u}, \boldsymbol{\gamma}) \;=\;
d\boldsymbol{u} \;\times\;
\prod_{k=1}^{g}\bigl[p_k\,\delta_{\gamma_k,1} + (1-p_k)\,\delta_{\gamma_k,0}\bigr],
$$

with $d\boldsymbol{u}$ the Lebesgue measure on the cube. The physical
parameter vector $\boldsymbol{\beta}$ is a deterministic function of
$(\boldsymbol{u},\boldsymbol{\gamma})$,

$$
\boldsymbol{\beta}_{G_k}
= \begin{cases} T(\boldsymbol{u})_{G_k} & \gamma_k = 1 \\
                \boldsymbol{\beta}_k^{\mathrm{off}} & \gamma_k = 0 \end{cases},
$$

with always-on indices always taking $T(\boldsymbol{u})_j$. The
likelihood seen by the chain is
$\mathcal{L}(\boldsymbol{\beta}(\boldsymbol{u},\boldsymbol{\gamma}), \boldsymbol{\gamma})$.
Note that for inactive groups the $\boldsymbol{u}_{G_k}$ coordinates
are *unobserved by the likelihood* — they are present in the state
but irrelevant.

### 2.2 β-space formulation (VdB+26)

Let $\mathcal{X}_\beta = \mathbb{R}^n \times \{0,1\}^g$ with the
mutually-singular dominating measure that puts a point mass at
$\boldsymbol{\beta}_k^{\mathrm{off}}$ on the $\gamma_k = 0$ subspace
and Lebesgue measure on the $\gamma_k = 1$ subspace. The prior density
w.r.t. this measure is

$$
\pi_\beta(\boldsymbol{\beta},\boldsymbol{\gamma})
= \prod_{k=1}^{g}\Bigl[
  (1-p_k)\,\mathbf{1}\{\boldsymbol{\beta}_{G_k} = \boldsymbol{\beta}_k^{\mathrm{off}}\}
  + p_k\,\pi_{\beta,k}(\boldsymbol{\beta}_{G_k})\,
        \mathbf{1}\{\boldsymbol{\beta}_{G_k} \ne \boldsymbol{\beta}_k^{\mathrm{off}}\}\Bigr],
$$

times the always-on prior factor. Here $\pi_{\beta,k}$ is the user's
continuous prior on the $k$-th group's coordinates — equivalently, the
pushforward of $\mathrm{Uniform}[0,1)^{|G_k|}$ under $T$. This is
exactly the spike-and-slab structure of VdB+26 §2.4 generalised from
$\boldsymbol{\beta}_k^{\mathrm{off}} = \mathbf{0}$ and scalar $\beta_i$
to vector groups with arbitrary off-vectors.

### 2.3 Equivalence

The pushforward of $\pi_u$ under
$(\boldsymbol{u},\boldsymbol{\gamma}) \mapsto
(\boldsymbol{\beta}(\boldsymbol{u},\boldsymbol{\gamma}),\boldsymbol{\gamma})$
is $\pi_\beta$. The point mass at $\boldsymbol{\beta}_k^{\mathrm{off}}$
on the $\gamma_k = 0$ slice has total mass $1-p_k$, accumulated from
the entire u-cube (which all maps to the same off-vector when
$\gamma_k = 0$). The slab density $\pi_{\beta,k}$ is the pushforward
of the uniform on the cube on the $\gamma_k = 1$ slice. Detailed-
balance arguments written in either space transfer to the other by
the change-of-variables formula; we will move freely between them.


## 3. The constrained prior under NS

### 3.1 Definition

Let $\mathcal{L}^\star \in \mathbb{R}$ be a likelihood threshold. The
**constrained prior** is

$$
\pi^{\star}(d\boldsymbol{\beta}, \boldsymbol{\gamma})
= \frac{1}{X(\mathcal{L}^\star)}\,
  \pi_\beta(d\boldsymbol{\beta}, \boldsymbol{\gamma})\,
  \mathbf{1}\bigl\{\mathcal{L}(\boldsymbol{\beta},\boldsymbol{\gamma}) > \mathcal{L}^\star\bigr\},
$$

normalised by the prior volume

$$
X(\mathcal{L}^\star) = \int \mathbf{1}\{\mathcal{L} > \mathcal{L}^\star\}\,d\pi_\beta.
$$

Skilling's nested sampling replaces the live point of lowest likelihood
by a fresh draw from $\pi^\star$ at the new threshold. The recursion
$X_i \approx \exp(-i/n_{\text{live}})$ and the evidence accumulation
$\log Z = \log\sum_i X_i \mathcal{L}_i$ are independent of the
parameter-space topology, so they apply unchanged to the joint
$(\boldsymbol{\beta},\boldsymbol{\gamma})$ space (cf. `recursion.py`).
The hard part — the part that depends on the model structure — is
producing that fresh draw.

### 3.2 Sufficient kernel condition

Let $K$ be a Markov kernel on $\mathcal{X}_\beta$ that satisfies
detailed balance w.r.t. the unconstrained prior $\pi_\beta$:

$$
\pi_\beta(d\boldsymbol{x})\, K(d\boldsymbol{x}'\mid \boldsymbol{x})
= \pi_\beta(d\boldsymbol{x}')\, K(d\boldsymbol{x}\mid \boldsymbol{x}'),
\qquad \boldsymbol{x} = (\boldsymbol{\beta},\boldsymbol{\gamma}).
$$

Let $K^\star$ be the kernel obtained by rejecting any move that
violates $\mathcal{L} > \mathcal{L}^\star$,

$$
K^\star(A\mid \boldsymbol{x})
= K(A \cap \{\mathcal{L} > \mathcal{L}^\star\}\mid \boldsymbol{x})
+ \mathbf{1}\{\boldsymbol{x} \in A\}\,
  K(\{\mathcal{L} \le \mathcal{L}^\star\}\mid \boldsymbol{x}),
$$

i.e. accept only proposals that stay in the constrained region;
otherwise self-transition. Provided we start from a state with
$\mathcal{L} > \mathcal{L}^\star$ (which the live cloud guarantees),
$K^\star$ satisfies detailed balance w.r.t. $\pi^\star$:

$$
\pi^\star(d\boldsymbol{x})\, K^\star(d\boldsymbol{x}'\mid \boldsymbol{x})
= \pi^\star(d\boldsymbol{x}')\, K^\star(d\boldsymbol{x}\mid \boldsymbol{x}').
$$

This is standard (see e.g. Tierney 1994 §3.2 for the constrained-domain
version of Metropolis): the indicator
$\mathbf{1}\{\mathcal{L} > \mathcal{L}^\star\}$ cancels symmetrically
on both sides of the balance equation when the chain is restricted to
states already in the constrained region. The implication is the
operationally important one for `daedalus`:

> **Reduction.** To sample $\pi^\star$ it suffices to design a kernel
> that satisfies detailed balance w.r.t. the **unconstrained** prior
> $\pi_\beta$ and to additionally reject any proposal with
> $\mathcal{L} \le \mathcal{L}^\star$.

The likelihood ratio that appears in standard MoMS (VdB+26 Eq. 6, where
the target is the posterior) is therefore replaced in `daedalus` by the
indicator $\mathbf{1}\{\mathcal{L} > \mathcal{L}^\star\}$. The
prior-and-proposal factors of Eq. 6 are kept verbatim.

### 3.3 Joint vs per-model evidence

The Skilling recursion in `daedalus` accumulates a single scalar
log-evidence,

$$
Z = \int_{\mathcal{X}} \mathcal{L}(\boldsymbol{\beta}, \boldsymbol{\gamma})\,
    d\pi(\boldsymbol{\beta}, \boldsymbol{\gamma})
= \sum_{\boldsymbol{\gamma}} P(\boldsymbol{\gamma})\, Z_{\boldsymbol{\gamma}},
$$

reported as `Results.log_Z`. This is the **joint** evidence over the
full trans-dim state space — the prior-weighted sum of per-model
evidences $Z_{\boldsymbol{\gamma}} = \int \mathcal{L}\,d\pi_{\beta\mid\gamma}$.
It is the right quantity for comparing against a different model class
altogether (a different parameterisation, a different physical model).
It is **not** the evidence of any single inclusion configuration, and
should not be quoted as such.

Per-model evidences are recovered from the chain output via Bayes'
theorem applied to the inclusion vector,

$$
\log Z_{\boldsymbol{\gamma}}
= \log P(\boldsymbol{\gamma}\mid y)
+ \log Z
- \log P(\boldsymbol{\gamma}),
$$

where $P(\boldsymbol{\gamma}\mid y)$ is the chain's posterior frequency
of the configuration and $P(\boldsymbol{\gamma}) = \prod_k p_k^{\gamma_k}(1-p_k)^{1-\gamma_k}$
is the prior on that configuration. `daedalus` exposes this as
`Results.model_evidences`, returning a dict
`{gamma_tuple: (log_Z_gamma, log_Z_gamma_err)}` over visited
configurations. The reported error combines the joint Skilling error
with the binomial Monte-Carlo error on $\log P(\boldsymbol{\gamma}\mid y)$
in quadrature.

> **Convention.** $Z_{\boldsymbol{\gamma}}$ above is the per-model
> *marginal likelihood* $p(y\mid\boldsymbol{\gamma}) = \int \mathcal{L}\, d\pi_{\beta\mid\gamma}$,
> i.e. the standard Bayesian-model-comparison evidence for inclusion
> configuration $\boldsymbol{\gamma}$, **excluding** the inclusion prior
> $P(\boldsymbol{\gamma})$. The joint evidence then decomposes as
> $Z = \sum_{\boldsymbol{\gamma}} P(\boldsymbol{\gamma}) Z_{\boldsymbol{\gamma}}$.
> Some literature usage instead bundles the inclusion prior into the
> per-model evidence ($Z'_{\boldsymbol{\gamma}} \equiv P(\boldsymbol{\gamma}) Z_{\boldsymbol{\gamma}}$,
> so that $Z = \sum_{\boldsymbol{\gamma}} Z'_{\boldsymbol{\gamma}}$); the two
> conventions differ by exactly $\log P(\boldsymbol{\gamma})$ per
> configuration. `Results.model_evidences` returns the former.

Bayes factors between trans-dim configurations need only the chain's
inclusion-vector frequencies (the joint $Z$ cancels),

$$
\log \mathrm{BF}(\boldsymbol{\gamma}_1, \boldsymbol{\gamma}_2)
= \log \frac{P(\boldsymbol{\gamma}_1\mid y)/P(\boldsymbol{\gamma}_1)}
            {P(\boldsymbol{\gamma}_2\mid y)/P(\boldsymbol{\gamma}_2)}.
$$


## 4. The trans-dim kernel: MoMS flips

A MoMS flip on group $k$ is a Metropolis step that toggles $\gamma_k$
and either draws (ADD: $0 \to 1$) or pins (DELETE: $1 \to 0$) the
corresponding $\boldsymbol{\beta}_{G_k}$. `daedalus` implements three
concrete kernels; all are special cases of the same M-H machinery on
$\mathcal{X}_\beta$.

We adopt the notation $R$ for the M-H ratio and write the inclusion
factor $p_k/(1-p_k)$ separately because every variant carries it.

### 4.1 Uniform-u birth (default)

Forward proposal (ADD):

- Draw $\boldsymbol{u}_{G_k}^\star \sim \mathrm{Uniform}[0,1)^{|G_k|}$.
- Push through $T$ to obtain
  $\boldsymbol{\beta}_{G_k}^\star = T(\boldsymbol{u}^\star)_{G_k}$.
- Set $\gamma_k = 1$.

Reverse (DELETE) is deterministic: pin
$\boldsymbol{\beta}_{G_k} = \boldsymbol{\beta}_k^{\mathrm{off}}$ and
set $\gamma_k = 0$.

The forward proposal density on the slab in $\beta$-space is the
pushforward of uniform on the cube — i.e. the user's continuous prior
itself, $q_{\mathrm{fwd}}(\boldsymbol{\beta}_{G_k}^\star) = \pi_{\beta,k}(\boldsymbol{\beta}_{G_k}^\star)$.
The reverse density on the spike sub-measure is $q_{\mathrm{rev}} = 1$
(the spike is a point mass, "density 1" in its singular base measure).
Substituting into VdB+26 Eq. 6 with the likelihood ratio replaced by
the constraint indicator (cf. §3.2):

$$
R_{\mathrm{ADD}}^{\,\mathrm{u}}
= \frac{p_k\,\pi_{\beta,k}(\boldsymbol{\beta}_{G_k}^\star)\cdot 1}
       {(1-p_k)\cdot 1\cdot \pi_{\beta,k}(\boldsymbol{\beta}_{G_k}^\star)}
= \frac{p_k}{1-p_k}.
$$

The continuous prior cancels exactly. The acceptance probability is

$$
\alpha_{\mathrm{ADD}}^{\,\mathrm{u}}
= \min\!\Bigl\{1,\;\frac{p_k}{1-p_k}\Bigr\}\cdot
  \mathbf{1}\bigl\{\mathcal{L}(\boldsymbol{\beta}^\star,\boldsymbol{\gamma}^\star)
                   > \mathcal{L}^\star\bigr\}.
$$

The DELETE direction is symmetric:
$\alpha_{\mathrm{DELETE}}^{\,\mathrm{u}} = \min\{1,(1-p_k)/p_k\}\cdot \mathbf{1}\{\mathcal{L}^\star \text{ constraint}\}$.

This recovers the operational simplification noted in VdB+26 §2.5
(page 14): when the auxiliary-variable proposal is the model's own
prior under the candidate model, the acceptance ratio collapses to
the prior ratio. In `daedalus` this is implemented in
`sampler.py:_moms_flip_once` along the `custom_birth = False` path.

### 4.2 GaussianRWBirth (paper-faithful Algorithm 2)

Forward proposal (ADD):
$\boldsymbol{\beta}_{G_k}^\star \sim \mathcal{N}(\boldsymbol{\beta}_k^{\mathrm{off}}, \mathrm{diag}(\boldsymbol{\tau}_k^2))$.
Reverse: deterministic DELETE.

Centering on the off-vector coincides with VdB+26 Algorithm 2's
"centered on current $\beta_i$" rule at the moment of an ADD move:
the chain is at $\gamma_k = 0$ and therefore at
$\boldsymbol{\beta}_{G_k} = \boldsymbol{\beta}_k^{\mathrm{off}}$. The
M-H ratio is

$$
R_{\mathrm{ADD}}^{\,\mathrm{RW}}
= \frac{p_k\,\pi_{\beta,k}(\boldsymbol{\beta}_{G_k}^\star)\cdot 1}
       {(1-p_k)\cdot 1\cdot
        \mathcal{N}(\boldsymbol{\beta}_{G_k}^\star;\,
                    \boldsymbol{\beta}_k^{\mathrm{off}},
                    \mathrm{diag}(\boldsymbol{\tau}_k^2))}
= \frac{p_k}{1-p_k}\cdot
  \frac{\pi_{\beta,k}(\boldsymbol{\beta}_{G_k}^\star)}
       {\mathcal{N}(\boldsymbol{\beta}_{G_k}^\star;\,
                    \boldsymbol{\beta}_k^{\mathrm{off}},
                    \mathrm{diag}(\boldsymbol{\tau}_k^2))},
$$

so the user must supply $\pi_{\beta,k}$ explicitly (this is the
`Group.log_prior_continuous` argument). The DELETE direction evaluates
the same Gaussian density at the *current* slab value
$\boldsymbol{\beta}_{G_k}$ and divides into rather than out of the
ratio; this is the customary M-H symmetry, implemented in
`sampler.py:_moms_flip_once` along the `custom_birth = True` path.

For scalar groups ($|G_k| = 1$) and
$\boldsymbol{\beta}_k^{\mathrm{off}} = 0$, this is exactly VdB+26
Algorithm 2. Vector groups and non-zero off-vectors are extensions of
the paper; the M-H derivation above shows they require no additional
machinery beyond what VdB+26 gives.

The proposal scale $\boldsymbol{\tau}_k$ is adapted via Robbins–Monro
(VdB+26 Eq. D1):

$$
\log \boldsymbol{\tau}_k^{(t+1)}
= \log \boldsymbol{\tau}_k^{(t)}
  + (t+1)^{-\phi}\,
    \bigl(\mathbf{1}\{\text{accept at step } t\} - \alpha_{\mathrm{target}}\bigr),
$$

with $\phi = 0.75$, $\alpha_{\mathrm{target}} = 0.44$ for $|G_k| = 1$
and $\alpha_{\mathrm{target}} = 0.234$ for vector groups (Roberts &
Rosenthal 2001 optimal scaling). `daedalus` runs the update on every
flip attempt (ADD or DELETE) involving $G_k$, since both directions
evaluate the same Gaussian scale — see `birth_proposals.GaussianRWBirth.update`.

### 4.3 Custom (informed) birth proposals

A user may supply any pair of densities $(q_{\mathrm{fwd}}, q_{\mathrm{rev}})$
defining a between-model proposal on the slab. The `BirthProposal`
Protocol exposes `propose` (sampling + forward density) and
`log_density` (reverse density at an arbitrary point); `daedalus`
plugs them into the same Eq. 6 ratio:

$$
R_{\mathrm{ADD}}^{\,\mathrm{custom}}
= \frac{p_k}{1-p_k}\cdot
  \frac{\pi_{\beta,k}(\boldsymbol{\beta}_{G_k}^\star)}
       {q_{\mathrm{fwd}}(\boldsymbol{\beta}_{G_k}^\star)},
$$

with the symmetric DELETE form. The user-supplied
`Group.log_prior_continuous` provides $\log\pi_{\beta,k}$. The
package automatically checks self-consistency at construction time
via `daedalus.validate_birth_consistency`: for each `Group` carrying
a custom `birth_proposal`, `NestedSampler` confirms

$$
\log\pi_{\beta,k}\bigl(T(\boldsymbol{u})|_{G_k}\bigr)
= -\log\bigl|\det\partial T_{G_k}/\partial \boldsymbol{u}_{G_k}\bigr|
$$

at a few hundred $\boldsymbol{u}$ draws under the standard
*coordinate-separable* assumption on `prior_transform` (the dynesty
convention: each output coord depends on one input coord). A
mismatch — wrong constant, wrong support, wrong shape, or coordinate
coupling — raises a `daedalus.BirthConsistencyWarning` with an
explanatory diagnostic, catching the silent miscalibration that
would otherwise go unnoticed. Pass `validate_births=False` to
`NestedSampler` to disable the check when its assumptions don't hold
(e.g. coupled multivariate Cholesky prior_transforms); call
`daedalus.validate_birth_consistency` directly to retrieve full
per-sample residuals.

Informed proposals shipped with `daedalus`:

- **`PriorDrawBirth`** — independent draw from a user-supplied prior
  sampler. When the supplied sampler equals $\pi_{\beta,k}$, the M-H
  ratio collapses to the inclusion ratio, recovering uniform-u up to
  numerical detail.
- **`BLSPeriodBirth`** — Box Least Squares periodogram-informed birth
  for *transit-style* periodic signals. Each candidate slot's
  coordinates are $(\log P, *)$ where the period sits at a configurable
  position; the proposal samples $\log P$ from peaks of a BLS
  spectrum of the *residuals* ($y - \text{model}(\beta_{\mathrm{active}})$),
  mixed with a uniform-on-prior fallback at weight $1 - \alpha$. The
  CDF is cached per `state.gamma`, so successive identical-gamma
  calls pay no periodogram cost. Optional harmonic suppression knocks
  down density near every active slot's $P/3, P/2, P, 2P, 3P$ to
  prevent multi-slot lockup on a single signal's harmonics. Used for
  the WASP-47 transit demo.
- **`GLSPeriodBirth`** — same machinery with Generalised Lomb–Scargle
  in place of BLS. Appropriate for *sinusoidal* signals (radial
  velocities, photometric variability, eccentric-orbit RVs with
  `nterms >= 2`) where BLS's box-shape assumption is wrong.

Both periodogram births require `astropy` at call time (lazily
imported); the package itself does not depend on `astropy`.


## 5. Within-model kernel

For an NS replacement, `daedalus` mixes $n_{\mathrm{mcmc}}$ steps:
each step is a MoMS flip with probability $f_{\text{td}}$
(`transdim_fraction`) and a within-model continuous step otherwise.
Within-model steps update $\boldsymbol{u}$ while keeping
$\boldsymbol{\gamma}$ fixed; the off-value override ensures
$\boldsymbol{\beta}_{G_k} = \boldsymbol{\beta}_k^{\mathrm{off}}$ for
inactive groups regardless of the proposed $\boldsymbol{u}$.

The within-model step itself is a standard NS constrained-MCMC kernel
— RW, slice, DE, or rejection from the bound. The prior on
$\boldsymbol{u}$ is uniform on the cube, so for symmetric proposals
(RW, DE) the M-H acceptance reduces to
$\mathbf{1}\{\boldsymbol{u}^\star \in [0,1)^n\}\cdot \mathbf{1}\{\mathcal{L} > \mathcal{L}^\star\}$
(cf. `samplers.py`). This machinery is independent of MoMS and reuses
the conventions of dynesty (Speagle 2020), MultiNest (Feroz+ 2009),
and PolyChord (Handley+ 2015).

VdB+26 Algorithm 2 uses Gibbs steps on the slab for the within-model
update; `daedalus` uses generic black-box constrained MCMC instead
because Gibbs requires conjugacy. This is the only point at which
`daedalus` deviates from VdB+26 within the variable-selection setting
where Gibbs is applicable; the trans-dim kernel itself is faithful
(§4.2) or a documented extension (§4.1, §4.3).

> **Warning — the acceptance rate is not a sufficient mixing diagnostic
> for the RW and DE kernels.** Both adapt their step scale toward a target
> acceptance (`target_accept`) by a multiplicative Robbins-Monro update.
> Because the NS constraint $\mathbf{1}\{\mathcal{L} >
> \mathcal{L}^\star\}$ is a *step* function, that controller has a
> degenerate solution: on a razor-thin likelihood ridge it can meet its
> target by shrinking the step toward zero rather than by mixing.
> Arbitrarily small steps barely change $\mathcal{L}$, so they are
> accepted at whatever rate is demanded while the walker stays put, and
> the acceptance statistic reads perfectly healthy while the chain is
> frozen.
>
> Measured on the WASP-47 transit benchmark before this was guarded:
> median acceptance exactly 0.500 against a 0.5 target, zero empty
> batches — and a step scale pinned at $10^{-5}$ for the majority of
> batches, with the run terminating 3105 nats below the true global
> optimum. The guard is the `scale_min` floor ($10^{-2}$, i.e. steps at
> least 1% of the bound's extent) in both `RandomWalkSampler` and
> `DifferentialEvolutionSampler`; $10^{-3}$ was measured and is not
> sufficient.
>
> This failure mode is specific to samplers with a hard constraint. Under
> a smooth target, acceptance $\to 1$ as the step $\to 0$, so the
> controller would *grow* the step and self-correct. When diagnosing a
> suspected mixing problem, log the **scale trajectory** and the fraction
> of batches sitting at `scale_min`, not the acceptance rate alone. The
> insertion-index test (`diagnostics`) is the rigorous check.


## 6. How to choose a birth proposal

### 6.1 Failure modes

Both default kernels (uniform-u and GaussianRWBirth) are valid. They
differ in *which proposal density goes into Eq. 6* and consequently
in where each fails:

- **Uniform-u** is an *independence Metropolis* sampler: forward
  proposal is $\pi_{\beta,k}$, target is
  $\pi_{\beta,k}\cdot \mathbf{1}\{\mathcal{L} > \mathcal{L}^\star\}$.
  Acceptance rate $\approx \Pr_{\pi_{\beta,k}}(\mathcal{L} > \mathcal{L}^\star)$
  = the prior volume $X$ restricted to group $k$. As $X$ shrinks
  (deep into the NS run), independence proposals stop landing in the
  constrained region and the trans-dim kernel quietly stops firing.
  This is the failure mode under **informative likelihoods**.
- **GaussianRWBirth** is *random-walk Metropolis* around the
  off-vector. Acceptance rate is steady (≈0.234–0.44 with adaptation)
  but mixing is diffusive. To reach a posterior mode at distance $d$
  from the off-vector, the chain needs $\sim (d/\tau)^2$ steps each
  clearing $\mathcal{L}^\star$. This is the failure mode when the
  **off-vector is far from the relevant mode** (transit periods, GW
  frequencies, asteroseismic modes, anything where no natural off
  sits in a high-likelihood basin).

The two failure modes are orthogonal — neither dominates. Concretely:

| Problem | Posterior shape | uniform-u | GaussianRWBirth |
|---|---|---|---|
| Diabetes (JZS, 10 preds) | broad, centered ≈ 0 | works | works (paper-faithful) |
| Spike-and-slab synthetic | ≈ prior | works | works |
| WASP-47 transit recovery | narrow spike at truth | fails ($X \ll 1$) | fails (RW from off can't span) |
| Multi-line spectroscopy | several narrow modes | fails | fails |
| Asteroseismic peak bagging | narrow modes near identified peaks | fails | fails |

Three of the five real-data demos in `daedalus` genuinely require an
informed birth (§4.3). The default choice is irrelevant for them.

#### A third failure mode: the birth channel closes

The two failure modes above are both about *whether a birth is accepted*.
There is a third, orthogonal to both, about *whether a birth is still
available at all* — and it defeats even a well-tuned informed birth.

A birth proposal only fires on an OFF $\to$ ON flip. The reverse move pins
$\boldsymbol{\beta}_{G_k}$ to $\boldsymbol{\beta}_k^{\mathrm{off}}$, so it
must satisfy $\mathcal{L}(\boldsymbol{\beta}^{\mathrm{off}}) >
\mathcal{L}^\star$. Once $\mathcal{L}^\star$ climbs above the best
likelihood the off-state can reach — which happens early for any
well-detected component — the delete move can *never* be accepted again.
$\gamma_k = 1$ becomes absorbing.

That is correct for the evidence: at high $\mathcal{L}^\star$ the
constrained posterior genuinely has $\gamma_k = 1$. But the side effect is
not benign. With deletes dead, births can never re-fire, so the group's
continuous block can thereafter move only by local within-model steps —
which cannot cross a likelihood valley. The component is frozen in
whichever mode its one accepted birth happened to land in, however early
and however poor.

Measured on WASP-47 single-planet recovery: the last accepted delete is
flip 1104 of 44011 (2.5% into the run), and of 168 accepted births only 3
were near the true period, because the $\pi/q$ correction in Eq. 6
systematically favours low-density *tail* proposals over the informative
periodogram peak while $\mathcal{L}^\star$ is still low. The chain then
drifts to a spurious mode ~2100 nats below the global optimum, with
$P(\gamma = 1) > 0.95$ throughout — the component is confidently detected,
and its parameters are wrong.

The remedy is a within-model **rejuvenation** move
(`run_nested(rejuvenate_fraction=...)`): re-propose an already-active
group's block from the same `BirthProposal` *without* touching
$\boldsymbol{\gamma}$, as an M-H move whose reverse density is evaluated
against the post-move state. This restores a mode-hopping proposal at any
$\mathcal{L}^\star$ for one likelihood call per attempt. On WASP-47 over 6
seeds it raises global-mode recovery from 2/6 to 5/6 at 3.6% *fewer*
likelihood calls.

It is a discovery aid, not a calibrated-inference setting. On the analytic
spike-and-slab the fraction that fixes the mode search (0.3) also carries a
small but statistically real bias ($\log Z$ $-0.058$, 4.3$\sigma$;
$P(\gamma=1)$ $+0.022$), while 0.0 is unbiased; fractions $\le 0.1$ show no
detectable bias but do not fix the search. Rejuvenate to *find* the mode,
then re-run at 0 — or with a confirmatory tight prior — to quote evidence.

This failure mode is a property of trans-dimensional samplers with a hard
likelihood constraint. A parallel-tempered trans-dim sampler is
structurally protected: hot chains see a softened likelihood, still accept
deletes, and swaps propagate that mobility down to $\beta = 1$ — provided
the ladder actually reaches a temperature where deletes are accepted, which
is worth measuring per rung rather than assuming.

### 6.2 Defaults

`daedalus` ships **uniform-u** as the default `birth_proposal=None`
for three reasons:

1. **No tuning required.** Uniform-u needs nothing from the user
   beyond `prior_transform`. GaussianRWBirth as the default would
   force every user to also supply `log_prior_continuous` and a
   starting $\boldsymbol{\tau}_k$.
2. **Robust to off-vector choice.** GaussianRWBirth around a
   poorly-chosen off can starve the chain. Uniform-u is invariant to
   the off-vector (only the spike location moves; the slab proposal
   is unchanged).
3. **In-paper precedent.** The independence-from-prior variant is
   exactly the simplification VdB+26 §2.5 (page 14) calls out as the
   "Gibbs step" case of MoMS: when the proposal density equals the
   model prior, the acceptance ratio collapses to the inclusion
   ratio. `daedalus` adopts this as the default; users wanting
   literal Algorithm 2 pass `birth_proposal=GaussianRWBirth(scale=...)`.

For the diabetes benchmark (the load-bearing trans-dim validation
against VdB+26 Table 1) `daedalus` runs both kernels and is required
to match enumeration to within the documented tolerance under each.
See `tests/test_e2e_diabetes.py`.


## 7. References

- van den Bergh, D., Clyde, M. A., Raftery, A. E., & Marsman, M.
  (2026). *Reversible Jump MCMC With No Regrets: Bayesian Variable
  Selection Using Mixtures of Mutually Singular Distributions.*
  arXiv:2604.27791.
- Skilling, J. (2006). *Nested Sampling for General Bayesian
  Computation.* Bayesian Analysis 1(4), 833–859.
- Green, P. J. (1995). *Reversible Jump Markov Chain Monte Carlo
  Computation and Bayesian Model Determination.* Biometrika 82,
  711–732.
- Roberts, G. O. & Rosenthal, J. S. (2001). *Optimal Scaling for
  Various Metropolis–Hastings Algorithms.* Statistical Science 16,
  351–367.
- Roberts, G. O. & Rosenthal, J. S. (2009). *Examples of Adaptive
  MCMC.* Journal of Computational and Graphical Statistics 18(2),
  349–367.
- Tierney, L. (1994). *Markov Chains for Exploring Posterior
  Distributions.* Annals of Statistics 22(4), 1701–1762.
- Speagle, J. S. (2020). *dynesty: a dynamic nested sampling package
  for estimating Bayesian posteriors and evidences.* MNRAS 493(3),
  3132–3158.
