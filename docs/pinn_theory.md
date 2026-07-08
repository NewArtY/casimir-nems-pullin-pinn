# PINN Theory and the Rapidity Substitution for the Pull-In Pole

Theoretical justification for the physics-informed neural-network (PINN)
treatment of the Casimir–electrostatic NEMS oscillator and for the **rapidity
substitution** `ξ = 1 − e^{−θ}` used to regularize the movable pull-in
singularity. Companion to `physics_notes.md` (derivation of the model) and
`bifurcation_notes.md` (equilibrium/fold structure). Over-dot denotes `d/dτ`
throughout, `τ = ω₀ t` the dimensionless time.

The governing equation (see `physics_notes.md §1.2`) is

$$
\ddot\xi + 2\zeta\,\dot\xi + \xi
= \frac{\alpha}{(1-\xi)^2} + \frac{\beta}{(1-\xi)^4},
\qquad \xi\in[0,1),\quad \alpha,\beta,\zeta\ge 0 .
\tag{Ξ}
$$

Above the fold (`bifurcation_notes.md §3`) no equilibrium exists in `[0,1)`; the
trajectory started from rest accelerates monotonically into the boundary and the
gap `1−ξ` closes at a **finite** time `τ_*`. This finite-time approach to
`ξ = 1` is a *movable-pole* singularity (its location `τ_*` depends on the
initial data and on `α, β, ζ`, not on the equation alone), and it is what makes
(Ξ) awkward to integrate and to approximate. The remainder of this note (i)
introduces the rapidity coordinate that pushes the pole to infinity, (ii)
derives the transformed ODE, (iii) states precisely why a mesh-free global
surrogate is preferable to a *fixed-step* classroom integrator near the pole
while being honest that a good *adaptive* integrator does not fail, (iv)
discusses loss weighting and collocation, and (v) records a citable Proposition
and Definition block for the article.

---

## 1. The rapidity substitution

### 1.1 Definition and bijectivity

Define the **rapidity** coordinate `θ` by

$$
\boxed{\;\xi = 1 - e^{-\theta}\;}\qquad\Longleftrightarrow\qquad
\theta = -\log(1-\xi),\qquad \theta\ge 0 .
\tag{R}
$$

Equivalently the dimensionless gap `u = 1-\xi` (used throughout the bifurcation
analysis) is the pure exponential

$$
u \equiv 1-\xi = e^{-\theta}.
$$

**Proposition-grade bijectivity.** The map `(R)` is a `C^\infty`
diffeomorphism of `[0,1)` onto `[0,\infty)`:

- it is strictly increasing, since `d\xi/d\theta = e^{-\theta} = 1-\xi > 0` on
  `[0,1)`;
- endpoints: `\xi=0 \leftrightarrow \theta=0`, and `\xi\to 1^- \leftrightarrow
  \theta\to +\infty`;
- the inverse `\theta=-\log(1-\xi)` is smooth on `[0,1)` because `1-\xi>0` there.

Thus **the entire admissible displacement interval `\xi\in[0,1)` is unfolded onto
the whole half-line `\theta\in[0,\infty)`**, and the pull-in barrier `\xi=1` — a
finite coordinate value sitting at the very edge of the state domain — is mapped
to `\theta=+\infty`. The "wall" is removed from any bounded search region: a
network with bounded output that must represent `\xi(\tau)` pressing flat against
`\xi=1` instead represents `\theta(\tau)` climbing a ramp that is monotone,
smooth, and merely *tall*, not *walled*.

### 1.2 Transformed derivatives

Differentiate `(R)` with respect to `\tau`:

$$
\dot\xi = e^{-\theta}\,\dot\theta,
\qquad
\ddot\xi = \frac{d}{d\tau}\!\big(e^{-\theta}\dot\theta\big)
         = e^{-\theta}\big(\ddot\theta - \dot\theta^{\,2}\big).
\tag{D}
$$

(The `-\dot\theta^{2}` term is the Jacobian curvature of the nonlinear chart; it
is the only "new" nonlinearity the substitution introduces on the left-hand
side.)

### 1.3 The transformed ODE

Insert `(D)` and the identities `\;1/(1-\xi)^2 = e^{2\theta}`,
`\;1/(1-\xi)^4 = e^{4\theta}`, `\;\xi = 1-e^{-\theta}` into `(Ξ)`:

$$
e^{-\theta}\big(\ddot\theta - \dot\theta^{2}\big)
 + 2\zeta\,e^{-\theta}\dot\theta
 + \big(1-e^{-\theta}\big)
 = \alpha\,e^{2\theta} + \beta\,e^{4\theta}.
$$

Multiply through by `e^{\theta}` (allowed, `e^{\theta}\ne 0`), and use
`e^{\theta}(1-e^{-\theta}) = e^{\theta}-1`:

$$
\ddot\theta - \dot\theta^{2} + 2\zeta\dot\theta + \big(e^{\theta}-1\big)
= \alpha\,e^{3\theta} + \beta\,e^{5\theta}.
$$

Solving for `\ddot\theta`:

$$
\boxed{\;
\ddot\theta = \dot\theta^{2} - 2\zeta\,\dot\theta
             - \big(e^{\theta}-1\big)
             + \alpha\,e^{3\theta} + \beta\,e^{5\theta}
\;}
\tag{Θ}
$$

i.e. `\ddot\theta = f(\theta,\dot\theta;\alpha,\beta,\zeta)` with

$$
f(\theta,\dot\theta) = \dot\theta^{2} - 2\zeta\dot\theta
   - e^{\theta} + 1 + \alpha e^{3\theta} + \beta e^{5\theta}.
$$

**Key structural fact.** The right-hand side `f` is a *finite sum of
exponentials and a quadratic in `\dot\theta`*: it is a **real-analytic (entire)
function of `(\theta,\dot\theta)` on all of `\mathbb R^2`**. It has **no pole at
any finite `\theta`**, in contrast to `(Ξ)`, whose vector field has a genuine
pole at the finite value `\xi=1`. The singular denominators `(1-\xi)^{-2}` and
`(1-\xi)^{-4}` have been traded for the *entire* factors `e^{3\theta}` and
`e^{5\theta}`. This is the analytic content of the regularization and is the
basis of Proposition 1 (§4).

**Consistency check (static limit).** Setting `\ddot\theta=\dot\theta=0` in `(Θ)`
gives `e^{\theta}-1 = \alpha e^{3\theta}+\beta e^{5\theta}`; multiplying by
`e^{-5\theta}` and writing `u=e^{-\theta}` returns
`u^4 - u^5 = \alpha u^2 + \beta`, which is exactly the equilibrium relation `(E)`
of `physics_notes.md §3` and `bifurcation_notes.md §2`. The coordinate change
preserves the fixed-point set, as it must.

**Initial conditions in `\theta`.** From `(R)`–`(D)`, an initial state
`(\xi_0,\dot\xi_0)` maps to

$$
\theta_0 = -\log(1-\xi_0),
\qquad
\dot\theta_0 = \frac{\dot\xi_0}{1-\xi_0} = \dot\xi_0\,e^{\theta_0}.
$$

The standard "released from rest at the flat electrode" datum `\xi_0=0`,
`\dot\xi_0=0` becomes simply `\theta_0=0`, `\dot\theta_0=0`.

### 1.4 The special-relativity analogy — precise and where it is only formal

In special relativity the physical velocity is a *bounded* quantity obtained
from an *unbounded* additive coordinate, the **rapidity** `\varphi`:

$$
\beta_v \equiv \frac{v}{c} = \tanh\varphi,\qquad \varphi\in[0,\infty)\ \to\ \beta_v\in[0,1),
$$

with the light cone `v=c` reached only as `\varphi\to\infty`, and the Lorentz
factor `\gamma = \cosh\varphi`, `1/\gamma = \operatorname{sech}\varphi`, playing
the role of the "gap" that vanishes at the barrier.

**The exact part of the analogy.** Both `(R)` and `\beta_v=\tanh\varphi` are
strictly increasing `C^\infty` bijections of the half-line `[0,\infty)` onto the
open interval `[0,1)` that send a physical, causally-unreachable barrier
(`\xi=1` / `\beta_v=1`) to the single point at infinity of the unbounded
coordinate. In both, the vanishing "gap" is exponentially small in the rapidity:

$$
1-\xi = e^{-\theta}
\qquad\text{vs.}\qquad
\frac{1}{\gamma} = \operatorname{sech}\varphi \sim 2\,e^{-\varphi}\ \ (\varphi\to\infty),
$$

so **asymptotically (large rapidity) the two charts coincide up to a constant
factor**: the pull-in gap `e^{-\theta}` decays exactly like the relativistic
`1/\gamma`. In this asymptotic, singularity-approaching regime — the regime the
PINN must resolve — the analogy is sharp: `\theta` is to the closing NEMS gap
what the relativistic rapidity is to the closing `1/\gamma`, and `\xi=1` is a
horizon-like boundary approached but never reached (cf. `bifurcation_notes.md
§4`).

**Where it is only formal.** (i) The specific chart differs: we use
`\xi = 1-e^{-\theta}`, not `\xi=\tanh\theta`. Either would map `[0,\infty)` onto
`[0,1)`; we choose the exponential chart because it renders the *gap* a pure
exponential `1-\xi=e^{-\theta}`, which diagonalizes the force terms into
`e^{2n\theta}` and yields the clean entire RHS of `(Θ)`. The two charts agree to
leading order for small `\theta` (`\tanh\theta\approx\theta\approx 1-e^{-\theta}`)
and both saturate at `1`, but their finite-`\theta` forms are not identical.
(ii) More importantly, `\theta` here carries **no group-theoretic meaning**:
there is no Lorentz group acting, no boost additivity, no invariant interval.
`\theta` is a smooth reparametrization of the state variable chosen for its
analytic properties, not a physical rapidity generated by a symmetry. The
analogy is therefore a *structural/geometric* one (unbounded coordinate,
exponential saturation, horizon at infinity), invoked to motivate the coordinate
and to connect to the companion relativistic manuscript, and should be presented
as such — not as a dynamical equivalence.

---

## 2. Why a PINN surrogate rather than fixed-step RK4 near the pole

This section is written to be defensible to a referee: it does **not** claim the
PINN is more accurate than a good adaptive integrator, and it is explicit about
which integrator degrades and why.

### 2.1 Fixed-step explicit RK4 suffers step-size collapse at the pole

Classical explicit RK4 with a *fixed* step `h` has local truncation error
`\ \varepsilon_{\text{loc}} = C\,h^5\,\Phi + O(h^6)`, where the error constant
`\Phi` is a fixed combination of elementary differentials of the vector field —
equivalently it scales with the fifth derivative of the solution along the step.
For `(Ξ)` the vector field contains `(1-\xi)^{-2}` and `(1-\xi)^{-4}`, whose
`k`-th `\xi`-derivatives scale like `(1-\xi)^{-4-k}`; the elementary
differentials entering `\Phi` therefore blow up as

$$
\Phi \sim (1-\xi)^{-p}\ \to\ \infty
\qquad (\xi\to 1),\quad p>0 .
$$

To hold `\varepsilon_{\text{loc}}` below a fixed tolerance one would need
`h \lesssim (1-\xi)^{p/5}\to 0`. A **fixed** `h` cannot do this: as the
trajectory nears `\tau_*`, one of two failures occurs — (a) the local error
grows past tolerance and the numerical solution loses all significant digits, or
(b) a single RK4 step **overshoots** the pole, evaluating the field at
`\xi>1`, where `(1-\xi)^{-4}` has the wrong sign / is being raised to a
non-integer power in downstream code, producing `NaN`/`Inf` and a spurious
"blow-up" that is a *numerical artifact*, not the true dynamics. This is the
classroom-RK4 failure that motivates a different representation.

### 2.2 An adaptive, high-order integrator does **not** blow up — and is the reference

Honesty requires stating the counterpoint. Error-controlled adaptive
integrators automatically shrink `h` exactly in the regime where `\Phi` grows:

- **Radau IIA** (implicit, `A`- and `L`-stable, 5th order; Hairer–Wanner) is
  built for stiff and near-singular problems and remains stable as `\tau\to\tau_*`;
- **DOP853** (adaptive explicit Runge–Kutta of order 8; Hairer–Nørsett–Wanner)
  likewise refines `h` under its embedded error estimate.

Neither of these "blows up" spuriously; they resolve the approach to the pole
accurately up to any prescribed `\xi_{\max}<1` and are the correct **reference
("ground-truth") solutions** against which the PINN must be validated. It would
be incorrect, and a referee would rightly object, to claim the PINN beats these
integrators in accuracy on the forward problem.

### 2.3 The fair, defensible statement of what the PINN provides

The PINN's advantage is not raw forward-accuracy over an adaptive solver; it is
*representational*. Trained in the rapidity coordinate `(Θ)`, where the field is
entire (§1.3, Prop. 1), the PINN yields a **smooth, mesh-free, closed-form,
analytically differentiable global surrogate** `\theta_{\rm NN}(\tau)` — and
hence `\xi_{\rm NN}(\tau) = 1 - e^{-\theta_{\rm NN}(\tau)}` — valid on the entire
window rather than a table of discrete samples requiring interpolation. Concretely:

1. **Singularity-regularized coordinate.** The surrogate is fitted to a residual
   that is bounded and smooth on the whole training domain (Prop. 1), because the
   pole has been mapped to `\theta=\infty`; there is no denominator to resolve.
2. **Mesh-free and continuous.** `\theta_{\rm NN}` and its exact derivatives are
   available at any `\tau` by automatic differentiation, useful for downstream
   sensitivity/optimization (e.g. `\partial\xi/\partial\alpha` for design), for
   inverse problems (fitting `\alpha,\beta,\zeta` to data), and for parametric
   sweeps where a single network can be conditioned on `(\alpha,\beta,\zeta)`.
3. **Same coordinate serves both manuscripts.** The identical rapidity PINN is
   used for the relativistic `v\to c` barrier in the companion Phys. Rev. D
   manuscript, so the method is shared, not bespoke.

The comparison to report in the paper is therefore: *fixed-step RK4 degrades and
can produce artifacts near pull-in; an adaptive integrator (Radau/DOP853) is
accurate and is used as reference; the PINN provides a smooth differentiable
surrogate in a pole-free coordinate, validated to agree with the adaptive
reference up to `\xi_{\max}`.* Framed this way the claim is both true and robust
to review.

---

## 3. Loss weighting and collocation

The composite loss is

$$
\mathcal L = \mathcal L_{\rm ODE}
           + \lambda_1\,\mathcal L_{\rm IC}
           + \lambda_2\,\mathcal L_{\rm pull\text{-}in},
$$

$$
\mathcal L_{\rm ODE}
 = \frac{1}{N_f}\sum_{i=1}^{N_f}
   \Big|\,\ddot\theta_{\rm NN}(\tau_i)
        - f\big(\theta_{\rm NN}(\tau_i),\dot\theta_{\rm NN}(\tau_i)\big)\Big|^2,
$$

with `\{\tau_i\}` the collocation points, `\mathcal L_{\rm IC}` enforcing
`\theta_{\rm NN}(0)=\theta_0`, `\dot\theta_{\rm NN}(0)=\dot\theta_0` (§1.3), and
`\mathcal L_{\rm pull\text{-}in}` the problem-specific pull-in anchor (below).

### 3.1 Gradient pathology and why the weights matter

A well-documented failure mode of PINNs is an **imbalance between the gradients
of the loss terms**: `\nabla_\Theta\mathcal L_{\rm ODE}` and
`\nabla_\Theta\mathcal L_{\rm IC}` can differ by orders of magnitude, so a plain
sum lets one term dominate the parameter update and the other is effectively
ignored, stalling training. For `(Θ)` this is aggravated by the `e^{5\theta}`
term: near pull-in the residual is numerically stiff and a handful of
large-`\theta` collocation points can swamp `\mathcal L_{\rm ODE}` itself.
Two mitigations, in order of preference:

1. **Hard-constrained initial conditions (recommended).** Use a trial-function
   ansatz that satisfies the ICs *identically* (Lagaris–Likas–Fotiadis 1998):

   $$
   \theta_{\rm NN}(\tau) = \theta_0 + \tau\,\dot\theta_0 + \tau^2\,\mathcal N(\tau;\Theta),
   $$

   where `\mathcal N` is the raw network. Then `\theta_{\rm NN}(0)=\theta_0` and
   `\dot\theta_{\rm NN}(0)=\dot\theta_0` hold by construction, `\mathcal L_{\rm IC}`
   is removed, and `\lambda_1` disappears — eliminating the worst of the
   imbalance at its source.

2. **Residual normalization.** Because `f` contains `e^{5\theta}`, weight or
   rescale the ODE residual so no point dominates — e.g. minimize the *relative*
   residual `\big(\ddot\theta - f\big)\big/\big(1+|f|\big)`, or restrict training
   to `\theta\in[0,\theta_{\max}]` with a modest `\theta_{\max}` (below). This is
   important in practice and should be reported.

3. **Adaptive loss weights** for whatever soft terms remain: learning-rate
   annealing of `\lambda_k` from gradient-norm statistics (Wang, Teng &
   Perdikaris 2021), NTK-based balancing (Wang, Yu & Perdikaris 2022), or
   pointwise self-adaptive weights (McClenny & Braga-Neto 2023). Prefer these
   over hand-tuned constants once the problem is set up.

### 3.2 Collocation points

- **Count.** `N_f \approx 512`–`1024` interior collocation points on `[0,T]`
  are ample for this smooth 1-D problem in the `\theta` coordinate; there is no
  benefit to very large `N_f` here, and it slows L-BFGS.
- **Placement.** Cluster points toward `\tau=T` (near pull-in), where
  `\theta,\dot\theta` are largest and the solution steepest — a graded or
  Chebyshev-like mesh, or Latin-hypercube sampling with a bias toward `T`.
- **Adaptive resampling.** Periodically add points where the residual is largest
  (residual-based adaptive refinement / RAR; see Lu et al., *DeepXDE*, 2021, and
  the sampling study of Wu et al., 2023 — *verify exact venue/pages*). A few
  refinement rounds typically suffice.
- **Domain / target.** Fix the window by a target maximum displacement
  `\xi_{\max}` rather than by `\tau_*` (which is unknown a priori). Choosing
  `\xi_{\max}=0.99` gives `\theta_{\max}=-\log(0.01)\approx 4.6`; training on
  `\theta\in[0,\theta_{\max}]` keeps `e^{5\theta}\lesssim e^{23}` bounded and the
  residual well-scaled while still capturing the full physical approach to
  pull-in.

### 3.3 The pull-in term and recommended weights

`\mathcal L_{\rm pull\text{-}in}` encodes the approach to collapse. Depending on
the formulation it is either (a) a *terminal anchor* pinning
`\theta_{\rm NN}(T)=\theta_{\max}` (equivalently `\xi_{\rm NN}(T)=\xi_{\max}`),
or (b) an *asymptotic/monotonicity* penalty enforcing `\dot\theta>0` and the
correct large-`\theta` growth. Either is a single, low-dimensional constraint, so
`\lambda_2` should be `O(1)`–`O(10)`; report exactly what it penalizes.

**Concrete starting recommendation** (then hand over to the adaptive schemes of
§3.1):

| Ingredient | Recommended value |
|---|---|
| IC handling | **Hard constraint** via `\theta_0+\tau\dot\theta_0+\tau^2\mathcal N` (drop `\mathcal L_{\rm IC}`) |
| `\lambda_1` (if soft IC instead) | `\approx 10^2`, then anneal |
| `\lambda_2` (pull-in anchor) | `\approx 10` (`O(1)`–`O(10)`) |
| `N_f` (collocation) | `512`–`1024`, graded toward `T`, RAR refinement |
| Domain | `\theta\in[0,\theta_{\max}]`, `\theta_{\max}=-\log(1-\xi_{\max})\approx 4.6` at `\xi_{\max}=0.99` |
| Residual | minimize relative residual `(\ddot\theta-f)/(1+|f|)` |
| Network | `\tanh` MLP, `\sim 4`–`5` layers `\times 50` units |
| Optimizer | Adam (warm-up) `\to` L-BFGS (polish) |

These are defaults to be tuned against the Radau/DOP853 reference of §2.2, not
claims of optimality.

---

## 4. Proposition (regularization of the pull-in pole)

> **Proposition 1 (Regularization of the pull-in pole).**
> Let `\xi:[0,\tau_*)\to[0,1)` be a solution of `(Ξ)` with `\xi(\tau)\to 1^-` as
> `\tau\uparrow\tau_*` (a movable pole). Under the rapidity map
> `\xi = 1-e^{-\theta}`, `\theta=-\log(1-\xi)`:
>
> 1. the map is a `C^\infty` diffeomorphism of `[0,1)` onto `[0,\infty)`,
>    strictly increasing, with `\xi=0\leftrightarrow\theta=0` and
>    `\xi\to1^-\leftrightarrow\theta\to+\infty`;
> 2. `\theta(\tau)` satisfies
>    `\ddot\theta = \dot\theta^{2} - 2\zeta\dot\theta - (e^{\theta}-1)
>     + \alpha e^{3\theta} + \beta e^{5\theta} =: f(\theta,\dot\theta)`,
>    whose right-hand side is real-analytic (entire) on all of `\mathbb R^2` and
>    has **no singularity at any finite `\theta`**, whereas the `(Ξ)`-field has a
>    pole at the finite value `\xi=1`;
> 3. consequently, for every `\theta_{\max}<\infty` and every `T` with
>    `\sup_{[0,T]}\theta\le\theta_{\max}`, the collocation residual
>    `R[\theta](\tau)=\ddot\theta-f(\theta,\dot\theta)` and hence
>    `\mathcal L_{\rm ODE}` are bounded and `C^\infty` on `[0,T]`. The PINN
>    optimization is thus posed on a **pole-free** (compact, singularity-
>    regularized) domain.

**Proof (sketch).**
*(1)* From `1-\xi=e^{-\theta}>0` the inverse `\theta=-\log(1-\xi)` is defined and
smooth on `[0,1)`; `d\xi/d\theta=e^{-\theta}=1-\xi>0` gives strict monotonicity
and (with its nonvanishing) the diffeomorphism property; the boundary limits are
immediate.
*(2)* Substituting `\dot\xi=e^{-\theta}\dot\theta`,
`\ddot\xi=e^{-\theta}(\ddot\theta-\dot\theta^{2})` and
`(1-\xi)^{-2n}=e^{2n\theta}` into `(Ξ)` and multiplying by `e^{\theta}` gives
`(Θ)` (see §1.3). The RHS `f` is a finite sum of `\exp(k\theta)` (`k=1,3,5`), a
constant, and the polynomial `\dot\theta^{2}-2\zeta\dot\theta`; each summand is
entire, and finite sums/products of entire functions are entire, so `f` is
real-analytic on `\mathbb R^2`. In particular `f` has no pole for any finite
`\theta`.
*(3)* On the compact set
`K=\{(\theta,\dot\theta):0\le\theta\le\theta_{\max},\ |\dot\theta|\le M\}` the
continuous function `f` attains a finite maximum; the network `\theta_{\rm NN}`
and its autodiff derivatives `\dot\theta_{\rm NN},\ddot\theta_{\rm NN}` are
`C^\infty` in `\tau`; hence `R=\ddot\theta_{\rm NN}-f(\theta_{\rm NN},\dot\theta_{\rm NN})`
is a `C^\infty`, bounded function on `[0,T]`, and its mean square
`\mathcal L_{\rm ODE}` is finite and smooth in the network parameters. `∎`

**Remark (honest scope).** Proposition 1 regularizes the *coordinate* pole: it
removes the singularity of the vector field at the finite state value `\xi=1`,
placing the loss on a pole-free domain. It does **not** claim to remove the
finite-time blow-up in the *independent* variable — `\theta(\tau)\to\infty` still
occurs as `\tau\to\tau_*`, since no change of the dependent variable can
eliminate a genuine finite-time escape of the `\tau`-flow. In practice one trains
on a fixed window `[0,T]` with `T<\tau_*` (equivalently up to `\xi_{\max}`), where
`\theta\le\theta_{\max}<\infty` and the hypotheses of part (3) hold. This is
precisely the regime of interest (the approach to pull-in), and the map converts
"approach a bounded wall with a diverging vector field" into "climb a smooth,
pole-free, unbounded ramp."

---

## 5. Definitions (for insertion into the article)

> **Definitions.** A *physics-informed neural network* (PINN) is a
> parametric surrogate `\theta_{\rm NN}(\tau;\Theta)`, here a `\tanh`
> multilayer perceptron with weights `\Theta`, trained by minimizing a loss in
> which the governing differential equation is imposed directly through the
> network's own automatic derivatives rather than through labelled solution
> data (Raissi, Perdikaris & Karniadakis 2019; Lagaris, Likas & Fotiadis 1998).
> The *collocation residual* is the pointwise defect
> `R[\theta](\tau_i)=\ddot\theta_{\rm NN}(\tau_i)-f(\theta_{\rm NN}(\tau_i),
> \dot\theta_{\rm NN}(\tau_i))` obtained by inserting the surrogate into the ODE
> `(Θ)` at a finite set of *collocation points* `\{\tau_i\}\subset[0,T]`; its
> mean square is the physics loss `\mathcal L_{\rm ODE}`. A *movable pole* (or
> movable singularity) is a singularity of the solution — here the finite-time
> collapse `\xi(\tau)\to 1` as `\tau\to\tau_*` — whose location `\tau_*` depends
> on the initial data and parameters rather than being fixed by the equation,
> and which is the dynamical signature of electrostatic–Casimir pull-in.

---

### Summary

- **Rapidity chart.** `\xi = 1-e^{-\theta}` (`\theta=-\log(1-\xi)`) is a
  `C^\infty` bijection `[0,1)\to[0,\infty)`; the gap is `1-\xi=e^{-\theta}` and
  the pull-in barrier `\xi=1` sits at `\theta=\infty`. Derivatives:
  `\dot\xi=e^{-\theta}\dot\theta`, `\ddot\xi=e^{-\theta}(\ddot\theta-\dot\theta^2)`.
- **Transformed ODE:**
  `\ddot\theta = \dot\theta^{2} - 2\zeta\dot\theta - (e^{\theta}-1)
  + \alpha e^{3\theta} + \beta e^{5\theta}`, with an **entire** RHS (no finite-`\theta`
  pole); its static limit reproduces the fold relation `u^4-u^5=\alpha u^2+\beta`.
- **Proposition 1:** the map regularizes the pole — the residual and
  `\mathcal L_{\rm ODE}` are bounded and `C^\infty` on any compact
  `\theta\in[0,\theta_{\max}]`, so the PINN is posed on a pole-free domain (with
  the honest caveat that the finite-time blow-up in `\tau` itself is not removed,
  only the coordinate singularity).
- **RK4 vs PINN (referee-safe):** *fixed-step* RK4 undergoes step-size
  collapse/overshoot at the pole; an *adaptive* integrator (Radau/DOP853) does
  not and is the reference; the PINN's value is a smooth, mesh-free,
  differentiable global surrogate in the pole-free `\theta` coordinate.
- **Recommended setup:** hard-constrained ICs (drop `\mathcal L_{\rm IC}`;
  else `\lambda_1\approx10^2`), `\lambda_2\approx10` for the pull-in anchor,
  `N_f\approx512`–`1024` collocation points graded toward `T` with residual-based
  refinement, domain `\theta\in[0,\theta_{\max}\approx4.6]` (`\xi_{\max}=0.99`),
  relative-residual scaling, `\tanh` MLP `4`–`5\times50`, Adam→L-BFGS, adaptive
  loss weights (Wang–Teng–Perdikaris 2021; Wang–Yu–Perdikaris 2022;
  McClenny–Braga-Neto 2023).

### References (verify exact volume/pages before typesetting)

- I. E. Lagaris, A. Likas, D. I. Fotiadis, "Artificial neural networks for
  solving ordinary and partial differential equations," *IEEE Trans. Neural
  Netw.* **9**(5), 987–1000 (1998).
- M. Raissi, P. Perdikaris, G. E. Karniadakis, "Physics-informed neural
  networks: A deep learning framework for solving forward and inverse problems
  involving nonlinear partial differential equations," *J. Comput. Phys.* **378**,
  686–707 (2019).
- S. Wang, Y. Teng, P. Perdikaris, "Understanding and mitigating gradient flow
  pathologies in physics-informed neural networks," *SIAM J. Sci. Comput.*
  **43**(5), A3055–A3081 (2021). *(verify)*
- S. Wang, X. Yu, P. Perdikaris, "When and why PINNs fail to train: A neural
  tangent kernel perspective," *J. Comput. Phys.* **449**, 110768 (2022). *(verify)*
- L. McClenny, U. Braga-Neto, "Self-adaptive physics-informed neural networks,"
  *J. Comput. Phys.* **474**, 111722 (2023). *(verify)*
- L. Lu, X. Meng, Z. Mao, G. E. Karniadakis, "DeepXDE: A deep learning library
  for solving differential equations," *SIAM Review* **63**(1), 208–228 (2021). *(verify)*
- C. Wu, M. Zhu, Q. Tan, Y. Kartha, L. Lu, "A comprehensive study of
  non-adaptive and residual-based adaptive sampling for physics-informed neural
  networks," *Comput. Methods Appl. Mech. Engrg.* **403**, 115671 (2023). *(verify)*
- E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations
  I: Nonstiff Problems* (Springer) — DOP853.
- E. Hairer, G. Wanner, *Solving Ordinary Differential Equations II: Stiff and
  Differential-Algebraic Problems* (Springer) — Radau IIA.
