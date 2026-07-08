# Bifurcation structure of the Casimir–electrostatic NEMS oscillator

Dimensionless equation of motion (over-dot = derivative w.r.t. dimensionless time):

$$
\ddot\xi + 2\zeta\,\dot\xi + \xi \;=\; \frac{\alpha}{(1-\xi)^2} + \frac{\beta}{(1-\xi)^4},
\qquad \xi\in[0,1),\quad \alpha,\beta,\zeta\ge 0 .
$$

`ξ` is the plate displacement normalized to the initial gap; `1−ξ` is the
instantaneous dimensionless gap. The linear `+ξ` on the left is the elastic
restoring (spring) force. On the right, `α/(1−ξ)²` is the parallel-plate
**electrostatic** attraction (`α ∝ V²`, the drive) and `β/(1−ξ)⁴` is the
**Casimir/van der Waals** attraction. Both attractive terms have a movable pole
at `ξ→1` (gap closes).

---

## 1. Equilibria and the effective potential

Static states satisfy `ξ̈ = ξ̇ = 0`, i.e. the net force vanishes:

$$
g(\xi) \equiv \frac{\alpha}{(1-\xi)^2} + \frac{\beta}{(1-\xi)^4} - \xi = 0 .
$$

The dynamics is gradient-like in an effective potential `V(ξ)` with
`V'(ξ) = -g(ξ)`, i.e.

$$
V(\xi) = \tfrac12\xi^2 - \frac{\alpha}{1-\xi} - \frac{\beta}{3(1-\xi)^3},
\qquad \ddot\xi = -V'(\xi) - 2\zeta\dot\xi .
$$

Linearizing about an equilibrium `ξ*`, `δξ̈ + 2ζ δξ̇ + V''(ξ*)δξ = 0` with
`V''(ξ*) = -g'(ξ*)`, where

$$
g'(\xi) = \frac{2\alpha}{(1-\xi)^3} + \frac{4\beta}{(1-\xi)^5} - 1 .
$$

Hence (for `ζ>0`) the equilibrium is:

- **stable** (damped oscillator / node-focus) when `V''>0 ⇔ g'(ξ*) < 0`;
- **unstable** (saddle) when `V''<0 ⇔ g'(ξ*) > 0`.

Because `g(0)=α+β>0` and `g(ξ)→+∞` as `ξ→1`, a smooth force curve that dips
below zero produces exactly **two** equilibria: the smaller root is the stable
node (`g` crosses `0` downward, `g'<0`), the larger is the saddle (`g` crosses
upward, `g'>0`). If the curve never reaches zero there are **no** equilibria.

---

## 2. Saddle-node (pull-in) condition and the closed-form fold locus

Pull-in is the **fold / saddle-node** bifurcation at which the stable node and
the saddle merge and annihilate — a double root of `g`:

$$
g(\xi)=0 \quad\text{and}\quad g'(\xi)=0 .
$$

Introduce the pull-in gap `u = 1-\xi_{\rm PI}\in(0,1]`. The two conditions become

$$
\underbrace{\frac{\alpha}{u^2} + \frac{\beta}{u^4} = 1-u}_{g=0},
\qquad
\underbrace{\frac{2\alpha}{u^3} + \frac{4\beta}{u^5} = 1}_{g'=0}.
$$

This is a **linear system in `(α,β)`**. Write it as

$$
\begin{pmatrix} u^{-2} & u^{-4}\\[2pt] 2u^{-3} & 4u^{-5}\end{pmatrix}
\begin{pmatrix}\alpha\\ \beta\end{pmatrix}
=\begin{pmatrix} 1-u\\ 1\end{pmatrix}.
$$

**Eliminate `α`.** Multiply the first row by `2/u` and subtract from the second:

$$
\Big(\tfrac{4}{u^5}-\tfrac{2}{u^5}\Big)\beta = 1 - \frac{2(1-u)}{u}
= \frac{u-2(1-u)}{u} = \frac{3u-2}{u}
\;\Longrightarrow\;
\frac{2\beta}{u^5} = \frac{3u-2}{u}.
$$

$$
\boxed{\;\beta_c(u) = \tfrac12\,u^4\,(3u-2)\;}
$$

**Back-substitute for `α`.** From the second equation
`2α/u³ = 1 - 4β/u⁵ = 1 - 2(3u-2)/u = (4-5u)/u`, so

$$
\boxed{\;\alpha_c(u) = \tfrac12\,u^2\,(4-5u)\;}
$$

(The determinant `4u^{-7}-2u^{-7}=2u^{-7}\ne0` for `u>0`, so the solution is
unique.) A direct residual check `g=g'=0` along `(α_c,β_c,ξ_PI)` is performed
in `bifurcation.py`; residuals are `~10⁻¹⁶`.

### Endpoints and the physical window

Positivity of the two forces bounds `u`:

- `β_c ≥ 0 ⇔ 3u-2 ≥ 0 ⇔ u ≥ 2/3` (i.e. `ξ_PI ≤ 1/3`);
- `α_c ≥ 0 ⇔ 4-5u ≥ 0 ⇔ u ≤ 4/5` (i.e. `ξ_PI ≥ 1/5`).

So the admissible both-forces-positive fold branch is

$$
u\in\Big[\tfrac23,\tfrac45\Big],\qquad
\xi_{\rm PI}=1-u\in\Big[\tfrac15,\tfrac13\Big].
$$

| limit | `u` | `ξ_PI` | `α_c` | `β_c` |
|---|---|---|---|---|
| pure electrostatic (`β=0`) | `2/3` | `1/3` | `4/27 ≈ 0.148148` | `0` |
| pure Casimir (`α=0`) | `4/5` | `1/5` | `0` | `256/3125 ≈ 0.081920` |

The classic electrostatic result `ξ_PI = 1/3`, `α_c = 4/27` is recovered
exactly at `β=0`. The pure-Casimir endpoint is `ξ_PI = 1/5`,
`β* = (4/5)⁵/4 = 256/3125 ≈ 0.08192`. For `u<2/3` the formula would demand
`β<0`; for `u>4/5` it demands `α<0` — both unphysical (flagged by the
`physical` mask returned from `fold_boundary(full=True)`).

### Physical interpretation

As the Casimir contribution `β` grows, the fold point slides from `ξ_PI = 1/3`
down to `ξ_PI = 1/5`: **the Casimir force lowers the maximum stable travel and
lowers the pull-in threshold.** The steeper `1/(1-ξ)⁴` attraction dominates at
smaller displacement, so the node and saddle collide earlier (smaller `ξ`) and
at smaller electrostatic drive `α`. This is the NEMS design constraint: at
sub-100-nm gaps the always-on Casimir attraction eats into the usable actuation
range and can cause stiction/collapse even at zero applied voltage once
`β ≥ β* = 256/3125`.

---

## 3. Three dynamical regimes

For a given `(α,β)` with `ζ>0`, starting from rest at `ξ=0`:

1. **Below the fold** (`(α,β)` inside the region bounded by the locus,
   equivalently `α<α_c`, `β<β_c` along the crossing ray): **two** equilibria — a
   stable node `ξ_s` and a saddle `ξ_u>ξ_s`. Since `g(0)>0`, the plate rolls
   outward and, with damping, settles at `ξ_s`. Device is **stable** (operable).
   `is_stable_from_rest → True`.

2. **On the fold** (`g=g'=0`): the node and saddle coalesce into a single
   degenerate (half-stable) equilibrium at `ξ_PI`. Marginal — the **pull-in
   threshold** itself. This is the codimension-one saddle-node set traced by
   `(α_c(u),β_c(u))`.

3. **Above the fold**: **no** equilibrium in `[0,1)`. The net force stays
   positive; the plate accelerates monotonically into the movable pole and
   **pulls in** (`ξ→1`, gap collapses / stiction). `is_stable_from_rest → False`.

`is_stable_from_rest` uses this equilibrium structure (root counting), not time
integration: a stable rest state exists iff the operating point lies below the
fold surface. (Note: the *dynamic* pull-in voltage under a suddenly applied step
is slightly lower than this quasi-static fold because the plate arrives at the
saddle with kinetic energy; the RK4 phase diagram cross-checked against
`stability_boundary_grid` quantifies that gap, controlled by `ζ`.)

---

## 4. Analogy: pull-in fold ↔ relativistic energy bifurcation

The force law carries a **movable singularity** at `ξ→1` (the gap closes), just
as a relativistic particle's energy/momentum carries a movable singularity at
`v→c`:

$$
\frac{1}{(1-\xi)^n}\quad\longleftrightarrow\quad
\frac{1}{\sqrt{1-(v/c)^2}}=\gamma .
$$

The dimensionless gap `1-ξ` plays the role of the Lorentz factor's `(1-v²/c²)`:
both are positive quantities that vanish at an unreachable boundary
(`ξ=1 ↔ v=c`), and both make the associated potential/energy diverge there.
The **fold** in the NEMS potential `V(ξ)` — the loss of a stable minimum as the
drive increases — is structurally the same catastrophe as the disappearance of
a bound (sub-luminal) energy branch as a forcing parameter pushes the system
toward its light-cone-like boundary: a control parameter (`α,β` here; the
accelerating field there) drives the system until the confining barrier (the
saddle) is destroyed at the movable pole and the state runs away toward the
singular boundary. In both cases the boundary `ξ=1` / `v=c` is an **essential
horizon** that the physical trajectory can only approach, never cross, and the
bifurcation is the moment the last stabilizing barrier between the operating
point and that horizon disappears.

---

## Function reference (`src/bifurcation.py`)

- `net_force(xi, alpha, beta)` — `g(ξ)`.
- `net_force_prime(xi, alpha, beta)` — `g'(ξ)`.
- `fold_boundary(n=400, full=False)` — closed-form `(alpha_c, beta_c, xi_pi,
  physical)`; default sweeps the physical branch `u∈[2/3,4/5]`, `full=True`
  sweeps `u∈(0,1]` with a `physical` mask marking `α,β≥0`.
- `equilibria(alpha, beta)` — all roots of `g=0` via dense scan + `brentq`,
  returned as `(stable, unstable)` split by `sign(g')`.
- `is_stable_from_rest(alpha, beta)` — `True` iff a stable equilibrium exists
  (operating point below the fold).
- `stability_boundary_grid(alpha_arr, beta_arr)` — boolean pull-in/stable grid
  (`rows=β, cols=α`) for cross-checking the RK4 phase diagram.

Running `python src/bifurcation.py` verifies the endpoints
(`α_c=4/27` at `β=0`, `β*=256/3125` at `α=0`), confirms
`is_stable_from_rest` agrees with the fold locus (inside → stable, outside →
pull-in), and writes `results/fold_boundary.csv` with columns
`u, xi_pi, alpha_c, beta_c`.
