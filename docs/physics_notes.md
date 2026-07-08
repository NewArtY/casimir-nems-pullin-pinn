# Physics Notes — Casimir–Electrostatic Pull-In Bifurcation in Nonlinear NEMS

Supporting notes for *"Casimir–Electrostatic Pull-In Bifurcation in Nonlinear
Nanomechanical Oscillators: Phase Diagrams and PINN Solutions with Relativistic
Lifshitz Corrections."*

These notes (i) re-derive the dimensionless equation of motion and the control
parameters α, β, ζ; (ii) tabulate realistic gold-coated NEMS device parameters
and evaluate the operating point in the (α, β) phase plane; (iii) verify the
static pull-in fold; (iv) quantify the thermal (Lifshitz) correction; and
(v) state the applied significance for *Physical Review Applied*.

Physical constants used throughout (CODATA 2018):

| Symbol | Value | Units |
|---|---|---|
| ε₀ | 8.854 187 8128 × 10⁻¹² | F m⁻¹ |
| ħ  | 1.054 571 817 × 10⁻³⁴ | J s |
| c  | 2.997 924 58 × 10⁸ | m s⁻¹ |
| k_B | 1.380 649 × 10⁻²³ | J K⁻¹ |
| ζ(3) (Apéry) | 1.202 056 903 | — |

---

## 1. Non-dimensionalization and the control parameters

### 1.1 Dimensional equation of motion

A single-mode (lumped) gold-coated actuator with effective mass *m*, mechanical
damping coefficient γ_d, and linear stiffness *k*, whose movable electrode is
displaced by *x* toward a fixed electrode across an initial vacuum gap *d*, obeys

$$
m\,\ddot x + \gamma_d\,\dot x + k\,x
= \underbrace{\frac{\varepsilon_0 A V_0^{2}}{2\,(d-x)^{2}}}_{\text{electrostatic}}
+ \underbrace{\frac{\pi^{2}\hbar c\,A}{240\,(d-x)^{4}}}_{\text{Casimir}} .
$$

Both forcing terms are *attractive* and *singular* as the gap `(d−x) → 0`. The
electrostatic term is the parallel-plate capacitor pressure `ε₀AV₀²/2(d−x)²`;
the Casimir term is the ideal-conductor (Casimir 1948) pressure
`π²ħc/240(d−x)⁴` integrated over area *A*.

### 1.2 Scaling

Introduce the dimensionless displacement, time and natural frequency

$$
\xi=\frac{x}{d},\qquad \tau=\omega_0 t,\qquad \omega_0=\sqrt{k/m}.
$$

Then, with `' ≡ d/dτ`,

$$
x=d\,\xi,\qquad \dot x=\frac{dx}{dt}=d\,\omega_0\,\xi',\qquad
\ddot x=d\,\omega_0^{2}\,\xi''.
$$

Substituting and using `d − x = d(1 − ξ)`:

$$
m d\omega_0^{2}\,\xi'' + \gamma_d d\omega_0\,\xi' + k d\,\xi
= \frac{\varepsilon_0 A V_0^{2}}{2 d^{2}(1-\xi)^{2}}
+ \frac{\pi^{2}\hbar c A}{240\, d^{4}(1-\xi)^{4}} .
$$

Divide through by `k d` and use `mω₀² = k`:

$$
\boxed{\;\xi'' + 2\zeta\,\xi' + \xi
= \frac{\alpha}{(1-\xi)^{2}} + \frac{\beta}{(1-\xi)^{4}}\;}
$$

with the three dimensionless groups

$$
\alpha=\frac{\varepsilon_0 A V_0^{2}}{2\,k\,d^{3}},\qquad
\beta =\frac{\pi^{2}\hbar c\,A}{240\,k\,d^{5}},\qquad
\zeta =\frac{\gamma_d}{2 m\omega_0}=\frac{\gamma_d\,\omega_0}{2k}=\frac{1}{2Q}.
$$

The damping term follows from `γ_d d ω₀ /(k d) = γ_d ω₀ /k = γ_d/(mω₀) = 2ζ`,
and by definition of the quality factor `Q = mω₀/γ_d`, so `ζ = 1/(2Q)`.

### 1.3 Dimensional (units) check

- **α** = ε₀·A·V₀² / (k·d³).
  Write `ε₀` in `C² N⁻¹ m⁻²`, `V = J C⁻¹ = N·m·C⁻¹`:
  numerator `= [C²N⁻¹m⁻²][m²][N²m²C⁻²] = N·m²`;
  denominator `k d³ = [N m⁻¹][m³] = N·m²`. Ratio **dimensionless.** ✔

- **β** = ħ·c·A / (k·d⁵).
  numerator `= [J s][m s⁻¹][m²] = J·m³`;
  denominator `k d⁵ = [N m⁻¹][m⁵] = N·m⁴ = J·m³`. Ratio **dimensionless.** ✔

- **ζ** = γ_d/(2mω₀): `[kg s⁻¹]/([kg][s⁻¹])` **dimensionless.** ✔

α is the electrostatic strength (∝ V₀², experimentally tunable), β the Casimir
strength (fixed by geometry/materials), ζ the damping (=1/2Q).

---

## 2. Realistic gold-coated NEMS devices and their operating point

The three devices below span the regime of published Casimir-MEMS/NEMS
oscillators — the micro-torsional oscillator of Chan *et al.* (Science **291**,
1941, 2001), the plate–plate configurations of Decca *et al.* (Phys. Rev. D
**75**, 077101, 2007), and stiff doubly-clamped NEMS beams (Ekinci &
Roukes, Rev. Sci. Instrum. **76**, 061101, 2005). Parameter ranges: gap
*d* = 50–200 nm, area *A* = 25–100 µm², stiffness *k* = 0.5–5 N m⁻¹,
*Q* = 10³–10⁴ (vacuum), *f₀* = 50 kHz–1 MHz. These are representative,
order-of-magnitude engineering values, not one specific fabricated device.

The effective mass follows from the chosen resonance, `m = k/ω₀²`,
`ω₀ = 2πf₀`. This gives fg–pg masses typical of NEMS.

| Device | A (µm²) | d (nm) | k (N/m) | f₀ | Q | m = k/ω₀² |
|---|---|---|---|---|---|---|
| **A** MEMS gold plate | 100 | 100 | 0.5 | 50 kHz | 10⁴ | 5.07 pg |
| **B** sub-100 nm NEMS | 25 | 50 | 2.0 | 1 MHz | 10³ | 50.7 fg |
| **C** stiff NEMS | 100 | 200 | 5.0 | 200 kHz | 5×10³ | 3.17 pg |

### 2.1 Evaluating α, β, ζ (worked arithmetic)

**Casimir strength** `β = π²ħcA/(240 k d⁵)`. For **Device A**
(A = 1×10⁻¹⁰ m², k = 0.5 N/m, d = 1×10⁻⁷ m):

```
numerator  = π²·ħ·c·A = 9.8696 · 1.0546e-34 · 2.9979e8 · 1.0e-10 = 3.120e-35
denominator= 240 · k · d⁵ = 240 · 0.5 · (1e-7)⁵ = 240·0.5·1e-35 = 1.200e-33
β_A = 3.120e-35 / 1.200e-33 = 0.0260
```

**Electrostatic strength** `α = ε₀AV₀²/(2 k d³)`. For Device A the geometric
prefactor is `ε₀A/(2 k d³) = (8.854e-12·1e-10)/(2·0.5·1e-21) = 0.8854 V⁻²`, so
`α_A = 0.8854 · V₀²`.

**Damping** `ζ = 1/(2Q)`: `ζ_A = 1/(2·10⁴) = 5.0×10⁻⁵`.

Repeating for all three devices:

| Device | β (Casimir) | ζ = 1/2Q | α/V₀² (V⁻²) | pull-in u* | ξ_PI | α_c | **V_PI** |
|---|---|---|---|---|---|---|---|
| **A** | 0.0260 | 5.0×10⁻⁵ | 0.885 | 0.728 | 0.272 | 0.0951 | **0.33 V** |
| **B** | 0.0520 | 5.0×10⁻⁴ | 0.443 | 0.767 | 0.233 | 0.0487 | **0.33 V** |
| **C** | 8.13×10⁻⁵ | 1.0×10⁻⁴ | 0.0111 | 0.667 | 0.333 | 0.148 | **3.66 V** |

*(V_PI computed from `V_PI = √(2 α_c k d³ / ε₀A)`, using the fold α_c of §3
evaluated at the device's own β.)*

**Reading of the phase diagram.** All three devices sit in the small-β corner
`β ≪ 0.082` (the Casimir-collapse ceiling, §3), i.e. below the roof of the
bistable region, so a genuine electrostatic pull-in fold exists and a finite
pull-in voltage can be defined. Device C (larger gap, stiffer) is essentially
electrostatic (β ~ 10⁻⁴, pull-in at the classical ξ = 1/3, V_PI ≈ 3.7 V).
Devices A and B, at 100 nm and 50 nm, carry β ~ 0.03–0.05: the Casimir force
pre-loads the system, lowering ξ_PI to ≈ 0.27 and 0.23 and cutting the pull-in
voltage below the naïve `β = 0` value. This is the applied payoff — at
sub-100-nm gaps the quantum-vacuum force measurably shifts the actuation
threshold, and predicting it correctly is required to design a *stable* gap.

---

## 3. Static pull-in fold — verification

Static equilibria are fixed points of the ODE (`ξ'' = ξ' = 0`):

$$
\xi=\frac{\alpha}{(1-\xi)^{2}}+\frac{\beta}{(1-\xi)^{4}} .
$$

Let `u = 1 − ξ` (the *normalized gap*, 0 < u < 1), so `ξ = 1 − u`. Then
`1 − u = α/u² + β/u⁴`. Multiply by `u⁴`:

$$
\text{(E, equilibrium)}\qquad \alpha u^{2}+\beta = u^{4}-u^{5}.
$$

The **fold / saddle-node** condition is tangency of the applied force
`F(ξ)=α/(1−ξ)²+β/(1−ξ)⁴` to the linear restoring force `ξ`, i.e. `dF/dξ = 1`.
Since `d/dξ = −d/du` and `F = αu⁻² + βu⁻⁴`,

$$
\frac{dF}{d\xi} = 2\alpha u^{-3}+4\beta u^{-5} = 1
\;\Longrightarrow\;
\text{(F, fold)}\qquad 2\alpha u^{2}+4\beta = u^{5}.
$$

**Solve (E) and (F) simultaneously for α(u), β(u).**
Multiply (E) by 2: `2αu² + 2β = 2u⁴ − 2u⁵`. Subtract (F):

$$
(2\alpha u^{2}+2\beta)-(2\alpha u^{2}+4\beta) = (2u^{4}-2u^{5})-u^{5}
\;\Rightarrow\; -2\beta = 2u^{4}-3u^{5},
$$

$$
\boxed{\;\beta_c(u)=\dfrac{u^{4}(3u-2)}{2}\;}
$$

Back-substitute into (F): `2αu² = u⁵ − 4β_c = u⁵ − 2u⁴(3u−2) = 4u⁴ − 5u⁵`, so

$$
\boxed{\;\alpha_c(u)=\dfrac{u^{2}(4-5u)}{2}\;}
$$

Both match the lead-author expressions. ✔

**Limits and physical range.**

- **β = 0** ⇒ `3u − 2 = 0` ⇒ `u = 2/3`, i.e. **ξ_PI = 1/3** (classical
  electrostatic pull-in), and `α_c = (2/3)²(4 − 10/3)/2 = (4/9)(2/3)/2 = 4/27
  ≈ 0.14815`. ✔
- **Positivity constraints** (physical bistable boundary needs α ≥ 0 and β ≥ 0):
  - α_c ≥ 0 ⇒ `4 − 5u ≥ 0` ⇒ `u ≤ 4/5` ⇒ **ξ_PI ≥ 1/5**.
  - β_c ≥ 0 ⇒ `3u − 2 ≥ 0` ⇒ `u ≥ 2/3` ⇒ **ξ_PI ≤ 1/3**.

  Hence the physically admissible fold locus is `u ∈ [2/3, 4/5]`, i.e.

$$
\boxed{\;\tfrac15 \le \xi_{\mathrm{PI}} \le \tfrac13\;}
$$

  The Casimir force **lowers** the pull-in displacement monotonically from
  ξ = 1/3 (no Casimir) toward ξ = 1/5 (Casimir-dominated).

- **α → 0 endpoint** (`u = 4/5`, ξ_PI = 1/5): pure-Casimir collapse. The
  maximum Casimir strength that still admits a static fold is

$$
\beta_{\max}= \beta_c\!\left(\tfrac45\right)
= \frac{(4/5)^{4}\,(3\cdot 4/5 - 2)}{2}
= \frac{256}{625}\cdot\frac{2}{5}\cdot\frac12
= \frac{256}{3125}\approx 0.08192 .
$$

  **For β > 256/3125 ≈ 0.0819 there is no equilibrium at any voltage** — the
  Casimir force alone drives the electrode to contact (spontaneous stiction).
  This is the hard geometric floor on miniaturization for a given (k, A). All
  three devices in §2 have β below this ceiling.

The boundary is a single-parameter curve `{α_c(u), β_c(u)} : u ∈ [2/3, 4/5]`
running from `(4/27, 0)` to `(0, 256/3125)` in the (α, β) plane — the pull-in
phase boundary plotted in the paper.

---

## 4. Thermal (Lifshitz) correction

Finite temperature adds the leading thermal correction to the Casimir pressure,
implemented as a multiplicative factor on β:

$$
\beta \;\longrightarrow\; \beta\Big[\,1+\delta\,\Big],\qquad
\delta=\frac{720\,\zeta(3)}{\pi^{3}}\,\frac{k_B T\,(d-x)}{\hbar c}.
$$

The correction is **linear in the instantaneous physical gap** `(d − x) = d(1−ξ)
= d·u`; it therefore couples *dynamically* — as the electrode moves the thermal
enhancement of the Casimir term breathes with `u(τ)`, softening the effective β
most near equilibrium (large u) and least near contact (u → 0).

**Prefactor.**
`720 ζ(3)/π³ = 720·1.202057/31.00628 = 27.913`, and
`k_B/(ħc) = 1.380649e-23 / (1.05457e-34·2.99792e8) = 436.70 m⁻¹ K⁻¹`. Hence

$$
\delta = 27.913 \cdot 436.70 \cdot T\,(d-x)
       = 1.219\times10^{4}\;\;T[\mathrm K]\,(d-x)[\mathrm m].
$$

### 4.1 Magnitude at d = 100 nm

Using the full gap `(d − x) = d = 100 nm` as an upper bound (electrode near rest):

| T | δ (full gap, d−x = d) | δ at fold (d−x = u·d, u ≈ 0.73) |
|---|---|---|
| 100 K | 0.122 (**12.2 %**) | ≈ 0.089 (**8.9 %**) |
| 300 K | 0.366 (**36.6 %**) | ≈ 0.266 (**26.6 %**) |

So the thermal Lifshitz term enhances the effective Casimir strength by
≈ 9–27 % at 100 nm over the 100–300 K range (evaluated at the fold, where
`d − x = u·d`). This is a **large, not negligible** correction at room
temperature and must be retained.

### 4.2 Resulting shift of the pull-in boundary (Device A)

Along the fold locus, `dα_c/dβ_c = (dα_c/du)/(dβ_c/du)`, which at Device A's
operating point (`β_A = 0.0260`, `u ≈ 0.728`) equals **−1.885**: raising the
effective Casimir strength lowers the electrostatic threshold. With
`Δβ = β_A·δ`:

| T | δ (at fold) | Δβ = β_A·δ | Δα_c = −1.885·Δβ | Δα_c/α_c | ΔV_PI/V_PI ≈ ½·Δα_c/α_c |
|---|---|---|---|---|---|
| 100 K | 0.089 | +2.3×10⁻³ | −4.4×10⁻³ | −4.6 % | **−2.3 %** |
| 300 K | 0.266 | +6.9×10⁻³ | −1.3×10⁻² | −13.7 % | **−6.9 %** |

Because `V_PI ∝ √α_c`, the room-temperature Lifshitz correction lowers the
predicted pull-in voltage of Device A by ≈ 7 % — a directly measurable,
temperature-dependent shift of the actuation threshold, and a genuine
prediction the paper can test.

### 4.3 The "≈ 7.6 µm at 300 K" crossover — confirmed, with a caveat

The bare thermal ratio `k_B T (d−x)/(ħc)` reaches unity when the gap equals the
**thermal (de Broglie–Wien) length**

$$
\lambda_T=\frac{\hbar c}{k_B T}.
$$

At T = 300 K: `λ_T = 3.1617e-26 / (1.380649e-23·300) = 7.633×10⁻⁶ m ≈ 7.6 µm`
(at 100 K, λ_T ≈ 22.9 µm). **The ≈ 7.6 µm claim is confirmed as this bare
thermal length.**

*Honest caveat:* this is the crossover of the *un-prefactored* ratio. The full
correction δ carries the factor `720ζ(3)/π³ ≈ 27.9`, so δ itself reaches order
unity already at `d = 1/(1.219×10⁴·300) ≈ 274 nm` at 300 K — more than an order
of magnitude tighter than λ_T. In practice, then: 7.6 µm marks where the bare
thermal scale becomes relevant, but the *actual* order-unity thermal
enhancement of β sets in near a few hundred nm at room temperature, which is
exactly the sub-µm regime of these devices. Both statements are consistent; the
paper should quote 7.6 µm as `λ_T` and ~270 nm as the effective onset.

---

## 5. Applied significance (for *Physical Review Applied*)

This model targets a concrete class of hardware: gold-coated MEMS/NEMS
capacitive actuators, torsional Casimir oscillators, RF-MEMS switches, and
tunable-gap resonant sensors operating at sub-100-nm gaps. For all of these,
**pull-in is the dominant failure and operating-limit mechanism**, and the
Casimir force is no longer a perturbation. Three applied consequences follow
directly from the analysis above:

1. **Pull-in voltage prediction.** The fold locus `{α_c(u), β_c(u)}` converts
   directly into a design curve for the maximum safe actuation voltage
   `V_PI = √(2α_c k d³/ε₀A)`. At 100 nm the Casimir pre-load cuts V_PI below the
   textbook `4/27` (β = 0) estimate; ignoring it over-predicts the stable
   voltage window and leads to unexpected stiction. Devices A/B above show
   V_PI ≈ 0.33 V, and the room-temperature Lifshitz term shaves a further ≈ 7 %
   — a difference designers must budget for.

2. **Casimir-force-limited miniaturization.** The ceiling `β_max = 256/3125`
   is a hard scaling law: since `β ∝ A/(k d⁵)`, shrinking the gap collapses the
   available voltage window catastrophically (∝ d⁻⁵). It quantifies the
   smallest gap a given (k, A) can hold open against the quantum vacuum before
   spontaneous collapse — the fundamental floor on capacitive-NEMS
   miniaturization, and a target for stiffness/area engineering.

3. **Sensing and metrology.** Near the fold the system is critically soft
   (`dF/dξ → 1`), so responsivity to force, voltage and — via the temperature
   dependence of δ — to *temperature* diverges. This underlies
   Casimir-force metrology, near-threshold force/mass sensing, and
   voltage/temperature transducers. The temperature-dependent boundary shift of
   §4 is itself a route to on-chip thermometry through the Lifshitz-modified
   pull-in voltage.

Together these make the (α, β, ζ) phase diagram, its fold, and the PINN
solutions of the transient dynamics an engineering design tool, not merely a
dynamical-systems curiosity — the applied contribution required by PRApplied.

---

### Summary of key numbers

- **Control parameters:** α = ε₀AV₀²/(2kd³), β = π²ħcA/(240kd⁵), ζ = 1/(2Q); all dimensionless (verified).
- **Fold:** α_c(u) = u²(4−5u)/2, β_c(u) = u⁴(3u−2)/2, u = 1−ξ. β = 0 ⇒ ξ_PI = 1/3, α_c = 4/27 ≈ 0.148.
- **Pull-in range:** 1/5 ≤ ξ_PI ≤ 1/3; Casimir ceiling β_max = 256/3125 ≈ 0.0819.
- **Representative devices:** (β, ζ, V_PI) = A (0.026, 5×10⁻⁵, 0.33 V), B (0.052, 5×10⁻⁴, 0.33 V), C (8.1×10⁻⁵, 1×10⁻⁴, 3.66 V).
- **Lifshitz shift at 100 nm (at fold):** δ ≈ 8.9 % (100 K), 26.6 % (300 K); ⇒ ΔV_PI/V_PI ≈ −2.3 % (100 K), −6.9 % (300 K).
- **Thermal length:** λ_T = ħc/k_BT = 7.63 µm at 300 K (bare crossover); prefactor 720ζ(3)/π³ ≈ 27.9 pulls effective onset to ≈ 274 nm.
