# Formal proofs for the notes' Section 5

Lean 4 proofs of the provable core of Section 5 of the unified-theory notes: the six propositions about Stuart–Landau networks, the obstruction behind P3's symmetry requirement, the local learning rule, the chaining closure, and the equilibrium-propagation identity that makes the local rule a gradient step. P3 and its consequences are proved for an arbitrary polynomial on-site potential, so the same theorems cover the cubic Stuart–Landau unit and the quintic bistable unit that a collaborator's model uses (`Oscillator/Potential.lean`). Everything here is checked by Lean against Mathlib; there is no `sorry` anywhere in `Oscillator/`.

## How to review this

A Lean `theorem` has two parts. The **statement** (everything before `:= by`) is the claim: its hypotheses and its conclusion. The **proof** (everything after) is checked mechanically by Lean's kernel, so a proof that compiles cannot be wrong about the statement it proves. What needs human review is therefore the statements: do they say what we want, under hypotheses we are willing to assume, about the objects we actually care about? This file lists, for every theorem, its hypotheses in plain words, the variables that appear in it, and a two-sentence picture of what it says in the stadium analogy from the `resonant` project. The Lean source has a docstring above each theorem saying the same thing more tersely.

Decisions taken here that the notes did not fix, and that are worth questioning:

* **The coupling is a complex Hermitian matrix, not a real symmetric one.** Hermitian (`Kⱼᵢ = conj(Kᵢⱼ)`) is the complex form of symmetric and is exactly the condition the energy needs; a real symmetric `K` is the special case. If the model will only ever use real couplings, nothing is lost by reading "Hermitian" as "symmetric".
* **The index set is any finite type `ι`, not `Fin N`.** This is what lets two modules be chained on `ι₁ ⊕ ι₂` and the torus be indexed by `ZMod m × ZMod m`. `N` means `card ι` throughout.
* **Solutions of the dynamics are hypotheses, never constructions.** Every "along a solution" theorem assumes a curve `w(t)` with `w'(t) = F(w(t))`. Existence and uniqueness are standard for this smooth field but are not proved, and nothing downstream (LaSalle convergence, the global attractor) that needs a flow is proved either.
* **The energy's factors.** With the notes' energy and the notes' field, the real gradient is `∇E = −2F`, not `−F`; see Conventions. The notes' `ẇ = −∂E/∂w̄` is the same statement in Wirtinger notation.
* **The coupling sum runs over all ordered pairs `(i, j)`**, so diagonal entries `Kᵢᵢ` act as shifts of `μᵢ`. Paper 02's diffusive coupling `Σⱼ Kᵢⱼ (wⱼ − wᵢ)` is this form with the self term folded into `μᵢ`, as the notes say.
* **The operator-norm hypothesis in P5** is the ℓ² operator norm written out as `‖Kv‖² ≤ κ²‖v‖²` for all `v`, rather than Mathlib's matrix norm instance.
* **Nondegeneracy in equilibrium propagation** (an invertible energy Hessian at the equilibrium) is a hypothesis, as it is in the original theorem; whether a given trained network satisfies it is an empirical question.

What is assumed and not proved, in one list: existence of solutions; convergence to equilibria (LaSalle); the global attractor; the nondegeneracy of equilibria; and P1 and P4 entirely (see below).

## Building it

```bash
brew install elan-init && elan-init -y   # Lean's version manager; puts ~/.elan/bin on your PATH
cd proofs
lake exe cache get                        # fetch Mathlib's prebuilt files once (a few GB)
lake build
```

`lake build` is the whole check; a theorem that did not prove would stop the build. The toolchain is pinned by `lean-toolchain` (Lean `v4.35.0`) and `lakefile.toml` (Mathlib `v4.35.0`, locked in `lake-manifest.json`); these are the latest stable releases as of 2026-10-18 (Lean `v4.36.0` exists only as a release candidate, which Mathlib's tagged releases do not yet target). To bump, change both pins to a matching Mathlib release tag and rerun `lake exe cache get`. Lean toolchains are per project, so the Homebrew formula only installs the manager `elan`; the Lean version a project uses is whatever its `lean-toolchain` file says.

## The objects and the symbols

The network is indexed by a finite type `ι` (the notes take `ι = Fin N`). The definitions live in `Oscillator/Basic.lean`.

| Symbol | Lean | Type | Meaning | In the stadium |
|---|---|---|---|---|
| `ι`, `N` | `ι`, `Fintype.card ι` | finite type, natural number | the units and how many there are | the fans in one section (paper 02: a 16×16 grid, `N = 256`) |
| `wᵢ` | `w : ι → ℂ` | one complex number per unit | the state of unit `i` in the rotating frame: `|wᵢ|` is the amplitude, `arg wᵢ` the phase | where fan `i`'s phone light points (phase) and how wide the arm's arc is (amplitude) |
| `μᵢ` | `μ : ι → ℝ` | one real number per unit | the threshold, the Hopf bifurcation parameter: below `0` the unit rests, above `0` it oscillates with amplitude `√μᵢ` | each fan's eagerness dial: how readily, and how widely, that fan swings at all |
| `Kᵢⱼ` | `K : Matrix ι ι ℂ` | complex `N×N` matrix | the coupling: how strongly unit `j` pulls on unit `i` | the peer-influence stencil, who sways whom and how hard |
| `uᵢ` | `u : ι → ℂ` | one complex number per unit | the input, constant in the rotating frame | the music from the stage, a steady shove on each fan |
| `ωᵢ` | `ω : ι → ℝ` | one real number per unit | the residual intrinsic frequency after going to the rotating frame of a common frequency | each fan's preferred tempo, measured relative to the crowd's shared tempo |
| `E(w)` | `energy μ K u w` | real number | the energy, a Lyapunov function | the altitude of the crowd's current pattern on a landscape |
| `F(w)` | `field μ K u w` | one complex number per unit | the vector field, `ẇ = F(w)`: `F(w)ᵢ = μᵢwᵢ − |wᵢ|²wᵢ + (Kw)ᵢ + uᵢ` | how every light is about to move |
| `F_ω(w)` | `rotField ω μ K u w` | as `F` | the field with the frequency term `iωᵢwᵢ` added | the same, with each fan also drifting at its own tempo |
| `⟨a, v⟩_ℝ` | `pairing a v` | real number | the real inner product `Σᵢ Re(conj(aᵢ) vᵢ)` on `ℂ^ι ≅ ℝ^{2ι}`; `pairing g` is the linear map `v ↦ ⟨g, v⟩_ℝ`, which is how "`g` is the gradient" is written | — |
| `S`, `‖w‖²` | `sumNormSq w` | real number | `Σᵢ |wᵢ|²`, the total squared amplitude | the total swing energy of the crowd |
| `t` | `t : ℝ` | real number | time along a solution `w : ℝ → ι → ℂ` | the concert clock |
| `ρ`, `Vᵢ(ρ)` | `V : ι → ℝ[X]` | real polynomial per unit | the on-site potential as a function of `ρ = |wᵢ|²` (`Oscillator/Potential.lean`) | the spring law of a fan's arm: how reach is rewarded or resisted |
| `gᵢ, sᵢ` | `g s : ι → ℝ` | real numbers per unit | the bistable unit's quadratic push and quintic restoring coefficients | an arm with two comfortable reaches |
| `κ, μ_max, U, g_max, s_min` | real numbers | bounds | `‖K‖ ≤ κ`, `μᵢ ≤ μ_max`, `|uᵢ| ≤ U`, `gᵢ ≤ g_max`, `sᵢ ≥ s_min` | the loudest shove, the strongest stencil, the most eager dial |
| `m`, `𝓕 v`, `W` | `ZMod m`, `ZMod.dft v`, `dftMatrix m` | — | the torus side length, the discrete Fourier transform, the DFT matrix (`Oscillator/Torus.lean`) | the stands wrap around with `m` seats per row; `𝓕` measures each wave pattern across the stands |
| `β`, `C`, `L(θ)` | `β : ℝ`, `cost`, `fun θ => C (w (θ, 0))` | — | the nudge strength, the cost on states, the loss as a function of the parameters `θ` (`Oscillator/EqProp.lean`) | how hard the coach pulls toward the right answer; the score; the score as a function of the dials |

**The stadium picture, once.** In `resonant`'s picture a section of the stadium is a grid of fans windmilling phone lights; where the light points is the fan's phase and, in the Stuart–Landau upgrade, how wide the arc is the amplitude, so `wᵢ` is one fan's light. Each fan has a preferred tempo (`ωᵢ`), copies neighbours through the section's peer-influence stencil (`K`), and is shoved by the music from the stage (`u`). The seating chart wraps around (the torus, P6). The theorems below are statements about that crowd: when its motion is a descent on a landscape (P3), when no landscape exists (the obstruction), how far arms can ever swing (P5), what the wrap-around seating does to the stencil (P6), what happens when sections are joined (chaining), and how a fan can adjust its own dial from what it alone experiences (the local rule and equilibrium propagation).

## The propositions

| P | Claim in the notes | Status | Where |
|---|---|---|---|
| P1 | Capacity bound: a linearly read system with `M` state variables has total information processing capacity at most `M` (Dambre et al. 2012) | not formalised | — |
| P2 | The threshold is the Hopf bifurcation parameter `μ`; one isolated unit has a stable origin for `μ < 0`, a limit cycle of radius `√μ` for `μ > 0`, and amplitude `F^{1/3}` under forcing `F` at `μ = 0` | the equilibria in all three regimes are proved; stability and the linear gain `1/|μ|` are not | `Oscillator/Unit.lean` |
| P3 | Energy function under symmetric coupling and a common frequency; `dE/dt ≤ 0`; `E` bounded below; local rules `∂E/∂μᵢ = −|wᵢ|²`, `∂E/∂Kᵢⱼ ∝ conj(wᵢ) wⱼ`; violated by asymmetric `K` and by heterogeneous `ωᵢ` | proved, including the two violations as theorems; proved again for any polynomial on-site potential `Vᵢ(|wᵢ|²)`, with the Stuart–Landau and the bistable unit as instances | `Oscillator/Energy.lean`, `Oscillator/LocalRule.lean`, `Oscillator/Obstruction.lean`, `Oscillator/Potential.lean` |
| P4 | Synchronisation transition: `K_c = 2γ` for all-to-all Kuramoto with Lorentzian frequencies, `r = √(1 − K_c/K)` above it | not formalised | — |
| P5 | Dissipativity: `d‖w‖²/dt ≤ 2(μ_max + κ)‖w‖² − (2/N)‖w‖⁴ + 2U√N‖w‖`, an absorbing ball, a global attractor | the differential inequality and the absorbing ball are proved, for the Stuart–Landau unit and, with the inequality restated, for the bistable unit; the attractor's existence is not | `Oscillator/Dissipativity.lean`, `Oscillator/Potential.lean` |
| P6 | Periodic boundaries make the kernel circulant on `Z_m × Z_m`, diagonalised by the two-dimensional DFT; the coupling ceiling is the kernel's operator norm | circulant structure and the diagonalisation are proved in one and two dimensions; the operator-norm consequence is not | `Oscillator/Torus.lean` |
| — | Chaining: two modules with Hermitian coupling joined by any inter-module coupling form one network with Hermitian coupling, so P3 applies to the chain | proved | `Oscillator/Chain.lean` |
| — | Equilibrium propagation: the loss gradient equals the `β → 0` limit of the contrast of `∂E/∂θ` between nudged and free equilibria | proved, for a `C¹` family of nudged equilibria and, via the implicit function theorem, at every nondegenerate equilibrium; instantiated as the network's threshold rule | `Oscillator/EqProp.lean` |

### Why P1 and P4 were not attempted

P1 is a statement about the space of functions of an input history and the fading-memory property of a driven dynamical system; Mathlib has none of this infrastructure (no reservoir theory, no information processing capacity), and the notes cite it rather than claim it. P4 is an infinite-population statement (the `N → ∞` Kuramoto self-consistency equation and the Ott–Antonsen reduction) about a different model (phase oscillators with random frequencies); it is the physics literature's result and lies far from what the finite-network definitions here express. Both are cited, not derived, in the notes, and the notes already mark P4 as "not a theorem".

## Theorem by theorem, with hypotheses in words

### `Oscillator/Energy.lean` (P3)

**Variables.** `w : ι → ℂ` the state; `μ : ι → ℝ` the thresholds; `K : Matrix ι ι ℂ` the coupling, assumed Hermitian; `u : ι → ℂ` the input; `v : ι → ℂ` a direction in state space; `w : ℝ → ι → ℂ` a curve of states and `t : ℝ` a time, when a solution is involved; `cᵢ = ½(Σⱼ|Kᵢⱼ| + Σⱼ|Kⱼᵢ|)` the coupling weight of unit `i` (`couplingWeight K i`), used only in the lower bound.

**Stadium picture.** Think of every possible pattern of lights as a point on a landscape and of `E` as its altitude. When each fan's pull on a neighbour is mirrored by that neighbour's pull back (Hermitian stencil) and the whole section windmills at one shared tempo (common frequency, so `ω = 0` in the rotating frame), the crowd's motion is always downhill on this landscape, at a rate equal to twice the squared speed of the lights, and the landscape has a floor. So the swirl must settle: the pattern cannot circulate forever. This is the structure that every learning rule in Section 7 that needs "settling" rests on.

* `hasFDerivAt_energy` — **the gradient structure.** Hypothesis: `K` is Hermitian (`K.IsHermitian`, the complex form of "real symmetric"). Conclusion: `E` is differentiable at every `w` with real gradient `−2 F(w)`, i.e. for every direction `v`, `dE(w)[v] = ⟨−2 F(w), v⟩_ℝ`. This is `F = −½∇E`, which in the notes' Wirtinger notation reads `Fᵢ = −∂E/∂w̄ᵢ` (see Conventions).
* `energy_hasDerivAt_solution` — Hypotheses: `K` Hermitian; a curve `w` with `w'(t) = F(w(t))` at time `t`. Conclusion: `d/dt E(w(t)) = −2 Σᵢ |F(w(t))ᵢ|²`.
* `energy_deriv_nonpos_solution` — same hypotheses; `d/dt E(w(t)) ≤ 0`.
* `energy_antitone_solution` — Hypotheses: `K` Hermitian; `w` solves the equation at every real time. Conclusion: `t ↦ E(w(t))` is non-increasing. `energy_antitoneOn_solution` is the same on a convex set of times for a solution defined there.
* `energyLowerBound_le`, `energy_bddBelow` — no hypotheses on `K`. `E(w) ≥ −Σᵢ (½(μᵢ + cᵢ + 1)² + |uᵢ|²)`, so `E` is bounded below.
* `fderiv_energy_eq_zero_iff` — `K` Hermitian. The critical points of `E` are exactly the equilibria `F(w) = 0`.
* `differentiable_energy`, `fderiv_energy_apply` — differentiability and the directional derivative written out.

LaSalle's conclusion in the notes ("every trajectory converges to the set of equilibria") is not formalised: Mathlib has no LaSalle invariance principle. What is proved is its two inputs, `dE/dt = −2Σ|Fᵢ|²` and `E` bounded below.

### `Oscillator/Obstruction.lean` (the symmetry requirement in P3)

**Variables.** In the general theorems: `V` any real normed space, `B : V →L[ℝ] V →L[ℝ] ℝ` a symmetric pairing (for us the real inner product), `G : V → V` a vector field, `φ : V → ℝ` a candidate potential, `G'` the Jacobian of `G` at a point `x`, `v w : V` directions. In the network theorems: `ω, μ, K, u` as above; `rotJacobian ω μ K w` is the Jacobian of `F_ω` at `w`.

**Stadium picture.** If fan A copies fan B harder than B copies A, or two fans keep different preferred tempos, then there is no landscape at all whose downhill direction is the crowd's motion, under any choice of altitude function: a hidden symmetry is violated. The symmetry is that of second derivatives (going east then north reaches the same altitude as north then east), which forces the cross-influences of any downhill flow to be mirror images. This is the precise content of the notes' "violated by heterogeneous `ωᵢ` and asymmetric `K`": with a trained, directed stencil the settling guarantee of P3 is gone, and convergence becomes empirical.

* `IsGradientFieldOf.jacobian_symm` — on any real normed space with a symmetric continuous pairing `B`: if `G` is the gradient field of some `φ` (`dφ(y) = B(G y)` for all `y`) and `G` is differentiable at `x` with Jacobian `G'`, then `B(G' v, w) = B(v, G' w)`. Proved from Mathlib's symmetry of second derivatives.
* `not_isGradientFieldOf_of_jacobian_not_symm` — the contrapositive: a non-symmetric Jacobian at one point rules out every potential.
* `jacobian_symm_of_hasGradientAt` — the same in a real inner product space for Mathlib's `HasGradientAt`.
* `hasFDerivAt_rotField`, `rotJacobian_apply` — the Stuart–Landau field with frequencies is differentiable, with Jacobian `v ↦ (μᵢ + iωᵢ − |wᵢ|²)vᵢ − 2Re(conj(wᵢ)vᵢ)wᵢ + (Kv)ᵢ`.
* `isHermitian_of_isGradientField_of_jacobian`, `isHermitian_of_isGradientField` — if a field whose Jacobian at the origin is `v ↦ (μᵢ + iωᵢ)vᵢ + (Kv)ᵢ` (in particular `F_ω`) is a gradient field for the real inner product, then `K + i·diag(ω)` is Hermitian.
* `not_isGradientField_of_not_isHermitian` — **asymmetric coupling.** `K` not Hermitian (common frequency): `F` is not a gradient field.
* `not_isGradientField_of_frequency` — **heterogeneous frequencies.** `K` Hermitian and some `ωᵢ ≠ 0`: `F_ω` is not a gradient field.
* `not_isGradientField_of_distinct_frequencies` — `K` Hermitian and two units with `ωᵢ ≠ ωⱼ`: there is no rotating frame `Ω` in which `F_{ω−Ω}` is a gradient field.

### `Oscillator/LocalRule.lean` (the local rules in P3)

**Variables.** `μ : ι → ℝ`, `K`, `u`, `w` as above; `δμ : ι → ℝ` a direction in threshold space; `δK : ι → ι → ℂ` a direction in coupling space (the coupling is handled as the function `Matrix.of K` so that the sup norm on `ι → ι → ℂ` provides the normed structure); `i j : ι` the units whose parameter is varied; `c : ℂ` the size of a single-entry change; `t : ℝ` the value of the one parameter being varied; `wβ, w0 : ι → ℂ` the nudged and free states and `β : ℝ` the nudge strength in the contrast statements.

**Stadium picture.** Each fan's eagerness dial `μᵢ` changes the landscape, and the slope of the landscape in that dial is just the square of that fan's own arm reach, `−|wᵢ|²`, with no term from any other fan. Likewise the slope in the stencil entry between fans `i` and `j` is the product of their two lights, `Re(conj(wᵢ)wⱼ)`, which is large when they swing together. So the energy's dependence on a parameter is readable at the parameter's own location, which is what makes a learning rule local: the dial can be adjusted from what that fan alone is doing, in two runs of the crowd (free and nudged), by comparing arm reaches.

* `hasFDerivAt_energy_mu`, `hasDerivAt_energy_mu_coord`, `deriv_energy_mu_coord` — **`∂E/∂μᵢ = −|wᵢ|²`**, exactly, with no hypothesis on `K`; in direction `δμ`, `dE[δμ] = −Σᵢ δμᵢ |wᵢ|²`.
* `hasFDerivAt_energy_K`, `fderiv_energy_K_apply` — in direction `δK`, `dE[δK] = −Σᵢⱼ Re(conj(wᵢ) δKᵢⱼ wⱼ)`.
* `fderiv_energy_K_single` — along the single entry `Kᵢⱼ ↦ Kᵢⱼ + c`: `dE = −Re(conj(wᵢ) wⱼ c)`. Read as a real gradient with respect to the complex entry, `∂E/∂Kᵢⱼ = −wᵢ conj(wⱼ) = −conj(conj(wᵢ) wⱼ)`.
* `fderiv_energy_K_symm` — along the notes' real symmetric direction `Kᵢⱼ = Kⱼᵢ = κ`: `dE/dκ = −2 Re(conj(wᵢ) wⱼ)`, the notes' `∂E/∂Kᵢⱼ`.
* `mu_contrast`, `K_contrast` — the equilibrium-propagation contrasts: `(∂E/∂μᵢ(w^β) − ∂E/∂μᵢ(w⁰))/β = −(|wᵢ^β|² − |wᵢ⁰|²)/β` and the coherence contrast for `K`, each local to the units involved. These follow by algebra from the derivatives above; the identity that makes the contrast the gradient of a loss is `eqprop_identity` in `Oscillator/EqProp.lean`.

### `Oscillator/Dissipativity.lean` (P5)

**Variables.** `ω, μ, K, u` as above (any frequencies, no symmetry of `K`); `κ ≥ 0` with `OpNormBound K κ`, meaning `‖Kv‖² ≤ κ²‖v‖²` for every `v`; `μ_max` with `μᵢ ≤ μ_max`; `U ≥ 0` with `|uᵢ| ≤ U`; `N = card ι`; a solution `w : ℝ → ι → ℂ` and a time `t`; `S = Σᵢ |wᵢ(t)|²`.

**Stadium picture.** However loud the stage and however strong the stencil, arms have a finite reach: the cubic restoring term `−|wᵢ|²wᵢ` (a fan cannot swing wider than the arm allows) grows faster than the eagerness dials, the peers' pulls and the shoves, all of which grow at most linearly in the reach. So the crowd's total swing energy `S` obeys a bound in which the negative `S²` term wins once `S` is large, and beyond an explicit radius `S` must decrease at a definite rate: the crowd lives in a ball it can never leave. This bounds the state, which the approximation results in paper 01 need; it says nothing about gradients.

* `dissipativity` — Hypotheses: `OpNormBound K κ`; `μᵢ ≤ μ_max` for all `i`; `0 ≤ U` and `|uᵢ| ≤ U` for all `i`; a curve `w` solving `w' = F_ω(w)` at time `t`. Conclusion: `d/dt Σᵢ|wᵢ|² ≤ 2(μ_max + κ)S − (2/N)S² + 2U√N√S`. `dissipativity_field` is the same without frequencies.
* `hasDerivAt_sumNormSq`, `hasDerivAt_sumNormSq_solution` — `d/dt Σ|wᵢ|² = 2 Σᵢ Re(conj(wᵢ) wᵢ')` along any differentiable curve, and along a solution; `dissipativity_bound` is the inequality at a fixed state.
* `sum_re_conj_mulVec_le`, `sum_re_conj_input_le`, `two_div_card_mul_sq_le` — the three bounds that make up the inequality: the coupling term by `κS` (Cauchy–Schwarz and the operator norm), the input term by `U√N√S` (Cauchy–Schwarz), and `(2/N)S² ≤ 2Σᵢ|wᵢ|⁴` (the square of a sum against the sum of squares).
* `dissipationBound_le_neg_two` — **the absorbing ball.** For `N > 0`, once `S ≥ max 1 (N(μ_max + κ + U√N + 1))` the right-hand side is at most `−2`.

The global attractor is not formalised; it needs a flow, which needs the existence theory that is left as a hypothesis throughout.

### `Oscillator/Torus.lean` (P6)

**Variables.** `G` a finite abelian group of positions (for us `ZMod m` or `ZMod m × ZMod m`); `K : Matrix G G R` a kernel with entries in a commutative ring `R` (for us `ℂ`); `v : G → R` the circulant's symbol, the stencil as a function of relative position (`circulant v` is the matrix `Kᵢⱼ = v(i − j)`); `χ : AddChar G R` an additive character of `G`; `m` the side length; `k : ZMod m` (or a pair) the wavenumber of a Fourier mode; `𝓕 v k` Mathlib's DFT of `v` at `k`; `W = dftMatrix m` the matrix whose columns are the modes; `dft2 m v k` the two-dimensional DFT.

**Stadium picture.** On a seating chart that wraps around, the stencil says "copy the fan two seats left and one row up with weight such-and-such", the same rule for every seat, so the stencil is a circulant and applying it is the convolution that `resonant` computes with an FFT. A plane wave across the stands (lights whose direction advances by a fixed angle per seat) is left looking exactly like itself by such a stencil, only scaled: the scale factor is the stencil's Fourier coefficient at that wave's wavelength. So the wave patterns are the natural coordinates of the torus, and the geometry acts through the kernel's spectrum, which is why the notes say geometry claims should be made through it.

* `circulant_of_translation_invariant` — a translation-invariant kernel on a finite abelian group (periodic boundary conditions) is circulant, `Kᵢⱼ = v(i − j)` with `v(d) = K d 0`.
* `circulant_mulVec_addChar` — every additive character `χ` of the group is an eigenvector of every circulant, with eigenvalue `Σ_l v(l) χ(−l)`.
* `circulant_mulVec_fourierMode` — **one dimension.** On `ZMod m`, `circulant v` maps the Fourier mode `j ↦ exp(2πi kj/m)` to `𝓕 v (k)` times itself, with Mathlib's `ZMod.dft`.
* `circulant_mul_dftMatrix`, `dftMatrix_mul_conjTranspose`, `circulant_eq_dftMatrix_mul_diagonal` — the matrix form: with `W` the DFT matrix, `circulant v · W = W · diag(𝓕 v)`, `W · Wᴴ = m·1`, hence `circulant v = W · diag(𝓕 v) · (m⁻¹ Wᴴ)`.
* `circulant_mulVec_fourierMode2`, `dft2_eq_iterated` — **two dimensions.** On `ZMod m × ZMod m` the product modes are eigenvectors with the two-dimensional DFT as eigenvalue, and that DFT is the iterated one-dimensional one. Paper 02's lattice is `m = 16`.
* `isHermitian_circulant`, `isHermitian_circulant_of_real_even` — a circulant is Hermitian iff `conj(v d) = v(−d)`; every real even kernel qualifies, so P3 applies to the torus network.

The notes' "coupling ceiling is the kernel's operator norm" (`= max_k |𝓕 v k|`, the spectral clamp in `resonant`) and the stability of the uniform state are not formalised; both read off the diagonalisation above.

### `Oscillator/Chain.lean` (chaining)

**Variables.** Two modules indexed by `ι₁` and `ι₂` with thresholds `μ₁, μ₂`, couplings `K₁, K₂` (Hermitian) and inputs `u₁, u₂`; `B : Matrix ι₁ ι₂ ℂ` the inter-module coupling, arbitrary (sparse, distance dependent, or anything else); `Bᴴ` its conjugate transpose, the return coupling; `chainCoupling K₁ K₂ B` the block matrix `[K₁ B; Bᴴ K₂]` on `ι₁ ⊕ ι₂`; `w₁ : ι₁ → ℂ`, `w₂ : ι₂ → ℂ` the two modules' states, glued as `Sum.elim w₁ w₂`.

**Stadium picture.** Two sections, each with a mirror-symmetric stencil, are joined by letting fans in one section glance at fans in the other through any stencil `B`, provided the glances back are the mirror image `Bᴴ`. The joined crowd is one bigger section with a mirror-symmetric stencil, so the landscape of P3 exists for it too, with altitude equal to the two sections' altitudes minus an interaction term; and the joined section can be joined again. This is the notes' "myelin gate" as sparse inter-module coupling, with the condition it must satisfy.

* `chainCoupling_isHermitian` — `K₁`, `K₂` Hermitian and `B` any inter-module coupling: the block matrix `[K₁ B; Bᴴ K₂]` is Hermitian. Repeating the construction chains any number of modules.
* `chain_field`, `chain_energy` — the chain's field is the two module fields plus `B w₂` and `Bᴴ w₁`; its energy is `E₁ + E₂ − 2Σᵢⱼ Re(conj(w₁ᵢ) Bᵢⱼ w₂ⱼ)`.
* `chain_energy_antitone`, `chain_hasFDerivAt_energy` — P3 for the chain.

### `Oscillator/Unit.lean` (P2)

**Variables.** One unit (`ι = Unit`), no coupling (`K = 0`); `μ : ℝ` its threshold; `F : ℂ` a constant input (real and positive in the compressive case); `z : ℂ` the unit's state.

**Stadium picture.** One fan alone. With the dial below zero the only resting state is the light hanging still; above zero the fan windmills with a steady reach `√μ` (in the rotating frame, every point on that circle is a resting state, since only the phase differs); exactly at zero, a steady shove of strength `F` produces a reach of `F^{1/3}`, so faint shoves are amplified and loud ones compressed, the cochlea's trick. P2 is a definition plus these facts; the stability of the resting states is not proved.

* `isEquilibrium_of_normSq_eq`, `isolated_equilibrium_iff_of_no_input` — `μ > 0`, no input: the circle `|w|² = μ` consists of equilibria in the rotating frame (the limit cycle of radius `√μ`), and the equilibria are exactly that circle and the origin.
* `eq_zero_of_isEquilibrium_of_neg` — `μ < 0`, no input: the origin is the only equilibrium.
* `norm_eq_rpow_third_of_isEquilibrium` — `μ = 0`, real forcing `F > 0`: every equilibrium has amplitude `|w| = F^{1/3}`.

### `Oscillator/EqProp.lean` (equilibrium propagation)

**Variables.** In the abstract setting: `Θ` the parameter space and `W` the state space, any real normed spaces (complete, for the implicit function theorem); `E : Θ → W → ℝ` the energy as a function of parameters and state; `C : W → ℝ` the cost (named `cost` where `C` would clash with polynomial notation); `β : ℝ` the nudge strength and `nudged E C θ β = E θ + β C` the nudged energy; `w : Θ × ℝ → W` the family of nudged equilibria, `w(θ, β)`; `θ₀` the current parameters and `h : Θ` a parameter direction; `L(θ) = C(w(θ, 0))` the loss. For the network: `Θ = ι → ℝ` (the thresholds), `W = ι → ℂ`, `E μ v = energy μ K u v`, `μ₀` the current thresholds, `w₀` the free equilibrium, and `i` a unit.

**Stadium picture.** Training without back-propagation: let the crowd settle on its own (the free run), then let it settle again while the coach gently pulls the lights toward the wanted answer (the nudged run, strength `β`), and have every fan compare its own arm reach between the two runs. The theorem says that this per-fan comparison, divided by `β`, converges as the pull gets gentle to exactly the slope of the score with respect to that fan's dial, the number back-propagation would compute by tracing blame through the whole swirl. The proof is the hidden symmetry again: the total energy of the nudged crowd, as a function of (dials, nudge), has symmetric second derivatives, and its two first derivatives are the dial sensitivity and the score.

* `eqprop_identity` — **the identity.** Hypotheses: `E` and `C` are `C²`; `w : Θ × ℝ → W` is `C¹` at `(θ₀, 0)` and, near `(θ₀, 0)`, `w(θ, β)` is a critical point of `E(θ, ·) + β C`. Conclusion: for every parameter direction `h`, the contrast function `β ↦ ∂_θE(θ₀, w(θ₀, β))·h` is differentiable at `β = 0` with derivative `d/dθ[C(w(θ, 0))]·h`, the gradient of the loss `L(θ) = C(w(θ, 0))`. Equivalently `(∂_θE(θ₀, w^β) − ∂_θE(θ₀, w⁰))·h / β → ∇L(θ₀)·h`. The proof is the symmetry of second derivatives (Mathlib's `ContDiffAt.isSymmSndFDerivAt`) of `Φ(θ, β) = E(θ, w) + β C(w)` at `w = w(θ, β)`, whose first derivative is `(∂_θE, C(w))` because the `∂_w` term vanishes at a critical point (the envelope theorem); `Φ` is `C²` for that reason even though `w` is only `C¹`.
* `exists_nudged_equilibria` — **the family exists.** Hypotheses: `Θ`, `W` complete; `E`, `C` `C²`; `w₀` a critical point of `E(θ₀, ·)` whose Hessian `fderiv (fderiv (E θ₀)) w₀` is invertible (a nondegenerate equilibrium). Conclusion: a `C¹` family `w(θ, β)` near `(θ₀, 0)` with `w(θ₀, 0) = w₀` of critical points of the nudged energies. This is Mathlib's `C¹` implicit function theorem (`ContDiffAt.implicitFunction`) applied to the nudged gradient `((θ, β), v) ↦ d_v(E(θ, ·) + β C)(v)`.
* `eqprop_of_nondegenerate` — the two combined: at a nondegenerate equilibrium the identity holds for the implicit family.
* `contDiff_energy` — the network's energy is `C²` (indeed polynomial) in `(μ, w)` jointly.
* `eqprop_threshold_rule` — **the notes' local threshold rule is a gradient step.** Hypotheses: `K` Hermitian; a `C²` cost `C` on states; `w₀` an equilibrium of the free network (`F(w₀) = 0`) whose energy Hessian is invertible. Conclusion: there is a `C¹` family `w(μ, β)` of nudged equilibria through `w₀` (critical points of `energy μ K u + β C`, i.e. equilibria of the nudged dynamics by `fderiv_energy_eq_zero_iff`), and for every unit `i` the amplitude-contrast function `β ↦ −|w(μ₀, β)ᵢ|²` has derivative at `0` equal to `∂L/∂μᵢ` with `L(μ) = C(w(μ, 0))`. So `(|wᵢ^β|² − |wᵢ⁰|²)/β → −∂L/∂μᵢ`: the update `Δμᵢ ∝ (|wᵢ^β|² − |wᵢ⁰|²)/β` descends the loss, and it is computed from unit `i` alone. The same argument with `K` as the parameter and `fderiv_energy_K_symm` gives the coupling rule; it is not written out separately. `eqprop_threshold_rule_V` in `Oscillator/Potential.lean` is the same statement for every potential of the threshold family, the bistable unit included.

What is not formalised here: that the nudged dynamics settle to `w(μ, β)` (that is the LaSalle step left open in P3), and the nondegeneracy of a given equilibrium, which is a hypothesis.

### `Oscillator/Potential.lean` (P3, P5 and the local rule for a general on-site potential)

**Variables.** `V : ι → ℝ[X]` one real polynomial per unit, the on-site potential, evaluated at `ρ = |wᵢ|²`; `Vᵢ'` its derivative (`Polynomial.derivative`); `energyV V K u w` and `fieldV V K u w` the energy and field with these potentials, `rotFieldV ω V K u w` with frequencies; `slPotential μ` and `bistablePotential μ g s` the two instances, with `g` the quadratic (subcritical) coefficient and `s` the quintic (restoring) coefficient; `thresholdPotential W μ` the family `Vᵢ(ρ) = −μᵢρ + Wᵢ(ρ)` with `W : ι → ℝ[X]` the fixed remainder; `g_max ≥ 0` and `s_min ≥ 0` bounds with `gᵢ ≤ g_max`, `sᵢ ≥ s_min`; `dissipationBoundBistable N μ_max g_max s_min κ U S` the restated P5 right-hand side.

**Stadium picture.** The spring law of a fan's arm need not be the Stuart–Landau cubic. Any polynomial law `V(ρ)` in the squared reach will do, and in particular the bistable law, where a fan has two comfortable reaches, small and wide, and a shove can flip it from one to the other and leave it there (a fan that remembers). Every statement above about the landscape, its floor, the obstruction, the dial slope `−|wᵢ|²` and the nudge-contrast rule holds for all such laws, with the same proofs; only the finite-reach bound of P5 changes shape, because the bistable law pushes outward at intermediate reach before its quintic term pulls back.

The energy with an arbitrary real polynomial on-site potential `Vᵢ`, evaluated at `ρ = |wᵢ|²`:

`E_V(w) = Σᵢ Vᵢ(|wᵢ|²) − Σᵢⱼ Re(conj(wᵢ) Kᵢⱼ wⱼ) − 2Σᵢ Re(conj(uᵢ) wᵢ)`, with gradient flow `F_V(w)ᵢ = −Vᵢ'(|wᵢ|²) wᵢ + (Kw)ᵢ + uᵢ` (`energyV`, `fieldV`, and `rotFieldV` with frequencies).

Two instances are spelled out. `slPotential μ` is `V(ρ) = −μρ + ½ρ²`, the Stuart–Landau unit with dynamics `(μ − |w|²)w`; it recovers `energy` and `field` exactly (`energyV_sl`, `fieldV_sl`), so everything in `Oscillator/Energy.lean` is the cubic instance of the theorems below. `bistablePotential μ g s` is `V(ρ) = −μρ − (g/2)ρ² + (s/3)ρ³` with `s > 0`, the subcritical, bistable unit with dynamics `(μ + g|w|² − s|w|⁴)w` (`fieldV_bistable`); **this quintic unit is the one a collaborator's model uses, so the proofs here cover both units.**

* `hasFDerivAt_energyV` — **P3 for any potential.** `K` Hermitian: `∇E_V = −2F_V` for the real inner product.
* `energyV_hasDerivAt_solution`, `energyV_deriv_nonpos_solution`, `energyV_antitone_solution` — along a curve assumed to solve `ẇ = F_V(w)`, `dE_V/dt = −2Σᵢ|F_V(w)ᵢ|² ≤ 0`, and `E_V` is non-increasing along global solutions.
* `fderiv_energyV_eq_zero_iff` — critical points of `E_V` are the zeros of `F_V`.
* `energyV_bddBelow` — **boundedness.** If every `Vᵢ` has degree at least `2` and positive leading coefficient, `E_V` is bounded below for every `K` and `u` (`exists_lowerBound_sub_linear`: such a polynomial dominates any linear function on `[0, ∞)`). The degree condition is necessary, since a linear potential loses to a strong enough coupling. `bistable_energy_bddBelow` is the instance `s > 0` (degree `3`, leading coefficient `s/3`).
* `hasFDerivAt_rotFieldV_zero`, `isHermitian_of_isGradientFieldV`, `not_isGradientFieldV_of_not_isHermitian`, `not_isGradientFieldV_of_frequency` — **the obstruction for any potential.** The Jacobian at the origin is `v ↦ (−Vᵢ'(0) + iωᵢ)vᵢ + (Kv)ᵢ`, so a non-Hermitian `K` or a nonzero residual frequency rules out every potential function, exactly as for the Stuart–Landau unit.
* `thresholdPotential W μ` — potentials `Vᵢ(ρ) = −μᵢρ + Wᵢ(ρ)` with the threshold as the learnable parameter; both instances have this form (`slPotential_eq_threshold`, `bistablePotential_eq_threshold`).
* `hasFDerivAt_energyV_mu`, `hasDerivAt_energyV_mu_coord`, `deriv_energyV_mu_coord` — **the local rule for any potential.** `∂E_V/∂μᵢ = −|wᵢ|²` exactly, whatever the remainder `Wᵢ`.
* `dissipativity_bistable` — **P5 for the bistable unit.** Hypotheses: `OpNormBound K κ`; `μᵢ ≤ μ_max`; `gᵢ ≤ g_max` with `g_max ≥ 0`; `sᵢ ≥ s_min ≥ 0`; `0 ≤ U`, `|uᵢ| ≤ U`; a curve solving `ẇ = F_{V,ω}(w)` at time `t` (frequencies allowed). Conclusion: `d/dt Σᵢ|wᵢ|² ≤ 2(μ_max + κ)S + 2g_max S² − (2s_min/N²)S³ + 2U√N√S`. The inequality changes from the cubic case in two places: the destabilising `+2g_max S²` from the subcritical term, and the restoring `−(2s_min/N²)S³` from the quintic term (`cube_sumNormSq_le`, the power-mean inequality `S³ ≤ N² Σᵢ|wᵢ|⁶`). `dissipationBoundBistable_le_neg_two` is the absorbing ball: for `s_min > 0`, once `S ≥ max 1 (N²(|μ_max + κ| + g_max + U√N + 1)/s_min)` the bound is at most `−2`.
* `contDiff_energyV_threshold`, `eqprop_threshold_rule_V` — equilibrium propagation for the whole threshold family: `(|wᵢ^β|² − |wᵢ⁰|²)/β → −∂L/∂μᵢ` at a nondegenerate equilibrium, for the bistable unit as for the Stuart–Landau unit.

## Conventions (align the notes to these)

1. `|z|²` is `Complex.normSq z = Re(z)² + Im(z)²`.
2. The coupling term of the energy is `−Σᵢⱼ Re(conj(wᵢ) Kᵢⱼ wⱼ)` over **all ordered pairs** `(i, j)`, for a Hermitian `K`. For a real symmetric `K` with zero diagonal it equals the notes' `−2Σ_{i<j} Kᵢⱼ Re(w̄ᵢ wⱼ)`. A diagonal entry `Kᵢᵢ` contributes `−Re(Kᵢᵢ)|wᵢ|²`, a shift of `μᵢ` by `Re Kᵢᵢ`.
3. The input term is `−2Σᵢ Re(conj(uᵢ) wᵢ)`, as in the notes.
4. The field is `F(w)ᵢ = μᵢwᵢ − |wᵢ|²wᵢ + (Kw)ᵢ + uᵢ`. With this `E` and this `F`, the **real gradient** of `E` for the inner product `⟨a, v⟩_ℝ = Σᵢ Re(conj(aᵢ) vᵢ)` on `ℂ^ι ≅ ℝ^{2ι}` is `∇E = −2F`, so `F = −½∇E`. The notes' `ẇᵢ = −∂E/∂w̄ᵢ` is the same statement in Wirtinger notation, because `∂/∂w̄ = ½(∂/∂x + i∂/∂y)` carries the factor `½`. Both give `dE/dt = −2Σᵢ|ẇᵢ|²`, which is what is proved.
5. `∂E/∂μᵢ = −|wᵢ|²` exactly. For the coupling, the real gradient with respect to the complex entry `Kᵢⱼ` is `−wᵢ conj(wⱼ) = −conj(conj(wᵢ) wⱼ)`, and along the symmetric direction `Kᵢⱼ = Kⱼᵢ = κ` the derivative is `−2Re(conj(wᵢ)wⱼ)`, which is the notes' `∂E/∂Kᵢⱼ`. The notes' `−conj(wᵢ)wⱼ` should be read as the symmetric-direction derivative up to the factor 2 and the conjugation of the real-gradient convention.
6. The operator norm in P5 is the ℓ² operator norm, stated as `‖Kv‖² ≤ κ²‖v‖²` for all `v`; `N` is `card ι`; `μ_max` and `U` are any upper bounds (`μᵢ ≤ μ_max`, `|uᵢ| ≤ U`, `U ≥ 0`).
7. The DFT on `ZMod m` is Mathlib's `ZMod.dft`, `𝓕 v (k) = Σⱼ exp(−2πi jk/m) v(j)`, with the Fourier mode `k` being `j ↦ exp(2πi kj/m)`; the DFT matrix `W` has these modes as columns and satisfies `W·Wᴴ = m·1`.
8. For a general on-site potential the energy's on-site term is `Vᵢ(|wᵢ|²)` with `Vᵢ` a real polynomial in `ρ = |wᵢ|²`, and the field is `F_V(w)ᵢ = −Vᵢ'(|wᵢ|²)wᵢ + (Kw)ᵢ + uᵢ`; the Stuart–Landau unit is `V(ρ) = −μρ + ½ρ²` and the bistable unit is `V(ρ) = −μρ − (g/2)ρ² + (s/3)ρ³`, `s > 0`. Convention 4 (`∇E_V = −2F_V`) and convention 5 (`∂E_V/∂μᵢ = −|wᵢ|²`) hold verbatim for every such potential.
