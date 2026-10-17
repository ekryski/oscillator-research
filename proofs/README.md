# Formal proofs for the notes' Section 5

Lean 4 proofs of the provable core of Section 5 of the unified-theory notes: the six propositions about Stuart–Landau networks, the obstruction behind P3's symmetry requirement, the local learning rule, and the chaining closure. Everything here is checked by Lean against Mathlib; there is no `sorry` anywhere in `Oscillator/`.

```bash
cd proofs
lake exe cache get   # fetch Mathlib's prebuilt files once (a few GB)
lake build
```

Pinned toolchain: Lean `v4.35.0` (`lean-toolchain`) and Mathlib `v4.35.0` (`lakefile.toml`, locked in `lake-manifest.json`). Lean's `lake build` is the whole check; a theorem that did not prove would stop the build.

## What the objects are

The network is indexed by any finite type `ι` (the notes take `ι = Fin N`; the general index is what lets two modules be chained on `ι₁ ⊕ ι₂`). A state is `w : ι → ℂ`, the thresholds are `μ : ι → ℝ`, the coupling is a complex matrix `K`, the input `u : ι → ℂ` is constant in the rotating frame, and intrinsic frequencies `ω : ι → ℝ` enter only where the notes discuss them. The definitions live in `Oscillator/Basic.lean`:

| Lean name | Meaning |
|---|---|
| `energy μ K u w` | `E(w) = Σᵢ(−μᵢ|wᵢ|² + ½|wᵢ|⁴) − Σᵢⱼ Re(conj(wᵢ) Kᵢⱼ wⱼ) − 2Σᵢ Re(conj(uᵢ) wᵢ)` |
| `field μ K u w` | `F(w)ᵢ = μᵢ wᵢ − |wᵢ|² wᵢ + (K w)ᵢ + uᵢ` |
| `rotField ω μ K u w` | `F_ω(w)ᵢ = F(w)ᵢ + i ωᵢ wᵢ`, the field with residual frequencies |
| `pairing a` | the real inner product `v ↦ ⟨a, v⟩_ℝ = Σᵢ Re(conj(aᵢ) vᵢ)` on `ℂ^ι ≅ ℝ^{2ι}` |
| `sumNormSq w` | `‖w‖² = Σᵢ |wᵢ|²` |
| `IsEquilibrium μ K u w` | `F(w) = 0` |

A solution of the dynamics is always a hypothesis, never a construction: a curve `w : ℝ → ι → ℂ` with `HasDerivAt w (field μ K u (w t)) t`. Existence of solutions is not proved; it is standard (the field is smooth) and the notes do not rest on it.

## The propositions

| P | Claim in the notes | Status | Where |
|---|---|---|---|
| P1 | Capacity bound: a linearly read system with `M` state variables has total information processing capacity at most `M` (Dambre et al. 2012) | not formalised | — |
| P2 | The threshold is the Hopf bifurcation parameter `μ`; one isolated unit has a stable origin for `μ < 0`, a limit cycle of radius `√μ` for `μ > 0`, and amplitude `F^{1/3}` under forcing `F` at `μ = 0` | the equilibria in all three regimes are proved; stability and the linear gain `1/|μ|` are not | `Oscillator/Unit.lean` |
| P3 | Energy function under symmetric coupling and a common frequency; `dE/dt ≤ 0`; `E` bounded below; local rules `∂E/∂μᵢ = −|wᵢ|²`, `∂E/∂Kᵢⱼ ∝ conj(wᵢ) wⱼ`; violated by asymmetric `K` and by heterogeneous `ωᵢ` | proved, including the two violations as theorems | `Oscillator/Energy.lean`, `Oscillator/LocalRule.lean`, `Oscillator/Obstruction.lean` |
| P4 | Synchronisation transition: `K_c = 2γ` for all-to-all Kuramoto with Lorentzian frequencies, `r = √(1 − K_c/K)` above it | not formalised | — |
| P5 | Dissipativity: `d‖w‖²/dt ≤ 2(μ_max + κ)‖w‖² − (2/N)‖w‖⁴ + 2U√N‖w‖`, an absorbing ball, a global attractor | the differential inequality and the absorbing ball are proved; the attractor's existence is not | `Oscillator/Dissipativity.lean` |
| P6 | Periodic boundaries make the kernel circulant on `Z_m × Z_m`, diagonalised by the two-dimensional DFT; the coupling ceiling is the kernel's operator norm | circulant structure and the diagonalisation are proved in one and two dimensions; the operator-norm consequence is not | `Oscillator/Torus.lean` |
| — | Chaining: two modules with Hermitian coupling joined by any inter-module coupling form one network with Hermitian coupling, so P3 applies to the chain | proved | `Oscillator/Chain.lean` |
| — | Equilibrium propagation: the loss gradient equals the `β → 0` limit of the contrast of `∂E/∂θ` between nudged and free equilibria | not formalised yet (see the end of this file) | — |

### Why P1 and P4 were not attempted

P1 is a statement about the space of functions of an input history and the fading-memory property of a driven dynamical system; Mathlib has none of this infrastructure (no reservoir theory, no information processing capacity), and the notes cite it rather than claim it. P4 is an infinite-population statement (the `N → ∞` Kuramoto self-consistency equation and the Ott–Antonsen reduction) about a different model (phase oscillators with random frequencies); it is the physics literature's result and lies far from what the finite-network definitions here express. Both are cited, not derived, in the notes, and the notes already mark P4 as "not a theorem".

## Theorem by theorem, with hypotheses in words

### `Oscillator/Energy.lean` (P3)

* `hasFDerivAt_energy` — **the gradient structure.** Hypothesis: `K` is Hermitian (`K.IsHermitian`, the complex form of "real symmetric"). Conclusion: `E` is differentiable at every `w` with real gradient `−2 F(w)`, i.e. for every direction `v`, `dE(w)[v] = ⟨−2 F(w), v⟩_ℝ`. This is `F = −½∇E`, which in the notes' Wirtinger notation reads `Fᵢ = −∂E/∂w̄ᵢ` (see Conventions).
* `energy_hasDerivAt_solution` — Hypotheses: `K` Hermitian; a curve `w` with `w'(t) = F(w(t))` at time `t`. Conclusion: `d/dt E(w(t)) = −2 Σᵢ |F(w(t))ᵢ|²`.
* `energy_deriv_nonpos_solution` — same hypotheses; `d/dt E(w(t)) ≤ 0`.
* `energy_antitone_solution` — Hypotheses: `K` Hermitian; `w` solves the equation at every real time. Conclusion: `t ↦ E(w(t))` is non-increasing. `energy_antitoneOn_solution` is the same on a convex set of times for a solution defined there.
* `energyLowerBound_le`, `energy_bddBelow` — no hypotheses on `K`. `E(w) ≥ −Σᵢ (½(μᵢ + cᵢ + 1)² + |uᵢ|²)` where `cᵢ = ½(Σⱼ|Kᵢⱼ| + Σⱼ|Kⱼᵢ|)` (`couplingWeight`), so `E` is bounded below.
* `fderiv_energy_eq_zero_iff` — `K` Hermitian. The critical points of `E` are exactly the equilibria `F(w) = 0`.
* `differentiable_energy`, `fderiv_energy_apply` — differentiability and the directional derivative written out.

LaSalle's conclusion in the notes ("every trajectory converges to the set of equilibria") is not formalised: Mathlib has no LaSalle invariance principle. What is proved is its two inputs, `dE/dt = −2Σ|Fᵢ|²` and `E` bounded below.

### `Oscillator/Obstruction.lean` (the symmetry requirement in P3)

* `IsGradientFieldOf.jacobian_symm` — on any real normed space with a symmetric continuous pairing `B`: if `G` is the gradient field of some `φ` (`dφ(y) = B(G y)` for all `y`) and `G` is differentiable at `x` with Jacobian `G'`, then `B(G' v, w) = B(v, G' w)`. Proved from Mathlib's symmetry of second derivatives.
* `not_isGradientFieldOf_of_jacobian_not_symm` — the contrapositive: a non-symmetric Jacobian at one point rules out every potential.
* `jacobian_symm_of_hasGradientAt` — the same in a real inner product space for Mathlib's `HasGradientAt`.
* `hasFDerivAt_rotField`, `rotJacobian_apply` — the Stuart–Landau field with frequencies is differentiable, with Jacobian `v ↦ (μᵢ + iωᵢ − |wᵢ|²)vᵢ − 2Re(conj(wᵢ)vᵢ)wᵢ + (Kv)ᵢ`.
* `isHermitian_of_isGradientField` — if `F_ω` is a gradient field for the real inner product, then `K + i·diag(ω)` is Hermitian.
* `not_isGradientField_of_not_isHermitian` — **asymmetric coupling.** `K` not Hermitian (common frequency): `F` is not a gradient field.
* `not_isGradientField_of_frequency` — **heterogeneous frequencies.** `K` Hermitian and some `ωᵢ ≠ 0`: `F_ω` is not a gradient field.
* `not_isGradientField_of_distinct_frequencies` — `K` Hermitian and two units with `ωᵢ ≠ ωⱼ`: there is no rotating frame `Ω` in which `F_{ω−Ω}` is a gradient field.

### `Oscillator/LocalRule.lean` (the local rules in P3)

* `hasFDerivAt_energy_mu`, `hasDerivAt_energy_mu_coord`, `deriv_energy_mu_coord` — **`∂E/∂μᵢ = −|wᵢ|²`**, exactly, with no hypothesis on `K`; in direction `δμ`, `dE[δμ] = −Σᵢ δμᵢ |wᵢ|²`.
* `hasFDerivAt_energy_K`, `fderiv_energy_K_apply` — in direction `δK`, `dE[δK] = −Σᵢⱼ Re(conj(wᵢ) δKᵢⱼ wⱼ)` (the coupling is taken as a function `ι → ι → ℂ`, i.e. `Matrix.of K`, to use the sup norm).
* `fderiv_energy_K_single` — along the single entry `Kᵢⱼ ↦ Kᵢⱼ + c`: `dE = −Re(conj(wᵢ) wⱼ c)`. Read as a real gradient with respect to the complex entry, `∂E/∂Kᵢⱼ = −wᵢ conj(wⱼ) = −conj(conj(wᵢ) wⱼ)`.
* `fderiv_energy_K_symm` — along the notes' real symmetric direction `Kᵢⱼ = Kⱼᵢ = κ`: `dE/dκ = −2 Re(conj(wᵢ) wⱼ)`, the notes' `∂E/∂Kᵢⱼ`.
* `mu_contrast`, `K_contrast` — the equilibrium-propagation contrasts: `(∂E/∂μᵢ(w^β) − ∂E/∂μᵢ(w⁰))/β = −(|wᵢ^β|² − |wᵢ⁰|²)/β` and the coherence contrast for `K`, each local to the units involved. These follow by algebra from the derivatives above; the identity that makes the contrast the gradient of a loss is the equilibrium-propagation theorem, see the end of this file.

### `Oscillator/Dissipativity.lean` (P5)

* `dissipativity` — Hypotheses: `‖K‖ ≤ κ` in ℓ² operator norm, stated as `OpNormBound K κ` (`κ ≥ 0` and `‖Kv‖² ≤ κ²‖v‖²` for all `v`); `μᵢ ≤ μ_max` for all `i`; `0 ≤ U` and `|uᵢ| ≤ U` for all `i`; a curve `w` solving `w' = F_ω(w)` at time `t` (any frequencies, no symmetry of `K`). Conclusion: `d/dt Σᵢ|wᵢ|² ≤ 2(μ_max + κ)S − (2/N)S² + 2U√N√S` with `S = Σᵢ|wᵢ(t)|²` and `N = card ι`. `dissipativity_field` is the same without frequencies.
* `hasDerivAt_sumNormSq_solution` — `d/dt Σ|wᵢ|² = 2 Σᵢ Re(conj(wᵢ) F_ω(w)ᵢ)`; `dissipativity_bound` is the inequality at a fixed state.
* `dissipationBound_le_neg_two` — **the absorbing ball.** For `N > 0`, once `S ≥ max 1 (N(μ_max + κ + U√N + 1))` the right-hand side is at most `−2`.

The global attractor is not formalised; it needs a flow, which needs the existence theory that is left as a hypothesis throughout.

### `Oscillator/Torus.lean` (P6)

* `circulant_of_translation_invariant` — a translation-invariant kernel on a finite abelian group (periodic boundary conditions) is circulant, `Kᵢⱼ = v(i − j)` with `v(d) = K d 0`.
* `circulant_mulVec_addChar` — every additive character `χ` of the group is an eigenvector of every circulant, with eigenvalue `Σ_l v(l) χ(−l)`.
* `circulant_mulVec_fourierMode` — **one dimension.** On `ZMod m`, `circulant v` maps the Fourier mode `j ↦ exp(2πi kj/m)` to `𝓕 v (k)` times itself, with Mathlib's `ZMod.dft`.
* `circulant_mul_dftMatrix`, `dftMatrix_mul_conjTranspose`, `circulant_eq_dftMatrix_mul_diagonal` — the matrix form: with `W` the DFT matrix, `circulant v · W = W · diag(𝓕 v)`, `W · Wᴴ = m·1`, hence `circulant v = W · diag(𝓕 v) · (m⁻¹ Wᴴ)`.
* `circulant_mulVec_fourierMode2`, `dft2_eq_iterated` — **two dimensions.** On `ZMod m × ZMod m` the product modes are eigenvectors with the two-dimensional DFT as eigenvalue, and that DFT is the iterated one-dimensional one. Paper 02's lattice is `m = 16`.
* `isHermitian_circulant`, `isHermitian_circulant_of_real_even` — a circulant is Hermitian iff `conj(v d) = v(−d)`; every real even kernel qualifies, so P3 applies to the torus network.

The notes' "coupling ceiling is the kernel's operator norm" (`= max_k |𝓕 v k|`) and the stability of the uniform state are not formalised; both read off the diagonalisation above.

### `Oscillator/Chain.lean` (chaining)

* `chainCoupling_isHermitian` — `K₁`, `K₂` Hermitian and `B` any inter-module coupling: the block matrix `[K₁ B; Bᴴ K₂]` is Hermitian. Repeating the construction chains any number of modules.
* `chain_field`, `chain_energy` — the chain's field is the two module fields plus `B w₂` and `Bᴴ w₁`; its energy is `E₁ + E₂ − 2Σᵢⱼ Re(conj(w₁ᵢ) Bᵢⱼ w₂ⱼ)`.
* `chain_energy_antitone`, `chain_hasFDerivAt_energy` — P3 for the chain.

### `Oscillator/Unit.lean` (P2)

For the isolated unit (`ι = Unit`, `K = 0`, constant input `F`):

* `isEquilibrium_of_normSq_eq`, `isolated_equilibrium_iff_of_no_input` — `μ > 0`, no input: the circle `|w|² = μ` consists of equilibria in the rotating frame (the limit cycle of radius `√μ`), and the equilibria are exactly that circle and the origin.
* `eq_zero_of_isEquilibrium_of_neg` — `μ < 0`, no input: the origin is the only equilibrium.
* `norm_eq_rpow_third_of_isEquilibrium` — `μ = 0`, real forcing `F > 0`: every equilibrium has amplitude `|w| = F^{1/3}`.

## Conventions (align the notes to these)

1. `|z|²` is `Complex.normSq z = Re(z)² + Im(z)²`.
2. The coupling term of the energy is `−Σᵢⱼ Re(conj(wᵢ) Kᵢⱼ wⱼ)` over **all ordered pairs** `(i, j)`, for a Hermitian `K`. For a real symmetric `K` with zero diagonal it equals the notes' `−2Σ_{i<j} Kᵢⱼ Re(w̄ᵢ wⱼ)`. A diagonal entry `Kᵢᵢ` contributes `−Re(Kᵢᵢ)|wᵢ|²`, a shift of `μᵢ` by `Re Kᵢᵢ`.
3. The input term is `−2Σᵢ Re(conj(uᵢ) wᵢ)`, as in the notes.
4. The field is `F(w)ᵢ = μᵢwᵢ − |wᵢ|²wᵢ + (Kw)ᵢ + uᵢ`. With this `E` and this `F`, the **real gradient** of `E` for the inner product `⟨a, v⟩_ℝ = Σᵢ Re(conj(aᵢ) vᵢ)` on `ℂ^ι ≅ ℝ^{2ι}` is `∇E = −2F`, so `F = −½∇E`. The notes' `ẇᵢ = −∂E/∂w̄ᵢ` is the same statement in Wirtinger notation, because `∂/∂w̄ = ½(∂/∂x + i∂/∂y)` carries the factor `½`. Both give `dE/dt = −2Σᵢ|ẇᵢ|²`, which is what is proved.
5. `∂E/∂μᵢ = −|wᵢ|²` exactly. For the coupling, the real gradient with respect to the complex entry `Kᵢⱼ` is `−wᵢ conj(wⱼ) = −conj(conj(wᵢ) wⱼ)`, and along the symmetric direction `Kᵢⱼ = Kⱼᵢ = κ` the derivative is `−2Re(conj(wᵢ)wⱼ)`, which is the notes' `∂E/∂Kᵢⱼ`. The notes' `−conj(wᵢ)wⱼ` should be read as the symmetric-direction derivative up to the factor 2 and the conjugation of the real-gradient convention.
6. The operator norm in P5 is the ℓ² operator norm, stated as `‖Kv‖² ≤ κ²‖v‖²` for all `v`; `N` is `card ι`; `μ_max` and `U` are any upper bounds (`μᵢ ≤ μ_max`, `|uᵢ| ≤ U`, `U ≥ 0`).
7. The DFT on `ZMod m` is Mathlib's `ZMod.dft`, `𝓕 v (k) = Σⱼ exp(−2πi jk/m) v(j)`, with the Fourier mode `k` being `j ↦ exp(2πi kj/m)`; the DFT matrix `W` has these modes as columns and satisfies `W·Wᴴ = m·1`.

## Equilibrium propagation

Not formalised in this version. What it would need: the equilibrium-propagation identity for a `C¹` family of nudged equilibria `w^β(θ)` of `E(θ, ·) + β C`, which follows from the symmetry of second derivatives of `Φ(θ, β) = E(θ, w^β) + β C(w^β)`, and the existence of that family near a nondegenerate equilibrium from the implicit function theorem (`ContDiffAt.implicitFunction` in this Mathlib). The contrasts that the identity turns into a gradient step are already proved (`mu_contrast`, `K_contrast`).
