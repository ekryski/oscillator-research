import Mathlib

/-!
# Stuart–Landau networks: the objects

A finite network of Stuart–Landau units indexed by a finite type `ι` (the notes take
`ι = Fin N`; the general index type is what lets two networks be chained on `ι₁ ⊕ ι₂`).
Each unit carries a complex state `w i : ℂ` (its amplitude in the rotating frame), a real
bifurcation parameter `μ i`, and the units are coupled through a matrix `K : Matrix ι ι ℂ`
and driven by an input `u : ι → ℂ` that is constant in the rotating frame.

## Conventions

These fix the factors left open in Section 5, P3 of the notes.

* `|z|²` is `Complex.normSq z = z.re ^ 2 + z.im ^ 2`.
* The energy is

  `E(w) = ∑ i (−μ_i |w_i|² + ½ |w_i|⁴) − ∑ i j Re(conj(w_i) K_ij w_j) − 2 ∑ i Re(conj(u_i) w_i)`.

  The coupling sum runs over *all* ordered pairs `(i, j)`.  For a real symmetric `K` with zero
  diagonal it equals the notes' `−2 ∑_{i<j} K_ij Re(conj(w_i) w_j)`; a diagonal entry `K_ii`
  contributes `−Re(K_ii) |w_i|²`, which is a shift of `μ_i` by `Re K_ii`.
* The vector field is `F(w)_i = μ_i w_i − |w_i|² w_i + (K w)_i + u_i`; with intrinsic
  frequencies it is `F_ω(w)_i = F(w)_i + i ω_i w_i`.
* The real inner product on `ℂ^ι ≅ ℝ^{2ι}` is `⟨a, v⟩_ℝ = ∑ i Re(conj(a_i) v_i)`.  With respect to
  it the gradient of `E` is `∇E = −2 F`, i.e. `F = −½ ∇E`.  In Wirtinger notation this is the
  notes' `F_i = −∂E/∂w̄_i`, because `∂/∂w̄ = ½ (∂/∂x + i ∂/∂y)` carries the factor `½`.
  Along a solution `ẇ = F(w)` one gets `dE/dt = ⟨∇E, F⟩_ℝ = −2 ∑ i |F_i|²`.
-/

namespace Oscillator

open Complex Matrix
open scoped ComplexConjugate

variable {ι : Type*} [Fintype ι]

/-- The energy of the network, see the module docstring for the factor conventions. -/
noncomputable def energy (μ : ι → ℝ) (K : Matrix ι ι ℂ) (u : ι → ℂ) (w : ι → ℂ) : ℝ :=
  (∑ i, (-μ i * normSq (w i) + (1 / 2 : ℝ) * normSq (w i) ^ 2))
    - (∑ i, ∑ j, (conj (w i) * K i j * w j).re)
    - 2 * ∑ i, (conj (u i) * w i).re

/-- The vector field `F(w)_i = μ_i w_i − |w_i|² w_i + (K w)_i + u_i` of the network in the
rotating frame of a common frequency. -/
noncomputable def field (μ : ι → ℝ) (K : Matrix ι ι ℂ) (u : ι → ℂ) (w : ι → ℂ) : ι → ℂ :=
  fun i => (μ i : ℂ) * w i - (normSq (w i) : ℂ) * w i + (K *ᵥ w) i + u i

/-- The vector field with residual intrinsic frequencies `ω`: `F_ω(w)_i = F(w)_i + i ω_i w_i`.
`ω = 0` recovers `field`. -/
noncomputable def rotField (ω μ : ι → ℝ) (K : Matrix ι ι ℂ) (u : ι → ℂ) (w : ι → ℂ) : ι → ℂ :=
  fun i => field μ K u w i + (ω i : ℂ) * I * w i

@[simp] theorem rotField_zero (μ : ι → ℝ) (K : Matrix ι ι ℂ) (u : ι → ℂ) :
    rotField 0 μ K u = field μ K u := by
  ext w i; simp [rotField]

/-- The real pairing `v ↦ ⟨a, v⟩_ℝ = ∑ i Re(conj(a_i) v_i)` as a continuous `ℝ`-linear functional.
A statement `HasFDerivAt E (pairing g) w` says exactly that `g` is the real gradient of `E` at
`w` for the inner product `⟨·,·⟩_ℝ` on `ℂ^ι ≅ ℝ^{2ι}`. -/
noncomputable def pairing (a : ι → ℂ) : (ι → ℂ) →L[ℝ] ℝ :=
  ∑ i, Complex.reCLM.comp
    (conj (a i) • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) i)

@[simp] theorem pairing_apply (a v : ι → ℂ) : pairing a v = ∑ i, (conj (a i) * v i).re := by
  simp [pairing]

/-- The squared Euclidean norm `‖w‖² = ∑ i |w_i|²` of a state. -/
noncomputable def sumNormSq (w : ι → ℂ) : ℝ := ∑ i, normSq (w i)

theorem sumNormSq_nonneg (w : ι → ℂ) : 0 ≤ sumNormSq w :=
  Finset.sum_nonneg fun i _ => normSq_nonneg (w i)

/-- An equilibrium of the network: a zero of its vector field. -/
def IsEquilibrium (μ : ι → ℝ) (K : Matrix ι ι ℂ) (u : ι → ℂ) (w : ι → ℂ) : Prop :=
  field μ K u w = 0

/-- For a Hermitian coupling the coupling form is symmetric under exchanging the two states:
`∑ i j Re(conj(w_i) K_ij v_j) = ∑ i j Re(conj(v_i) K_ij w_j)`. -/
theorem couplingForm_swap {K : Matrix ι ι ℂ} (hK : K.IsHermitian) (w v : ι → ℂ) :
    ∑ i, ∑ j, (conj (w i) * K i j * v j).re = ∑ i, ∑ j, (conj (v i) * K i j * w j).re := by
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => ?_
  have h : K j i = conj (K i j) := by
    have := hK.apply i j
    rw [Complex.star_def] at this
    rw [← this, Complex.conj_conj]
  rw [h]
  simp only [Complex.mul_re, Complex.mul_im, Complex.conj_re, Complex.conj_im]
  ring

end Oscillator
