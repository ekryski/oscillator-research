import Oscillator.Energy

/-!
# P6: the torus lattice

Periodic boundary conditions make a coupling kernel translation invariant, and a translation
invariant kernel on a finite abelian group `G` is circulant (`circulant_of_translation_invariant`).
Every circulant on `G` is diagonalised by the characters of `G`: for an additive character `χ`,
`circulant v *ᵥ χ = (∑ l v_l χ(−l)) • χ` (`circulant_mulVec_addChar`).

* One dimension, `G = ZMod m`: the characters are the Fourier modes `j ↦ exp(2πi kj/m)` and the
  eigenvalue is Mathlib's discrete Fourier transform `ZMod.dft v k`
  (`circulant_mulVec_fourierMode`).  In matrix form, with `W` the DFT matrix,
  `circulant v * W = W * diagonal (𝓕 v)` and `W * Wᴴ = m • 1`, so
  `circulant v = W * diagonal (𝓕 v) * (m⁻¹ • Wᴴ)` (`circulant_eq_dftMatrix_mul_diagonal`).
* Two dimensions, `G = ZMod m × ZMod m` (paper 02's `Z_16 × Z_16` is `m = 16`): the characters are
  the products of one-dimensional modes and the eigenvalue is the two-dimensional DFT, which is the
  iterated one-dimensional DFT (`circulant_mulVec_fourierMode2`, `dft2_eq_iterated`).

A circulant kernel is Hermitian exactly when its symbol satisfies `conj (v d) = v (−d)`, which holds
for every real even kernel such as a distance-dependent one; then P3 applies to the torus network
(`isHermitian_circulant`).  The operator-norm consequence (the coupling ceiling is `max_k |𝓕 v k|`)
is not formalised; it follows from the unitary diagonalisation above by Parseval's identity.
-/

namespace Oscillator

open Matrix
open scoped ComplexConjugate

/-! ### Translation-invariant kernels and characters on a finite abelian group -/

section General

variable {G : Type*} [AddCommGroup G] [Fintype G] {R : Type*} [CommRing R]

omit [Fintype G] [CommRing R] in
/-- A translation-invariant kernel on a finite abelian group is circulant, with symbol
`d ↦ K d 0`. -/
theorem circulant_of_translation_invariant (K : Matrix G G R)
    (hK : ∀ i j t, K (i + t) (j + t) = K i j) :
    K = circulant (fun d => K d 0) := by
  ext i j
  rw [circulant_apply]
  have h := hK (i - j) 0 j
  rw [sub_add_cancel, zero_add] at h
  exact h

/-- **Characters are eigenvectors of circulants.**  For an additive character `χ : G → R`,
`circulant v *ᵥ χ = (∑ l, v l * χ (−l)) • χ`. -/
theorem circulant_mulVec_addChar (v : G → R) (χ : AddChar G R) :
    circulant v *ᵥ ⇑χ = (∑ l, v l * χ (-l)) • ⇑χ := by
  funext i
  simp only [mulVec, dotProduct, circulant_apply, Pi.smul_apply, smul_eq_mul]
  have hb : Function.Bijective (fun l : G => i - l) :=
    (Function.Involutive.bijective fun l => sub_sub_cancel i l)
  rw [← Fintype.sum_bijective (fun l => i - l) hb (fun l => v (i - (i - l)) * χ (i - l))
    (fun j => v (i - j) * χ j) (fun l => rfl)]
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [sub_sub_cancel, sub_eq_add_neg, AddChar.map_add_eq_mul]
  ring

end General

/-! ### One dimension: `ZMod m` and Mathlib's DFT -/

section OneDim

open ZMod

variable (m : ℕ) [NeZero m]

/-- The `k`-th Fourier mode on `ZMod m`: `j ↦ exp(2πi kj/m)`. -/
noncomputable def fourierMode (k : ZMod m) : ZMod m → ℂ := fun j => stdAddChar (k * j)

theorem fourierMode_eq (k : ZMod m) :
    fourierMode m k = ⇑((stdAddChar (N := m)).mulShift k) := by
  funext j
  simp [fourierMode]

/-- **P6, one dimension.**  The circulant with symbol `v` acts on the `k`-th Fourier mode by
multiplication with the `k`-th DFT coefficient `𝓕 v k`. -/
theorem circulant_mulVec_fourierMode (v : ZMod m → ℂ) (k : ZMod m) :
    circulant v *ᵥ fourierMode m k = (dft v k) • fourierMode m k := by
  rw [fourierMode_eq, circulant_mulVec_addChar, dft_apply]
  congr 1
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [AddChar.mulShift_apply, mul_neg, mul_comm k l, smul_eq_mul, mul_comm]

/-- The DFT matrix: column `k` is the `k`-th Fourier mode. -/
noncomputable def dftMatrix : Matrix (ZMod m) (ZMod m) ℂ := fun j k => stdAddChar (k * j)

/-- `circulant v * W = W * diagonal (𝓕 v)`. -/
theorem circulant_mul_dftMatrix (v : ZMod m → ℂ) :
    circulant v * dftMatrix m = dftMatrix m * diagonal (dft v) := by
  ext i k
  rw [mul_diagonal, mul_apply]
  have h := congrFun (circulant_mulVec_fourierMode m v k) i
  simp only [mulVec, dotProduct, circulant_apply, fourierMode, Pi.smul_apply, smul_eq_mul] at h
  simp only [dftMatrix, circulant_apply]
  rw [h]
  ring

theorem stdAddChar_conj (x : ZMod m) : conj (stdAddChar x) = stdAddChar (-x) := by
  rw [stdAddChar_apply, stdAddChar_apply, AddChar.map_neg_eq_inv, Circle.coe_inv_eq_conj]

/-- The DFT matrix is unitary up to the factor `m`: `W * Wᴴ = m • 1`. -/
theorem dftMatrix_mul_conjTranspose :
    dftMatrix m * (dftMatrix m)ᴴ = (m : ℂ) • (1 : Matrix (ZMod m) (ZMod m) ℂ) := by
  ext i i'
  rw [mul_apply]
  simp only [dftMatrix, conjTranspose_apply, Complex.star_def, stdAddChar_conj]
  have h : ∀ k : ZMod m, stdAddChar (k * i) * stdAddChar (-(k * i')) = stdAddChar (k * (i - i')) := by
    intro k
    rw [← AddChar.map_add_eq_mul]
    congr 1
    ring
  simp only [h]
  rw [AddChar.sum_mulShift _ (isPrimitive_stdAddChar m)]
  by_cases hii : i = i'
  · subst hii
    simp [ZMod.card]
  · have : i - i' ≠ 0 := sub_ne_zero.mpr hii
    simp [this, hii]

/-- **P6, one dimension, matrix form.**  The circulant is diagonalised by the DFT matrix:
`circulant v = W * diagonal (𝓕 v) * (m⁻¹ • Wᴴ)`. -/
theorem circulant_eq_dftMatrix_mul_diagonal (v : ZMod m → ℂ) :
    circulant v = dftMatrix m * diagonal (dft v) * ((m : ℂ)⁻¹ • (dftMatrix m)ᴴ) := by
  have hm : (m : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne m)
  have hinv : dftMatrix m * ((m : ℂ)⁻¹ • (dftMatrix m)ᴴ) = 1 := by
    rw [Matrix.mul_smul, dftMatrix_mul_conjTranspose, smul_smul, inv_mul_cancel₀ hm, one_smul]
  calc circulant v = circulant v * (dftMatrix m * ((m : ℂ)⁻¹ • (dftMatrix m)ᴴ)) := by
        rw [hinv, Matrix.mul_one]
    _ = circulant v * dftMatrix m * ((m : ℂ)⁻¹ • (dftMatrix m)ᴴ) := by rw [Matrix.mul_assoc]
    _ = dftMatrix m * diagonal (dft v) * ((m : ℂ)⁻¹ • (dftMatrix m)ᴴ) := by
        rw [circulant_mul_dftMatrix]

end OneDim

/-! ### Two dimensions: `ZMod m × ZMod m` -/

section TwoDim

/-- The product of two additive characters is a character of the product group. -/
def prodChar {G₁ G₂ R : Type*} [AddMonoid G₁] [AddMonoid G₂] [CommMonoid R]
    (χ₁ : AddChar G₁ R) (χ₂ : AddChar G₂ R) : AddChar (G₁ × G₂) R where
  toFun p := χ₁ p.1 * χ₂ p.2
  map_zero_eq_one' := by simp
  map_add_eq_mul' p q := by
    simp only [Prod.fst_add, Prod.snd_add, AddChar.map_add_eq_mul]
    exact mul_mul_mul_comm _ _ _ _

@[simp] theorem prodChar_apply {G₁ G₂ R : Type*} [AddMonoid G₁] [AddMonoid G₂] [CommMonoid R]
    (χ₁ : AddChar G₁ R) (χ₂ : AddChar G₂ R) (p : G₁ × G₂) :
    prodChar χ₁ χ₂ p = χ₁ p.1 * χ₂ p.2 := rfl

open ZMod

variable (m : ℕ) [NeZero m]

/-- The two-dimensional Fourier mode `(k₁, k₂)` on the torus `ZMod m × ZMod m`. -/
noncomputable def fourierMode2 (k : ZMod m × ZMod m) : ZMod m × ZMod m → ℂ :=
  fun j => stdAddChar (k.1 * j.1) * stdAddChar (k.2 * j.2)

theorem fourierMode2_eq (k : ZMod m × ZMod m) :
    fourierMode2 m k
      = ⇑(prodChar ((stdAddChar (N := m)).mulShift k.1) ((stdAddChar (N := m)).mulShift k.2)) := by
  funext j
  simp [fourierMode2]

/-- The two-dimensional discrete Fourier transform on `ZMod m × ZMod m`. -/
noncomputable def dft2 (v : ZMod m × ZMod m → ℂ) (k : ZMod m × ZMod m) : ℂ :=
  ∑ l, v l * stdAddChar (-(k.1 * l.1)) * stdAddChar (-(k.2 * l.2))

/-- The two-dimensional DFT is the iterated one-dimensional DFT. -/
theorem dft2_eq_iterated (v : ZMod m × ZMod m → ℂ) (k : ZMod m × ZMod m) :
    dft2 m v k = dft (fun l₁ => dft (fun l₂ => v (l₁, l₂)) k.2) k.1 := by
  simp only [dft2, dft_apply, smul_eq_mul, Fintype.sum_prod_type, Finset.mul_sum]
  refine Finset.sum_congr rfl fun l₁ _ => Finset.sum_congr rfl fun l₂ _ => ?_
  rw [mul_comm l₁ k.1, mul_comm l₂ k.2]
  ring

/-- **P6, two dimensions.**  A circulant kernel on the torus `ZMod m × ZMod m` (periodic boundary
conditions) acts on each two-dimensional Fourier mode by its two-dimensional DFT coefficient. -/
theorem circulant_mulVec_fourierMode2 (v : ZMod m × ZMod m → ℂ) (k : ZMod m × ZMod m) :
    circulant v *ᵥ fourierMode2 m k = dft2 m v k • fourierMode2 m k := by
  rw [fourierMode2_eq, circulant_mulVec_addChar]
  congr 1
  refine Finset.sum_congr rfl fun l _ => ?_
  simp only [prodChar_apply, AddChar.mulShift_apply, Prod.fst_neg, Prod.snd_neg, mul_neg]
  ring

end TwoDim

/-! ### Hermitian circulants: where P3 applies on the torus -/

section Hermitian

variable {G : Type*} [AddCommGroup G]

/-- A circulant kernel is Hermitian exactly when `conj (v d) = v (−d)`; in particular every real
even kernel (`v (−d) = v d`, `v` real) gives a Hermitian coupling, so P3 applies. -/
theorem isHermitian_circulant (v : G → ℂ) (hv : ∀ d, conj (v d) = v (-d)) :
    (circulant v).IsHermitian := by
  refine Matrix.IsHermitian.ext fun i j => ?_
  rw [Complex.star_def, circulant_apply, circulant_apply, hv, neg_sub]

theorem isHermitian_circulant_of_real_even (v : G → ℝ) (hv : ∀ d, v (-d) = v d) :
    (circulant (fun d => (v d : ℂ))).IsHermitian :=
  isHermitian_circulant _ fun d => by rw [Complex.conj_ofReal, hv]

end Hermitian

end Oscillator
