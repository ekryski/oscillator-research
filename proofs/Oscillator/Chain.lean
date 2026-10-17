import Oscillator.Energy

/-!
# Chaining closure

Two networks on index types `ι₁` and `ι₂`, with Hermitian couplings `K₁` and `K₂`, joined by an
arbitrary inter-module coupling `B : Matrix ι₁ ι₂ ℂ` (sparse, distance dependent, or anything else)
and its adjoint `Bᴴ` back, form one network on `ι₁ ⊕ ι₂` whose coupling is the block matrix
`[K₁ B; Bᴴ K₂]`.  This coupling is Hermitian (`chainCoupling_isHermitian`), so everything in P3
applies to the chain: its energy is a Lyapunov function along solutions
(`chain_energy_antitone`).  Since the chain is again a network with Hermitian coupling, the
construction can be repeated to chain any number of modules.

The chain's vector field is the two module fields plus the inter-module terms (`chain_field`),
and its energy is the sum of the two module energies minus the interaction energy
`2 ∑ i j Re(conj(w₁_i) B_ij w₂_j)` (`chain_energy`).
-/

namespace Oscillator

open Complex Matrix
open scoped ComplexConjugate

variable {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]

/-- The coupling of the chained network: the block matrix `[K₁ B; Bᴴ K₂]`. -/
def chainCoupling (K₁ : Matrix ι₁ ι₁ ℂ) (K₂ : Matrix ι₂ ι₂ ℂ) (B : Matrix ι₁ ι₂ ℂ) :
    Matrix (ι₁ ⊕ ι₂) (ι₁ ⊕ ι₂) ℂ :=
  fromBlocks K₁ B Bᴴ K₂

variable {μ₁ : ι₁ → ℝ} {μ₂ : ι₂ → ℝ} {K₁ : Matrix ι₁ ι₁ ℂ} {K₂ : Matrix ι₂ ι₂ ℂ}
  {u₁ : ι₁ → ℂ} {u₂ : ι₂ → ℂ}

omit [Fintype ι₁] [Fintype ι₂] in
/-- **Chaining closure.**  The chain of two Hermitian couplings through any inter-module coupling
is Hermitian. -/
theorem chainCoupling_isHermitian (h₁ : K₁.IsHermitian) (h₂ : K₂.IsHermitian)
    (B : Matrix ι₁ ι₂ ℂ) : (chainCoupling K₁ K₂ B).IsHermitian :=
  isHermitian_fromBlocks_iff.2 ⟨h₁, rfl, conjTranspose_conjTranspose B, h₂⟩

/-- The chain's vector field: each module's own field plus the coupling from the other module. -/
theorem chain_field (B : Matrix ι₁ ι₂ ℂ) (w₁ : ι₁ → ℂ) (w₂ : ι₂ → ℂ) :
    field (Sum.elim μ₁ μ₂) (chainCoupling K₁ K₂ B) (Sum.elim u₁ u₂) (Sum.elim w₁ w₂)
      = Sum.elim (field μ₁ K₁ u₁ w₁ + B *ᵥ w₂) (field μ₂ K₂ u₂ w₂ + Bᴴ *ᵥ w₁) := by
  funext i
  cases i with
  | inl i =>
    simp only [field, chainCoupling, fromBlocks_mulVec, Sum.elim_comp_inl, Sum.elim_comp_inr,
      Sum.elim_inl, Pi.add_apply]
    ring
  | inr i =>
    simp only [field, chainCoupling, fromBlocks_mulVec, Sum.elim_comp_inl, Sum.elim_comp_inr,
      Sum.elim_inr, Pi.add_apply]
    ring

/-- The coupling form through `Bᴴ` equals the coupling form through `B` with the roles of the
two states exchanged. -/
theorem couplingForm_conjTranspose (B : Matrix ι₁ ι₂ ℂ) (w₁ : ι₁ → ℂ) (w₂ : ι₂ → ℂ) :
    ∑ i, ∑ j, (conj (w₂ i) * Bᴴ i j * w₁ j).re = ∑ i, ∑ j, (conj (w₁ i) * B i j * w₂ j).re := by
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => ?_
  rw [conjTranspose_apply, Complex.star_def]
  simp only [Complex.mul_re, Complex.mul_im, Complex.conj_re, Complex.conj_im]
  ring

/-- The chain's energy: the two module energies minus the interaction energy. -/
theorem chain_energy (B : Matrix ι₁ ι₂ ℂ) (w₁ : ι₁ → ℂ) (w₂ : ι₂ → ℂ) :
    energy (Sum.elim μ₁ μ₂) (chainCoupling K₁ K₂ B) (Sum.elim u₁ u₂) (Sum.elim w₁ w₂)
      = energy μ₁ K₁ u₁ w₁ + energy μ₂ K₂ u₂ w₂
        - 2 * ∑ i, ∑ j, (conj (w₁ i) * B i j * w₂ j).re := by
  simp only [energy, chainCoupling, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr,
    fromBlocks_apply₁₁, fromBlocks_apply₁₂, fromBlocks_apply₂₁, fromBlocks_apply₂₂,
    Finset.sum_add_distrib]
  rw [couplingForm_conjTranspose]
  ring

/-- **P3 for the chain.**  Along any solution of the chained network the energy is non-increasing. -/
theorem chain_energy_antitone (h₁ : K₁.IsHermitian) (h₂ : K₂.IsHermitian) (B : Matrix ι₁ ι₂ ℂ)
    {w : ℝ → ι₁ ⊕ ι₂ → ℂ}
    (hw : ∀ t, HasDerivAt w
      (field (Sum.elim μ₁ μ₂) (chainCoupling K₁ K₂ B) (Sum.elim u₁ u₂) (w t)) t) :
    Antitone (fun t => energy (Sum.elim μ₁ μ₂) (chainCoupling K₁ K₂ B) (Sum.elim u₁ u₂) (w t)) :=
  energy_antitone_solution (chainCoupling_isHermitian h₁ h₂ B) hw

/-- The gradient structure of the chain: `∇E = −2 F` for the chained network. -/
theorem chain_hasFDerivAt_energy (h₁ : K₁.IsHermitian) (h₂ : K₂.IsHermitian)
    (B : Matrix ι₁ ι₂ ℂ) (w : ι₁ ⊕ ι₂ → ℂ) :
    HasFDerivAt (energy (Sum.elim μ₁ μ₂) (chainCoupling K₁ K₂ B) (Sum.elim u₁ u₂))
      (pairing ((-2 : ℝ) • field (Sum.elim μ₁ μ₂) (chainCoupling K₁ K₂ B) (Sum.elim u₁ u₂) w))
      w :=
  hasFDerivAt_energy (chainCoupling_isHermitian h₁ h₂ B) w

end Oscillator
