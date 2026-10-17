import Oscillator.Basic

/-!
# P3: gradient structure and energy descent

For a Hermitian coupling `K` (the complex form of the notes' "real symmetric `K`") the vector
field `F = field μ K u` is a gradient field: with respect to the real inner product
`⟨a, v⟩_ℝ = ∑ i Re(conj(a_i) v_i)` on `ℂ^ι ≅ ℝ^{2ι}`,

  `∇E(w) = −2 F(w)`, i.e. `F = −½ ∇E` (the notes' Wirtinger form `F_i = −∂E/∂w̄_i`).

Consequences, each for a curve `w : ℝ → ι → ℂ` that is assumed to solve `ẇ = F(w)` (existence of
solutions is not proved here, it is a hypothesis):

* `energy_hasDerivAt_solution`: `d/dt E(w(t)) = −2 ∑ i |F(w(t))_i|²`;
* `energy_deriv_nonpos_solution`: this is `≤ 0`;
* `energy_antitone_solution`: `E` is non-increasing along every global solution;
* `energy_bddBelow`: `E` is bounded below, with the explicit constant
  `−∑ i (½ (μ_i + c_i + 1)² + |u_i|²)`, where `c_i = ½ (∑ j |K_ij| + ∑ j |K_ji|)`.

The critical points of `E` are exactly the equilibria of the network
(`fderiv_energy_eq_zero_iff`).
-/

namespace Oscillator

open Complex Matrix
open scoped ComplexConjugate

variable {ι : Type*} [Fintype ι]

/-! ### The pairing is `ℝ`-linear in its vector argument -/

theorem pairing_single [DecidableEq ι] (i : ι) (a : ℂ) (v : ι → ℂ) :
    pairing (Pi.single i a) v = (conj a * v i).re := by
  rw [pairing_apply, Finset.sum_eq_single i]
  · simp
  · intro j _ hj
    simp [hj]
  · intro h
    exact absurd (Finset.mem_univ i) h

/-- `pairing` as an `ℝ`-linear map in the vector argument. -/
noncomputable def pairingₗ : (ι → ℂ) →ₗ[ℝ] ((ι → ℂ) →L[ℝ] ℝ) where
  toFun := pairing
  map_add' a b := by
    ext v
    simp [add_mul, Finset.sum_add_distrib]
  map_smul' c a := by
    ext v
    simp [Finset.mul_sum, Complex.real_smul, mul_assoc]

@[simp] theorem pairingₗ_apply (a : ι → ℂ) : pairingₗ a = pairing a := rfl

theorem pairing_add (a b : ι → ℂ) : pairing (a + b) = pairing a + pairing b :=
  map_add pairingₗ a b

theorem pairing_smul (c : ℝ) (a : ι → ℂ) : pairing (c • a) = c • pairing a :=
  map_smul pairingₗ c a

theorem pairing_sum {κ : Type*} (s : Finset κ) (a : κ → ι → ℂ) :
    pairing (∑ k ∈ s, a k) = ∑ k ∈ s, pairing (a k) :=
  map_sum pairingₗ a s

theorem pairing_sub (a b : ι → ℂ) : pairing (a - b) = pairing a - pairing b :=
  map_sub pairingₗ a b

/-- Pairing a rescaled vector with the vector itself gives the squared norm. -/
theorem pairing_smul_self (c : ℝ) (a : ι → ℂ) : pairing (c • a) a = c * sumNormSq a := by
  rw [pairing_apply, sumNormSq, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp [Complex.real_smul, normSq_apply]
  ring

/-! ### Derivatives of the pieces of the energy -/

omit [Fintype ι] in
theorem hasFDerivAt_coord (i : ι) (w : ι → ℂ) :
    HasFDerivAt (fun w : ι → ℂ => w i)
      (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) i) w :=
  hasFDerivAt_apply i w

theorem hasFDerivAt_conj_coord (i : ι) (w : ι → ℂ) :
    HasFDerivAt (fun w : ι → ℂ => conj (w i))
      ((conjCLE : ℂ →L[ℝ] ℂ).comp
        (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) i)) w :=
  (conjCLE.hasFDerivAt).comp w (hasFDerivAt_coord i w)

section Pieces

variable [DecidableEq ι]

/-- Derivative of `w ↦ Re(conj(w_i) c w_j)`. -/
theorem hasFDerivAt_reConjMul (i j : ι) (c : ℂ) (w : ι → ℂ) :
    HasFDerivAt (fun w : ι → ℂ => (conj (w i) * c * w j).re)
      (pairing (Pi.single i (c * w j) + Pi.single j (conj c * w i))) w := by
  have h := (reCLM.hasFDerivAt).comp w
    (((hasFDerivAt_conj_coord i w).mul_const c).mul (hasFDerivAt_coord j w))
  refine h.congr_fderiv ?_
  ext v
  simp only [pairing_add, _root_.add_apply, pairing_single]
  simp
  ring

/-- Derivative of `w ↦ |w_i|²`. -/
theorem hasFDerivAt_normSq_coord (i : ι) (w : ι → ℂ) :
    HasFDerivAt (fun w : ι → ℂ => normSq (w i)) (pairing (Pi.single i (2 * w i))) w := by
  have hfun : (fun w : ι → ℂ => normSq (w i)) = fun w => (conj (w i) * 1 * w i).re := by
    funext w
    simp [normSq_apply]
  rw [hfun]
  refine (hasFDerivAt_reConjMul i i 1 w).congr_fderiv ?_
  ext v
  simp only [pairing_add, _root_.add_apply, pairing_single]
  simp
  ring

/-- Derivative of `w ↦ |w_i|⁴ = (|w_i|²)²`. -/
theorem hasFDerivAt_normSq_sq_coord (i : ι) (w : ι → ℂ) :
    HasFDerivAt (fun w : ι → ℂ => normSq (w i) ^ 2)
      (pairing (Pi.single i (4 * normSq (w i) * w i))) w := by
  refine ((hasFDerivAt_normSq_coord i w).pow 2).congr_fderiv ?_
  ext v
  simp only [pairing_single, _root_.smul_apply, smul_eq_mul]
  simp
  ring

/-- Derivative of the input term `w ↦ Re(conj(a) w_i)`. -/
theorem hasFDerivAt_input (i : ι) (a : ℂ) (w : ι → ℂ) :
    HasFDerivAt (fun w : ι → ℂ => (conj a * w i).re) (pairing (Pi.single i a)) w := by
  have h := (reCLM.hasFDerivAt).comp w ((hasFDerivAt_coord i w).const_mul (conj a))
  refine h.congr_fderiv ?_
  ext v
  simp only [pairing_single]
  simp

end Pieces

/-! ### P3: the gradient of the energy is `−2 F` -/

variable {μ : ι → ℝ} {K : Matrix ι ι ℂ} {u : ι → ℂ}

/-- The vector identity behind P3: the sum of the per-term gradients is `−2 F(w)`.  This is where
the Hermitian hypothesis enters, through `conj(K_ik) = K_ki`. -/
theorem gradient_vector_identity [DecidableEq ι] (hK : K.IsHermitian) (w : ι → ℂ) :
    (∑ i, ((-μ i) • (Pi.single i (2 * w i) : ι → ℂ)
        + (1 / 2 : ℝ) • (Pi.single i (4 * normSq (w i) * w i) : ι → ℂ)))
      - (∑ i, ∑ j, ((Pi.single i (K i j * w j) : ι → ℂ)
          + (Pi.single j (conj (K i j) * w i) : ι → ℂ)))
      - (2 : ℝ) • ∑ i, (Pi.single i (u i) : ι → ℂ) = (-2 : ℝ) • field μ K u w := by
  funext k
  have hKk : ∀ i, conj (K i k) = K k i := fun i => by
    have := hK.apply k i
    rwa [Complex.star_def] at this
  simp only [Pi.sub_apply, Pi.smul_apply, Finset.sum_apply, Pi.add_apply, Pi.single_apply,
    Finset.sum_add_distrib, smul_ite, smul_zero, Finset.sum_ite_eq, Finset.mem_univ, ite_true,
    field, mulVec, dotProduct, Complex.real_smul]
  rw [Finset.sum_comm (f := fun i j => if k = i then K i j * w j else 0)]
  simp only [Finset.sum_ite_eq, Finset.mem_univ, ite_true, hKk]
  push_cast
  ring

/-- **P3, gradient structure.**  For a Hermitian coupling the energy is differentiable with
real gradient `∇E(w) = −2 F(w)`: for every direction `v`,
`dE(w)[v] = ⟨−2 F(w), v⟩_ℝ = −2 ∑ i Re(conj(F(w)_i) v_i)`. -/
theorem hasFDerivAt_energy (hK : K.IsHermitian) (w : ι → ℂ) :
    HasFDerivAt (energy μ K u) (pairing ((-2 : ℝ) • field μ K u w)) w := by
  classical
  have h1 : HasFDerivAt
      (fun w : ι → ℂ => ∑ i, (-μ i * normSq (w i) + (1 / 2 : ℝ) * normSq (w i) ^ 2))
      (∑ i, ((-μ i) • pairing (Pi.single i (2 * w i))
        + (1 / 2 : ℝ) • pairing (Pi.single i (4 * normSq (w i) * w i)))) w :=
    HasFDerivAt.fun_sum fun i _ =>
      ((hasFDerivAt_normSq_coord i w).const_mul (-μ i)).fun_add
        ((hasFDerivAt_normSq_sq_coord i w).const_mul (1 / 2 : ℝ))
  have h2 : HasFDerivAt (fun w : ι → ℂ => ∑ i, ∑ j, (conj (w i) * K i j * w j).re)
      (∑ i, ∑ j, pairing (Pi.single i (K i j * w j) + Pi.single j (conj (K i j) * w i))) w :=
    HasFDerivAt.fun_sum fun i _ => HasFDerivAt.fun_sum fun j _ =>
      hasFDerivAt_reConjMul i j (K i j) w
  have h3 : HasFDerivAt (fun w : ι → ℂ => 2 * ∑ i, (conj (u i) * w i).re)
      ((2 : ℝ) • ∑ i, pairing (Pi.single i (u i))) w :=
    (HasFDerivAt.fun_sum fun i _ => hasFDerivAt_input i (u i) w).const_mul (2 : ℝ)
  refine ((h1.fun_sub h2).fun_sub h3).congr_fderiv ?_
  simp only [← pairingₗ_apply, ← map_smul, ← map_add, ← map_sum, ← map_sub]
  exact congrArg pairingₗ (gradient_vector_identity (μ := μ) hK w)

/-- The energy is differentiable everywhere (for a Hermitian coupling). -/
theorem differentiable_energy (hK : K.IsHermitian) : Differentiable ℝ (energy μ K u) :=
  fun w => (hasFDerivAt_energy (μ := μ) (u := u) hK w).differentiableAt

/-- The directional derivative of the energy, written out. -/
theorem fderiv_energy_apply (hK : K.IsHermitian) (w v : ι → ℂ) :
    fderiv ℝ (energy μ K u) w v = -2 * ∑ i, (conj (field μ K u w i) * v i).re := by
  rw [(hasFDerivAt_energy hK w).fderiv, pairing_apply, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp [Complex.real_smul]
  ring

/-- The critical points of `E` are exactly the equilibria of the network. -/
theorem fderiv_energy_eq_zero_iff (hK : K.IsHermitian) (w : ι → ℂ) :
    fderiv ℝ (energy μ K u) w = 0 ↔ IsEquilibrium μ K u w := by
  rw [(hasFDerivAt_energy hK w).fderiv, IsEquilibrium]
  constructor
  · intro h
    have h1 : pairing ((-2 : ℝ) • field μ K u w) (field μ K u w) = 0 := by
      rw [h]; rfl
    rw [pairing_smul_self, sumNormSq] at h1
    have h2 : ∑ i, normSq (field μ K u w i) = 0 := by linarith
    rw [Finset.sum_eq_zero_iff_of_nonneg (fun i _ => normSq_nonneg _)] at h2
    funext i
    exact normSq_eq_zero.mp (h2 i (Finset.mem_univ i))
  · intro h
    rw [h]
    ext v
    simp [pairing_apply]

/-! ### Energy descent along solutions -/

/-- Along a curve that solves `ẇ = F(w)` at time `t`: `d/dt E(w(t)) = −2 ∑ i |F(w(t))_i|²`. -/
theorem energy_hasDerivAt_solution (hK : K.IsHermitian) {w : ℝ → ι → ℂ} {t : ℝ}
    (hw : HasDerivAt w (field μ K u (w t)) t) :
    HasDerivAt (fun s => energy μ K u (w s)) (-2 * sumNormSq (field μ K u (w t))) t := by
  have h := (hasFDerivAt_energy (μ := μ) (u := u) hK (w t)).comp_hasDerivAt t hw
  rw [pairing_smul_self] at h
  exact h

/-- The energy does not increase along solutions: `d/dt E(w(t)) ≤ 0`. -/
theorem energy_deriv_nonpos_solution (hK : K.IsHermitian) {w : ℝ → ι → ℂ} {t : ℝ}
    (hw : HasDerivAt w (field μ K u (w t)) t) :
    deriv (fun s => energy μ K u (w s)) t ≤ 0 := by
  rw [(energy_hasDerivAt_solution hK hw).deriv]
  have := sumNormSq_nonneg (field μ K u (w t))
  linarith

/-- Along a global solution of `ẇ = F(w)` the energy is non-increasing in time. -/
theorem energy_antitone_solution (hK : K.IsHermitian) {w : ℝ → ι → ℂ}
    (hw : ∀ t, HasDerivAt w (field μ K u (w t)) t) :
    Antitone (fun t => energy μ K u (w t)) := by
  refine antitone_of_deriv_nonpos (fun t => ?_) (fun t => energy_deriv_nonpos_solution hK (hw t))
  exact (energy_hasDerivAt_solution hK (hw t)).differentiableAt

/-- Along a solution defined on a convex set of times `D` (an interval), continuous on `D` and
solving the equation in its interior, the energy is non-increasing on `D`. -/
theorem energy_antitoneOn_solution (hK : K.IsHermitian) {w : ℝ → ι → ℂ} {D : Set ℝ}
    (hD : Convex ℝ D) (hc : ContinuousOn w D)
    (hw : ∀ t ∈ interior D, HasDerivAt w (field μ K u (w t)) t) :
    AntitoneOn (fun t => energy μ K u (w t)) D := by
  refine antitoneOn_of_deriv_nonpos hD ?_ ?_ ?_
  · exact (differentiable_energy hK).continuous.comp_continuousOn hc
  · intro t ht
    exact (energy_hasDerivAt_solution hK (hw t ht)).differentiableAt.differentiableWithinAt
  · intro t ht
    exact energy_deriv_nonpos_solution hK (hw t ht)

/-! ### The energy is bounded below -/

/-- `Re(conj(a) c b) ≤ |c| (|a|² + |b|²) / 2`. -/
theorem re_conj_mul_mul_le (a c b : ℂ) :
    (conj a * c * b).re ≤ ‖c‖ * (normSq a + normSq b) / 2 := by
  calc (conj a * c * b).re ≤ ‖conj a * c * b‖ := Complex.re_le_norm _
    _ = ‖a‖ * ‖c‖ * ‖b‖ := by rw [norm_mul, norm_mul, Complex.norm_conj]
    _ ≤ ‖c‖ * (normSq a + normSq b) / 2 := by
        rw [normSq_eq_norm_sq, normSq_eq_norm_sq]
        nlinarith [sq_nonneg (‖a‖ - ‖b‖), norm_nonneg c, norm_nonneg a, norm_nonneg b,
          mul_nonneg (norm_nonneg c) (sq_nonneg (‖a‖ - ‖b‖))]

/-- `2 Re(conj(a) b) ≤ |a|² + |b|²`. -/
theorem two_re_conj_mul_le (a b : ℂ) : 2 * (conj a * b).re ≤ normSq a + normSq b := by
  have h := re_conj_mul_mul_le a 1 b
  simp only [mul_one, norm_one, one_mul] at h
  linarith

/-- The coupling weight at unit `i`: half the sum of the absolute values of row `i` and
column `i` of `K`. -/
noncomputable def couplingWeight (K : Matrix ι ι ℂ) (i : ι) : ℝ :=
  ((∑ j, ‖K i j‖) + ∑ j, ‖K j i‖) / 2

/-- An explicit lower bound for the energy, `−∑ i (½ (μ_i + c_i + 1)² + |u_i|²)`. -/
noncomputable def energyLowerBound (μ : ι → ℝ) (K : Matrix ι ι ℂ) (u : ι → ℂ) : ℝ :=
  -∑ i, ((1 / 2 : ℝ) * (μ i + couplingWeight K i + 1) ^ 2 + normSq (u i))

/-- The coupling form is bounded by the coupling weights. -/
theorem couplingForm_le (K : Matrix ι ι ℂ) (w : ι → ℂ) :
    ∑ i, ∑ j, (conj (w i) * K i j * w j).re ≤ ∑ i, couplingWeight K i * normSq (w i) := by
  have e1 : ∑ i, ∑ j, ‖K i j‖ * (normSq (w i) + normSq (w j)) / 2
      = ∑ i, ∑ j, ‖K i j‖ * normSq (w i) / 2 + ∑ i, ∑ j, ‖K i j‖ * normSq (w j) / 2 := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun j _ => ?_
    ring
  have e2 : ∑ i, ∑ j, ‖K i j‖ * normSq (w j) / 2 = ∑ i, ∑ j, ‖K j i‖ * normSq (w i) / 2 :=
    Finset.sum_comm
  have e3 : ∀ i, ∑ j, ‖K i j‖ * normSq (w i) / 2 + ∑ j, ‖K j i‖ * normSq (w i) / 2
      = couplingWeight K i * normSq (w i) := by
    intro i
    simp only [couplingWeight, ← Finset.sum_div, ← Finset.sum_mul]
    ring
  calc ∑ i, ∑ j, (conj (w i) * K i j * w j).re
      ≤ ∑ i, ∑ j, ‖K i j‖ * (normSq (w i) + normSq (w j)) / 2 :=
        Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => re_conj_mul_mul_le _ _ _
    _ = ∑ i, couplingWeight K i * normSq (w i) := by
        rw [e1, e2, ← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun i _ => e3 i

/-- **P3, boundedness.**  The energy is bounded below by `energyLowerBound μ K u`. -/
theorem energyLowerBound_le (w : ι → ℂ) : energyLowerBound μ K u ≤ energy μ K u w := by
  have hcoup := couplingForm_le K w
  have hin : 2 * ∑ i, (conj (u i) * w i).re ≤ ∑ i, (normSq (u i) + normSq (w i)) := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun i _ => two_re_conj_mul_le _ _
  have hq : ∀ i, -((1 / 2 : ℝ) * (μ i + couplingWeight K i + 1) ^ 2 + normSq (u i))
      ≤ (-μ i * normSq (w i) + (1 / 2 : ℝ) * normSq (w i) ^ 2)
        - couplingWeight K i * normSq (w i) - (normSq (u i) + normSq (w i)) := by
    intro i
    nlinarith [sq_nonneg (normSq (w i) - (μ i + couplingWeight K i + 1))]
  have hsum := Finset.sum_le_sum fun i (_ : i ∈ Finset.univ) => hq i
  rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, Finset.sum_neg_distrib] at hsum
  unfold energy energyLowerBound
  linarith

/-- The energy is bounded below. -/
theorem energy_bddBelow : BddBelow (Set.range (energy μ K u)) :=
  ⟨energyLowerBound μ K u, by
    rintro _ ⟨w, rfl⟩
    exact energyLowerBound_le w⟩

end Oscillator
