import Oscillator.Energy

/-!
# P5: dissipativity

Along any solution of `ẇ = F_ω(w)` (frequencies allowed, no symmetry assumed on `K`), the
squared norm `S(t) = ∑ i |w_i(t)|²` satisfies

  `dS/dt ≤ 2 (μ_max + κ) S − (2/N) S² + 2 U √N √S`,

where `N = card ι`, `‖K‖ ≤ κ` in ℓ² operator norm (`OpNormBound K κ`), `μ_i ≤ μ_max` and
`|u_i| ≤ U` (`dissipativity`).  The right-hand side is `dissipationBound`; it is at most `−2`
once `S ≥ max 1 (N (μ_max + κ + U √N + 1))` (`dissipationBound_le_neg_two`), which is the
absorbing ball.  The existence of the global attractor is not formalised; it follows from the
absorbing ball by the standard argument once a flow exists, and the notes cite it as such.
-/

namespace Oscillator

open Complex Matrix
open scoped ComplexConjugate

variable {ι : Type*} [Fintype ι]
variable {μ ω : ι → ℝ} {K : Matrix ι ι ℂ} {u : ι → ℂ}

/-- `κ ≥ 0` bounds the ℓ² operator norm of `K`: `‖K v‖² ≤ κ² ‖v‖²` for every `v`. -/
def OpNormBound (K : Matrix ι ι ℂ) (κ : ℝ) : Prop :=
  0 ≤ κ ∧ ∀ v : ι → ℂ, sumNormSq (K *ᵥ v) ≤ κ ^ 2 * sumNormSq v

/-- The right-hand side of the dissipativity bound, as a function of `S = ‖w‖²`. -/
noncomputable def dissipationBound (N : ℕ) (μmax κ U S : ℝ) : ℝ :=
  2 * (μmax + κ) * S - (2 / N) * S ^ 2 + 2 * U * Real.sqrt N * Real.sqrt S

/-- The derivative of the squared norm along a solution: `dS/dt = 2 ∑ i Re(conj(w_i) F_ω(w)_i)`. -/
theorem hasDerivAt_sumNormSq_solution {w : ℝ → ι → ℂ} {t : ℝ}
    (hw : HasDerivAt w (rotField ω μ K u (w t)) t) :
    HasDerivAt (fun s => sumNormSq (w s))
      (2 * ∑ i, (conj (w t i) * rotField ω μ K u (w t) i).re) t := by
  have hi : ∀ i, HasDerivAt (fun s => normSq (w s i))
      (2 * (conj (w t i) * rotField ω μ K u (w t) i).re) t := by
    intro i
    have h := (hasDerivAt_pi.1 hw i).norm_sq
    rw [Complex.inner] at h
    simp only [← Complex.normSq_eq_norm_sq] at h
    convert h using 3
    exact mul_comm _ _
  unfold sumNormSq
  have h := HasDerivAt.fun_sum (u := Finset.univ) fun i _ => hi i
  rw [← Finset.mul_sum] at h
  exact h

/-- The frequency term drops out of `Re(conj(w_i) F_ω(w)_i)`, which splits into its four parts. -/
theorem re_conj_mul_rotField (w : ι → ℂ) (i : ι) :
    (conj (w i) * rotField ω μ K u w i).re
      = μ i * normSq (w i) - normSq (w i) ^ 2
        + (conj (w i) * (K *ᵥ w) i).re + (conj (w i) * u i).re := by
  simp only [rotField, field, normSq_apply, Complex.mul_re, Complex.mul_im, Complex.add_re,
    Complex.add_im, Complex.sub_re, Complex.sub_im, Complex.conj_re, Complex.conj_im,
    Complex.ofReal_re, Complex.ofReal_im, Complex.I_re, Complex.I_im]
  ring

/-- The bound on `dS/dt`, as an inequality at a fixed state `w`. -/
theorem dissipativity_bound {κ μmax U : ℝ} (hK : OpNormBound K κ) (hμ : ∀ i, μ i ≤ μmax)
    (hU : 0 ≤ U) (hu : ∀ i, ‖u i‖ ≤ U) (w : ι → ℂ) :
    2 * ∑ i, (conj (w i) * rotField ω μ K u w i).re
      ≤ dissipationBound (Fintype.card ι) μmax κ U (sumNormSq w) := by
  have hexp : ∑ i, (conj (w i) * rotField ω μ K u w i).re
      = ∑ i, μ i * normSq (w i) - ∑ i, normSq (w i) ^ 2
        + ∑ i, (conj (w i) * (K *ᵥ w) i).re + ∑ i, (conj (w i) * u i).re := by
    simp only [re_conj_mul_rotField, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  have hA : ∑ i, μ i * normSq (w i) ≤ μmax * sumNormSq w := by
    rw [sumNormSq, Finset.mul_sum]
    exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hμ i) (normSq_nonneg _)
  have hB : (2 / (Fintype.card ι : ℝ)) * sumNormSq w ^ 2 ≤ 2 * ∑ i, normSq (w i) ^ 2 := by
    rcases Nat.eq_zero_or_pos (Fintype.card ι) with h0 | hpos
    · rw [h0]
      simp only [Nat.cast_zero, div_zero, zero_mul]
      exact mul_nonneg (by norm_num) (Finset.sum_nonneg fun i _ => sq_nonneg _)
    · have hcs := sq_sum_le_card_mul_sum_sq (s := Finset.univ) (f := fun i => normSq (w i))
      rw [Finset.card_univ] at hcs
      have hNpos : (0 : ℝ) < Fintype.card ι := by exact_mod_cast hpos
      rw [sumNormSq, div_mul_eq_mul_div, div_le_iff₀ hNpos]
      nlinarith [hcs]
  have hC : ∑ i, (conj (w i) * (K *ᵥ w) i).re ≤ κ * sumNormSq w := by
    have h1 : ∑ i, (conj (w i) * (K *ᵥ w) i).re ≤ ∑ i, ‖w i‖ * ‖(K *ᵥ w) i‖ := by
      refine Finset.sum_le_sum fun i _ => ?_
      calc (conj (w i) * (K *ᵥ w) i).re ≤ ‖conj (w i) * (K *ᵥ w) i‖ := Complex.re_le_norm _
        _ = ‖w i‖ * ‖(K *ᵥ w) i‖ := by rw [norm_mul, Complex.norm_conj]
    have h2 : (∑ i, ‖w i‖ * ‖(K *ᵥ w) i‖) ^ 2
        ≤ (∑ i, ‖w i‖ ^ 2) * ∑ i, ‖(K *ᵥ w) i‖ ^ 2 :=
      Finset.sum_mul_sq_le_sq_mul_sq _ _ _
    have h3 : ∑ i, ‖w i‖ ^ 2 = sumNormSq w := by
      simp [sumNormSq, Complex.normSq_eq_norm_sq]
    have h4 : ∑ i, ‖(K *ᵥ w) i‖ ^ 2 = sumNormSq (K *ᵥ w) := by
      simp [sumNormSq, Complex.normSq_eq_norm_sq]
    rw [h3, h4] at h2
    have h5 : (∑ i, ‖w i‖ * ‖(K *ᵥ w) i‖) ^ 2 ≤ (κ * sumNormSq w) ^ 2 := by
      calc (∑ i, ‖w i‖ * ‖(K *ᵥ w) i‖) ^ 2 ≤ sumNormSq w * sumNormSq (K *ᵥ w) := h2
        _ ≤ sumNormSq w * (κ ^ 2 * sumNormSq w) :=
          mul_le_mul_of_nonneg_left (hK.2 w) (sumNormSq_nonneg w)
        _ = (κ * sumNormSq w) ^ 2 := by ring
    have h6 : ∑ i, ‖w i‖ * ‖(K *ᵥ w) i‖ ≤ κ * sumNormSq w :=
      le_of_sq_le_sq h5 (mul_nonneg hK.1 (sumNormSq_nonneg w))
    linarith
  have hD : ∑ i, (conj (w i) * u i).re
      ≤ U * Real.sqrt (Fintype.card ι) * Real.sqrt (sumNormSq w) := by
    have h1 : ∑ i, (conj (w i) * u i).re ≤ U * ∑ i, ‖w i‖ := by
      rw [Finset.mul_sum]
      refine Finset.sum_le_sum fun i _ => ?_
      calc (conj (w i) * u i).re ≤ ‖conj (w i) * u i‖ := Complex.re_le_norm _
        _ = ‖w i‖ * ‖u i‖ := by rw [norm_mul, Complex.norm_conj]
        _ ≤ ‖w i‖ * U := mul_le_mul_of_nonneg_left (hu i) (norm_nonneg _)
        _ = U * ‖w i‖ := mul_comm _ _
    have h2 : ∑ i, ‖w i‖ ≤ Real.sqrt (Fintype.card ι) * Real.sqrt (sumNormSq w) := by
      rw [← Real.sqrt_mul (Nat.cast_nonneg _)]
      rw [Real.le_sqrt (Finset.sum_nonneg fun i _ => norm_nonneg _)
        (mul_nonneg (Nat.cast_nonneg _) (sumNormSq_nonneg w))]
      have hcs := sq_sum_le_card_mul_sum_sq (s := Finset.univ) (f := fun i => ‖w i‖)
      rw [Finset.card_univ] at hcs
      calc (∑ i, ‖w i‖) ^ 2 ≤ (Fintype.card ι : ℝ) * ∑ i, ‖w i‖ ^ 2 := hcs
        _ = Fintype.card ι * sumNormSq w := by simp [sumNormSq, Complex.normSq_eq_norm_sq]
    calc ∑ i, (conj (w i) * u i).re ≤ U * ∑ i, ‖w i‖ := h1
      _ ≤ U * (Real.sqrt (Fintype.card ι) * Real.sqrt (sumNormSq w)) :=
        mul_le_mul_of_nonneg_left h2 hU
      _ = U * Real.sqrt (Fintype.card ι) * Real.sqrt (sumNormSq w) := by ring
  rw [hexp, dissipationBound]
  nlinarith [hA, hB, hC, hD]

/-- **P5, dissipativity.**  Along a solution of `ẇ = F_ω(w)` at time `t`, with `‖K‖ ≤ κ`,
`μ_i ≤ μ_max` and `|u_i| ≤ U`:
`d/dt ∑ i |w_i|² ≤ 2 (μ_max + κ) S − (2/N) S² + 2 U √N √S`, where `S = ∑ i |w_i(t)|²`. -/
theorem dissipativity {κ μmax U : ℝ} (hK : OpNormBound K κ) (hμ : ∀ i, μ i ≤ μmax)
    (hU : 0 ≤ U) (hu : ∀ i, ‖u i‖ ≤ U) {w : ℝ → ι → ℂ} {t : ℝ}
    (hw : HasDerivAt w (rotField ω μ K u (w t)) t) :
    deriv (fun s => sumNormSq (w s)) t
      ≤ dissipationBound (Fintype.card ι) μmax κ U (sumNormSq (w t)) := by
  rw [(hasDerivAt_sumNormSq_solution hw).deriv]
  exact dissipativity_bound hK hμ hU hu (w t)

/-- P5 for the field without frequencies. -/
theorem dissipativity_field {κ μmax U : ℝ} (hK : OpNormBound K κ) (hμ : ∀ i, μ i ≤ μmax)
    (hU : 0 ≤ U) (hu : ∀ i, ‖u i‖ ≤ U) {w : ℝ → ι → ℂ} {t : ℝ}
    (hw : HasDerivAt w (field μ K u (w t)) t) :
    deriv (fun s => sumNormSq (w s)) t
      ≤ dissipationBound (Fintype.card ι) μmax κ U (sumNormSq (w t)) := by
  have hw' : HasDerivAt w (rotField 0 μ K u (w t)) t := by rwa [rotField_zero]
  exact dissipativity hK hμ hU hu hw'

/-- **The absorbing ball.**  Once `S ≥ max 1 (N (μ_max + κ + U √N + 1))` the bound is at most
`−2`, so `S` decreases at rate at least `2` outside this ball. -/
theorem dissipationBound_le_neg_two {N : ℕ} {μmax κ U S : ℝ} (hU : 0 ≤ U) (hN : 0 < N)
    (hS : max 1 (N * (μmax + κ + U * Real.sqrt N + 1)) ≤ S) :
    dissipationBound N μmax κ U S ≤ -2 := by
  have h1 : (1 : ℝ) ≤ S := le_trans (le_max_left _ _) hS
  have h2 : (N : ℝ) * (μmax + κ + U * Real.sqrt N + 1) ≤ S := le_trans (le_max_right _ _) hS
  have hsqrt : Real.sqrt S ≤ S := by
    rw [Real.sqrt_le_left (by linarith)]
    nlinarith
  have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
  have h4 : 0 ≤ U * Real.sqrt N := mul_nonneg hU (Real.sqrt_nonneg _)
  have h3 : 2 * S * (μmax + κ + U * Real.sqrt N + 1) ≤ (2 / N) * S ^ 2 := by
    rw [div_mul_eq_mul_div, le_div_iff₀ hNpos]
    nlinarith [mul_le_mul_of_nonneg_left h2 (by linarith : (0 : ℝ) ≤ 2 * S)]
  unfold dissipationBound
  nlinarith [h3, mul_le_mul_of_nonneg_left hsqrt h4, h1]

end Oscillator
