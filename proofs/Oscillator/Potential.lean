import Oscillator.Energy
import Oscillator.Obstruction
import Oscillator.LocalRule
import Oscillator.Dissipativity
import Oscillator.EqProp

/-!
# P3 for a general polynomial on-site potential

The energy of P3 with an arbitrary real polynomial on-site potential `V_i(ρ)`, `ρ = |w_i|²`:

  `E_V(w) = ∑ i V_i(|w_i|²) − ∑ i j Re(conj(w_i) K_ij w_j) − 2 ∑ i Re(conj(u_i) w_i)`,

whose gradient flow is `F_V(w)_i = −V_i'(|w_i|²) w_i + (K w)_i + u_i`.  Two instances:

* `slPotential μ`, `V(ρ) = −μρ + ½ρ²`: the Stuart–Landau unit, `F = (μ − |w|²) w + …`; this
  recovers `energy` and `field` exactly (`energyV_sl`, `fieldV_sl`), so every theorem of
  `Oscillator.Energy` is the cubic instance of the theorems here;
* `bistablePotential μ g s`, `V(ρ) = −μρ − (g/2)ρ² + (s/3)ρ³` with `s > 0`: the subcritical,
  bistable unit, `F = (μ + g|w|² − s|w|⁴) w + …` (`fieldV_bistable`).

Proved for every polynomial potential: the gradient structure `∇E_V = −2 F_V` for Hermitian `K`
(`hasFDerivAt_energyV`), the descent identity `dE_V/dt = −2 ∑ |F_V(w)_i|²` along solutions
(`energyV_hasDerivAt_solution`, `energyV_antitone_solution`), critical points = equilibria
(`fderiv_energyV_eq_zero_iff`), and boundedness below when every `V_i` has degree at least `2`
and positive leading coefficient (`energyV_bddBelow`); the degree condition is needed, since a
linear potential loses to a strong enough coupling.
-/

namespace Oscillator

open Complex Matrix Polynomial
open scoped ComplexConjugate

variable {ι : Type*} [Fintype ι]

/-! ### Definitions and the two instances -/

/-- The energy with on-site potentials `V i`, evaluated at `|w_i|²`. -/
noncomputable def energyV (V : ι → ℝ[X]) (K : Matrix ι ι ℂ) (u : ι → ℂ) (w : ι → ℂ) : ℝ :=
  (∑ i, (V i).eval (normSq (w i)))
    - (∑ i, ∑ j, (conj (w i) * K i j * w j).re)
    - 2 * ∑ i, (conj (u i) * w i).re

/-- The vector field `F_V(w)_i = −V_i'(|w_i|²) w_i + (K w)_i + u_i`. -/
noncomputable def fieldV (V : ι → ℝ[X]) (K : Matrix ι ι ℂ) (u : ι → ℂ) (w : ι → ℂ) : ι → ℂ :=
  fun i => -(((derivative (V i)).eval (normSq (w i)) : ℝ) : ℂ) * w i + (K *ᵥ w) i + u i

/-- The field with residual frequencies. -/
noncomputable def rotFieldV (ω : ι → ℝ) (V : ι → ℝ[X]) (K : Matrix ι ι ℂ) (u : ι → ℂ)
    (w : ι → ℂ) : ι → ℂ :=
  fun i => fieldV V K u w i + (ω i : ℂ) * I * w i

/-- The Stuart–Landau potential `V(ρ) = −μρ + ½ρ²`. -/
noncomputable def slPotential (μ : ℝ) : ℝ[X] := C (-μ) * X + C (1 / 2) * X ^ 2

/-- The subcritical, bistable potential `V(ρ) = −μρ − (g/2)ρ² + (s/3)ρ³`. -/
noncomputable def bistablePotential (μ g s : ℝ) : ℝ[X] :=
  C (-μ) * X + C (-(g / 2)) * X ^ 2 + C (s / 3) * X ^ 3

@[simp] theorem slPotential_eval (μ ρ : ℝ) :
    (slPotential μ).eval ρ = -μ * ρ + 1 / 2 * ρ ^ 2 := by
  simp [slPotential]

@[simp] theorem derivative_slPotential_eval (μ ρ : ℝ) :
    (derivative (slPotential μ)).eval ρ = -μ + ρ := by
  simp [slPotential, derivative_mul]
  ring

@[simp] theorem bistablePotential_eval (μ g s ρ : ℝ) :
    (bistablePotential μ g s).eval ρ = -μ * ρ - g / 2 * ρ ^ 2 + s / 3 * ρ ^ 3 := by
  simp [bistablePotential]
  ring

@[simp] theorem derivative_bistablePotential_eval (μ g s ρ : ℝ) :
    (derivative (bistablePotential μ g s)).eval ρ = -μ - g * ρ + s * ρ ^ 2 := by
  simp [bistablePotential, derivative_mul]
  ring

variable {K : Matrix ι ι ℂ} {u : ι → ℂ}

/-- The Stuart–Landau instance recovers `energy`. -/
theorem energyV_sl (μ : ι → ℝ) : energyV (fun i => slPotential (μ i)) K u = energy μ K u := by
  funext w
  simp [energyV, energy]

/-- The Stuart–Landau instance recovers `field`. -/
theorem fieldV_sl (μ : ι → ℝ) : fieldV (fun i => slPotential (μ i)) K u = field μ K u := by
  funext w i
  simp only [fieldV, field, derivative_slPotential_eval]
  push_cast
  ring

theorem rotFieldV_sl (ω μ : ι → ℝ) :
    rotFieldV ω (fun i => slPotential (μ i)) K u = rotField ω μ K u := by
  funext w i
  simp [rotFieldV, rotField, fieldV_sl]

/-- The bistable instance: `F(w)_i = (μ_i + g_i |w_i|² − s_i |w_i|⁴) w_i + (K w)_i + u_i`. -/
theorem fieldV_bistable (μ g s : ι → ℝ) (w : ι → ℂ) :
    fieldV (fun i => bistablePotential (μ i) (g i) (s i)) K u w
      = fun i => ((μ i : ℂ) + g i * normSq (w i) - s i * normSq (w i) ^ 2) * w i
        + (K *ᵥ w) i + u i := by
  funext i
  simp only [fieldV, derivative_bistablePotential_eval]
  push_cast
  ring

/-! ### The gradient structure -/

/-- Derivative of the on-site term `w ↦ V(|w_i|²)`: `2 V'(|w_i|²) Re(conj(w_i) v_i)`. -/
theorem hasFDerivAt_onsite [DecidableEq ι] (V : ℝ[X]) (i : ι) (w : ι → ℂ) :
    HasFDerivAt (fun w : ι → ℂ => V.eval (normSq (w i)))
      (pairing (Pi.single i (2 * (((derivative V).eval (normSq (w i)) : ℝ) : ℂ) * w i))) w := by
  have h := (V.hasDerivAt (normSq (w i))).comp_hasFDerivAt w (hasFDerivAt_normSq_coord i w)
  refine h.congr_fderiv ?_
  ext v
  simp only [_root_.smul_apply, pairing_single, smul_eq_mul]
  simp
  ring

/-- The vector identity behind the gradient structure for a general potential. -/
theorem gradient_vector_identityV [DecidableEq ι] (hK : K.IsHermitian) (V : ι → ℝ[X])
    (w : ι → ℂ) :
    (∑ i, (Pi.single i (2 * (((derivative (V i)).eval (normSq (w i)) : ℝ) : ℂ) * w i) : ι → ℂ))
      - (∑ i, ∑ j, ((Pi.single i (K i j * w j) : ι → ℂ)
          + (Pi.single j (conj (K i j) * w i) : ι → ℂ)))
      - (2 : ℝ) • ∑ i, (Pi.single i (u i) : ι → ℂ) = (-2 : ℝ) • fieldV V K u w := by
  funext k
  have hKk : ∀ i, conj (K i k) = K k i := fun i => by
    have := hK.apply k i
    rwa [Complex.star_def] at this
  simp only [Pi.sub_apply, Pi.smul_apply, Finset.sum_apply, Pi.add_apply, Pi.single_apply,
    Finset.sum_add_distrib, Finset.sum_ite_eq, Finset.mem_univ, ite_true,
    fieldV, mulVec, dotProduct, Complex.real_smul]
  rw [Finset.sum_comm (f := fun i j => if k = i then K i j * w j else 0)]
  simp only [Finset.sum_ite_eq, Finset.mem_univ, ite_true, hKk]
  push_cast
  ring

/-- **P3 for a general potential.**  For a Hermitian coupling, `∇E_V(w) = −2 F_V(w)`. -/
theorem hasFDerivAt_energyV (hK : K.IsHermitian) (V : ι → ℝ[X]) (w : ι → ℂ) :
    HasFDerivAt (energyV V K u) (pairing ((-2 : ℝ) • fieldV V K u w)) w := by
  classical
  have h1 : HasFDerivAt (fun w : ι → ℂ => ∑ i, (V i).eval (normSq (w i)))
      (∑ i, pairing (Pi.single i (2 * (((derivative (V i)).eval (normSq (w i)) : ℝ) : ℂ) * w i))) w :=
    HasFDerivAt.fun_sum fun i _ => hasFDerivAt_onsite (V i) i w
  have h2 : HasFDerivAt (fun w : ι → ℂ => ∑ i, ∑ j, (conj (w i) * K i j * w j).re)
      (∑ i, ∑ j, pairing (Pi.single i (K i j * w j) + Pi.single j (conj (K i j) * w i))) w :=
    HasFDerivAt.fun_sum fun i _ => HasFDerivAt.fun_sum fun j _ =>
      hasFDerivAt_reConjMul i j (K i j) w
  have h3 : HasFDerivAt (fun w : ι → ℂ => 2 * ∑ i, (conj (u i) * w i).re)
      ((2 : ℝ) • ∑ i, pairing (Pi.single i (u i))) w :=
    (HasFDerivAt.fun_sum fun i _ => hasFDerivAt_input i (u i) w).const_mul (2 : ℝ)
  refine ((h1.fun_sub h2).fun_sub h3).congr_fderiv ?_
  simp only [← pairingₗ_apply, ← map_smul, ← map_sum, ← map_sub]
  exact congrArg pairingₗ (gradient_vector_identityV hK V w)

theorem differentiable_energyV (hK : K.IsHermitian) (V : ι → ℝ[X]) :
    Differentiable ℝ (energyV V K u) :=
  fun w => (hasFDerivAt_energyV (u := u) hK V w).differentiableAt

/-- Along a curve solving `ẇ = F_V(w)` at time `t`: `d/dt E_V(w(t)) = −2 ∑ i |F_V(w(t))_i|²`. -/
theorem energyV_hasDerivAt_solution (hK : K.IsHermitian) (V : ι → ℝ[X]) {w : ℝ → ι → ℂ} {t : ℝ}
    (hw : HasDerivAt w (fieldV V K u (w t)) t) :
    HasDerivAt (fun s => energyV V K u (w s)) (-2 * sumNormSq (fieldV V K u (w t))) t := by
  have h := (hasFDerivAt_energyV (u := u) hK V (w t)).comp_hasDerivAt t hw
  rw [pairing_smul_self] at h
  exact h

theorem energyV_deriv_nonpos_solution (hK : K.IsHermitian) (V : ι → ℝ[X]) {w : ℝ → ι → ℂ}
    {t : ℝ} (hw : HasDerivAt w (fieldV V K u (w t)) t) :
    deriv (fun s => energyV V K u (w s)) t ≤ 0 := by
  rw [(energyV_hasDerivAt_solution hK V hw).deriv]
  have := sumNormSq_nonneg (fieldV V K u (w t))
  linarith

/-- Along a global solution the energy is non-increasing. -/
theorem energyV_antitone_solution (hK : K.IsHermitian) (V : ι → ℝ[X]) {w : ℝ → ι → ℂ}
    (hw : ∀ t, HasDerivAt w (fieldV V K u (w t)) t) :
    Antitone (fun t => energyV V K u (w t)) := by
  refine antitone_of_deriv_nonpos (fun t => ?_)
    (fun t => energyV_deriv_nonpos_solution hK V (hw t))
  exact (energyV_hasDerivAt_solution hK V (hw t)).differentiableAt

/-- The critical points of `E_V` are exactly the zeros of `F_V`. -/
theorem fderiv_energyV_eq_zero_iff (hK : K.IsHermitian) (V : ι → ℝ[X]) (w : ι → ℂ) :
    fderiv ℝ (energyV V K u) w = 0 ↔ fieldV V K u w = 0 := by
  rw [(hasFDerivAt_energyV hK V w).fderiv]
  constructor
  · intro h
    have h1 : pairing ((-2 : ℝ) • fieldV V K u w) (fieldV V K u w) = 0 := by
      rw [h]; rfl
    rw [pairing_smul_self, sumNormSq] at h1
    have h2 : ∑ i, normSq (fieldV V K u w i) = 0 := by linarith
    rw [Finset.sum_eq_zero_iff_of_nonneg (fun i _ => normSq_nonneg _)] at h2
    funext i
    exact normSq_eq_zero.mp (h2 i (Finset.mem_univ i))
  · intro h
    rw [h]
    ext v
    simp [pairing_apply]

/-! ### Boundedness below from the leading coefficient -/

/-- A real polynomial of degree at least `2` with positive leading coefficient dominates every
linear function from below on `[0, ∞)`. -/
theorem exists_lowerBound_sub_linear (p : ℝ[X]) (hdeg : 2 ≤ p.natDegree)
    (hlc : 0 < p.leadingCoeff) (a : ℝ) :
    ∃ m : ℝ, ∀ ρ : ℝ, 0 ≤ ρ → m ≤ p.eval ρ - a * ρ := by
  set q : ℝ[X] := p - C a * X with hq
  have hp0 : p ≠ 0 := leadingCoeff_ne_zero.1 hlc.ne'
  have hdegp : (2 : WithBot ℕ) ≤ p.degree := by
    rw [degree_eq_natDegree hp0]
    exact_mod_cast hdeg
  have hlt : (C a * X).degree < p.degree :=
    lt_of_le_of_lt (degree_C_mul_X_le a) (lt_of_lt_of_le (by norm_num) hdegp)
  have hqdeg : 0 < q.degree := by
    rw [hq, degree_sub_eq_left_of_degree_lt hlt]
    exact lt_of_lt_of_le (by norm_num) hdegp
  have hqlc : 0 ≤ q.leadingCoeff := by
    rw [hq, leadingCoeff_sub_of_degree_lt hlt]
    exact hlc.le
  have ht := q.tendsto_atTop_of_leadingCoeff_nonneg hqdeg hqlc
  obtain ⟨R, hR⟩ := Filter.eventually_atTop.1 (ht.eventually_ge_atTop 0)
  have hc : IsCompact ((fun x => q.eval x) '' Set.Icc (0 : ℝ) R) :=
    isCompact_Icc.image q.continuous
  obtain ⟨m, hm⟩ := hc.bddBelow
  refine ⟨min m 0, fun ρ hρ => ?_⟩
  have hqρ : q.eval ρ = p.eval ρ - a * ρ := by
    simp [hq]
  rw [← hqρ]
  by_cases hρR : ρ ≤ R
  · have hmem : q.eval ρ ∈ (fun x => q.eval x) '' Set.Icc 0 R := ⟨ρ, ⟨hρ, hρR⟩, rfl⟩
    exact le_trans (min_le_left _ _) (hm hmem)
  · exact le_trans (min_le_right _ _) (hR ρ (le_of_not_ge hρR))

/-- **P3, boundedness for a general potential.**  If every `V_i` has degree at least `2` and a
positive leading coefficient, `E_V` is bounded below (for any `K` and `u`). -/
theorem energyV_bddBelow (V : ι → ℝ[X]) (hdeg : ∀ i, 2 ≤ (V i).natDegree)
    (hlc : ∀ i, 0 < (V i).leadingCoeff) :
    ∃ m : ℝ, ∀ w, m ≤ energyV V K u w := by
  choose m hm using fun i =>
    exists_lowerBound_sub_linear (V i) (hdeg i) (hlc i) (couplingWeight K i + 1)
  refine ⟨∑ i, m i - ∑ i, normSq (u i), fun w => ?_⟩
  have hcoup := couplingForm_le K w
  have hin : 2 * ∑ i, (conj (u i) * w i).re ≤ ∑ i, normSq (u i) + ∑ i, normSq (w i) := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_le_sum fun i _ => two_re_conj_mul_le _ _
  have hsum : ∑ i, m i ≤ ∑ i, (V i).eval (normSq (w i))
      - (∑ i, couplingWeight K i * normSq (w i) + ∑ i, normSq (w i)) := by
    have h1 : ∑ i, m i ≤ ∑ i, ((V i).eval (normSq (w i))
        - (couplingWeight K i + 1) * normSq (w i)) :=
      Finset.sum_le_sum fun i _ => hm i _ (normSq_nonneg _)
    have h2 : ∑ i, (couplingWeight K i + 1) * normSq (w i)
        = ∑ i, couplingWeight K i * normSq (w i) + ∑ i, normSq (w i) := by
      rw [← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring
    rw [Finset.sum_sub_distrib, h2] at h1
    exact h1
  unfold energyV
  linarith

theorem energyV_bddBelow' (V : ι → ℝ[X]) (hdeg : ∀ i, 2 ≤ (V i).natDegree)
    (hlc : ∀ i, 0 < (V i).leadingCoeff) : BddBelow (Set.range (energyV V K u)) := by
  obtain ⟨m, hm⟩ := energyV_bddBelow (K := K) (u := u) V hdeg hlc
  exact ⟨m, by rintro _ ⟨w, rfl⟩; exact hm w⟩

/-- The bistable potential with `s > 0` has degree `3` and leading coefficient `s/3`, so its
energy is bounded below. -/
theorem bistablePotential_natDegree (μ g s : ℝ) (hs : s ≠ 0) :
    (bistablePotential μ g s).natDegree = 3 := by
  unfold bistablePotential
  compute_degree!

theorem bistablePotential_leadingCoeff (μ g s : ℝ) (hs : s ≠ 0) :
    (bistablePotential μ g s).leadingCoeff = s / 3 := by
  rw [leadingCoeff, bistablePotential_natDegree μ g s hs]
  unfold bistablePotential
  simp [coeff_X_pow]

theorem bistable_energy_bddBelow (μ g s : ι → ℝ) (hs : ∀ i, 0 < s i) :
    ∃ m : ℝ, ∀ w, m ≤ energyV (fun i => bistablePotential (μ i) (g i) (s i)) K u w :=
  energyV_bddBelow _ (fun i => by rw [bistablePotential_natDegree _ _ _ (hs i).ne']; norm_num)
    (fun i => by
      rw [bistablePotential_leadingCoeff _ _ _ (hs i).ne']
      exact div_pos (hs i) (by norm_num))

/-! ### The obstruction for a general potential -/

/-- At the origin the Jacobian of `F_{V,ω}` is `v ↦ (−V_i'(0) + i ω_i) v_i + (K v)_i`, the same
shape as for the Stuart–Landau field with `μ_i = −V_i'(0)`. -/
theorem hasFDerivAt_rotFieldV_zero (ω : ι → ℝ) (V : ι → ℝ[X]) :
    HasFDerivAt (rotFieldV ω V K u)
      (rotJacobian ω (fun i => -(derivative (V i)).eval 0) K 0) 0 := by
  classical
  refine hasFDerivAt_pi'.2 fun i => ?_
  have hc : HasFDerivAt (fun w : ι → ℂ => (((derivative (V i)).eval (normSq (w i)) : ℝ) : ℂ))
      (ofRealCLM.comp (pairing (Pi.single i
        (2 * (((derivative (derivative (V i))).eval (normSq ((0 : ι → ℂ) i)) : ℝ) : ℂ)
          * (0 : ι → ℂ) i)))) 0 :=
    ofRealCLM.hasFDerivAt.comp (0 : ι → ℂ) (hasFDerivAt_onsite (derivative (V i)) i 0)
  have hn := (hc.fun_neg).mul (hasFDerivAt_coord i (0 : ι → ℂ))
  have hKv : HasFDerivAt (fun w : ι → ℂ => (K *ᵥ w) i)
      (∑ j, K i j • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) j) 0 :=
    HasFDerivAt.fun_sum fun j _ => (hasFDerivAt_coord j 0).const_mul (K i j)
  have hu : HasFDerivAt (fun _ : ι → ℂ => u i) (0 : (ι → ℂ) →L[ℝ] ℂ) 0 := hasFDerivAt_const _ _
  have hω : HasFDerivAt (fun w : ι → ℂ => (ω i : ℂ) * I * w i)
      (((ω i : ℂ) * I) • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) i) 0 :=
    (hasFDerivAt_coord i 0).const_mul _
  have h := ((hn.fun_add hKv).fun_add hu).fun_add hω
  refine h.congr_fderiv ?_
  ext v
  simp only [ContinuousLinearMap.comp_apply, _root_.add_apply, _root_.smul_apply]
  simp [rotJacobian]
  ring

/-- If `F_{V,ω}` is a gradient field for the real inner product, `K + i·diag(ω)` is Hermitian. -/
theorem isHermitian_of_isGradientFieldV [DecidableEq ι] (ω : ι → ℝ) (V : ι → ℝ[X])
    (h : ∃ φ, IsGradientFieldOf pairingCLM (rotFieldV ω V K u) φ) :
    (K + diagonal (fun i => (ω i : ℂ) * I)).IsHermitian :=
  isHermitian_of_isGradientField_of_jacobian (hasFDerivAt_rotFieldV_zero (u := u) ω V) h

/-- **Asymmetric coupling, any potential.**  With a common frequency, a non-Hermitian coupling
gives a field that is not a gradient field. -/
theorem not_isGradientFieldV_of_not_isHermitian [DecidableEq ι] (V : ι → ℝ[X])
    (hK : ¬ K.IsHermitian) : ¬ ∃ φ, IsGradientFieldOf pairingCLM (fieldV V K u) φ := by
  intro h
  apply hK
  have h' : ∃ φ, IsGradientFieldOf pairingCLM (rotFieldV 0 V K u) φ := by
    have e : rotFieldV 0 V K u = fieldV V K u := by
      funext w i
      simp [rotFieldV]
    rw [e]
    exact h
  have := isHermitian_of_isGradientFieldV 0 V h'
  simpa using this

/-- **Residual frequencies, any potential.**  With a Hermitian coupling, a nonzero residual
frequency at any unit gives a field that is not a gradient field. -/
theorem not_isGradientFieldV_of_frequency [DecidableEq ι] (V : ι → ℝ[X]) (hK : K.IsHermitian)
    {ω : ι → ℝ} {i : ι} (hi : ω i ≠ 0) :
    ¬ ∃ φ, IsGradientFieldOf pairingCLM (rotFieldV ω V K u) φ := by
  intro h
  have hM := isHermitian_of_isGradientFieldV ω V h
  have hD : (diagonal (fun i => (ω i : ℂ) * I)).IsHermitian := by
    have := hM.sub hK
    simpa using this
  have := hD.apply i i
  simp [Complex.ext_iff] at this
  exact hi (by linarith)

/-! ### The threshold as a learnable parameter: `V_i(ρ) = −μ_i ρ + W_i(ρ)` -/

/-- Potentials with a learnable threshold `μ_i` in front of `ρ` and a fixed remainder `W_i`.
Both instances have this form. -/
noncomputable def thresholdPotential (W : ι → ℝ[X]) (μ : ι → ℝ) : ι → ℝ[X] :=
  fun i => C (-μ i) * X + W i

omit [Fintype ι] in
theorem slPotential_eq_threshold (μ : ι → ℝ) :
    (fun i => slPotential (μ i)) = thresholdPotential (fun _ => C (1 / 2) * X ^ 2) μ := rfl

omit [Fintype ι] in
theorem bistablePotential_eq_threshold (μ g s : ι → ℝ) :
    (fun i => bistablePotential (μ i) (g i) (s i))
      = thresholdPotential (fun i => C (-(g i / 2)) * X ^ 2 + C (s i / 3) * X ^ 3) μ := by
  funext i
  simp only [bistablePotential, thresholdPotential]
  ring

theorem energyV_threshold (W : ι → ℝ[X]) (μ : ι → ℝ) (w : ι → ℂ) :
    energyV (thresholdPotential W μ) K u w
      = (∑ i, (-μ i * normSq (w i) + (W i).eval (normSq (w i))))
        - (∑ i, ∑ j, (conj (w i) * K i j * w j).re) - 2 * ∑ i, (conj (u i) * w i).re := by
  simp [energyV, thresholdPotential]

/-- The derivative of `E_V` in the thresholds: `dE[δμ] = −∑ i δμ_i |w_i|²`, for any remainder `W`. -/
theorem hasFDerivAt_energyV_mu (W : ι → ℝ[X]) (K : Matrix ι ι ℂ) (u w : ι → ℂ) (μ : ι → ℝ) :
    HasFDerivAt (fun μ : ι → ℝ => energyV (thresholdPotential W μ) K u w)
      (-∑ i, normSq (w i) • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℝ) i) μ := by
  have e : (fun μ : ι → ℝ => energyV (thresholdPotential W μ) K u w)
      = fun μ => (∑ i, (-μ i * normSq (w i) + (W i).eval (normSq (w i))))
        - (∑ i, ∑ j, (conj (w i) * K i j * w j).re) - 2 * ∑ i, (conj (u i) * w i).re := by
    funext μ
    exact energyV_threshold W μ w
  rw [e]
  have h1 : HasFDerivAt
      (fun μ : ι → ℝ => ∑ i, (-μ i * normSq (w i) + (W i).eval (normSq (w i))))
      (∑ i, normSq (w i) • -(ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℝ) i)) μ :=
    HasFDerivAt.fun_sum fun i _ =>
      (((hasFDerivAt_apply i μ).fun_neg).mul_const (normSq (w i))).add_const _
  have h2 : HasFDerivAt (fun _ : ι → ℝ => ∑ i, ∑ j, (conj (w i) * K i j * w j).re)
      (0 : (ι → ℝ) →L[ℝ] ℝ) μ := hasFDerivAt_const _ _
  have h3 : HasFDerivAt (fun _ : ι → ℝ => 2 * ∑ i, (conj (u i) * w i).re)
      (0 : (ι → ℝ) →L[ℝ] ℝ) μ := hasFDerivAt_const _ _
  refine ((h1.fun_sub h2).fun_sub h3).congr_fderiv ?_
  ext δ
  simp [Finset.sum_neg_distrib]

theorem fderiv_energyV_mu_apply (W : ι → ℝ[X]) (w : ι → ℂ) (μ δ : ι → ℝ) :
    fderiv ℝ (fun μ : ι → ℝ => energyV (thresholdPotential W μ) K u w) μ δ
      = -∑ i, δ i * normSq (w i) := by
  rw [(hasFDerivAt_energyV_mu W K u w μ).fderiv]
  simp [mul_comm]

/-- **The local threshold rule for any potential.**  `∂E_V/∂μ_i = −|w_i|²`. -/
theorem hasDerivAt_energyV_mu_coord [DecidableEq ι] (W : ι → ℝ[X]) (μ : ι → ℝ) (w : ι → ℂ)
    (i : ι) (t : ℝ) :
    HasDerivAt (fun t : ℝ => energyV (thresholdPotential W (Function.update μ i t)) K u w)
      (-normSq (w i)) t := by
  have h := (hasFDerivAt_energyV_mu W K u w (Function.update μ i t)).comp_hasDerivAt t
    (hasDerivAt_update μ i t)
  have e : (-∑ j, normSq (w j) • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℝ) j)
      (Pi.single i (1 : ℝ)) = -normSq (w i) := by
    simp [Pi.single_apply]
  rw [e] at h
  exact h

theorem deriv_energyV_mu_coord [DecidableEq ι] (W : ι → ℝ[X]) (μ : ι → ℝ) (w : ι → ℂ)
    (i : ι) (t : ℝ) :
    deriv (fun t : ℝ => energyV (thresholdPotential W (Function.update μ i t)) K u w) t
      = -normSq (w i) :=
  (hasDerivAt_energyV_mu_coord W μ w i t).deriv

/-! ### P5 for the bistable unit -/

/-- The frequency term drops out of `Re(conj(w_i) F_{V,ω}(w)_i)`. -/
theorem re_conj_mul_rotFieldV (ω : ι → ℝ) (V : ι → ℝ[X]) (w : ι → ℂ) (i : ι) :
    (conj (w i) * rotFieldV ω V K u w i).re
      = -(derivative (V i)).eval (normSq (w i)) * normSq (w i)
        + (conj (w i) * (K *ᵥ w) i).re + (conj (w i) * u i).re := by
  simp only [rotFieldV, fieldV, normSq_apply, Complex.mul_re, Complex.mul_im, Complex.add_re,
    Complex.add_im, Complex.neg_re, Complex.neg_im, Complex.conj_re, Complex.conj_im,
    Complex.ofReal_re, Complex.ofReal_im, Complex.I_re, Complex.I_im]
  ring

/-- `∑ i |w_i|⁴ ≤ (∑ i |w_i|²)²`. -/
theorem sum_sq_normSq_le (w : ι → ℂ) : ∑ i, normSq (w i) ^ 2 ≤ sumNormSq w ^ 2 := by
  have h : ∀ i, normSq (w i) ^ 2 ≤ normSq (w i) * sumNormSq w := fun i => by
    have hle : normSq (w i) ≤ sumNormSq w :=
      Finset.single_le_sum (fun j _ => normSq_nonneg (w j)) (Finset.mem_univ i)
    nlinarith [normSq_nonneg (w i)]
  calc ∑ i, normSq (w i) ^ 2 ≤ ∑ i, normSq (w i) * sumNormSq w :=
        Finset.sum_le_sum fun i _ => h i
    _ = sumNormSq w ^ 2 := by
        rw [← Finset.sum_mul]
        unfold sumNormSq
        ring

/-- `(∑ i |w_i|²)³ ≤ N² ∑ i |w_i|⁶` (the power-mean inequality). -/
theorem cube_sumNormSq_le (w : ι → ℂ) :
    sumNormSq w ^ 3 ≤ (Fintype.card ι : ℝ) ^ 2 * ∑ i, normSq (w i) ^ 3 := by
  rcases Nat.eq_zero_or_pos (Fintype.card ι) with h0 | hpos
  · have : IsEmpty ι := Fintype.card_eq_zero_iff.1 h0
    simp [sumNormSq]
  · have hN : (0 : ℝ) < Fintype.card ι := by exact_mod_cast hpos
    have hpm := Real.pow_arith_mean_le_arith_mean_pow (s := Finset.univ)
      (fun _ => 1 / (Fintype.card ι : ℝ)) (fun i => normSq (w i)) (fun i _ => by positivity)
      (by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
        exact mul_one_div_cancel hN.ne')
      (fun i _ => normSq_nonneg _) 3
    simp only [← Finset.mul_sum] at hpm
    unfold sumNormSq
    have e1 : (1 / (Fintype.card ι : ℝ) * ∑ i, normSq (w i)) ^ 3
        = (∑ i, normSq (w i)) ^ 3 / (Fintype.card ι : ℝ) ^ 3 := by
      field_simp
    rw [e1, div_le_iff₀ (by positivity)] at hpm
    have e2 : 1 / (Fintype.card ι : ℝ) * (∑ i, normSq (w i) ^ 3) * (Fintype.card ι : ℝ) ^ 3
        = (Fintype.card ι : ℝ) ^ 2 * ∑ i, normSq (w i) ^ 3 := by
      field_simp
    linarith [hpm, e2]

/-- The right-hand side of the dissipativity bound for the bistable unit. -/
noncomputable def dissipationBoundBistable (N : ℕ) (μmax gmax smin κ U S : ℝ) : ℝ :=
  2 * (μmax + κ) * S + 2 * gmax * S ^ 2 - (2 * smin / (N : ℝ) ^ 2) * S ^ 3
    + 2 * U * Real.sqrt N * Real.sqrt S

variable {μ g s ω : ι → ℝ}

/-- The bound on `dS/dt` for the bistable unit, at a fixed state. -/
theorem dissipativity_bistable_bound {κ μmax gmax smin U : ℝ} (hK : OpNormBound K κ)
    (hμ : ∀ i, μ i ≤ μmax) (hg : ∀ i, g i ≤ gmax) (hgmax : 0 ≤ gmax) (hs : ∀ i, smin ≤ s i)
    (hsmin : 0 ≤ smin) (hU : 0 ≤ U) (hu : ∀ i, ‖u i‖ ≤ U) (w : ι → ℂ) :
    2 * ∑ i, (conj (w i) * rotFieldV ω (fun i => bistablePotential (μ i) (g i) (s i)) K u w i).re
      ≤ dissipationBoundBistable (Fintype.card ι) μmax gmax smin κ U (sumNormSq w) := by
  have hterm : ∀ i,
      (conj (w i) * rotFieldV ω (fun i => bistablePotential (μ i) (g i) (s i)) K u w i).re
        = μ i * normSq (w i) + g i * normSq (w i) ^ 2 - s i * normSq (w i) ^ 3
          + (conj (w i) * (K *ᵥ w) i).re + (conj (w i) * u i).re := by
    intro i
    rw [re_conj_mul_rotFieldV, derivative_bistablePotential_eval]
    ring
  have hexp : ∑ i, (conj (w i) * rotFieldV ω (fun i => bistablePotential (μ i) (g i) (s i))
      K u w i).re
      = ∑ i, μ i * normSq (w i) + ∑ i, g i * normSq (w i) ^ 2 - ∑ i, s i * normSq (w i) ^ 3
        + ∑ i, (conj (w i) * (K *ᵥ w) i).re + ∑ i, (conj (w i) * u i).re := by
    simp only [hterm, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  have hA : ∑ i, μ i * normSq (w i) ≤ μmax * sumNormSq w := by
    rw [sumNormSq, Finset.mul_sum]
    exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hμ i) (normSq_nonneg _)
  have hB : ∑ i, g i * normSq (w i) ^ 2 ≤ gmax * sumNormSq w ^ 2 := by
    calc ∑ i, g i * normSq (w i) ^ 2 ≤ ∑ i, gmax * normSq (w i) ^ 2 :=
          Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hg i) (sq_nonneg _)
      _ = gmax * ∑ i, normSq (w i) ^ 2 := by rw [Finset.mul_sum]
      _ ≤ gmax * sumNormSq w ^ 2 := mul_le_mul_of_nonneg_left (sum_sq_normSq_le w) hgmax
  have hC : (2 * smin / (Fintype.card ι : ℝ) ^ 2) * sumNormSq w ^ 3
      ≤ 2 * ∑ i, s i * normSq (w i) ^ 3 := by
    have h1 : smin * ∑ i, normSq (w i) ^ 3 ≤ ∑ i, s i * normSq (w i) ^ 3 := by
      rw [Finset.mul_sum]
      exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hs i) (pow_nonneg (normSq_nonneg _) 3)
    rcases Nat.eq_zero_or_pos (Fintype.card ι) with h0 | hpos
    · rw [h0]
      simp only [Nat.cast_zero, zero_pow (by norm_num : (2 : ℕ) ≠ 0), div_zero, zero_mul]
      have : 0 ≤ ∑ i, normSq (w i) ^ 3 := Finset.sum_nonneg fun i _ => pow_nonneg (normSq_nonneg _) 3
      nlinarith
    · have hN : (0 : ℝ) < (Fintype.card ι : ℝ) ^ 2 := by positivity
      have h2 := cube_sumNormSq_le w
      rw [div_mul_eq_mul_div, div_le_iff₀ hN]
      nlinarith [mul_le_mul_of_nonneg_left h2 hsmin]
  have hD := sum_re_conj_mulVec_le hK w
  have hE := sum_re_conj_input_le hU hu w
  rw [hexp, dissipationBoundBistable]
  nlinarith [hA, hB, hC, hD, hE]

/-- **P5 for the bistable unit.**  Along a solution of `ẇ = F_{V,ω}(w)` with the bistable
potential, `‖K‖ ≤ κ`, `μ_i ≤ μ_max`, `g_i ≤ g_max` (`g_max ≥ 0`), `s_i ≥ s_min ≥ 0`, `|u_i| ≤ U`:
`d/dt ∑ i |w_i|² ≤ 2(μ_max + κ) S + 2 g_max S² − (2 s_min / N²) S³ + 2U√N√S`. -/
theorem dissipativity_bistable {κ μmax gmax smin U : ℝ} (hK : OpNormBound K κ)
    (hμ : ∀ i, μ i ≤ μmax) (hg : ∀ i, g i ≤ gmax) (hgmax : 0 ≤ gmax) (hs : ∀ i, smin ≤ s i)
    (hsmin : 0 ≤ smin) (hU : 0 ≤ U) (hu : ∀ i, ‖u i‖ ≤ U) {w : ℝ → ι → ℂ} {t : ℝ}
    (hw : HasDerivAt w (rotFieldV ω (fun i => bistablePotential (μ i) (g i) (s i)) K u (w t)) t) :
    deriv (fun s => sumNormSq (w s)) t
      ≤ dissipationBoundBistable (Fintype.card ι) μmax gmax smin κ U (sumNormSq (w t)) := by
  rw [(hasDerivAt_sumNormSq hw).deriv]
  exact dissipativity_bistable_bound hK hμ hg hgmax hs hsmin hU hu (w t)

/-- **The absorbing ball for the bistable unit.**  For `s_min > 0`, once
`S ≥ max 1 (N² (|μ_max + κ| + g_max + U√N + 1) / s_min)` the bound is at most `−2`. -/
theorem dissipationBoundBistable_le_neg_two {N : ℕ} {μmax gmax smin κ U S : ℝ}
    (hsmin : 0 < smin) (hU : 0 ≤ U) (hN : 0 < N)
    (hS : max 1 ((N : ℝ) ^ 2 * (|μmax + κ| + gmax + U * Real.sqrt N + 1) / smin) ≤ S) :
    dissipationBoundBistable N μmax gmax smin κ U S ≤ -2 := by
  have h1 : (1 : ℝ) ≤ S := le_trans (le_max_left _ _) hS
  have h2 : (N : ℝ) ^ 2 * (|μmax + κ| + gmax + U * Real.sqrt N + 1) / smin ≤ S :=
    le_trans (le_max_right _ _) hS
  have hNpos : (0 : ℝ) < (N : ℝ) ^ 2 := by positivity
  have hsqrt : Real.sqrt S ≤ S := by
    rw [Real.sqrt_le_left (by linarith)]
    nlinarith
  have hS2 : S ≤ S ^ 2 := by nlinarith
  have hS2' : 1 ≤ S ^ 2 := by nlinarith
  have hUN : 0 ≤ U * Real.sqrt N := mul_nonneg hU (Real.sqrt_nonneg _)
  have h2' : (N : ℝ) ^ 2 * (|μmax + κ| + gmax + U * Real.sqrt N + 1) ≤ S * smin :=
    (div_le_iff₀ hsmin).1 h2
  have hkey : 2 * (|μmax + κ| + gmax + U * Real.sqrt N + 1) * S ^ 2
      ≤ (2 * smin / (N : ℝ) ^ 2) * S ^ 3 := by
    rw [div_mul_eq_mul_div, le_div_iff₀ hNpos]
    nlinarith [mul_le_mul_of_nonneg_right h2' (sq_nonneg S)]
  have hμκ : 2 * (μmax + κ) * S ≤ 2 * |μmax + κ| * S ^ 2 := by
    have := le_abs_self (μmax + κ)
    nlinarith [abs_nonneg (μmax + κ)]
  have hUS : 2 * U * Real.sqrt N * Real.sqrt S ≤ 2 * U * Real.sqrt N * S ^ 2 :=
    mul_le_mul_of_nonneg_left (le_trans hsqrt hS2) (by positivity)
  unfold dissipationBoundBistable
  nlinarith [hkey, hμκ, hUS, hS2']

/-! ### Equilibrium propagation for the threshold family -/

section EqPropV

open Filter Topology

@[fun_prop] theorem contDiff_poly_eval {n : WithTop ℕ∞} (p : ℝ[X]) :
    ContDiff ℝ n (fun x : ℝ => p.eval x) := by
  have h := Polynomial.contDiff_aeval (𝕜 := ℝ) p n
  have e : (fun x : ℝ => p.aeval x) = fun x => p.eval x := by
    funext x
    exact congrFun (Polynomial.coe_aeval_eq_eval x) p
  rwa [e] at h

/-- The energy with any threshold family is `C²` in `(μ, w)` jointly. -/
theorem contDiff_energyV_threshold (W : ι → ℝ[X]) :
    ContDiff ℝ 2 (Function.uncurry fun μ : ι → ℝ => energyV (thresholdPotential W μ) K u) := by
  have e : (Function.uncurry fun μ : ι → ℝ => energyV (thresholdPotential W μ) K u)
      = fun p : (ι → ℝ) × (ι → ℂ) =>
        (∑ i, (-p.1 i * normSq (p.2 i) + (W i).eval (normSq (p.2 i))))
          - (∑ i, ∑ j, (conj (p.2 i) * K i j * p.2 j).re) - 2 * ∑ i, (conj (u i) * p.2 i).re := by
    funext p
    exact energyV_threshold W p.1 p.2
  rw [e]
  fun_prop

/-- **The threshold rule is a gradient step, for any potential of the threshold family**
(the Stuart–Landau and the bistable unit included).  Hypotheses: `K` Hermitian; a `C²` cost;
`w₀` an equilibrium of the free network with invertible energy Hessian.  Conclusion: a `C¹` family
of nudged equilibria `w(μ, β)` through `w₀`, along which `β ↦ −|w(μ₀, β)_i|²` has derivative
`∂L/∂μ_i` at `β = 0`, where `L(μ) = cost(w(μ, 0))`. -/
theorem eqprop_threshold_rule_V [DecidableEq ι] (hK : K.IsHermitian) (W : ι → ℝ[X])
    {cost : (ι → ℂ) → ℝ} (hcost : ContDiff ℝ 2 cost) {μ₀ : ι → ℝ} {w₀ : ι → ℂ}
    (h0 : fieldV (thresholdPotential W μ₀) K u w₀ = 0)
    (hH : (fderiv ℝ (fderiv ℝ (energyV (thresholdPotential W μ₀) K u)) w₀).IsInvertible) :
    ∃ w : (ι → ℝ) × ℝ → (ι → ℂ), w (μ₀, 0) = w₀ ∧ ContDiffAt ℝ 1 w (μ₀, 0) ∧
      (∀ᶠ p in 𝓝 (μ₀, 0),
        fderiv ℝ (fun v => energyV (thresholdPotential W p.1) K u v + p.2 * cost v) (w p) = 0) ∧
      ∀ i, HasDerivAt (fun β => -normSq (w (μ₀, β) i))
        (fderiv ℝ (fun μ => cost (w (μ, 0))) μ₀ (Pi.single i 1)) 0 := by
  have h0' : fderiv ℝ (energyV (thresholdPotential W μ₀) K u) w₀ = 0 :=
    (fderiv_energyV_eq_zero_iff hK _ w₀).2 h0
  obtain ⟨w, hw0, hw1, heq, hder⟩ :=
    eqprop_of_nondegenerate (E := fun μ => energyV (thresholdPotential W μ) K u) (C := cost)
      (contDiff_energyV_threshold W) hcost h0' hH
  refine ⟨w, hw0, hw1, heq, fun i => ?_⟩
  refine (hder (Pi.single i 1)).congr_of_eventuallyEq (Eventually.of_forall fun β => ?_)
  show -normSq (w (μ₀, β) i)
    = fderiv ℝ (Function.uncurry fun μ : ι → ℝ => energyV (thresholdPotential W μ) K u)
        (μ₀, w (μ₀, β)) (Pi.single i 1, 0)
  rw [fderiv_uncurry_inl ((contDiff_energyV_threshold W).differentiable (by norm_num))]
  have e : fderiv ℝ (fun θ : ι → ℝ =>
      (fun μ : ι → ℝ => energyV (thresholdPotential W μ) K u) θ (w (μ₀, β))) μ₀
      = fderiv ℝ (fun θ : ι → ℝ => energyV (thresholdPotential W θ) K u (w (μ₀, β))) μ₀ := rfl
  rw [e, fderiv_energyV_mu_apply]
  simp [Pi.single_apply]

end EqPropV

end Oscillator
