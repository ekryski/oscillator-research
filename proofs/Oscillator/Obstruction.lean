import Oscillator.Energy

/-!
# The obstruction: non-symmetric Jacobians are not gradients

A `C¹` vector field whose Jacobian is not symmetric at some point is not the gradient of any
function, because second derivatives of a function are symmetric
(`second_derivative_symmetric` in Mathlib).  We state this for a general real normed space `V`
with a symmetric pairing `B`, and for a real inner product space.

Applied to the Stuart–Landau field with intrinsic frequencies, `rotField ω μ K u`, the Jacobian
at `w = 0` is `v ↦ (μ_i + i ω_i) v_i + (K v)_i`, and its symmetry for the real inner product
forces `K + i·diag(ω)` to be Hermitian (`isHermitian_of_isGradientField`).  Hence

* with `ω = 0`, a non-Hermitian `K` gives a field that is not a gradient field
  (`not_isGradientField_of_not_isHermitian`);
* with a Hermitian `K`, any nonzero residual frequency gives a field that is not a gradient field
  (`not_isGradientField_of_frequency`), and if two units have different frequencies there is no
  rotating frame at all in which the field is a gradient field
  (`not_isGradientField_of_distinct_frequencies`).

This is the precise content of the notes' "violated by heterogeneous `ω_i` ... and asymmetric
`K`" in P3.
-/

namespace Oscillator

open Complex Matrix
open scoped ComplexConjugate

/-! ### The general statement -/

section General

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- `G` is the gradient field of `φ` with respect to the pairing `B`:
`dφ(y)[v] = B (G y) v` for all `y` and `v`. -/
def IsGradientFieldOf (B : V →L[ℝ] V →L[ℝ] ℝ) (G : V → V) (φ : V → ℝ) : Prop :=
  ∀ y, HasFDerivAt φ (B (G y)) y

/-- **Symmetry of the Jacobian of a gradient field.**  If `G = ∇φ` for a symmetric pairing `B`
and `G` is differentiable at `x` with derivative `G'`, then `B (G' v) w = B v (G' w)`. -/
theorem IsGradientFieldOf.jacobian_symm {B : V →L[ℝ] V →L[ℝ] ℝ} (hB : ∀ a b, B a b = B b a)
    {G : V → V} {φ : V → ℝ} (hφ : IsGradientFieldOf B G φ) {G' : V →L[ℝ] V} {x : V}
    (hG : HasFDerivAt G G' x) (v w : V) : B (G' v) w = B v (G' w) := by
  have h2 : HasFDerivAt (fun y => B (G y)) (B.comp G') x := B.hasFDerivAt.comp x hG
  have h3 := second_derivative_symmetric (f := φ) hφ h2 v w
  simp only [ContinuousLinearMap.comp_apply] at h3
  rw [h3, hB]

/-- **The obstruction.**  A vector field whose Jacobian at some point is not symmetric for a
symmetric pairing is not a gradient field for that pairing. -/
theorem not_isGradientFieldOf_of_jacobian_not_symm {B : V →L[ℝ] V →L[ℝ] ℝ}
    (hB : ∀ a b, B a b = B b a) {G : V → V} {G' : V →L[ℝ] V} {x : V}
    (hG : HasFDerivAt G G' x) (h : ∃ v w, B (G' v) w ≠ B v (G' w)) :
    ¬ ∃ φ, IsGradientFieldOf B G φ := by
  rintro ⟨φ, hφ⟩
  obtain ⟨v, w, hvw⟩ := h
  exact hvw (hφ.jacobian_symm hB hG v w)

end General

section InnerProduct

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

/-- The same statement in a real inner product space, for Mathlib's `HasGradientAt`:
the Jacobian of a gradient field is self-adjoint. -/
theorem jacobian_symm_of_hasGradientAt {G : V → V} {φ : V → ℝ}
    (hφ : ∀ y, HasGradientAt φ (G y) y) {G' : V →L[ℝ] V} {x : V}
    (hG : HasFDerivAt G G' x) (v w : V) : inner ℝ (G' v) w = inner ℝ v (G' w) := by
  have hφ' : IsGradientFieldOf (innerSL ℝ) G φ := fun y => by
    have h := (hasGradientAt_iff_hasFDerivAt.1 (hφ y))
    refine h.congr_fderiv ?_
    ext z
    rfl
  exact hφ'.jacobian_symm (fun a b => real_inner_comm b a) hG v w

end InnerProduct

/-! ### The pairing as a continuous bilinear map, and P3 in this language -/

variable {ι : Type*} [Fintype ι]

/-- The real pairing `⟨a, v⟩_ℝ` as a continuous bilinear map. -/
noncomputable def pairingCLM : (ι → ℂ) →L[ℝ] ((ι → ℂ) →L[ℝ] ℝ) :=
  LinearMap.toContinuousLinearMap pairingₗ

@[simp] theorem pairingCLM_apply (a : ι → ℂ) : pairingCLM a = pairing a := by
  simp [pairingCLM]

theorem pairing_comm (a b : ι → ℂ) : pairing a b = pairing b a := by
  simp only [pairing_apply]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp only [Complex.mul_re, Complex.conj_re, Complex.conj_im]
  ring

theorem pairingCLM_comm (a b : ι → ℂ) : pairingCLM a b = pairingCLM b a := by
  rw [pairingCLM_apply, pairingCLM_apply, pairing_comm]

variable {μ ω : ι → ℝ} {K : Matrix ι ι ℂ} {u : ι → ℂ}

/-- P3 restated: for Hermitian `K`, `−2 F` is the gradient field of `E`. -/
theorem isGradientFieldOf_energy (hK : K.IsHermitian) :
    IsGradientFieldOf pairingCLM (fun w => (-2 : ℝ) • field μ K u w) (energy μ K u) := by
  intro w
  rw [pairingCLM_apply]
  exact hasFDerivAt_energy hK w

/-! ### The Jacobian of the Stuart–Landau field -/

/-- The Jacobian (real Fréchet derivative) of `rotField ω μ K u` at `w`:
`v ↦ (μ_i + i ω_i − |w_i|²) v_i − 2 Re(conj(w_i) v_i) w_i + (K v)_i`. -/
noncomputable def rotJacobian (ω μ : ι → ℝ) (K : Matrix ι ι ℂ) (w : ι → ℂ) :
    (ι → ℂ) →L[ℝ] (ι → ℂ) :=
  ContinuousLinearMap.pi fun i =>
    ((μ i : ℂ) + (ω i : ℂ) * I - (normSq (w i) : ℂ))
        • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) i
      - (reCLM.comp (conj (w i) • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) i)).smulRight
          (2 * w i)
      + ∑ j, K i j • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) j

theorem rotJacobian_apply (w v : ι → ℂ) (i : ι) :
    rotJacobian ω μ K w v i = ((μ i : ℂ) + (ω i : ℂ) * I - (normSq (w i) : ℂ)) * v i
      - ((conj (w i) * v i).re : ℂ) * (2 * w i) + (K *ᵥ v) i := by
  simp [rotJacobian, mulVec, dotProduct, Complex.real_smul]

/-- The Stuart–Landau field (with frequencies) is differentiable, with Jacobian `rotJacobian`. -/
theorem hasFDerivAt_rotField (w : ι → ℂ) :
    HasFDerivAt (rotField ω μ K u) (rotJacobian ω μ K w) w := by
  classical
  refine hasFDerivAt_pi'.2 fun i => ?_
  have hμ : HasFDerivAt (fun w : ι → ℂ => (μ i : ℂ) * w i)
      ((μ i : ℂ) • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) i) w :=
    (hasFDerivAt_coord i w).const_mul _
  have hn : HasFDerivAt (fun w : ι → ℂ => ((normSq (w i) : ℝ) : ℂ) * w i)
      (((normSq (w i) : ℝ) : ℂ) • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) i
        + w i • (ofRealCLM.comp (pairing (Pi.single i (2 * w i))))) w :=
    (ofRealCLM.hasFDerivAt.comp w (hasFDerivAt_normSq_coord i w)).mul (hasFDerivAt_coord i w)
  have hKv : HasFDerivAt (fun w : ι → ℂ => (K *ᵥ w) i)
      (∑ j, K i j • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) j) w :=
    HasFDerivAt.fun_sum fun j _ => (hasFDerivAt_coord j w).const_mul (K i j)
  have hu : HasFDerivAt (fun _ : ι → ℂ => u i) (0 : (ι → ℂ) →L[ℝ] ℂ) w := hasFDerivAt_const _ _
  have hω : HasFDerivAt (fun w : ι → ℂ => (ω i : ℂ) * I * w i)
      (((ω i : ℂ) * I) • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) i) w :=
    (hasFDerivAt_coord i w).const_mul _
  have h := (((hμ.fun_sub hn).fun_add hKv).fun_add hu).fun_add hω
  refine h.congr_fderiv ?_
  ext v
  simp only [ContinuousLinearMap.comp_apply, _root_.add_apply, _root_.sub_apply, _root_.smul_apply,
    pairing_single]
  simp [rotJacobian, Complex.real_smul]
  ring

theorem differentiable_rotField : Differentiable ℝ (rotField ω μ K u) :=
  fun w => (hasFDerivAt_rotField (u := u) w).differentiableAt

theorem hasFDerivAt_field (w : ι → ℂ) :
    HasFDerivAt (field μ K u) (rotJacobian 0 μ K w) w := by
  have := hasFDerivAt_rotField (ω := 0) (μ := μ) (K := K) (u := u) w
  rwa [rotField_zero] at this

/-- At the origin the Jacobian is the linear part `v ↦ (μ_i + i ω_i) v_i + (K v)_i`. -/
theorem rotJacobian_zero_apply (v : ι → ℂ) :
    rotJacobian ω μ K 0 v = fun i => ((μ i : ℂ) + (ω i : ℂ) * I) * v i + (K *ᵥ v) i := by
  funext i
  simp [rotJacobian_apply]

/-! ### The obstruction for the Stuart–Landau field -/

theorem pairing_apply_single [DecidableEq ι] (a : ι → ℂ) (k : ι) (b : ℂ) :
    pairing a (Pi.single k b) = (conj (a k) * b).re := by
  rw [pairing_comm, pairing_single]
  simp
  ring

/-- If the field with frequencies `ω` is a gradient field for the real inner product, then
`K + i·diag(ω)` is Hermitian. -/
theorem isHermitian_of_isGradientField [DecidableEq ι]
    (h : ∃ φ, IsGradientFieldOf pairingCLM (rotField ω μ K u) φ) :
    (K + diagonal (fun i => (ω i : ℂ) * I)).IsHermitian := by
  obtain ⟨φ, hφ⟩ := h
  have hsym : ∀ v w, pairing (rotJacobian ω μ K 0 v) w = pairing v (rotJacobian ω μ K 0 w) := by
    intro v w
    have := hφ.jacobian_symm pairingCLM_comm (hasFDerivAt_rotField (u := u) 0) v w
    simpa using this
  refine Matrix.IsHermitian.ext fun j k => ?_
  have h1 := hsym (Pi.single j 1) (Pi.single k 1)
  have h2 := hsym (Pi.single j 1) (Pi.single k I)
  simp only [rotJacobian_zero_apply, pairing_single, pairing_apply_single] at h1 h2
  by_cases hjk : j = k
  · subst hjk
    simp [mulVec, dotProduct, Pi.single_apply, Complex.ext_iff] at h1 h2 ⊢
    linarith
  · have hkj : k ≠ j := fun h => hjk h.symm
    simp [mulVec, dotProduct, Pi.single_apply, hjk, hkj, Complex.ext_iff] at h1 h2 ⊢
    constructor <;> linarith

/-- **Corollary (asymmetric coupling).**  With a common frequency (`ω = 0`), a non-Hermitian
coupling gives a field that is not a gradient field for the real inner product. -/
theorem not_isGradientField_of_not_isHermitian [DecidableEq ι] (hK : ¬ K.IsHermitian) :
    ¬ ∃ φ, IsGradientFieldOf pairingCLM (field μ K u) φ := by
  intro h
  apply hK
  have h' : ∃ φ, IsGradientFieldOf pairingCLM (rotField 0 μ K u) φ := by
    simpa [rotField_zero] using h
  have := isHermitian_of_isGradientField h'
  simpa using this

/-- **Corollary (residual frequencies).**  With a Hermitian coupling, a nonzero residual
frequency at any unit gives a field that is not a gradient field. -/
theorem not_isGradientField_of_frequency [DecidableEq ι] (hK : K.IsHermitian) {i : ι}
    (hi : ω i ≠ 0) :
    ¬ ∃ φ, IsGradientFieldOf pairingCLM (rotField ω μ K u) φ := by
  intro h
  have hM := isHermitian_of_isGradientField h
  have hD : (diagonal (fun i => (ω i : ℂ) * I)).IsHermitian := by
    have := hM.sub hK
    simpa using this
  have := hD.apply i i
  simp [Complex.ext_iff] at this
  exact hi (by linarith)

/-- **Corollary (distinct frequencies).**  With a Hermitian coupling and two units of different
intrinsic frequency, there is no rotating frame `Ω` in which the field is a gradient field. -/
theorem not_isGradientField_of_distinct_frequencies [DecidableEq ι] (hK : K.IsHermitian)
    {i j : ι} (hij : ω i ≠ ω j) (Ω : ℝ) :
    ¬ ∃ φ, IsGradientFieldOf pairingCLM (rotField (fun k => ω k - Ω) μ K u) φ := by
  rcases ne_or_eq (ω i - Ω) 0 with h | h
  · exact not_isGradientField_of_frequency hK (i := i) h
  · have h' : ω j - Ω ≠ 0 := fun h' => hij (by linarith)
    exact not_isGradientField_of_frequency hK (i := j) h'

end Oscillator
