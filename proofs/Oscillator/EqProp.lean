import Oscillator.LocalRule

/-!
# Equilibrium propagation

Scellier & Bengio's identity, in the form the notes use.  Let `E(θ, v)` be an energy with
parameters `θ`, `C(v)` a cost, and `w(θ, β)` the equilibrium of the nudged energy
`E(θ, ·) + β C` (so `w(θ, 0)` is the free equilibrium).  Then for every parameter direction `h`

  `d/dθ [C(w(θ, 0))] · h = d/dβ |_{β=0} [∂_θ E(θ, w(θ, β)) · h]`,

i.e. the gradient of the loss `L(θ) = C(w(θ, 0))` is the `β → 0` limit of the contrast of
`∂E/∂θ` between the nudged and the free equilibrium (`eqprop_identity`).  The proof is the
symmetry of second derivatives of `Φ(θ, β) = E(θ, w) + β C(w)` at `w = w(θ, β)`, whose first
derivative is `(∂_θ E, C(w))` by the equilibrium condition.

The family `w` exists and is `C¹` near any nondegenerate equilibrium (invertible Hessian) by the
implicit function theorem (`exists_nudged_equilibria`, from Mathlib's
`ContDiffAt.implicitFunction`), which gives the unconditional statement
`eqprop_of_nondegenerate`.

For the Stuart–Landau network with the thresholds `μ` as parameters, `∂E/∂μ_i = −|w_i|²`
(`Oscillator.LocalRule`), so the identity says that the amplitude contrast
`(|w_i^β|² − |w_i^0|²)/β` converges, as `β → 0`, to minus the gradient of the loss in `μ_i`
(`eqprop_threshold_rule`): the notes' local threshold rule is a gradient step on the loss.

What is assumed: `E` and `C` are `C²` (true for the network's polynomial energy, proved in
`contDiff_energy`), and the equilibrium is nondegenerate.  What is not formalised: that the
nudged *dynamics* settle to `w(θ, β)` (convergence is the LaSalle step left open in P3).
-/

namespace Oscillator

open Filter Topology
open scoped ContDiff

section Abstract

variable {Θ W : Type*} [NormedAddCommGroup Θ] [NormedSpace ℝ Θ]
  [NormedAddCommGroup W] [NormedSpace ℝ W]

/-- The nudged energy `E(θ, v) + β C(v)`. -/
def nudged (E : Θ → W → ℝ) (C : W → ℝ) (θ : Θ) (β : ℝ) (v : W) : ℝ := E θ v + β * C v

omit [NormedAddCommGroup Θ] [NormedSpace ℝ Θ] [NormedAddCommGroup W] [NormedSpace ℝ W] in
theorem nudged_zero (E : Θ → W → ℝ) (C : W → ℝ) (θ : Θ) : nudged E C θ 0 = E θ := by
  funext v
  simp [nudged]

/-- The derivative of an uncurried function in its second argument. -/
theorem fderiv_uncurry_inr {E : Θ → W → ℝ} (hE : Differentiable ℝ (Function.uncurry E))
    (a : Θ) (b k : W) :
    fderiv ℝ (Function.uncurry E) (a, b) (0, k) = fderiv ℝ (E a) b k := by
  have h := (hE (a, b)).hasFDerivAt.comp b (hasFDerivAt_prodMk_right (𝕜 := ℝ) a b)
  have hfun : (Function.uncurry E ∘ fun v : W => (a, v)) = E a := by
    funext v
    rfl
  rw [hfun] at h
  rw [h.fderiv]
  simp

/-- The derivative of an uncurried function in its first argument. -/
theorem fderiv_uncurry_inl {E : Θ → W → ℝ} (hE : Differentiable ℝ (Function.uncurry E))
    (a : Θ) (b : W) (h : Θ) :
    fderiv ℝ (Function.uncurry E) (a, b) (h, 0) = fderiv ℝ (fun θ => E θ b) a h := by
  have h' := (hE (a, b)).hasFDerivAt.comp a (hasFDerivAt_prodMk_left (𝕜 := ℝ) a b)
  have hfun : (Function.uncurry E ∘ fun θ : Θ => (θ, b)) = fun θ => E θ b := by
    funext θ
    rfl
  rw [hfun] at h'
  rw [h'.fderiv]
  simp

/-- **The equilibrium-propagation identity.**  Hypotheses: `E` and `C` are `C²`; `w` is a `C¹`
family near `(θ₀, 0)` of critical points of the nudged energy `E(θ, ·) + β C`.  Conclusion: for
every parameter direction `h`, the function `β ↦ ∂_θ E(θ₀, w(θ₀, β)) · h` is differentiable at
`β = 0` with derivative `d/dθ [C(w(θ, 0))] · h`, the gradient of the loss. -/
theorem eqprop_identity {E : Θ → W → ℝ} {C : W → ℝ} {w : Θ × ℝ → W} {θ₀ : Θ}
    (hE : ContDiff ℝ 2 (Function.uncurry E)) (hC : ContDiff ℝ 2 C)
    (hw : ContDiffAt ℝ 1 w (θ₀, 0))
    (heq : ∀ᶠ p in 𝓝 (θ₀, 0), fderiv ℝ (nudged E C p.1 p.2) (w p) = 0) (h : Θ) :
    HasDerivAt (fun β => fderiv ℝ (Function.uncurry E) (θ₀, w (θ₀, β)) (h, 0))
      (fderiv ℝ (fun θ => C (w (θ, 0))) θ₀ h) 0 := by
  have hEd : Differentiable ℝ (Function.uncurry E) := hE.differentiable (by norm_num)
  have hCd : Differentiable ℝ C := hC.differentiable (by norm_num)
  have hE1 : ∀ a, Differentiable ℝ (E a) := fun a =>
    hEd.comp ((differentiable_const a).prodMk differentiable_id)
  -- the total function and its derivative
  let Φ : Θ × ℝ → ℝ := fun p => nudged E C p.1 p.2 (w p)
  let Φ' : Θ × ℝ → (Θ × ℝ →L[ℝ] ℝ) := fun p =>
    (fderiv ℝ (Function.uncurry E) (p.1, w p)).comp
        ((ContinuousLinearMap.inl ℝ Θ W).comp (ContinuousLinearMap.fst ℝ Θ ℝ))
      + C (w p) • ContinuousLinearMap.snd ℝ Θ ℝ
  -- Step 1: `Φ'` is the derivative of `Φ` near the base point (the envelope theorem).
  have hΦ : ∀ᶠ p in 𝓝 (θ₀, 0), HasFDerivAt Φ (Φ' p) p := by
    filter_upwards [heq, hw.eventually (by simp)] with p hp hwp
    have hwd : HasFDerivAt w (fderiv ℝ w p) p := (hwp.differentiableAt (by norm_num)).hasFDerivAt
    have hg : HasFDerivAt (fun p : Θ × ℝ => (p.1, w p))
        ((ContinuousLinearMap.fst ℝ Θ ℝ).prod (fderiv ℝ w p)) p :=
      hasFDerivAt_fst.prodMk hwd
    have h1' := (hEd (p.1, w p)).hasFDerivAt.comp p hg
    have h1 : HasFDerivAt (fun p : Θ × ℝ => Function.uncurry E (p.1, w p))
        ((fderiv ℝ (Function.uncurry E) (p.1, w p)).comp
          ((ContinuousLinearMap.fst ℝ Θ ℝ).prod (fderiv ℝ w p))) p := h1'
    have h2 : HasFDerivAt (fun p : Θ × ℝ => p.2 * C (w p))
        (p.2 • ((fderiv ℝ C (w p)).comp (fderiv ℝ w p))
          + C (w p) • ContinuousLinearMap.snd ℝ Θ ℝ) p :=
      hasFDerivAt_snd.mul ((hCd (w p)).hasFDerivAt.comp p hwd)
    have hp' : ∀ k, fderiv ℝ (Function.uncurry E) (p.1, w p) (0, k)
        + p.2 * fderiv ℝ C (w p) k = 0 := by
      intro k
      have e1 : fderiv ℝ (nudged E C p.1 p.2) (w p)
          = fderiv ℝ (E p.1) (w p) + p.2 • fderiv ℝ C (w p) :=
        ((hE1 p.1 (w p)).hasFDerivAt.add ((hCd (w p)).hasFDerivAt.const_mul p.2)).fderiv
      have e2 := congrArg (fun L : W →L[ℝ] ℝ => L k) (e1.symm.trans hp)
      simp only [_root_.add_apply, _root_.smul_apply, smul_eq_mul, _root_.zero_apply] at e2
      rw [fderiv_uncurry_inr hEd]
      exact e2
    refine (h1.fun_add h2).congr_fderiv ?_
    refine ContinuousLinearMap.ext fun δ => ?_
    simp only [Φ', _root_.add_apply, ContinuousLinearMap.comp_apply,
      ContinuousLinearMap.prod_apply, _root_.smul_apply, smul_eq_mul,
      ContinuousLinearMap.coe_fst', ContinuousLinearMap.coe_snd', ContinuousLinearMap.inl_apply]
    have hsplit : ((δ.1, fderiv ℝ w p δ) : Θ × W) = (δ.1, (0 : W)) + ((0 : Θ), fderiv ℝ w p δ) := by
      simp
    rw [hsplit, map_add]
    linarith [hp' (fderiv ℝ w p δ)]
  -- Step 2: `Φ'` is `C¹`.
  have hΦ'1 : ContDiffAt ℝ 1 Φ' (θ₀, 0) := by
    have hA : ContDiffAt ℝ 1 (fun p : Θ × ℝ => fderiv ℝ (Function.uncurry E) (p.1, w p)) (θ₀, 0) :=
      (hE.fderiv_right (m := 1) (by norm_num)).contDiffAt.comp (θ₀, 0) (contDiffAt_fst.prodMk hw)
    have hB : ContDiffAt ℝ 1 (fun p : Θ × ℝ => C (w p)) (θ₀, 0) :=
      (hC.of_le (by norm_num)).contDiffAt.comp (θ₀, 0) hw
    exact (hA.clm_comp contDiffAt_const).add (hB.smul contDiffAt_const)
  -- Step 3: hence `Φ` is `C²`.
  have hΦ2 : ContDiffAt ℝ 2 Φ (θ₀, 0) := by
    have h2 : (2 : ℕ∞ω) = ((1 : ℕ) : ℕ∞ω) + 1 := by norm_num
    rw [h2, contDiffAt_succ_iff_hasFDerivAt]
    exact ⟨Φ', eventually_iff_exists_mem.1 hΦ, hΦ'1⟩
  -- Step 4: symmetry of the second derivative, transported to `Φ'`.
  have hsymm := hΦ2.isSymmSndFDerivAt (by simp)
  have hfd : fderiv ℝ Φ =ᶠ[𝓝 (θ₀, 0)] Φ' := hΦ.mono fun p hp => hp.fderiv
  have key : fderiv ℝ Φ' (θ₀, 0) (0, 1) (h, 0) = fderiv ℝ Φ' (θ₀, 0) (h, 0) (0, 1) := by
    have := hsymm (0, 1) (h, 0)
    rwa [hfd.fderiv_eq] at this
  -- Step 5: identify the two mixed derivatives.
  have hΦ'd : DifferentiableAt ℝ Φ' (θ₀, 0) := hΦ'1.differentiableAt (by norm_num)
  have hA : fderiv ℝ (fun p => Φ' p (h, 0)) (θ₀, 0) (0, 1)
      = fderiv ℝ Φ' (θ₀, 0) (0, 1) (h, 0) := by
    rw [fderiv_clm_apply hΦ'd (differentiableAt_const _)]
    simp
  have hB : fderiv ℝ (fun p => Φ' p (0, 1)) (θ₀, 0) (h, 0)
      = fderiv ℝ Φ' (θ₀, 0) (h, 0) (0, 1) := by
    rw [fderiv_clm_apply hΦ'd (differentiableAt_const _)]
    simp
  have hBfun : (fun p : Θ × ℝ => Φ' p (0, 1)) = fun p => C (w p) := by
    funext p
    simp [Φ', Prod.mk_zero_zero]
  have hL : fderiv ℝ (fun θ => C (w (θ, 0))) θ₀ h
      = fderiv ℝ (fun p : Θ × ℝ => C (w p)) (θ₀, 0) (h, 0) := by
    have hCw : DifferentiableAt ℝ (fun p : Θ × ℝ => C (w p)) (θ₀, 0) :=
      (hCd _).comp _ (hw.differentiableAt (by norm_num))
    have h' := hCw.hasFDerivAt.comp θ₀ (hasFDerivAt_prodMk_left (𝕜 := ℝ) θ₀ (0 : ℝ))
    have this : HasFDerivAt (fun θ => C (w (θ, 0)))
        ((fderiv ℝ (fun p : Θ × ℝ => C (w p)) (θ₀, 0)).comp (ContinuousLinearMap.inl ℝ Θ ℝ)) θ₀ :=
      h'
    rw [this.fderiv]
    simp
  have hAfun : ∀ β, Φ' (θ₀, β) (h, 0) = fderiv ℝ (Function.uncurry E) (θ₀, w (θ₀, β)) (h, 0) := by
    intro β
    simp [Φ']
  -- Step 6: the derivative in `β` of the contrast function.
  have hder : HasDerivAt (fun β => Φ' (θ₀, β) (h, 0))
      (fderiv ℝ (fun p => Φ' p (h, 0)) (θ₀, 0) (0, 1)) 0 := by
    have hd : DifferentiableAt ℝ (fun p => Φ' p (h, 0)) (θ₀, 0) :=
      hΦ'd.clm_apply (differentiableAt_const _)
    have h' := hd.hasFDerivAt.comp_hasDerivAt (0 : ℝ)
      ((hasFDerivAt_prodMk_right (𝕜 := ℝ) θ₀ (0 : ℝ)).hasDerivAt)
    have e : (ContinuousLinearMap.inr ℝ Θ ℝ) (1 : ℝ) = ((0 : Θ), (1 : ℝ)) := by simp
    rw [e] at h'
    exact h'
  have hval : fderiv ℝ (fun p => Φ' p (h, 0)) (θ₀, 0) (0, 1)
      = fderiv ℝ (fun θ => C (w (θ, 0))) θ₀ h := by
    rw [hA, key, ← hB, hBfun, hL]
  rw [hval] at hder
  exact hder.congr_of_eventuallyEq (Eventually.of_forall fun β => (hAfun β).symm)

/-! ### Existence of the nudged equilibria: the implicit function theorem -/

variable [CompleteSpace Θ] [CompleteSpace W]

/-- The gradient of the nudged energy in the state, as a map of `((θ, β), v)`. -/
noncomputable def nudgedGrad (E : Θ → W → ℝ) (C : W → ℝ) : (Θ × ℝ) × W → (W →L[ℝ] ℝ) :=
  fun q => fderiv ℝ (nudged E C q.1.1 q.1.2) q.2

omit [CompleteSpace Θ] [CompleteSpace W] in
theorem contDiff_nudgedGrad {E : Θ → W → ℝ} {C : W → ℝ} (hE : ContDiff ℝ 2 (Function.uncurry E))
    (hC : ContDiff ℝ 2 C) : ContDiff ℝ 1 (nudgedGrad E C) := by
  have hf : ContDiff ℝ 2
      (Function.uncurry fun (q : (Θ × ℝ) × W) (v : W) => nudged E C q.1.1 q.1.2 v) := by
    have e : (Function.uncurry fun (q : (Θ × ℝ) × W) (v : W) => nudged E C q.1.1 q.1.2 v)
        = fun x : ((Θ × ℝ) × W) × W => Function.uncurry E (x.1.1.1, x.2) + x.1.1.2 * C x.2 := by
      funext x
      rfl
    rw [e]
    exact (hE.comp (contDiff_fst.fst.fst.prodMk contDiff_snd)).add
      (contDiff_fst.fst.snd.mul (hC.comp contDiff_snd))
  exact hf.fderiv contDiff_snd (by norm_num)

/-- **Existence of the nudged equilibria.**  Hypotheses: `E` and `C` are `C²`; `w₀` is a critical
point of `E(θ₀, ·)` whose Hessian is invertible (a nondegenerate equilibrium).  Conclusion: there is
a `C¹` family `w(θ, β)` near `(θ₀, 0)` with `w(θ₀, 0) = w₀` consisting of critical points of the
nudged energies `E(θ, ·) + β C`.  This is Mathlib's implicit function theorem
(`ContDiffAt.implicitFunction`). -/
theorem exists_nudged_equilibria {E : Θ → W → ℝ} {C : W → ℝ}
    (hE : ContDiff ℝ 2 (Function.uncurry E)) (hC : ContDiff ℝ 2 C) {θ₀ : Θ} {w₀ : W}
    (h0 : fderiv ℝ (E θ₀) w₀ = 0) (hH : (fderiv ℝ (fderiv ℝ (E θ₀)) w₀).IsInvertible) :
    ∃ w : Θ × ℝ → W, w (θ₀, 0) = w₀ ∧ ContDiffAt ℝ 1 w (θ₀, 0) ∧
      ∀ᶠ p in 𝓝 (θ₀, 0), fderiv ℝ (nudged E C p.1 p.2) (w p) = 0 := by
  have hf : ContDiffAt ℝ 1 (nudgedGrad E C) ((θ₀, 0), w₀) := (contDiff_nudgedGrad hE hC).contDiffAt
  have hslice : (fun v => nudgedGrad E C ((θ₀, 0), v)) = fderiv ℝ (E θ₀) := by
    funext v
    simp only [nudgedGrad, nudged_zero]
  have hinv : (fderiv ℝ (nudgedGrad E C) ((θ₀, 0), w₀)
      ∘L ContinuousLinearMap.inr ℝ (Θ × ℝ) W).IsInvertible := by
    have h' := (hf.differentiableAt (by norm_num)).hasFDerivAt.comp w₀
      (hasFDerivAt_prodMk_right (𝕜 := ℝ) (θ₀, 0) w₀)
    have h'' : HasFDerivAt (fun v => nudgedGrad E C ((θ₀, 0), v))
        (fderiv ℝ (nudgedGrad E C) ((θ₀, 0), w₀) ∘L ContinuousLinearMap.inr ℝ (Θ × ℝ) W) w₀ := h'
    rw [← h''.fderiv, hslice]
    exact hH
  have hone : (1 : ℕ∞ω) ≠ 0 := one_ne_zero
  refine ⟨hf.implicitFunction hone hinv, hf.implicitFunction_apply_self hone hinv,
    hf.contDiffAt_implicitFunction hone hinv, ?_⟩
  have h0' : nudgedGrad E C ((θ₀, 0), w₀) = 0 := by
    simp only [nudgedGrad, nudged_zero]
    exact h0
  filter_upwards [hf.eventually_apply_implicitFunction hone hinv] with p hp
  rw [h0'] at hp
  exact hp

/-- **Equilibrium propagation at a nondegenerate equilibrium.**  Hypotheses: `E`, `C` are `C²`;
`w₀` is a nondegenerate critical point of `E(θ₀, ·)`.  Conclusion: there is a `C¹` family of nudged
equilibria `w(θ, β)` through `w₀`, and for every parameter direction `h` the contrast function
`β ↦ ∂_θ E(θ₀, w(θ₀, β)) · h` has derivative at `0` equal to the gradient of the loss
`θ ↦ C(w(θ, 0))` in direction `h`. -/
theorem eqprop_of_nondegenerate {E : Θ → W → ℝ} {C : W → ℝ}
    (hE : ContDiff ℝ 2 (Function.uncurry E)) (hC : ContDiff ℝ 2 C) {θ₀ : Θ} {w₀ : W}
    (h0 : fderiv ℝ (E θ₀) w₀ = 0) (hH : (fderiv ℝ (fderiv ℝ (E θ₀)) w₀).IsInvertible) :
    ∃ w : Θ × ℝ → W, w (θ₀, 0) = w₀ ∧ ContDiffAt ℝ 1 w (θ₀, 0) ∧
      (∀ᶠ p in 𝓝 (θ₀, 0), fderiv ℝ (nudged E C p.1 p.2) (w p) = 0) ∧
      ∀ h : Θ, HasDerivAt (fun β => fderiv ℝ (Function.uncurry E) (θ₀, w (θ₀, β)) (h, 0))
        (fderiv ℝ (fun θ => C (w (θ, 0))) θ₀ h) 0 := by
  obtain ⟨w, hw0, hw1, heq⟩ := exists_nudged_equilibria (C := C) hE hC h0 hH
  exact ⟨w, hw0, hw1, heq, fun h => eqprop_identity hE hC hw1 heq h⟩

end Abstract

/-! ### The network: the threshold rule is a gradient step -/

section Network

open Complex
open scoped ComplexConjugate

@[fun_prop] theorem contDiff_complex_re {n : ℕ∞ω} : ContDiff ℝ n Complex.re :=
  Complex.reCLM.contDiff

@[fun_prop] theorem contDiff_complex_im {n : ℕ∞ω} : ContDiff ℝ n Complex.im :=
  Complex.imCLM.contDiff

@[fun_prop] theorem contDiff_complex_conj {n : ℕ∞ω} : ContDiff ℝ n (starRingEnd ℂ) :=
  Complex.conjCLE.contDiff

@[fun_prop] theorem contDiff_complex_normSq {n : ℕ∞ω} : ContDiff ℝ n Complex.normSq := by
  have e : (Complex.normSq : ℂ → ℝ) = fun z => z.re * z.re + z.im * z.im := by
    funext z
    exact Complex.normSq_apply z
  rw [e]
  fun_prop

variable {ι : Type*} [Fintype ι] {K : Matrix ι ι ℂ} {u : ι → ℂ}

/-- The energy is smooth in the thresholds and the state together. -/
theorem contDiff_energy : ContDiff ℝ 2 (Function.uncurry fun μ : ι → ℝ => energy μ K u) := by
  have e : (Function.uncurry fun μ : ι → ℝ => energy μ K u)
      = fun p : (ι → ℝ) × (ι → ℂ) => energy p.1 K u p.2 := rfl
  rw [e]
  unfold energy
  fun_prop

/-- **The threshold rule is a gradient step on the loss.**  Hypotheses: `K` Hermitian; a `C²` cost
`C`; `w₀` an equilibrium of the free network whose energy Hessian is invertible.  Conclusion: there
is a `C¹` family `w(μ, β)` of nudged equilibria through `w₀` (critical points of
`energy μ K u + β C`), and for every unit `i` the amplitude-contrast function
`β ↦ −|w(μ₀, β)_i|²` has derivative at `β = 0` equal to `∂L/∂μ_i`, the gradient of the loss
`L(μ) = C(w(μ, 0))`.  So `(|w_i^β|² − |w_i^0|²)/β → −∂L/∂μ_i` as `β → 0`: the notes' rule
`Δμ_i ∝ (|w_i^β|² − |w_i^0|²)/β` is a descent step on the loss, computed from unit `i` alone. -/
theorem eqprop_threshold_rule [DecidableEq ι] (hK : K.IsHermitian) {C : (ι → ℂ) → ℝ}
    (hC : ContDiff ℝ 2 C)
    {μ₀ : ι → ℝ} {w₀ : ι → ℂ} (h0 : IsEquilibrium μ₀ K u w₀)
    (hH : (fderiv ℝ (fderiv ℝ (energy μ₀ K u)) w₀).IsInvertible) :
    ∃ w : (ι → ℝ) × ℝ → (ι → ℂ), w (μ₀, 0) = w₀ ∧ ContDiffAt ℝ 1 w (μ₀, 0) ∧
      (∀ᶠ p in 𝓝 (μ₀, 0), fderiv ℝ (fun v => energy p.1 K u v + p.2 * C v) (w p) = 0) ∧
      ∀ i, HasDerivAt (fun β => -normSq (w (μ₀, β) i))
        (fderiv ℝ (fun μ => C (w (μ, 0))) μ₀ (Pi.single i 1)) 0 := by
  classical
  have h0' : fderiv ℝ (energy μ₀ K u) w₀ = 0 := (fderiv_energy_eq_zero_iff hK w₀).2 h0
  obtain ⟨w, hw0, hw1, heq, hder⟩ :=
    eqprop_of_nondegenerate (E := fun μ => energy μ K u) (C := C) contDiff_energy hC h0' hH
  refine ⟨w, hw0, hw1, heq, fun i => ?_⟩
  refine (hder (Pi.single i 1)).congr_of_eventuallyEq (Eventually.of_forall fun β => ?_)
  show -normSq (w (μ₀, β) i)
    = fderiv ℝ (Function.uncurry fun μ : ι → ℝ => energy μ K u) (μ₀, w (μ₀, β)) (Pi.single i 1, 0)
  rw [fderiv_uncurry_inl (contDiff_energy.differentiable (by norm_num))]
  have e : fderiv ℝ (fun θ : ι → ℝ => (fun μ : ι → ℝ => energy μ K u) θ (w (μ₀, β))) μ₀
      = fderiv ℝ (fun θ : ι → ℝ => energy θ K u (w (μ₀, β))) μ₀ := rfl
  rw [e, fderiv_energy_mu_apply]
  simp [Pi.single_apply]

end Network

end Oscillator
