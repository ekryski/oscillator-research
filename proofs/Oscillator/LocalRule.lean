import Oscillator.Energy

/-!
# The local rule: derivatives of the energy in the parameters

The energy is affine in the thresholds `μ` and real-linear in the coupling `K`, so its
parameter derivatives are exact and local:

* `∂E/∂μ_i = −|w_i|²` (`hasDerivAt_energy_mu_coord`, `deriv_energy_mu_coord`), and in direction
  `δμ`: `dE[δμ] = −∑ i δμ_i |w_i|²` (`hasFDerivAt_energy_mu`);
* in direction `δK`: `dE[δK] = −∑ i j Re(conj(w_i) δK_ij w_j)` (`hasFDerivAt_energy_K`); along the
  single entry `K_ij ↦ K_ij + c` this is `−Re(conj(w_i) w_j c)` (`fderiv_energy_K_single`), so the
  real gradient of `E` with respect to the complex entry `K_ij` is `−w_i conj(w_j) =
  −conj(conj(w_i) w_j)`; along the notes' symmetric direction `K_ij = K_ji = κ` it is
  `−2 Re(conj(w_i) w_j)` (`fderiv_energy_K_symm`).

Equilibrium propagation updates each parameter by the contrast of its energy derivative between
the nudged and the free equilibrium, divided by the nudge `β`.  With the derivatives above that
contrast is the amplitude contrast for the threshold (`mu_contrast`) and the coherence contrast
for the coupling (`K_contrast`); both are local to the units involved.  The identity that makes
this contrast the gradient of the loss is `Oscillator.EqProp`.
-/

namespace Oscillator

open Complex Matrix
open scoped ComplexConjugate

variable {ι : Type*} [Fintype ι]
variable {μ : ι → ℝ} {K : Matrix ι ι ℂ} {u w : ι → ℂ}

/-! ### Derivative in the thresholds `μ` -/

/-- The derivative of `E` in the thresholds `μ`: `dE[δμ] = −∑ i δμ_i |w_i|²`. -/
theorem hasFDerivAt_energy_mu (K : Matrix ι ι ℂ) (u w : ι → ℂ) (μ : ι → ℝ) :
    HasFDerivAt (fun μ : ι → ℝ => energy μ K u w)
      (-∑ i, normSq (w i) • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℝ) i) μ := by
  have h1 : HasFDerivAt
      (fun μ : ι → ℝ => ∑ i, (-μ i * normSq (w i) + (1 / 2 : ℝ) * normSq (w i) ^ 2))
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

theorem fderiv_energy_mu_apply (δ : ι → ℝ) :
    fderiv ℝ (fun μ : ι → ℝ => energy μ K u w) μ δ = -∑ i, δ i * normSq (w i) := by
  rw [(hasFDerivAt_energy_mu K u w μ).fderiv]
  simp [mul_comm]

/-- **The local threshold rule.**  `∂E/∂μ_i = −|w_i|²`, as the derivative of the energy in the
single parameter `μ_i` (all other parameters and the state held fixed). -/
theorem hasDerivAt_energy_mu_coord [DecidableEq ι] (i : ι) (t : ℝ) :
    HasDerivAt (fun t : ℝ => energy (Function.update μ i t) K u w) (-normSq (w i)) t := by
  have h := (hasFDerivAt_energy_mu K u w (Function.update μ i t)).comp_hasDerivAt t
    (hasDerivAt_update μ i t)
  have e : (-∑ j, normSq (w j) • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℝ) j)
      (Pi.single i (1 : ℝ)) = -normSq (w i) := by
    simp [Pi.single_apply]
  rw [e] at h
  exact h

theorem deriv_energy_mu_coord [DecidableEq ι] (i : ι) (t : ℝ) :
    deriv (fun t : ℝ => energy (Function.update μ i t) K u w) t = -normSq (w i) :=
  (hasDerivAt_energy_mu_coord i t).deriv

/-- The equilibrium-propagation contrast for the threshold `μ_i`: the difference of `∂E/∂μ_i`
between a nudged state `wβ` and a free state `w0`, divided by the nudge `β`, is the amplitude
contrast `−(|wβ_i|² − |w0_i|²)/β`.  It involves unit `i` alone. -/
theorem mu_contrast [DecidableEq ι] (i : ι) (β : ℝ) (wβ w0 : ι → ℂ) :
    (deriv (fun t : ℝ => energy (Function.update μ i t) K u wβ) (μ i)
      - deriv (fun t : ℝ => energy (Function.update μ i t) K u w0) (μ i)) / β
      = -(normSq (wβ i) - normSq (w0 i)) / β := by
  rw [deriv_energy_mu_coord, deriv_energy_mu_coord]
  ring

/-! ### Derivative in the coupling `K` -/

/-- The linear functional `δK ↦ −∑ i j Re(conj(w_i) δK_ij w_j)` on `ι → ι → ℂ`. -/
noncomputable def couplingGradient (w : ι → ℂ) : (ι → ι → ℂ) →L[ℝ] ℝ :=
  -∑ i, ∑ j, Complex.reCLM.comp ((conj (w i) * w j) •
    ((ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) j).comp
      (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ι → ℂ) i)))

theorem couplingGradient_apply (w : ι → ℂ) (δ : ι → ι → ℂ) :
    couplingGradient w δ = -∑ i, ∑ j, (conj (w i) * δ i j * w j).re := by
  simp only [couplingGradient, _root_.neg_apply, _root_.sum_apply,
    ContinuousLinearMap.comp_apply, _root_.smul_apply, ContinuousLinearMap.proj_apply,
    Complex.reCLM_apply, smul_eq_mul]
  congr 1
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  ring_nf

/-- The derivative of `E` in the coupling: `dE[δK] = −∑ i j Re(conj(w_i) δK_ij w_j)`.  The
coupling is taken as a function `ι → ι → ℂ` (the matrix `Matrix.of K`) so that the sup norm on
`ι → ι → ℂ` provides the normed structure. -/
theorem hasFDerivAt_energy_K (μ : ι → ℝ) (u w : ι → ℂ) (K : ι → ι → ℂ) :
    HasFDerivAt (fun K : ι → ι → ℂ => energy μ (Matrix.of K) u w) (couplingGradient w) K := by
  have h1 : HasFDerivAt
      (fun _ : ι → ι → ℂ => ∑ i, (-μ i * normSq (w i) + (1 / 2 : ℝ) * normSq (w i) ^ 2))
      (0 : (ι → ι → ℂ) →L[ℝ] ℝ) K := hasFDerivAt_const _ _
  have h2 : HasFDerivAt (fun K : ι → ι → ℂ => ∑ i, ∑ j, (conj (w i) * Matrix.of K i j * w j).re)
      (∑ i, ∑ j, Complex.reCLM.comp ((conj (w i) * w j) •
        ((ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) j).comp
          (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ι → ℂ) i)))) K := by
    refine HasFDerivAt.fun_sum fun i _ => HasFDerivAt.fun_sum fun j _ => ?_
    have hij : HasFDerivAt (fun K : ι → ι → ℂ => K i j)
        ((ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) j).comp
          (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ι → ℂ) i)) K :=
      ((ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ℂ) j).comp
          (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => ι → ℂ) i)).hasFDerivAt
    have h := reCLM.hasFDerivAt.comp K ((hij.const_mul (conj (w i))).mul_const (w j))
    refine h.congr_fderiv ?_
    ext δ
    simp
    ring
  have h3 : HasFDerivAt (fun _ : ι → ι → ℂ => 2 * ∑ i, (conj (u i) * w i).re)
      (0 : (ι → ι → ℂ) →L[ℝ] ℝ) K := hasFDerivAt_const _ _
  refine ((h1.fun_sub h2).fun_sub h3).congr_fderiv ?_
  simp [couplingGradient]

theorem fderiv_energy_K_apply (K : ι → ι → ℂ) (δ : ι → ι → ℂ) :
    fderiv ℝ (fun K : ι → ι → ℂ => energy μ (Matrix.of K) u w) K δ
      = -∑ i, ∑ j, (conj (w i) * δ i j * w j).re := by
  rw [(hasFDerivAt_energy_K μ u w K).fderiv, couplingGradient_apply]

/-- **The local coupling rule.**  Along the single entry direction `K_ij ↦ K_ij + c`,
`dE = −Re(conj(w_i) w_j c)`.  Read as a real gradient with respect to the complex entry `K_ij`
(pairing `⟨a, c⟩_ℝ = Re(conj(a) c)`), this says `∂E/∂K_ij = −w_i conj(w_j) = −conj(conj(w_i) w_j)`;
it involves units `i` and `j` alone. -/
theorem fderiv_energy_K_single [DecidableEq ι] (K : ι → ι → ℂ) (i j : ι) (c : ℂ) :
    fderiv ℝ (fun K : ι → ι → ℂ => energy μ (Matrix.of K) u w) K (Pi.single i (Pi.single j c))
      = -(conj (w i) * w j * c).re := by
  rw [fderiv_energy_K_apply]
  rw [Finset.sum_eq_single i (fun b _ hb => by simp [hb]) (fun h => absurd (Finset.mem_univ i) h)]
  rw [Finset.sum_eq_single j (fun b _ hb => by simp [hb]) (fun h => absurd (Finset.mem_univ j) h)]
  simp
  ring

/-- The notes' real symmetric parameterization: along the direction `K_ij = K_ji = κ`
(moving both entries together), `dE/dκ = −2 Re(conj(w_i) w_j)`. -/
theorem fderiv_energy_K_symm [DecidableEq ι] (K : ι → ι → ℂ) (i j : ι) :
    fderiv ℝ (fun K : ι → ι → ℂ => energy μ (Matrix.of K) u w) K
      (Pi.single i (Pi.single j 1) + Pi.single j (Pi.single i 1))
      = -2 * (conj (w i) * w j).re := by
  rw [map_add, fderiv_energy_K_single, fderiv_energy_K_single]
  simp only [mul_one, Complex.mul_re, Complex.conj_re, Complex.conj_im]
  ring

/-- The equilibrium-propagation contrast for the coupling `K_ij`, along the symmetric direction:
the coherence contrast `−2 (Re(conj(wβ_i) wβ_j) − Re(conj(w0_i) w0_j))/β`. -/
theorem K_contrast [DecidableEq ι] (K : ι → ι → ℂ) (i j : ι) (β : ℝ) (wβ w0 : ι → ℂ) :
    (fderiv ℝ (fun K : ι → ι → ℂ => energy μ (Matrix.of K) u wβ) K
        (Pi.single i (Pi.single j 1) + Pi.single j (Pi.single i 1))
      - fderiv ℝ (fun K : ι → ι → ℂ => energy μ (Matrix.of K) u w0) K
        (Pi.single i (Pi.single j 1) + Pi.single j (Pi.single i 1))) / β
      = -2 * ((conj (wβ i) * wβ j).re - (conj (w0 i) * w0 j).re) / β := by
  rw [fderiv_energy_K_symm, fderiv_energy_K_symm]
  ring

end Oscillator
