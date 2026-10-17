import Oscillator.Energy

/-!
# P2: the isolated unit

P2 is a definition (the threshold is the bifurcation parameter `μ`) plus the equilibria of the
isolated Stuart–Landau unit `ẇ = μ w − |w|² w + F` in its three regimes.  Taking `ι = Unit`,
`K = 0` and a constant input `F`:

* `μ > 0`, `F = 0`: the circle `|w|² = μ` consists of equilibria in the rotating frame, the limit
  cycle of radius `√μ` (`isEquilibrium_of_normSq_eq`), and the only other equilibrium is the origin
  (`isolated_equilibrium_iff_of_no_input`);
* `μ < 0`, `F = 0`: the origin is the only equilibrium (`eq_zero_of_isEquilibrium_of_neg`);
* `μ = 0`, real forcing `F > 0`: every equilibrium has amplitude `|w| = F^{1/3}`
  (`norm_eq_rpow_third_of_isEquilibrium`), the compressive nonlinearity of the hearing literature.

Stability of these equilibria and the linear gain `1/|μ|` are not formalised.
-/

namespace Oscillator

open Complex

/-- The vector field of the isolated unit. -/
theorem field_unit (μ : ℝ) (F : ℂ) (w : Unit → ℂ) :
    field (fun _ : Unit => μ) 0 (fun _ => F) w () = (μ : ℂ) * w () - (normSq (w ()) : ℂ) * w () + F := by
  simp [field]

theorem isolated_equilibrium_iff (μ : ℝ) (F : ℂ) (z : ℂ) :
    IsEquilibrium (fun _ : Unit => μ) 0 (fun _ => F) (fun _ => z)
      ↔ (μ : ℂ) * z - (normSq z : ℂ) * z + F = 0 := by
  constructor
  · intro h
    have := congrFun h ()
    rwa [field_unit] at this
  · intro h
    funext i
    cases i
    rw [field_unit]
    exact h

/-- **The limit cycle.**  For `μ > 0` and no input, every state on the circle `|w|² = μ` is an
equilibrium in the rotating frame. -/
theorem isEquilibrium_of_normSq_eq {μ : ℝ} {z : ℂ} (hz : normSq z = μ) :
    IsEquilibrium (fun _ : Unit => μ) 0 (fun _ => 0) (fun _ => z) := by
  rw [isolated_equilibrium_iff, hz]
  ring

/-- Without input the equilibria of the isolated unit are the origin and the circle `|w|² = μ`. -/
theorem isolated_equilibrium_iff_of_no_input (μ : ℝ) (z : ℂ) :
    IsEquilibrium (fun _ : Unit => μ) 0 (fun _ => 0) (fun _ => z) ↔ z = 0 ∨ normSq z = μ := by
  rw [isolated_equilibrium_iff, add_zero]
  constructor
  · intro h
    have h' : ((μ : ℂ) - normSq z) * z = 0 := by rw [← h]; ring
    rcases mul_eq_zero.1 h' with h1 | h1
    · right
      have := sub_eq_zero.1 h1
      exact_mod_cast this.symm
    · left
      exact h1
  · rintro (h | h)
    · simp [h]
    · rw [h]
      ring

/-- **The stable origin.**  For `μ < 0` and no input the origin is the only equilibrium. -/
theorem eq_zero_of_isEquilibrium_of_neg {μ : ℝ} (hμ : μ < 0) {z : ℂ}
    (h : IsEquilibrium (fun _ : Unit => μ) 0 (fun _ => 0) (fun _ => z)) : z = 0 := by
  rcases (isolated_equilibrium_iff_of_no_input μ z).1 h with h | h
  · exact h
  · have := normSq_nonneg z
    linarith

/-- **The compressive nonlinearity.**  At `μ = 0` with a real forcing `F > 0`, every equilibrium
has amplitude `|w| = F^{1/3}`. -/
theorem norm_eq_rpow_third_of_isEquilibrium {F : ℝ} (hF : 0 < F) {z : ℂ}
    (h : IsEquilibrium (fun _ : Unit => 0) 0 (fun _ => (F : ℂ)) (fun _ => z)) :
    ‖z‖ = F ^ ((3 : ℕ) : ℝ)⁻¹ := by
  rw [isolated_equilibrium_iff] at h
  have h1 : (normSq z : ℂ) * z = F := by
    have : (normSq z : ℂ) * z = (0 : ℝ) * z - ((0 : ℝ) * z - (normSq z : ℂ) * z + F) + F := by
      ring
    rw [this, h]
    simp
  have h2 : ‖z‖ ^ 3 = F := by
    have := congrArg norm h1
    rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (normSq_nonneg z),
      normSq_eq_norm_sq, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hF] at this
    rw [← this]
    ring
  rw [← h2, Real.pow_rpow_inv_natCast (norm_nonneg z) (by norm_num)]

end Oscillator
