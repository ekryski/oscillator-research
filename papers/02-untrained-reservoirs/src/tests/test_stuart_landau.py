""" Stuart-Landau core contracts: cubic solver, phase-reduction match,
amplitude boundedness, semi-implicit drive ordering, physics-only invariant."""

import math
import sys

import torch

sys.path.insert(0, ".")
from harness import OscillatorField
from harness.models.stuart_landau import SLCore, implicit_amplitude_root

TWO_PI = 2 * math.pi


def test_cardano_root_solves_the_cubic():
    gen = torch.Generator().manual_seed(0)
    r_old = torch.rand(64, generator=gen) * 2
    dt_beta = torch.rand(64, generator=gen) * 0.2 + 0.01
    one_minus = 1 - torch.rand(64, generator=gen) * 0.05
    r = implicit_amplitude_root(r_old, dt_beta, one_minus)
    assert (r > 0).all()
    residual = dt_beta * r ** 3 + one_minus * r - r_old
    assert residual.abs().max() < 1e-4


def test_fixedamp_sl_converges_to_phase_core_as_dt_shrinks():
    # Arm B (amplitude frozen at 1) is the phase core's equal-amplitude slice
    # in CONTINUOUS time; the discrete maps differ because 's recipe
    # integrates the omega rotation exactly (z * e^{i dt w}) while the phase
    # core folds omega into a single Euler line — operator splitting with
    # LOCAL O(dt^2), GLOBAL O(dt) error. Measured: gap halves with dt
    # (0.051 / 0.024 / 0.009 / 0.005 at dt 0.1/0.05/0.02/0.01). Contract:
    # ~linear shrinkage at fixed physical time per frame.
    def gap(dt, substeps):
        rows = torch.randn(2, 3, 16, generator=torch.Generator().manual_seed(11)) * 0.3
        torch.manual_seed(7)
        phase = OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0,
                         dt=dt, substeps=substeps)
        torch.manual_seed(7)
        sl = OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0,
                      dt=dt, substeps=substeps, core="sl-fixedamp")
        drv = (rows * phase.gain).view(2, 3, 1, 16, 1).expand(2, 3, 2, 16, 16)
        f_phase, _ = phase.core.forward_scan(drv)
        f_sl, _ = sl.core.forward_scan(drv)
        return (f_phase - f_sl).abs().max().item()

    coarse, fine = gap(0.1, 1), gap(0.01, 10)  # same physical time per frame
    assert fine < 0.01, fine  # near-identical at fine dt (measured 0.0046)
    assert coarse / fine > 6, (coarse, fine)  # ~dt shrinkage (measured 11x)


def test_sl_amplitude_stays_bounded_and_is_used():
    torch.manual_seed(0)
    core = SLCore(channels=2, grid=8, damping=0.1)
    state = core.init_state(1)
    state = torch.stack((state[:, 0] * 1.5, state[:, 1] * 0.4), dim=1)  # perturb radius
    drives = torch.sin(TWO_PI * 0.03 * torch.arange(100)).view(1, 100, 1, 1, 1).expand(1, 100, 2, 8, 8)
    feats, final = core.forward_scan(drives, state)
    assert torch.isfinite(feats).all()
    r = torch.sqrt(final[:, 0] ** 2 + final[:, 1] ** 2)
    assert 0.2 < r.min() and r.max() < 3.0  # bounded by the dynamics, no clamp needed
    # amplitude actually moves under drive (arm C's mechanism is live)
    d = 2 * 8 * 8
    x, y = feats[..., d:].view(1, 100, 2, 8, 8), feats[..., :d].view(1, 100, 2, 8, 8)
    r_t = torch.sqrt(x ** 2 + y ** 2)
    assert r_t.std(dim=1).mean() > 1e-3


def test_drive_ordering_is_semi_implicit():
    # Pins the intended sequential update (SLCore docstring, "Discretization
    # ordering"): the y line reads the ALREADY-UPDATED x. With coupling,
    # pinning, and omega zeroed the substep has a closed form —
    #   x1 = x0 − h·y0,  y1 = y0 + h·x1   (h = dt·drive)
    # whereas substep-start (Jacobi) ordering would give y0 + h·x0. The two
    # differ by h²·y0 ≈ 1e-2 here, far above float noise, so this fails loudly
    # if the ordering is ever "fixed" — which would also break the TS-port
    # parity fixtures (viz/src/physics/sl.ts mirrors this ordering exactly).
    dt, d = 0.1, 1.0
    core = SLCore(channels=1, grid=4, dt=dt, damping=0.0, amplitude_frozen=True)
    with torch.no_grad():
        core.kernel.zero_()
        core.natural_freqs.zero_()
    state = core.init_state(1)
    drive = torch.full((1, 1, 4, 4), d)
    new_state, _ = core.step_frame(state, drive, core.prepare_coupling())

    x0, y0 = state[:, 0], state[:, 1]
    h = dt * d
    x1 = x0 - h * y0
    y_semi = y0 + h * x1  # ordering under test: uses the new x
    y_jac = y0 + h * x0   # the ordering we document as NOT used
    for y1, expect_match in ((y_semi, True), (y_jac, False)):
        r = torch.sqrt(x1 ** 2 + y1 ** 2)  # arm B renormalizes each substep
        diff = torch.stack((x1 / r, y1 / r), dim=1) - new_state
        assert (diff.abs().max() < 1e-6) == expect_match, diff.abs().max()


def test_frozen_amplitude_is_exactly_renormalized():
    # Arm B contract: |z| = 1 after every frame regardless of drive. This is
    # the step that erases the sequential ordering's O((dt·d)²) radial residue
    # (see the docstring's "Discretization ordering" note).
    torch.manual_seed(0)
    core = SLCore(channels=2, grid=8, damping=0.1, amplitude_frozen=True)
    drives = torch.randn(1, 50, 2, 8, 8)  # O(1) drives, like scale
    _, final = core.forward_scan(drives)
    r = torch.sqrt(final[:, 0] ** 2 + final[:, 1] ** 2)
    assert (r - 1).abs().max() < 1e-5


def test_sl_trainables_are_physics_only():
    torch.manual_seed(0)
    m = OscillatorField(channels=2, grid=8, n_classes=4, probe_seed=0, core="sl")
    trainable = sorted(n for n, p in m.named_parameters() if p.requires_grad)
    assert trainable == ["core.alpha", "core.beta_hat", "core.kernel", "core.natural_freqs"]
    torch.manual_seed(0)
    b = OscillatorField(channels=2, grid=8, n_classes=4, probe_seed=0, core="sl-fixedamp")
    trainable_b = sorted(n for n, p in b.named_parameters() if p.requires_grad)
    assert trainable_b == ["core.kernel", "core.natural_freqs"]


def test_designed_omega_applies_to_sl_core():
    from harness import tonotopic_omega
    torch.manual_seed(0)
    m = OscillatorField(channels=2, grid=8, n_classes=4, probe_seed=0, core="sl")
    with torch.no_grad():
        m.core.natural_freqs.copy_(tonotopic_omega(2, 8, 0.1, 1, torch.Generator().manual_seed(1)))
    f = m.features(torch.randn(2, 20, 8) * 0.5)
    assert torch.isfinite(f).all()


def test_sl_phase_trajectory_and_features_shapes():
    torch.manual_seed(0)
    m = OscillatorField(channels=2, grid=8, n_classes=4, probe_seed=0, core="sl")
    rows = torch.randn(3, 40, 8) * 0.5
    th = m.phase_trajectory(rows)
    assert th.shape == (3, 40, 2, 8, 8) and torch.isfinite(th).all()
    f = m.features(rows)
    assert f.shape == (3, 4 * 2 * 8 * 8) and torch.isfinite(f).all()


def test_exact_drive_matches_phase_core_at_high_drive():
    # The semi-implicit drive line rotates by 2·arcsin(h/2) per frame, not h, so at a
    # strong drive the fixed-amplitude core drifts from the phase core; with the drive
    # applied as an exact rotation (drive="rotation") the two agree at every drive up to
    # the operator split's O(dt) error, which the drive's size does not enlarge.
    def gap(core, scale):
        rows = torch.randn(2, 40, 16, generator=torch.Generator().manual_seed(11)).abs() * scale
        torch.manual_seed(7)
        phase = OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0, gain=1.0)
        torch.manual_seed(7)
        sl = OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0, gain=1.0, core=core)
        drv = rows.view(2, 40, 1, 16, 1).expand(2, 40, 2, 16, 16)
        f_phase, _ = phase.core.forward_scan(drv)
        f_sl, _ = sl.core.forward_scan(drv)
        return (f_phase - f_sl).abs().mean().item()

    weak_semi, weak_exact = gap("sl-fixedamp", 0.3), gap("sl-fixedamp-exact", 0.3)
    strong_semi, strong_exact = gap("sl-fixedamp", 12.0), gap("sl-fixedamp-exact", 12.0)
    assert weak_exact < 0.05 and strong_exact < 0.05, (weak_exact, strong_exact)
    assert strong_semi > 3 * strong_exact, (strong_semi, strong_exact)


def test_exact_drive_rotates_by_exactly_h():
    # Coupling, restoring pull and natural frequency zeroed, a constant drive d: under the
    # exact-rotation drive every frame turns the phase by exactly dt·d. Under the semi-implicit
    # drive one frame's turn depends on the starting phase (the map is conjugate to a rotation,
    # not a rotation), and over many frames the phase advances by 2·arcsin(dt·d/2) per frame
    # on average, the rotation number of the map (x, y) -> (x − h·y, y + h·(x − h·y)).
    dt, d, frames = 0.1, 15.0, 400  # h = 1.5
    h = dt * d
    turns = {}
    for drive in ("rotation", "semi-implicit"):
        core = SLCore(channels=1, grid=4, dt=dt, damping=0.0, amplitude_frozen=True, drive=drive)
        with torch.no_grad():
            core.kernel.zero_()
            core.natural_freqs.zero_()
        state = core.init_state(1)
        coup = core.prepare_coupling()
        steps = []
        for _ in range(frames):
            new_state, _ = core.step_frame(state, torch.full((1, 1, 4, 4), d), coup)
            before = torch.atan2(state[:, 1], state[:, 0])
            after = torch.atan2(new_state[:, 1], new_state[:, 0])
            steps.append(torch.remainder(after - before, TWO_PI))
            state = new_state
        turns[drive] = torch.stack(steps)  # [frames, 1, 1, 4, 4]
    assert (turns["rotation"] - h).abs().max() < 1e-4, turns["rotation"].flatten()[:4]
    semi = turns["semi-implicit"]
    assert semi.std(0).mean() > 0.1, semi.std(0).mean()                  # a single frame's turn varies with the phase
    assert abs(semi.mean().item() - 2 * math.asin(h / 2)) < 0.02, (semi.mean().item(), 2 * math.asin(h / 2))


def test_free_amplitude_exact_drive_runs_and_is_recorded_distinctly():
    from harness.experiment import arms as am
    from harness.experiment import terms
    for coupling in ("stuart-landau-exact", "stuart-landau-fixed-exact"):
        arm = am.Arm("network", coupling=coupling)
        model = am.build_untrained(arm, 2.0, 0)
        assert model.core.drive_step == "rotation"
        assert model.core.amplitude_frozen == coupling.endswith("fixed-exact")
        label = arm.label()
        assert "exact" in label and "exact-rotation drive" in terms.arm(label), terms.arm(label)
    assert terms.arm("coupled-stuart-landau-fixed-exact-torus-random-restoring0.3-ceiling1") == (
        "coupled oscillator network (Stuart–Landau, fixed amplitude, exact-rotation drive)")
