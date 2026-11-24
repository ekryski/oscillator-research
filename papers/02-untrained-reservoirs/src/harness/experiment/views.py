"""Views of an oscillator network's signals: what the readout is allowed to see of z = r·e^{iθ}.

The readout normally sees every oscillator's sin θ and cos θ, which for a
Stuart–Landau oscillator are r·sin θ and r·cos θ. Three views take that apart,
to ask what the amplitude gives the readout:

    full       the signals as they are
    phase      z/|z|: the phase alone, every oscillator at unit radius
    amplitude  |z| alone, one signal per oscillator
    slaved     a phase network's sin θ and cos θ scaled by an amplitude that
               follows the Stuart–Landau radial equation driven by those phases,
               with no feedback on them: the coupling's radial pull on an
               oscillator out of step with its neighbours and the restoring
               pull, then the same amplitude relaxation as the Stuart–Landau
               core (alpha = beta = 0.05, attracting radius 1). If reading a
               Kuramoto network this way reproduces the Stuart–Landau network's
               lead, the amplitude's gain to the readout is the weighting of
               each oscillator by its local coherence.

The slaved amplitude is computed frame by frame from the phase trajectory with
the network's own coupling operator and restoring strength.
"""

from __future__ import annotations

import torch

from harness.models.stuart_landau import implicit_amplitude_root

VIEWS = ("full", "phase", "amplitude", "slaved")
#: the Stuart–Landau core's amplitude relaxation rate, and attracting radius 1 (beta = alpha)
ALPHA = 0.05


def apply_view(arm, model, signals: torch.Tensor, view: str, dt: float = 0.1) -> torch.Tensor:
    """[B, T, 2N] sin-then-cos signals -> the view's signals ([B, T, N] for the amplitude view)."""
    if view not in VIEWS:
        raise ValueError(f"unknown view '{view}'")
    if view == "full" or arm.kind != "network":
        return signals
    n = signals.shape[-1] // 2
    s, c = signals[..., :n], signals[..., n:]
    if view == "amplitude":
        return torch.sqrt(s * s + c * c)
    if view == "phase":
        r = torch.sqrt(s * s + c * c).clamp_min(1e-8)
        return torch.cat((s / r, c / r), dim=-1)
    return slaved(arm, model, signals, dt)


def slaved(arm, model, signals: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """A phase network's signals scaled by an amplitude slaved to its phases."""
    block = model.core.blocks[0]
    channels, grid = block.channels, block.grid
    coup = block.prepare_coupling()
    ones = signals.new_ones(1, 1, channels, grid, grid)
    s0 = block._apply_conv(ones, coup)[0, 0]                     # [C, G, G], the operator's DC response
    lam = arm.restoring
    b, t, d = signals.shape
    n = d // 2
    s = signals[..., :n].reshape(b, t, channels, grid, grid)
    c = signals[..., n:].reshape(b, t, channels, grid, grid)
    r = torch.ones(b, channels, grid, grid, device=signals.device, dtype=signals.dtype)
    dt_beta = torch.tensor(dt * ALPHA, device=signals.device, dtype=signals.dtype)
    one_minus = torch.tensor(1 - dt * ALPHA, device=signals.device, dtype=signals.dtype)
    out_s, out_c = [], []
    for k in range(t):
        x, y = r * c[:, k], r * s[:, k]                            # z = r·e^{iθ} at the network's phase
        conv = block._apply_conv(torch.stack((x, y), dim=1), coup)  # [B, 2, C, G, G]
        # the Stuart–Landau core's coupling and restoring step, without the drive (the phase already has it)
        x2 = x + dt * (conv[:, 0] - s0 * x + lam * (1 - x))
        y2 = y + dt * (conv[:, 1] - s0 * y - lam * y)
        r_old = torch.sqrt(x2 * x2 + y2 * y2).clamp_min(1e-8)
        r = implicit_amplitude_root(r_old, dt_beta, one_minus)
        out_s.append((r * s[:, k]).flatten(1))
        out_c.append((r * c[:, k]).flatten(1))
    return torch.cat((torch.stack(out_s, 1), torch.stack(out_c, 1)), dim=-1)
