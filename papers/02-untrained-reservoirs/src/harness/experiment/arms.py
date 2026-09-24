"""The arms: what sits between the front end and the shared readout.

An arm's only job is to expose signals over time, [B, T, D]. Everything after
that is shared: the same statistics over the same frames (harness.measurement
.features), the same projection and the same ridge (harness.experiment.readout).

    baseline     nothing: the front end's rows themselves (the spectrogram-only baseline)
    network      the untrained coupled oscillator network; `coupled=False` zeroes
                 its coupling kernels and changes nothing else (the uncoupled network)
    bank         the untrained leaky-integrator bank, the non-oscillating control
    trained      a trained baseline, trained with a learned linear head on the
                 very statistics the ridge will read

Every arm is read over one fixed window, the same frames for every clip: from
the end of the integrator warm-up to the end of the padded clip. The baseline is
also read from the very first frame, the harder control, since it then sees
everything the reservoirs were driven with.

The window is fixed because a per-clip window leaks. Read over each clip's own
length, an undriven network still tells clips apart: it keeps rotating, and its
statistics over a span encode the span's length, which in speech carries the
digit. Over a fixed window, an arm with no input reads exactly chance, and
everything a read carries arrives through the arm's response to the input.
`span="clip"` restores a per-clip window, for the gate that measures the leak.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from harness.measurement import features as ft
from harness.measurement.instruments import analytic_row_phase
from harness.models.baselines import CNNBaseline, GRUBaseline, S4DBaseline, TCNBaseline, TransformerBaseline
from harness.models.field import OscillatorField, physics_block, tonotopic_omega
from harness.models.leaky_bank import LeakyBank
from harness.utils.constants import PLV_LOCK_THRESH, WARMUP_FRAMES

GRID = 16
#: the Kuramoto-Sakaguchi phase lag and the second-harmonic weight
ALPHA, BETA = 0.7853981633974483, 0.5   # pi/4, 0.5
N_CLASSES = 10
#: how much of each clip a read covers
SPANS = ("fixed", "clip")
DT, SUBSTEPS = 0.1, 1
#: windows per read: four for recognition, one (the order-free read) for order
WINDOWS = {"recognition": 4, "order": 1}
TRAINED = {"gru": GRUBaseline, "tcn": TCNBaseline, "cnn": CNNBaseline,
           "transformer": TransformerBaseline, "s4d": S4DBaseline}
#: the trained baselines' recipe
EPOCHS, BATCH, LR, CLIP, LR_FLOOR = 30, 64, 3e-3, 1.0, 0.1
#: a trained baseline that did not cut its loss by this much failed to train,
#: which is a different claim from "cannot do the task"
HEALTHY_LOSS_DROP = 0.20
#: seed offsets for the readout probe buffer and the tonotopic frequencies' jitter
PROBE_SEED, TONOTOPIC_SEED = 5000, 8000
#: the model layer's names for each coupling function: (core, phase coupling law)
CORES = {"kuramoto": ("phase", "kuramoto"), "kuramoto-sakaguchi": ("phase", "sakaguchi"),
         "second-harmonic": ("phase", "harmonic2"), "winfree": ("phase", "winfree"),
         "stuart-landau": ("sl", "kuramoto"), "stuart-landau-fixed": ("sl-fixedamp", "kuramoto")}
#: the bank's label by channel count: the state-matched and the width-matched bank
BANKS = {4: "bank-state", 8: "bank-width"}


@dataclass(frozen=True)
class Arm:
    """One arm, fully specified. The label is its identity in the record."""
    kind: str                       # baseline | network | bank | trained
    coupling: str = "kuramoto"      # a key of CORES
    geometry: str = "torus"         # torus | cylinder | sheet | helix | cube | sphere
    frequencies: str = "random"     # random | tonotopic | identical
    restoring: float = 0.3          # the restoring strength, lambda
    ceiling: float = 1.0            # the coupling ceiling
    channels: int = 4
    coupled: bool = True            # False: the uncoupled network, its coupling kernels zeroed
    arch: str = ""                  # a trained baseline's architecture

    def label(self) -> str:
        if self.kind == "baseline":
            return "baseline"
        if self.kind == "bank":
            return BANKS.get(self.channels, f"bank-ch{self.channels}")
        if self.kind == "trained":
            return f"trained-{self.arch}"
        name = "coupled" if self.coupled else "uncoupled"
        size = "" if self.channels == 4 else f"-ch{self.channels}"
        return (f"{name}-{self.coupling}-{self.geometry}-{self.frequencies}"
                f"-restoring{self.restoring:g}-ceiling{self.ceiling:g}{size}")

    @property
    def uses_gain(self) -> bool:
        """Gain scales the input into a reservoir; the baseline and the trained baselines read rows as they are."""
        return self.kind in ("network", "bank")

    def as_dict(self) -> dict:
        return asdict(self)


def _statistics(task: str) -> tuple[tuple[str, int], ...]:
    """(name, windows): recognition reads four windows and the whole span; order only the whole span."""
    return (("windowed", WINDOWS["recognition"]), ("pooled", 1)) if task == "recognition" else (("pooled", 1),)


def reads(arm: Arm, task: str) -> dict[str, list[str]]:
    """Each read a run records for an arm, as the feature blocks it concatenates.

    The order task reads only the whole span, the order-free read. An oscillator
    network adds each read with its rotation rates. The baseline is read over
    the arms' span and over the whole clip. A block is stored once however many
    reads use it.
    """
    out: dict[str, list[str]] = {}
    for name, _ in _statistics(task):
        out[name] = [name]
        if arm.kind == "baseline":
            out[f"{name}@wholeclip"] = [f"{name}@wholeclip"]
        if arm.kind == "network":
            out[f"{name}+rate"] = [name, f"{name}:rate"]
    return out


# ---------------------------------------------------------------------------
# Untrained arms
# ---------------------------------------------------------------------------

def build_untrained(arm: Arm, gain: float, seed: int, device: str = "cpu") -> nn.Module | None:
    """An untrained arm, built from its seed."""
    if arm.kind == "baseline":
        return None
    if arm.kind == "bank":
        return LeakyBank(channels=arm.channels, grid=GRID, gain=gain, seed=seed).to(device)
    if arm.kind != "network":
        raise ValueError(f"{arm.kind} is not an untrained arm")
    core, coupling = CORES[arm.coupling]
    torch.manual_seed(seed)              # the kernel and natural-frequency draws
    network = OscillatorField(
        channels=arm.channels, grid=GRID, coupling=coupling, damping=arm.restoring,
        spectral_clamp=arm.ceiling, substeps=SUBSTEPS, dt=DT, n_classes=N_CLASSES,
        probe_seed=PROBE_SEED + seed, gain=gain, seed=seed, core=core, boundary=arm.geometry,
        sakaguchi_alpha=ALPHA if coupling == "sakaguchi" else 0.0,
        harmonic2_beta=BETA if coupling == "harmonic2" else 0.0)
    block = physics_block(network.core)
    with torch.no_grad():
        if arm.frequencies == "tonotopic":
            block.natural_freqs.copy_(tonotopic_omega(
                arm.channels, GRID, DT, SUBSTEPS, torch.Generator().manual_seed(TONOTOPIC_SEED + seed)))
        elif arm.frequencies == "identical":
            block.natural_freqs.fill_(1.0)
        if not arm.coupled:
            block.kernel.zero_()
    for p in network.parameters():
        p.requires_grad_(False)
    return network.eval().to(device)


def untrained_signals(arm: Arm, model: nn.Module | None, rows: torch.Tensor) -> torch.Tensor:
    """[B, T, D]: the rows (baseline), the network's sin/cos state, or the bank's states."""
    if arm.kind == "baseline":
        return rows.flatten(2)            # quadrature rows [B,T,G,2] read as 2G signals
    if arm.kind == "network":
        return model._scan(rows)
    return model.signals(rows)


def _end(tvalid: torch.Tensor, span: str) -> torch.Tensor | None:
    """Where a read stops: the end of the padded clip, or (diagnostic) each clip's own end."""
    if span not in SPANS:
        raise ValueError(f"unknown span '{span}'")
    return tvalid if span == "clip" else None


def untrained_features(arm: Arm, signals: torch.Tensor, tvalid: torch.Tensor, task: str,
                       span: str = "fixed") -> dict[str, torch.Tensor]:
    """One batch's feature blocks for an untrained arm, keyed as `reads` names them."""
    hi = _end(tvalid, span)
    out: dict[str, torch.Tensor] = {}
    for name, windows in _statistics(task):
        out[name] = ft.windowed(signals, windows, lo=WARMUP_FRAMES, hi=hi)
        if arm.kind == "baseline":
            out[f"{name}@wholeclip"] = ft.windowed(signals, windows, lo=0, hi=hi)
        if arm.kind == "network":
            out[f"{name}:rate"] = ft.rotation_rate(signals, windows, lo=WARMUP_FRAMES, hi=hi)
    return out


def drive_phase(rows: torch.Tensor, pathway: str) -> torch.Tensor:
    """[B, T, G]: the phase of each band's own delivered drive, the reference an oscillator can lock to.

    The quadrature pathway carries its phase explicitly; the band energies have
    none, so theirs is the analytic phase.
    """
    if pathway == "quadrature":
        return torch.atan2(rows[..., 1], rows[..., 0])
    return analytic_row_phase(rows)


def network_instruments(signals: torch.Tensor, rows: torch.Tensor, pathway: str, channels: int,
                        grid: int = GRID, lo: int = WARMUP_FRAMES) -> dict[str, torch.Tensor]:
    """Per clip, over the read's frames: how synchronized the network is and how locked to its input.

    R is the global order parameter |<e^{i theta}>| of each channel, averaged over
    channels and frames. plv is each oscillator's phase-locking value to its own
    row's drive, |<e^{i(theta - phi)}>| over frames, averaged over oscillators, and
    entrained the share of oscillators whose value exceeds PLV_LOCK_THRESH.
    amplitude is the mean |z|: 1 for a phase core, free for a Stuart-Landau one.
    Diagnostics only: nothing here reaches the readout.
    """
    b, t, _ = signals.shape
    n = channels * grid * grid
    s = signals[:, lo:, :n].reshape(b, t - lo, channels, grid, grid)
    c = signals[:, lo:, n:2 * n].reshape(b, t - lo, channels, grid, grid)
    amp = torch.sqrt(s * s + c * c)
    theta = torch.atan2(s, c)
    r = torch.sqrt(torch.cos(theta).mean((-2, -1)) ** 2 + torch.sin(theta).mean((-2, -1)) ** 2).mean((1, 2))
    diff = theta - drive_phase(rows, pathway)[:, lo:t, None, :, None]
    plv = torch.sqrt(torch.cos(diff).mean(1) ** 2 + torch.sin(diff).mean(1) ** 2).flatten(1)
    return {"R": r, "plv": plv.mean(1), "entrained": (plv > PLV_LOCK_THRESH).float().mean(1),
            "amplitude": amp.mean((1, 2, 3, 4))}


def meta(arm: Arm, model: nn.Module | None) -> dict:
    """What the record says about an arm's size: states, and parameters stored and trained."""
    if model is None:
        return {"states": 0, "stored_params": 0, "trained_params": 0}
    stored = sum(p.numel() for p in model.parameters())
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    states = arm.channels * GRID * GRID if arm.kind in ("network", "bank") else None
    out = {"states": states, "stored_params": stored, "trained_params": trained}
    if arm.kind == "network" and not arm.coupled:
        # the zeroed kernel is still stored, but it no longer does anything
        out["effective_params"] = stored - physics_block(model.core).kernel.numel()
    return out


# ---------------------------------------------------------------------------
# Trained baselines
# ---------------------------------------------------------------------------

def trained_blocks(backbone: nn.Module, rows: torch.Tensor, tvalid: torch.Tensor, task: str,
                   span: str = "fixed") -> dict[str, torch.Tensor]:
    """A trained baseline's feature blocks, keyed as `reads` names them."""
    return {name: trained_features(backbone, rows, tvalid, windows, span) for name, windows in _statistics(task)}


def trained_features(backbone: nn.Module, rows: torch.Tensor, tvalid: torch.Tensor, windows: int,
                     span: str = "fixed") -> torch.Tensor:
    """A network's hidden trajectory through the shared statistics.

    The baselines already drop the warm-up frames from their trajectories, so
    the read starts at their first frame: the same frames every untrained arm
    is read over.
    """
    hi = _end(tvalid, span)
    return ft.windowed(backbone._hidden(rows), windows, lo=0, hi=None if hi is None else hi - WARMUP_FRAMES)


def train_baseline(arm: Arm, rows: torch.Tensor, tvalid: torch.Tensor, labels: torch.Tensor,
                   task: str, seed: int, n_classes: int = N_CLASSES, epochs: int = EPOCHS,
                   span: str = "fixed", device: str = "cpu") -> tuple[nn.Module, nn.Module, dict]:
    """Train a baseline end to end, with a learned linear head on the shared statistics.

    Conventional practice is a learned head; putting that head on the same
    statistics the ridge will read means the network is trained for the read it
    is judged by, and a recurrent network cannot hand the head its last state.
    Returns (backbone, head, health). The weights are drawn on the CPU and the
    batches in the same order on every device, so a GPU run differs from the
    CPU's only by floating-point rounding.
    """
    torch.manual_seed(seed)
    backbone = TRAINED[arm.arch](grid=GRID, n_classes=n_classes, probe_seed=PROBE_SEED + seed)
    windows = WINDOWS[task]
    with torch.no_grad():
        width = trained_features(backbone, rows[:2], tvalid[:2], windows, span).shape[1]
    # the head reads the statistics standardized, as the ridge does: unstandardized, their spread between
    # clips can be a thousandth of their size (with noise especially), and the loss never leaves chance
    head = nn.Sequential(nn.BatchNorm1d(width, affine=False), nn.Linear(width, n_classes))
    backbone, head = backbone.to(device), head.to(device)
    rows, tvalid, labels = rows.to(device), tvalid.to(device), labels.to(device)
    params = list(backbone.parameters()) + list(head.parameters())
    steps = epochs * ((len(rows) + BATCH - 1) // BATCH)
    opt = torch.optim.AdamW(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=LR_FLOOR * LR)
    gen = torch.Generator().manual_seed(seed)
    losses = []
    backbone.train(); head.train()
    for _ in range(epochs):
        perm = torch.randperm(len(rows), generator=gen).to(device)
        total = 0.0
        for i in range(0, len(rows), BATCH):
            idx = perm[i:i + BATCH]
            logits = head(trained_features(backbone, rows[idx], tvalid[idx], windows, span))
            loss = nn.functional.cross_entropy(logits, labels[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{arm.label()} seed {seed}: non-finite loss")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        losses.append(total / len(rows))
    backbone.eval(); head.eval()
    drop = (losses[0] - losses[-1]) / losses[0] if losses[0] else 0.0
    health = {"first_epoch_loss": losses[0], "last_epoch_loss": losses[-1], "loss_drop": drop,
              "healthy": bool(drop >= HEALTHY_LOSS_DROP), "epochs": epochs}
    return backbone, head, health
