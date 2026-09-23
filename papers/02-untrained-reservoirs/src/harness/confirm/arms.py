"""The arms: what sits between the front end and the shared readout.

An arm's only job is to expose signals over time, [B, T, D]. Everything after
that is shared: the same statistics over the same frames (harness.measurement
.features), the same projection and the same ridge (harness.confirm.readout).

    floor        nothing: the front-end rows themselves
    field        the frozen coupled oscillator field; `severed` zeroes its
                 coupling kernel and changes nothing else
    bank         the frozen leaky-integrator bank, the non-oscillating control
    ann          a conventional network, trained with a learned linear head on
                 the very statistics the ridge will read

Every arm is read over one fixed window, the same frames for every clip: from
the end of the integrator warm-up to the end of the padded clip. The floor is
also read from the very first frame, the harder control, since it then sees
everything the arms were driven with.

The window is fixed because a per-clip window leaks. Read over each clip's own
length, an undriven field still tells clips apart: it keeps rotating, and its
statistics over a span encode the span's length, which in speech carries the
digit. That hands a duration code to an autonomous oscillator and to nothing
else. Over a fixed window, an arm with no input reads exactly chance, and
everything a read carries arrives through the arm's response to the input.
`span="clip"` restores the per-clip window of the exploratory harness, for the
diagnostic that measures how much it inflated the exploratory reads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from harness.measurement import features as ft
from harness.models.baselines import CNNBaseline, GRUBaseline, S4DBaseline, TCNBaseline, TransformerBaseline
from harness.models.field import OscillatorField, physics_block, tonotopic_omega
from harness.models.leaky_bank import LeakyBank
from harness.sweep import ALPHA, BETA
from harness.utils.constants import WARMUP_FRAMES

GRID = 16
N_CLASSES = 10
#: how much of each clip a read covers
SPANS = ("fixed", "clip")
DT, SUBSTEPS = 0.1, 1
#: windows per read: four for recognition, one (the order-free read) for order
WINDOWS = {"recognition": 4, "order": 1}
ANNS = {"gru": GRUBaseline, "tcn": TCNBaseline, "cnn": CNNBaseline,
        "transformer": TransformerBaseline, "s4d": S4DBaseline}
#: the exploratory trained-head recipe (scripts/trained_head_baselines.py)
EPOCHS, BATCH, LR, CLIP, LR_FLOOR = 30, 64, 3e-3, 1.0, 0.1
#: a trained arm that did not cut its loss by this much failed to optimize,
#: which is a different claim from "cannot do the task"
HEALTHY_LOSS_DROP = 0.20
#: the seed offsets the exploratory runner used for these draws
PROBE_SEED, DESIGNED_OMEGA_SEED = 5000, 8000


@dataclass(frozen=True)
class Arm:
    """One arm, fully specified. The label is its identity in the record."""
    kind: str                       # floor | field | bank | ann
    physics: str = "kuramoto"       # kuramoto | sakaguchi | harmonic2 | winfree | sl | sl-fixedamp
    boundary: str = "torus"
    omega: str = "random"           # random | designed | uniform
    damping: float = 0.3
    clamp: float = 1.0
    channels: int = 4
    severed: bool = False
    arch: str = ""

    def label(self) -> str:
        if self.kind == "floor":
            return "floor"
        if self.kind == "bank":
            return f"bank-c{self.channels}"
        if self.kind == "ann":
            return f"ann-{self.arch}"
        name = "severed" if self.severed else "field"
        size = "" if self.channels == 4 else f"-c{self.channels}"
        return (f"{name}-{self.physics}-{self.boundary}-{self.omega}"
                f"-lam{self.damping:g}-clamp{self.clamp:g}{size}")

    @property
    def uses_gain(self) -> bool:
        """Gain scales the drive into a frozen dynamical arm; the floor and the networks read rows as they are."""
        return self.kind in ("field", "bank")

    def as_dict(self) -> dict:
        return asdict(self)


def _statistics(task: str) -> tuple[tuple[str, int], ...]:
    """(name, windows): recognition reads four windows and the whole span; order only the whole span."""
    return (("windowed", WINDOWS["recognition"]), ("pooled", 1)) if task == "recognition" else (("pooled", 1),)


def reads(arm: Arm, task: str) -> dict[str, list[str]]:
    """Each read a run records for an arm, as the feature blocks it concatenates.

    The order task reads only the whole span, the order-free read. The field
    adds each read with its rotation rates, the read that favours it. The floor
    is read over the arms' span and over the whole clip. A block is stored once
    however many reads use it.
    """
    out: dict[str, list[str]] = {}
    for name, _ in _statistics(task):
        out[name] = [name]
        if arm.kind == "floor":
            out[f"{name}@wholeclip"] = [f"{name}@wholeclip"]
        if arm.kind == "field":
            out[f"{name}+rate"] = [name, f"{name}:rate"]
    return out


# ---------------------------------------------------------------------------
# Frozen arms
# ---------------------------------------------------------------------------

def build_frozen(arm: Arm, gain: float, seed: int, device: str = "cpu",
                 rate_hz: float | None = None) -> nn.Module | None:
    """The frozen arm, built from its seed exactly as the exploratory runner built it."""
    if arm.kind == "floor":
        return None
    if arm.kind == "bank":
        extra = {} if rate_hz is None else {"rate_hz": rate_hz}
        return LeakyBank(channels=arm.channels, grid=GRID, gain=gain, seed=seed, **extra).to(device)
    if arm.kind != "field":
        raise ValueError(f"{arm.kind} is not a frozen arm")
    core, coupling = ((arm.physics, "kuramoto") if arm.physics in ("sl", "sl-fixedamp")
                      else ("phase", arm.physics))
    torch.manual_seed(seed)              # the kernel and natural-frequency draws
    field = OscillatorField(
        channels=arm.channels, grid=GRID, coupling=coupling, damping=arm.damping,
        spectral_clamp=arm.clamp, substeps=SUBSTEPS, dt=DT, n_classes=N_CLASSES,
        probe_seed=PROBE_SEED + seed, gain=gain, seed=seed, core=core, boundary=arm.boundary,
        sakaguchi_alpha=ALPHA if arm.physics == "sakaguchi" else 0.0,
        harmonic2_beta=BETA if arm.physics == "harmonic2" else 0.0)
    block = physics_block(field.core)
    with torch.no_grad():
        if arm.omega == "designed":
            block.natural_freqs.copy_(tonotopic_omega(
                arm.channels, GRID, DT, SUBSTEPS, torch.Generator().manual_seed(DESIGNED_OMEGA_SEED + seed)))
        elif arm.omega == "uniform":
            block.natural_freqs.fill_(1.0)
        if arm.severed:
            block.kernel.zero_()
    for p in field.parameters():
        p.requires_grad_(False)
    return field.eval().to(device)


def frozen_signals(arm: Arm, model: nn.Module | None, rows: torch.Tensor) -> torch.Tensor:
    """[B, T, D]: the rows (floor), the field's sin/cos state, or the bank's states."""
    if arm.kind == "floor":
        return rows.flatten(2)            # quadrature rows [B,T,G,2] read as 2G signals
    if arm.kind == "field":
        return model._scan(rows)
    return model.signals(rows)


def _end(tvalid: torch.Tensor, span: str) -> torch.Tensor | None:
    """Where a read stops: the end of the padded clip, or (diagnostic) each clip's own end."""
    if span not in SPANS:
        raise ValueError(f"unknown span '{span}'")
    return tvalid if span == "clip" else None


def frozen_features(arm: Arm, signals: torch.Tensor, tvalid: torch.Tensor, task: str,
                    span: str = "fixed") -> dict[str, torch.Tensor]:
    """One batch's feature blocks for a frozen arm, keyed as `reads` names them."""
    hi = _end(tvalid, span)
    out: dict[str, torch.Tensor] = {}
    for name, windows in _statistics(task):
        out[name] = ft.windowed(signals, windows, lo=WARMUP_FRAMES, hi=hi)
        if arm.kind == "floor":
            out[f"{name}@wholeclip"] = ft.windowed(signals, windows, lo=0, hi=hi)
        if arm.kind == "field":
            out[f"{name}:rate"] = ft.rotation_rate(signals, windows, lo=WARMUP_FRAMES, hi=hi)
    return out


def meta(arm: Arm, model: nn.Module | None) -> dict:
    """What the record says about an arm's size: states, and parameters stored and trained."""
    if model is None:
        return {"states": 0, "stored_params": 0, "trained_params": 0}
    stored = sum(p.numel() for p in model.parameters())
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    states = arm.channels * GRID * GRID if arm.kind in ("field", "bank") else None
    out = {"states": states, "stored_params": stored, "trained_params": trained}
    if arm.severed:
        # the zeroed kernel is still stored, but it no longer does anything
        out["effective_params"] = stored - physics_block(model.core).kernel.numel()
    return out


# ---------------------------------------------------------------------------
# Trained arms
# ---------------------------------------------------------------------------

def ann_blocks(backbone: nn.Module, rows: torch.Tensor, tvalid: torch.Tensor, task: str,
               span: str = "fixed") -> dict[str, torch.Tensor]:
    """A trained network's feature blocks, keyed as `reads` names them."""
    return {name: ann_features(backbone, rows, tvalid, windows, span) for name, windows in _statistics(task)}


def ann_features(backbone: nn.Module, rows: torch.Tensor, tvalid: torch.Tensor, windows: int,
                 span: str = "fixed") -> torch.Tensor:
    """A network's hidden trajectory through the shared statistics.

    The baselines already drop the warm-up frames from their trajectories, so
    the read starts at their first frame: the same frames every frozen arm is
    read over.
    """
    hi = _end(tvalid, span)
    return ft.windowed(backbone._hidden(rows), windows, lo=0, hi=None if hi is None else hi - WARMUP_FRAMES)


def train_ann(arm: Arm, rows: torch.Tensor, tvalid: torch.Tensor, labels: torch.Tensor,
              task: str, seed: int, n_classes: int = N_CLASSES, epochs: int = EPOCHS,
              span: str = "fixed") -> tuple[nn.Module, nn.Module, dict]:
    """Train a network end to end, with a learned linear head on the shared statistics.

    Conventional practice is a learned head; putting that head on the same
    statistics the ridge will read means the network is trained for the read it
    is judged by, and a recurrent network cannot hand the head its last state.
    Returns (backbone, head, health).
    """
    torch.manual_seed(seed)
    backbone = ANNS[arm.arch](grid=GRID, n_classes=n_classes, probe_seed=PROBE_SEED + seed)
    windows = WINDOWS[task]
    with torch.no_grad():
        width = ann_features(backbone, rows[:2], tvalid[:2], windows, span).shape[1]
    head = nn.Linear(width, n_classes)
    params = list(backbone.parameters()) + list(head.parameters())
    steps = epochs * ((len(rows) + BATCH - 1) // BATCH)
    opt = torch.optim.AdamW(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=LR_FLOOR * LR)
    gen = torch.Generator().manual_seed(seed)
    losses = []
    backbone.train(); head.train()
    for _ in range(epochs):
        perm = torch.randperm(len(rows), generator=gen)
        total = 0.0
        for i in range(0, len(rows), BATCH):
            idx = perm[i:i + BATCH]
            logits = head(ann_features(backbone, rows[idx], tvalid[idx], windows, span))
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
