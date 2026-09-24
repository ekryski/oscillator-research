"""The arms: what sits between the front end and the shared readout.

An arm's only job is to expose signals over time, [B, T, D]. Everything after
that is shared: the same statistics over the same frames (harness.measurement
.features), the same projection and the same ridge (harness.confirm.readout).
The record keeps paper 02's labels; `harness.confirm.terms` gives the paper's
terms for them.

    floor        the spectrogram-only baseline: the front-end rows themselves
    field        the coupled oscillator network; `severed` is the uncoupled
                 network, its coupling kernels set to zero and nothing else changed
    bank         the leaky-integrator bank; at C channels of a G x G lattice it
                 has the network's C * G * G states and 2 * C * G * G parameters
    ann          a trained baseline, trained with a learned linear head on the
                 very statistics the ridge will read, and sized to the network
                 whose lattice and channel count its label names

Every arm has a lattice (grid x grid per channel), a channel count and a band
mapping (`bands`: 0 drives each of the grid rows with its own mel band, 16
maps paper 02's 16 bands onto the rows). Paper 02's arms are the 16 x 16,
4-channel ones, and keep exactly the labels paper 02 recorded them under.

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

from dataclasses import asdict, dataclass, replace

import torch
from torch import nn

from harness.measurement import features as ft
from harness.models.baselines import CNNBaseline, GRUBaseline, S4DBaseline, TCNBaseline, TransformerBaseline
from harness.models.field import OscillatorField, physics_block, tonotopic_omega
from harness.models.leaky_bank import LeakyBank
from harness.utils.constants import ALPHA, BETA, WARMUP_FRAMES

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
#: the seed offset paper 02 used for tonotopic natural frequencies
DESIGNED_OMEGA_SEED = 8000
#: paper 02's network: the channel count and lattice every unsuffixed label means
CHANNELS = 4
#: a trained baseline within this share of its budget is matched; the record flags any that is not
BUDGET_TOLERANCE = 0.15


@dataclass(frozen=True)
class Arm:
    """One arm, fully specified. The label is its identity in the record."""
    kind: str                       # floor | field | bank | ann
    physics: str = "kuramoto"       # kuramoto | sakaguchi | harmonic2 | winfree | sl | sl-fixedamp
    boundary: str = "torus"
    omega: str = "random"           # random | designed | uniform
    damping: float = 0.3
    clamp: float = 1.0
    channels: int = CHANNELS
    severed: bool = False
    arch: str = ""
    grid: int = GRID                # the lattice is grid x grid per channel
    bands: int = 0                  # mel bands in the front end; 0 is one per lattice row

    @property
    def n_bands(self) -> int:
        return self.bands or self.grid

    def _lattice(self) -> str:
        """'' at the registered 16 x 16 with a band per row, so every registered label is unchanged."""
        out = "" if self.grid == GRID else f"-{self.grid}x{self.grid}"
        return out + ("" if self.n_bands == self.grid else f"-{self.n_bands}bands")

    def label(self) -> str:
        if self.kind == "floor":
            return "floor" + self._lattice()
        if self.kind == "bank":
            return f"bank-c{self.channels}" + self._lattice()
        if self.kind == "ann":
            # the network it is sized to; paper 02's 4 channels of 16 x 16 keep the plain label
            size = "" if self.channels == CHANNELS else f"-c{self.channels}"
            return f"ann-{self.arch}{size}{self._lattice()}"
        name = "severed" if self.severed else "field"
        size = "" if self.channels == 4 else f"-c{self.channels}"
        return (f"{name}-{self.physics}-{self.boundary}-{self.omega}"
                f"-lam{self.damping:g}-clamp{self.clamp:g}{size}{self._lattice()}")

    @property
    def states(self) -> int:
        return self.channels * self.grid * self.grid if self.kind in ("field", "bank") else 0

    @property
    def budget(self) -> int:
        """The coupled network's parameter count at this lattice and channel count: 2 * C * G * G
        (its coupling kernels and natural frequencies). The state-matched bank has it by
        construction; a trained baseline is sized to it."""
        return 2 * self.channels * self.grid * self.grid

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
                 rate_hz: float | None = None, kernel_scaling: str = "exact") -> nn.Module | None:
    """The untrained arm, built from its seed exactly as paper 02 built it.

    `kernel_scaling="cap"` rebuilds paper 02's network, whose kernels were only
    ever scaled down to the coupling ceiling; paper 03 scales every kernel to
    it exactly. At 16 x 16 and above the two are the same network, bit for bit.
    """
    if arm.kind == "floor":
        return None
    if arm.kind == "bank":
        extra = {} if rate_hz is None else {"rate_hz": rate_hz}
        return LeakyBank(channels=arm.channels, grid=arm.grid, gain=gain, seed=seed, **extra).to(device)
    if arm.kind != "field":
        raise ValueError(f"{arm.kind} is not an untrained arm")
    core, coupling = ((arm.physics, "kuramoto") if arm.physics in ("sl", "sl-fixedamp")
                      else ("phase", arm.physics))
    torch.manual_seed(seed)              # the kernel and natural-frequency draws
    field = OscillatorField(
        channels=arm.channels, grid=arm.grid, coupling=coupling, damping=arm.damping,
        spectral_clamp=arm.clamp, substeps=SUBSTEPS, dt=DT, gain=gain, seed=seed, core=core,
        boundary=arm.boundary, sakaguchi_alpha=ALPHA if arm.physics == "sakaguchi" else 0.0,
        harmonic2_beta=BETA if arm.physics == "harmonic2" else 0.0, kernel_scaling=kernel_scaling)
    block = physics_block(field.core)
    with torch.no_grad():
        if arm.omega == "designed":
            block.natural_freqs.copy_(tonotopic_omega(
                arm.channels, arm.grid, DT, SUBSTEPS, torch.Generator().manual_seed(DESIGNED_OMEGA_SEED + seed)))
        elif arm.omega == "uniform":
            block.natural_freqs.fill_(1.0)
        if arm.severed:
            block.kernel.zero_()
    for p in field.parameters():
        p.requires_grad_(False)
    return field.eval().to(device)


def channel(arm: Arm, model: nn.Module, c: int) -> tuple[Arm, nn.Module]:
    """Channel c of an untrained network or bank, as a one-channel arm of its own.

    Channels are independent: each has its own kernel, natural frequencies,
    starting phases (or its own leak rates and input weights), all receive
    the same input, and none acts on another. Channel c's trajectory is
    therefore the full arm's channel-c slice, which lets the streamed read
    (harness.confirm.stream) simulate a large arm one channel at a time. Every
    stored tensor is sliced along the one axis where a one-channel arm
    differs from the full one, so nothing is redrawn.
    """
    one = replace(arm, channels=1)
    if arm.channels == 1:
        return one, model
    with torch.random.fork_rng(devices=[]):
        sub = build_frozen(one, getattr(model, "gain", 0.0), 0, "cpu", getattr(model, "rate_hz", None),
                           _scaling(model))
    full, part = model.state_dict(), sub.state_dict()
    sliced = {}
    for key, value in part.items():
        whole = full[key]
        axes = [d for d in range(whole.dim()) if whole.shape[d] != value.shape[d]]
        if not axes:
            sliced[key] = whole
            continue
        if len(axes) != 1 or whole.shape[axes[0]] != arm.channels:
            raise ValueError(f"{key}: cannot find the channel axis in {tuple(whole.shape)}")
        sliced[key] = whole.narrow(axes[0], c, 1).clone()
    sub.load_state_dict(sliced)
    for p in sub.parameters():
        p.requires_grad_(False)
    return one, sub.eval().to(next(iter(model.parameters())).device)


def _scaling(model: nn.Module) -> str:
    core = getattr(model, "core", None)
    if core is None:
        return "exact"
    return physics_block(core).kernel_scaling


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
    states = arm.states if arm.kind in ("field", "bank") else None
    out = {"states": states, "stored_params": stored, "trained_params": trained, "budget": arm.budget}
    if arm.kind == "ann":
        out["width"] = ann_width(arm.arch, arm.grid, arm.budget)
        out["within_budget_tolerance"] = abs(trained / arm.budget - 1) <= BUDGET_TOLERANCE
    if arm.severed:
        # the zeroed kernel is still stored, but it no longer does anything
        out["effective_params"] = stored - physics_block(model.core).kernel.numel()
    return out


# ---------------------------------------------------------------------------
# Trained arms
# ---------------------------------------------------------------------------

#: each architecture's one width knob, the smallest width it takes, and the step between widths
_WIDTH = {"gru": ("hidden", 1, 1), "tcn": ("hidden", 1, 1), "cnn": ("hidden", 1, 1),
          "transformer": ("d", 2, 2), "s4d": ("hidden", 1, 1)}
#: the CNN takes the narrowest width at or over the budget; the rest the widest at or under it
_ROUND_UP = ("cnn",)


def _kwargs(arch: str, width: int) -> dict:
    knob = _WIDTH[arch][0]
    return {knob: width, "n_states": width} if arch == "s4d" else {knob: width}


def ann_params(arch: str, rows: int, width: int) -> int:
    """Trainable parameters of a trained baseline on `rows` input rows at `width`, head excluded."""
    with torch.device("meta"):
        return sum(p.numel() for p in ANNS[arch](grid=rows, **_kwargs(arch, width)).parameters())


def ann_width(arch: str, rows: int, budget: int) -> int:
    """The width that sizes a trained baseline to the network's parameter count.

    Each architecture has one width knob (the GRU's hidden size, the TCN's and
    the CNN's hidden channels, the transformer's model width with two heads,
    the S4D's width with as many states per channel as its width) and the
    rest of its design fixed. It takes the widest width whose parameter count
    does not exceed the budget; the CNN, which is the TCN's form, takes the
    narrowest that reaches it, so the two stay one width apart. On 16 rows at
    2,048 parameters this gives paper 02's widths exactly: 18, 12, 13, 16 and
    16. Parameter counts rise with width, so a bisection finds it.
    """
    _, lo, step = _WIDTH[arch]
    count = lambda w: ann_params(arch, rows, w)  # noqa: E731
    if count(lo) > budget:
        return lo
    hi = lo
    while count(hi) <= budget:
        hi = lo + 2 * (hi - lo) + step
    while hi - lo > step:                       # count(lo) <= budget < count(hi)
        mid = lo + (hi - lo) // (2 * step) * step
        mid = mid if mid > lo else lo + step
        lo, hi = (mid, hi) if count(mid) <= budget else (lo, mid)
    if arch in _ROUND_UP and count(lo) < budget:
        return hi
    return lo


def build_ann(arm: Arm) -> nn.Module:
    """A trained baseline sized to the network its label names, on that network's rows."""
    return ANNS[arm.arch](grid=arm.grid, **_kwargs(arm.arch, ann_width(arm.arch, arm.grid, arm.budget)))

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
              span: str = "fixed", device: str = "cpu") -> tuple[nn.Module, nn.Module, dict]:
    """Train a network end to end, with a learned linear head on the shared statistics.

    Conventional practice is a learned head; putting that head on the same
    statistics the ridge will read means the network is trained for the read it
    is judged by, and a recurrent network cannot hand the head its last state.
    Both are built on the CPU, so every device starts from the same weights,
    and trained on `device`; the clips stay on the CPU and move a batch at a
    time. Returns (backbone, head, health), on `device`.
    """
    torch.manual_seed(seed)
    backbone = build_ann(arm)
    windows = WINDOWS[task]
    with torch.no_grad():
        width = ann_features(backbone, rows[:2], tvalid[:2], windows, span).shape[1]
    head = nn.Linear(width, n_classes)
    backbone, head = backbone.to(device), head.to(device)
    params = list(backbone.parameters()) + list(head.parameters())
    steps = epochs * ((len(rows) + BATCH - 1) // BATCH)
    opt = torch.optim.AdamW(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=LR_FLOOR * LR)
    gen = torch.Generator().manual_seed(seed)
    losses = []
    backbone.train()
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(len(rows), generator=gen)
        total = 0.0
        for i in range(0, len(rows), BATCH):
            idx = perm[i:i + BATCH]
            logits = head(ann_features(backbone, rows[idx].to(device), tvalid[idx].to(device), windows, span))
            loss = nn.functional.cross_entropy(logits, labels[idx].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{arm.label()} seed {seed}: non-finite loss")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        losses.append(total / len(rows))
    backbone.eval()
    head.eval()
    drop = (losses[0] - losses[-1]) / losses[0] if losses[0] else 0.0
    health = {"first_epoch_loss": losses[0], "last_epoch_loss": losses[-1], "loss_drop": drop,
              "healthy": bool(drop >= HEALTHY_LOSS_DROP), "epochs": epochs}
    return backbone, head, health
