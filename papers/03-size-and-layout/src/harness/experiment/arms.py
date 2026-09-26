"""The arms: what sits between the front end and the shared readout.

An arm's only job is to expose signals over time, [B, T, D]. Everything after
that is shared: the same statistics over the same frames (harness.measurement
.features), the same projection and the same ridge (harness.experiment.readout).
The labels follow the paper's glossary in short form, as paper 02's do;
`harness.experiment.terms` gives the words in full.

    baseline     nothing: the front end's rows themselves (the spectrogram-only baseline)
    network      the untrained coupled oscillator network; `coupled=False` zeroes
                 its coupling kernels and changes nothing else (the uncoupled network)
    bank         the untrained leaky-integrator bank; at C channels of a G x G
                 lattice it has the network's C * G * G states and 2 * C * G * G parameters
    trained      a trained baseline, trained with a learned linear head on the
                 very statistics the ridge will read, and sized to the network
                 whose lattice and channel count its label names

Every arm has a lattice (grid x grid per channel), a channel count, a band
mapping (`bands`: 0 drives each of the grid rows with its own mel band, 16
maps paper 02's 16 bands onto the rows) and an analysis window (`window`: 0 is
paper 02's 512 samples). Paper 02's arms are the 16 x 16, 4-channel ones with
a band per row and its window, and keep exactly the labels paper 02 records
them under.

Every arm is read over one fixed window, the same frames for every clip: from
the end of the integrator warm-up to the end of the padded clip. The baseline
is also read from the very first frame, the harder control, since it then sees
everything the reservoirs were driven with.

The window is fixed because a per-clip window leaks. Read over each clip's own
length, an undriven network still tells clips apart: it keeps rotating, and its
statistics over a span encode the span's length, which in speech carries the
digit. Over a fixed window, an arm with no input reads exactly chance, and
everything a read carries arrives through the arm's response to the input.
`span="clip"` restores a per-clip window, for a diagnostic that measures the leak.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
from torch import nn

from harness.measurement import features as ft
from harness.measurement.instruments import analytic_row_phase
from harness.models.baselines import CNNBaseline, GRUBaseline, S4DBaseline, TCNBaseline, TransformerBaseline
from harness.models.field import OscillatorField, physics_block
from harness.models.leaky_bank import LeakyBank
from harness.utils.constants import ALPHA, BETA, PLV_LOCK_THRESH, WARMUP_FRAMES

GRID = 16
#: paper 02's analysis window, in samples
WINDOW = 512
N_CLASSES = 10
#: how much of each clip a read covers
SPANS = ("fixed", "clip")
DT, SUBSTEPS = 0.1, 1
#: windows per read: four for recognition, one (the order-free read) for order
WINDOWS = {"recognition": 4, "order": 1, "sequence": 1}
#: each task's primary read: the four windows for recognition, the whole span, order-free, for the memory tasks
PRIMARY_READ = {"recognition": "windowed", "order": "pooled", "sequence": "pooled"}
TRAINED = {"gru": GRUBaseline, "tcn": TCNBaseline, "cnn": CNNBaseline,
           "transformer": TransformerBaseline, "s4d": S4DBaseline}
#: the trained baselines' recipe, paper 02's
EPOCHS, BATCH, LR, CLIP, LR_FLOOR = 30, 64, 3e-3, 1.0, 0.1
#: a trained baseline that did not cut its loss by this much failed to train,
#: which is a different claim from "cannot do the task"
HEALTHY_LOSS_DROP = 0.20
#: paper 02's network: the channel count and lattice every unsuffixed label means
CHANNELS = 4
#: a trained baseline within this share of its budget is matched; the record flags any that is not
BUDGET_TOLERANCE = 0.15
#: the model layer's names for each coupling function: (core, phase coupling law)
CORES = {"kuramoto": ("phase", "kuramoto"), "kuramoto-sakaguchi": ("phase", "sakaguchi"),
         "second-harmonic": ("phase", "harmonic2"), "winfree": ("phase", "winfree"),
         "stuart-landau": ("sl", "kuramoto"), "stuart-landau-fixed": ("sl-fixedamp", "kuramoto")}
#: the bank's label by channel count, as paper 02 names its two banks (at 16 x 16 the state-matched and the
#: width-matched bank of the 4-channel network; at every lattice, the banks of 4 and 8 channels)
BANKS = {4: "bank-state", 8: "bank-width"}


@dataclass(frozen=True)
class Arm:
    """One arm, fully specified. The label is its identity in the record."""
    kind: str                       # baseline | network | bank | trained
    coupling: str = "kuramoto"      # a key of CORES
    geometry: str = "torus"         # a key of geometries.GEOMETRIES: torus, ..., sphere, coil, cochlea, ...
    frequencies: str = "random"     # paper 03 draws them at random only, as paper 02's reference does
    restoring: float = 0.3          # the restoring strength, lambda
    ceiling: float = 1.0            # the coupling ceiling
    channels: int = CHANNELS
    coupled: bool = True            # False: the uncoupled network, its coupling kernels zeroed
    arch: str = ""                  # a trained baseline's architecture
    grid: int = GRID                # the lattice is grid x grid per channel
    bands: int = 0                  # mel bands in the front end; 0 is one per lattice row
    window: int = 0                 # the front end's analysis window in samples; 0 is paper 02's 512

    @property
    def n_bands(self) -> int:
        return self.bands or self.grid

    @property
    def n_window(self) -> int:
        return self.window or WINDOW

    def _lattice(self) -> str:
        """'' at paper 02's 16 x 16 with a band per row and its window, so paper 02's labels are unchanged."""
        out = "" if self.grid == GRID else f"-{self.grid}x{self.grid}"
        out += "" if self.n_bands == self.grid else f"-{self.n_bands}bands"
        return out + ("" if self.n_window == WINDOW else f"-w{self.n_window}")

    def label(self) -> str:
        """Paper 02's label, then the size where it is not paper 02's: `-ch<C>` for a channel count
        other than 4 (a bank says its channels in its name), `-<G>x<G>`, `-16bands` and `-w<N>`."""
        size = "" if self.channels == CHANNELS else f"-ch{self.channels}"
        if self.kind == "baseline":
            return "baseline" + self._lattice()
        if self.kind == "bank":
            return BANKS.get(self.channels, f"bank-ch{self.channels}") + self._lattice()
        if self.kind == "trained":
            # the network it is sized to; paper 02's 4 channels of 16 x 16 keep the plain label
            return f"trained-{self.arch}{size}{self._lattice()}"
        name = "coupled" if self.coupled else "uncoupled"
        return (f"{name}-{self.coupling}-{self.geometry}-{self.frequencies}"
                f"-restoring{self.restoring:g}-ceiling{self.ceiling:g}{size}{self._lattice()}")

    @property
    def states(self) -> int:
        return self.channels * self.grid * self.grid if self.kind in ("network", "bank") else 0

    @property
    def budget(self) -> int:
        """The coupled network's parameter count at this lattice and channel count: 2 * C * G * G
        (its coupling kernels and natural frequencies). The state-matched bank has it by
        construction; a trained baseline is sized to it."""
        return 2 * self.channels * self.grid * self.grid

    @property
    def uses_gain(self) -> bool:
        """Gain scales the input into a reservoir; the baseline and the trained baselines read rows as they are."""
        return self.kind in ("network", "bank")

    def as_dict(self) -> dict:
        return asdict(self)


def _statistics(task: str) -> tuple[tuple[str, int], ...]:
    """(name, windows): recognition reads four windows and the whole span; the memory tasks (order,
    sequence) only the whole span, the read that does not depend on the order of the frames."""
    return (("windowed", WINDOWS["recognition"]), ("pooled", 1)) if task == "recognition" else (("pooled", 1),)


def reads(arm: Arm, task: str) -> dict[str, list[str]]:
    """Each read a run records for an arm, as the feature blocks it concatenates.

    The memory tasks read only the whole span, the order-free read. An
    oscillator network adds each read with its rotation rates. The baseline is
    read over the arms' span and over the whole clip. A block is stored once
    however many reads use it.
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

def build_untrained(arm: Arm, gain: float, seed: int, device: str = "cpu",
                    kernel_scaling: str = "exact") -> nn.Module | None:
    """The untrained arm, built from its seed exactly as paper 02 built it.

    `kernel_scaling="cap"` rebuilds paper 02's network, whose kernels were only
    ever scaled down to the coupling ceiling; paper 03 scales every kernel to
    it exactly. At 16 x 16 and above the two are the same network, bit for bit.
    """
    if arm.kind == "baseline":
        return None
    if arm.kind == "bank":
        return LeakyBank(channels=arm.channels, grid=arm.grid, gain=gain, seed=seed).to(device)
    if arm.kind != "network":
        raise ValueError(f"{arm.kind} is not an untrained arm")
    if arm.frequencies != "random":
        raise ValueError("paper 03 runs random natural frequencies only")
    core, coupling = CORES[arm.coupling]
    torch.manual_seed(seed)              # the kernel and natural-frequency draws
    network = OscillatorField(
        channels=arm.channels, grid=arm.grid, coupling=coupling, damping=arm.restoring,
        spectral_clamp=arm.ceiling, substeps=SUBSTEPS, dt=DT, gain=gain, seed=seed, core=core,
        boundary=arm.geometry, sakaguchi_alpha=ALPHA if coupling == "sakaguchi" else 0.0,
        harmonic2_beta=BETA if coupling == "harmonic2" else 0.0, kernel_scaling=kernel_scaling)
    block = physics_block(network.core)
    with torch.no_grad():
        if not arm.coupled:
            block.kernel.zero_()
    for p in network.parameters():
        p.requires_grad_(False)
    return network.eval().to(device)


def channel(arm: Arm, model: nn.Module, c: int) -> tuple[Arm, nn.Module]:
    """Channel c of an untrained network or bank, as a one-channel arm of its own.

    Channels are independent: each has its own kernel, natural frequencies,
    starting phases (or its own leak rates and input weights), all receive
    the same input, and none acts on another. Channel c's trajectory is
    therefore the full arm's channel-c slice, which lets the streamed read
    (harness.experiment.stream) simulate a large arm one channel at a time. Every
    stored tensor is sliced along the one axis where a one-channel arm
    differs from the full one, so nothing is redrawn.
    """
    one = replace(arm, channels=1)
    if arm.channels == 1:
        return one, model
    with torch.random.fork_rng(devices=[]):
        sub = build_untrained(one, getattr(model, "gain", 0.0), 0, "cpu", _scaling(model))
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

    Paper 02's instruments. R is the global order parameter |<e^{i theta}>| of
    each channel, averaged over channels and frames. plv is each oscillator's
    phase-locking value to its own row's drive, |<e^{i(theta - phi)}>| over
    frames, averaged over oscillators, and entrained the share of oscillators
    whose value exceeds PLV_LOCK_THRESH. amplitude is the mean |z|: 1 for a
    phase core, free for a Stuart-Landau one. Diagnostics only: nothing here
    reaches the readout.
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
    states = arm.states if arm.kind in ("network", "bank") else None
    out = {"states": states, "stored_params": stored, "trained_params": trained, "budget": arm.budget}
    if arm.kind == "trained":
        out["width"] = trained_width(arm.arch, arm.grid, arm.budget)
        out["within_budget_tolerance"] = abs(trained / arm.budget - 1) <= BUDGET_TOLERANCE
    if arm.kind == "network" and not arm.coupled:
        # the zeroed kernel is still stored, but it no longer does anything
        out["effective_params"] = stored - physics_block(model.core).kernel.numel()
    return out


# ---------------------------------------------------------------------------
# Trained baselines
# ---------------------------------------------------------------------------

#: each architecture's one width knob, the smallest width it takes, and the step between widths
_WIDTH = {"gru": ("hidden", 1, 1), "tcn": ("hidden", 1, 1), "cnn": ("hidden", 1, 1),
          "transformer": ("d", 2, 2), "s4d": ("hidden", 1, 1)}
#: the CNN takes the narrowest width at or over the budget, as paper 02's does; the rest the widest at or under it
_ROUND_UP = ("cnn",)


def _kwargs(arch: str, width: int) -> dict:
    knob = _WIDTH[arch][0]
    return {knob: width, "n_states": width} if arch == "s4d" else {knob: width}


def trained_params(arch: str, rows: int, width: int) -> int:
    """Trainable parameters of a trained baseline on `rows` input rows at `width`, head excluded."""
    with torch.device("meta"):
        return sum(p.numel() for p in TRAINED[arch](grid=rows, **_kwargs(arch, width)).parameters())


def trained_width(arch: str, rows: int, budget: int) -> int:
    """The width that sizes a trained baseline to the network's parameter count.

    Each architecture has one width knob (the GRU's hidden size, the TCN's and
    the CNN's hidden channels, the transformer's model width with two heads,
    the S4D's width with as many states per channel as its width) and the
    rest of its design fixed. It takes the widest width whose parameter count
    does not exceed the budget; the CNN takes the narrowest that reaches it,
    as paper 02's CNN, one width over its budget, does. On 16 rows at 2,048
    parameters this gives paper 02's widths exactly: 18, 10, 13, 16 and 16.
    Parameter counts rise with width, so a bisection finds it.
    """
    _, lo, step = _WIDTH[arch]
    count = lambda w: trained_params(arch, rows, w)  # noqa: E731
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


def build_trained(arm: Arm) -> nn.Module:
    """A trained baseline sized to the network its label names, on that network's rows."""
    return TRAINED[arm.arch](grid=arm.grid, **_kwargs(arm.arch, trained_width(arm.arch, arm.grid, arm.budget)))


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
    Both are built on the CPU, so every device starts from the same weights,
    and trained on `device`; the clips stay on the CPU and move a batch at a
    time. Returns (backbone, head, health), on `device`.
    """
    torch.manual_seed(seed)
    backbone = build_trained(arm)
    windows = WINDOWS[task]
    with torch.no_grad():
        width = trained_features(backbone, rows[:2], tvalid[:2], windows, span).shape[1]
    # the head reads the statistics standardized, as the ridge does: unstandardized, their spread between
    # clips can be a thousandth of their size (with noise especially), and the loss never leaves chance
    head = nn.Sequential(nn.BatchNorm1d(width, affine=False), nn.Linear(width, n_classes))
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
            logits = head(trained_features(backbone, rows[idx].to(device), tvalid[idx].to(device), windows, span))
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
