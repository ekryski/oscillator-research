"""One run: one arm, one condition, one seed, read at every width.

A run streams its clips through the arm in batches, keeps each feature block
once, hands them to the shared readout, and records every cell with its
per-clip correctness where the plan asks for it. The training block and the
test block are batched separately, so a test clip's features are computed in
the same batch in every tier that uses it. An arm with more than
`stream.STREAM_STATES` states is instead read channel by channel
(`harness.experiment.stream`).

The record lives under results/, one file per tier, task, input
pathway and lattice (and coupling function in the design tiers). A run's
identity is derived from its specification alone, so a sweep can be stopped
and restarted and each run lands in the same place exactly once. Run
identities have paper 02's form, so a paper 02 cell and the paper 03 cell it
stands for share one.
"""

from __future__ import annotations

import fcntl
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from harness.experiment import arms as am
from harness.experiment import protocol as pr
from harness.experiment import readout as ro
from harness.experiment import stream as st
from harness.utils.device import describe, resolve
from harness.utils.paths import results_root

#: paper 02's common widths; an arm is never read wider than its native width
WIDTHS = (192, 1024, 4096)
#: the size every verdict is read at
PRIMARY_SIZE = 2048
PRIMARY_STAT = am.PRIMARY_READ
#: clips per batch, by task and drive; the carrier runs at 16 kHz, so its batches
#: are small, and a GPU (CUDA or MPS) holds four times as many of its 16,000-frame trajectories
BATCH = {"recognition": 512, "order": 256, "sequence": 128, "carrier": 8, "carrier-gpu": 32}
#: a batch holds this many states' trajectories at the sizes above; larger arms (or channels) shrink it
BATCH_STATES = 1024
#: above this many states a batch shrinks in proportion (paper 02's rule, unchanged up to 4,096 states)
LARGE_STATES = 4096
#: the fewest clips a batch shrinks to
MIN_BATCH = 16
#: the carrier path drives the network at the audio sample rate
CARRIER_RATE_HZ = 16000.0


@dataclass(frozen=True)
class Spec:
    """Everything that decides a run's numbers, and nothing else."""
    tier: str
    task: str                      # recognition | order | sequence
    drive: str                     # envelope | quadrature | carrier: the input pathway
    noise_db: float | None         # None is clean audio
    gain: float | None             # None for arms that read the rows as they are
    seed: int
    arm: am.Arm
    protocol: str = "A"            # paper 02's Protocol A, the only one paper 03 uses
    pair: tuple = ()               # the order task's digit pair
    length: int = 0                # the digit-sequence task's number of digits
    sizes: tuple = (PRIMARY_SIZE,)
    widths: tuple = WIDTHS
    native_sizes: tuple = (PRIMARY_SIZE,)
    bits: str = "primary"          # all | primary
    span: str = "fixed"            # fixed | clip (the exploratory per-clip span, a diagnostic)
    reads: tuple = ()              # the reads to record; empty records every read the arm has

    def group(self) -> str:
        """The record file: tier, task (unless recognition), pathway and lattice, and the coupling
        function in the design tiers."""
        a = self.arm
        task = "" if self.task == "recognition" else f"-{self.task}"
        name = f"{self.tier}{task}-{self.drive}-{a.grid}x{a.grid}"
        return f"{name}-{a.physics}" if self.tier.startswith("design") and a.kind == "field" else name

    def run_id(self) -> str:
        parts = [self.protocol]
        if self.task == "order":
            parts.append(f"pair{self.pair[0]}{self.pair[1]}")
        if self.task == "sequence":
            parts.append(f"seq{self.length}")
        parts.append("clean" if self.noise_db is None else f"{self.noise_db:g}db")
        if self.gain is not None:
            parts.append(f"g{self.gain:g}")
        parts += [f"s{self.seed}", self.arm.label()]
        if self.arm.kind == "ann":
            parts.append(f"n{self.sizes[0]}")
        if self.span != "fixed":
            parts.append(f"{self.span}span")
        return "/".join(parts)

    @property
    def streamed(self) -> bool:
        """Read channel by channel: an untrained network or bank above STREAM_STATES states."""
        return self.arm.kind in ("field", "bank") and self.arm.states > st.STREAM_STATES

    def as_dict(self) -> dict:
        d = asdict(self)
        d["arm"] = self.arm.as_dict()
        return d


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def record_root() -> Path:
    """Where the record lives, read per call: OSC_RESULTS_DIR redirects it, e.g. to reproduce into a fresh tree."""
    return results_root()


def group_path(group: str) -> Path:
    return record_root() / f"{group}.json"


def load_group(group: str) -> dict:
    path = group_path(group)
    return json.loads(path.read_text()) if path.exists() else {"group": group, "runs": {}}


def recorded_ids(group: str) -> set[str]:
    return set(load_group(group)["runs"])


def write(spec: Spec, record: dict) -> None:
    """Merge one run into its group file: exclusive lock, merge, atomic swap."""
    path = group_path(spec.group())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".json.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = load_group(spec.group())
            data["runs"][spec.run_id()] = record
            data["runs"] = dict(sorted(data["runs"].items()))
            tmp = tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False)
            with tmp:
                json.dump(data, tmp, indent=1)
                tmp.write("\n")
            os.replace(tmp.name, path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def environment(device: str) -> dict:
    """What produced the numbers: library versions, the device (and which GPU), and the code's commit."""
    here = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=here, capture_output=True, text=True)
        return out.stdout.strip()

    return {"torch": torch.__version__, "python": sys.version.split()[0],
            "platform": platform.platform(), "machine": platform.machine(), **describe(device),
            "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "."))}


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------

@dataclass
class Clips:
    """A run's clips in readout order: training (seed order), then test.

    `labels` is [clips] for recognition and the order task, and [clips, positions]
    for the digit-sequence task, one 10-way label per position."""
    blocks: list                   # per block: a callable (start, stop) -> (rows, valid frames)
    sizes: list[int]
    labels: torch.Tensor
    layout: ro.Layout
    n_classes: int


def _cached_or_built(spec: Spec, path, n: int, build):
    """Rows maker for a joined-clip set: from its cache when there is one, else built batch by batch."""
    bands, grid, window = spec.arm.n_bands, spec.arm.grid, spec.arm.n_window
    cache = pr.load_rows(path, n) if window == am.WINDOW and spec.drive == "envelope" else None
    if cache is not None:
        return lambda a, b: (pr.to_rows(cache["rows"][a:b], grid), cache["tvalid"][a:b])

    def make(a, b):
        waves, lens = build(a, b)
        return pr.to_rows(pr.front_end(waves, spec.drive, bands, window), grid), pr.valid_frames(lens, spec.drive)
    return make


def _order_block(bank: dict, spec: Spec, pool: torch.Tensor, n: int, code: int):
    """(rows maker, labels) for one order-task set, as paper 02 built it."""
    first, second, labels = pr.order_set(bank, pool, spec.pair, n, code)
    path = pr.order_rows_path(spec.pair, code, spec.noise_db, spec.arm.n_bands)
    return _cached_or_built(spec, path, n, lambda a, b: pr.order_clips(
        bank, first[a:b], second[a:b], spec.pair, code, a, spec.noise_db)), labels


def _sequence_block(bank: dict, spec: Spec, pool: torch.Tensor, n: int, code: int):
    """(rows maker, digits [n, length]) for one digit-sequence set."""
    idx, digits = pr.sequence_set(bank, pool, spec.length, n, code)
    path = pr.sequence_rows_path(spec.length, code, spec.noise_db, spec.arm.n_bands)
    return _cached_or_built(spec, path, n, lambda a, b: pr.sequence_clips(
        bank, idx[a:b], spec.length, code, a, spec.noise_db)), digits


def _recognition_block(bank: dict, spec: Spec, idx: torch.Tensor):
    """Rows maker for bank clips `idx`, from the pathway's cache when there is one."""
    bands, grid, window = spec.arm.n_bands, spec.arm.grid, spec.arm.n_window
    cache = (pr.load_rows(pr.rows_path(spec.drive, spec.noise_db, bands, window), len(bank["labels"]))
             if spec.drive in pr.CACHED_DRIVES else None)
    if cache is not None:
        return lambda a, b: (pr.to_rows(cache["rows"][idx[a:b]], grid), cache["tvalid"][idx[a:b]])

    def make(a, b):
        waves, lens, _ = pr.recognition_clips(bank, idx[a:b], spec.noise_db)
        rows = pr.front_end(waves, spec.drive, bands, window)
        return pr.to_rows(rows, grid), pr.valid_frames(lens, spec.drive)
    return make


def assemble(spec: Spec, bank: dict) -> Clips:
    if spec.protocol != "A":
        raise ValueError("paper 03 reads Protocol A only")
    train_pool, test_pool = pr.protocol_a(bank)
    if spec.task in ("order", "sequence"):
        if spec.drive != "envelope":
            raise ValueError("the memory tasks run on the band-energy pathway")
        block, n_tr, n_te = ((_order_block, pr.ORDER_TRAIN, pr.ORDER_TEST) if spec.task == "order"
                             else (_sequence_block, pr.SEQUENCE_TRAIN, pr.SEQUENCE_TEST))
        tr, y_tr = block(bank, spec, train_pool, n_tr, spec.seed + 1)
        te, y_te = block(bank, spec, test_pool, n_te, 0)
        classes = 2 if spec.task == "order" else pr.DIGIT_CHOICES
        return Clips([tr, te], [n_tr, n_te], torch.cat((y_tr, y_te)), ro.Layout(n_tr, 0, n_te), classes)
    if spec.task != "recognition":
        raise ValueError(f"unknown task '{spec.task}'")
    order = pr.training_order(train_pool, spec.seed)[:max(spec.sizes)]
    parts = (order, test_pool)
    layout = ro.Layout(len(order), 0, len(test_pool))
    labels = torch.cat([bank["labels"][p] for p in parts])
    return Clips([_recognition_block(bank, spec, p) for p in parts], [len(p) for p in parts],
                 labels, layout, am.N_CLASSES)


def batch_size(spec: Spec, device: str = "cpu", states: int | None = None) -> int:
    """Clips per batch: paper 02's sizes, shrunk in proportion above LARGE_STATES states.

    `states` is what one pass simulates: the whole arm, or one channel of a streamed arm.
    """
    carrier = "carrier-gpu" if device.startswith(("cuda", "mps")) else "carrier"
    size = BATCH[carrier] if spec.drive == "carrier" else BATCH[spec.task]
    states = spec.arm.states if states is None else states
    if spec.drive != "carrier" and states > LARGE_STATES:
        size = max(MIN_BATCH, size * LARGE_STATES // states)      # bound a batch's trajectory memory
    if spec.drive == "carrier" and states > BATCH_STATES:
        size = max(1, size * BATCH_STATES // states)
    return size


def batches(spec: Spec, clips: Clips, device: str = "cpu") -> Iterator[tuple[torch.Tensor, torch.Tensor, slice]]:
    """(rows, valid frames, rows' place in the readout order), block by block."""
    size = batch_size(spec, device)
    at = 0
    for make, n in zip(clips.blocks, clips.sizes, strict=True):
        for a in range(0, n, size):
            b = min(a + size, n)
            rows, tvalid = make(a, b)
            yield rows, tvalid, slice(at + a, at + b)
        at += n


def channel_batches(spec: Spec, clips: Clips, n_train: int, device: str = "cpu"):
    """For the streamed read: `blocks(c, part)` yields one channel's batches of the first `n_train`
    training clips (part 0) or of the test clips (part 1), each slice relative to its part."""
    size = batch_size(spec, device, states=spec.arm.grid * spec.arm.grid)

    def blocks(c: int, part: int):
        make, n = clips.blocks[part], (n_train if part == 0 else clips.sizes[part])
        for a in range(0, n, size):
            b = min(a + size, n)
            rows, tvalid = make(a, b)
            yield rows, tvalid, slice(a, b)
    return blocks


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def readout_device(device: str) -> str:
    """Where the readout's projection runs: on the run's GPU if it has one. The ridge is always CPU float64."""
    return device if device.startswith(("cuda", "mps")) else "cpu"


def keep_bits(spec: Spec):
    if spec.bits == "all":
        return lambda read, width, n: True
    primary = PRIMARY_STAT[spec.task]
    return lambda read, width, n: n == PRIMARY_SIZE and read.split("@")[0] == primary


def _store(buffers: dict, feats: dict, where: slice, total: int) -> None:
    for key, value in feats.items():
        if key not in buffers:
            buffers[key] = torch.zeros(total, value.shape[1])
        buffers[key][where] = value.cpu()


def instrument_summary(per_clip: dict[str, torch.Tensor], labels: torch.Tensor) -> dict:
    """A network's instruments over the test clips: mean, spread, and (with one label per clip) the mean
    per class."""
    out = {}
    for name, v in per_clip.items():
        v = v.double()
        out[name] = {"mean": v.mean().item(), "sd": v.std().item()}
        if labels.dim() == 1:
            out[name]["by_class"] = [v[labels == k].mean().item() for k in range(int(labels.max()) + 1)]
    return out


def execute(spec: Spec, device: str = "cpu", bank: dict | None = None) -> dict:
    """Run one spec and return its record (not yet written). `device` may be "auto" (harness.utils.device)."""
    t0 = time.perf_counter()
    device = resolve(device)
    bank = pr.load_bank() if bank is None else bank
    clips = assemble(spec, bank)
    total = len(clips.labels)
    arm, timing, health, extra = spec.arm, {}, None, {}
    buffers: dict[str, torch.Tensor] = {}

    if arm.kind == "ann":
        if clips.labels.dim() != 1:
            raise ValueError("the trained baselines are not run on the digit-sequence task")
        rows, tvalid = [], []
        for r, tv, _ in batches(spec, clips):
            rows.append(r)
            tvalid.append(tv)
        rows, tvalid = torch.cat(rows), torch.cat(tvalid)
        n_tr = spec.sizes[0]
        t1 = time.perf_counter()
        backbone, head, health = am.train_ann(arm, rows[:n_tr], tvalid[:n_tr], clips.labels[:n_tr],
                                              spec.task, spec.seed, clips.n_classes, span=spec.span, device=device)
        timing["train_s"] = time.perf_counter() - t1
        with torch.no_grad():
            for a in range(0, total, BATCH[spec.task]):
                b = min(a + BATCH[spec.task], total)
                _store(buffers, am.ann_blocks(backbone, rows[a:b].to(device), tvalid[a:b].to(device), spec.task,
                                              spec.span), slice(a, b), total)
            primary = buffers[PRIMARY_STAT[spec.task]]
            test = clips.layout.test
            predicted = head(primary[test].to(device)).argmax(1).cpu()
            extra["head_acc"] = (predicted == clips.labels[test]).double().mean().item()
        model = backbone
    else:
        rate = CARRIER_RATE_HZ if spec.drive == "carrier" else None
        model = am.build_frozen(arm, spec.gain if spec.gain is not None else 0.0, spec.seed, device, rate)
        if spec.streamed:
            return _execute_streamed(spec, clips, model, device, t0)
        # store only the blocks the recorded reads use: a large arm's unread blocks run to gigabytes
        keep = {k for read, keys in am.reads(arm, spec.task).items() if not spec.reads or read in spec.reads
                for k in keys}
        t1 = time.perf_counter()
        per_clip: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for rows, tvalid, where in batches(spec, clips, device):
                sig = am.frozen_signals(arm, model, rows.to(device))
                if arm.kind == "field":
                    for name, v in am.field_instruments(sig, rows.to(device), spec.drive, arm.channels,
                                                        arm.grid).items():
                        per_clip.setdefault(name, torch.zeros(total))[where] = v.cpu()
                feats = am.frozen_features(arm, sig, tvalid.to(device), spec.task, spec.span)
                _store(buffers, {k: v for k, v in feats.items() if k in keep}, where, total)
        timing["simulate_s"] = time.perf_counter() - t1
        if per_clip:
            test = clips.layout.test
            extra["instruments"] = instrument_summary({k: v[test] for k, v in per_clip.items()},
                                                      clips.labels[test])

    t2 = time.perf_counter()
    read_blocks = {read: [buffers[k] for k in keys] for read, keys in am.reads(arm, spec.task).items()
                   if not spec.reads or read in spec.reads}
    cells = ro.read_cells(read_blocks, clips.labels, clips.layout, spec.sizes, spec.widths,
                          spec.native_sizes, clips.n_classes, keep_bits(spec), seed=spec.seed,
                          device=readout_device(device))
    timing["readout_s"] = time.perf_counter() - t2
    timing["total_s"] = time.perf_counter() - t0
    return {"spec": spec.as_dict(), "arm_meta": am.meta(arm, model), "read": "in memory",
            "native_widths": {read: sum(b.shape[1] for b in bl) for read, bl in read_blocks.items()},
            "n_test": clips.layout.n_test, "cells": cells, "health": health, **extra,
            "timing": timing, "env": environment(device)}


def _execute_streamed(spec: Spec, clips: Clips, model: torch.nn.Module, device: str, t0: float) -> dict:
    """An arm too large to hold, read channel by channel (harness.experiment.stream), one read at a time.

    Each read simulates the channels again, so that only one read's features
    and matrices are held at once; a network's instruments are recorded on
    the first read's pass.
    """
    own = am.reads(spec.arm, spec.task)
    reads = list(spec.reads) if spec.reads else list(own)
    if (len(spec.sizes) != 1 or spec.native_sizes or spec.span != "fixed"
            or any(r not in own or "@" in r for r in reads)):
        raise ValueError("the streamed read fits one training size, the fixed span and the arm's own reads, "
                         "with no native cell")
    n = spec.sizes[0]
    labels = torch.cat((clips.labels[:n], clips.labels[clips.layout.test]))
    layout = ro.Layout(n, 0, clips.layout.n_test)
    positions = [None] if labels.dim() == 1 else list(range(labels.shape[1]))
    cells, natives, extra, simulate_s, readout_s = [], {}, {}, 0.0, 0.0
    for k, read in enumerate(reads):
        t1 = time.perf_counter()
        with torch.no_grad():
            projected, native, instruments = st.streamed_read(
                spec.arm, model, channel_batches(spec, clips, n, device), n, clips.layout.n_test, spec.widths,
                spec.task, spec.seed, device, read=read, drive=spec.drive if k == 0 else None)
        t2 = time.perf_counter()
        if max(spec.widths) >= native:
            raise ValueError(f"a streamed read keeps no unprojected features, so every width must be below the "
                             f"arm's {native:,}; got {spec.widths}")
        natives[read] = native
        wanted = ro.wanted_widths(spec.widths, native, False)
        for projection, (p_tr, p_te) in projected.items():
            for p in positions:
                y = labels if p is None else labels[:, p]
                fitted = ro.fit_widths(read, n, wanted, native, (p_tr, p_te, None), None, y, layout,
                                       clips.n_classes, keep_bits(spec), projection)
                cells += fitted if p is None else [{**c, "position": p} for c in fitted]
        if instruments:
            extra["instruments"] = instrument_summary(instruments, clips.labels[clips.layout.test])
        simulate_s += t2 - t1
        readout_s += time.perf_counter() - t2
    timing = {"simulate_and_project_s": simulate_s, "readout_s": readout_s, "total_s": time.perf_counter() - t0}
    return {"spec": spec.as_dict(), "arm_meta": am.meta(spec.arm, model), "read": "streamed by channel",
            "native_widths": natives, "n_test": clips.layout.n_test, "cells": cells, "health": None, **extra,
            "timing": timing, "env": environment(device)}


def run(spec: Spec, device: str = "cpu", threads: int = 1) -> str:
    """Execute and record one spec; returns where it landed."""
    torch.set_num_threads(threads)
    write(spec, execute(spec, device))
    return f"{spec.group()}/{spec.run_id()}"
