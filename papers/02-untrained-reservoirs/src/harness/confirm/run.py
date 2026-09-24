"""One confirmatory run: one arm, one condition, one seed, read at every size and width.

A run streams its clips through the arm in batches, keeps each feature block
once, hands them to the shared readout, and records every cell with its
per-clip correctness where the registration asks for it. The training block
and the test block are batched separately, so a test clip's features are
computed in the same batch in every tier that uses it.

The record lives under results/confirmatory/, one file per tier, task and
drive, and never touches the exploratory record beside it. A run's identity is
derived from its specification alone, so a sweep can be stopped and restarted
and each run lands in the same place exactly once.
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

from harness.confirm import arms as am
from harness.confirm import protocol as pr
from harness.confirm import readout as ro
from harness.results import results_root

#: the registered common widths; an arm is never read wider than its native width
WIDTHS = (192, 1024, 4096)
#: the size every verdict is read at
PRIMARY_SIZE = 2048
PRIMARY_STAT = {"recognition": "windowed", "order": "pooled"}
#: clips per batch, by task and drive; the carrier runs at 16 kHz, so its batches
#: are small, and a GPU holds four times as many of its 16,000-frame trajectories
BATCH = {"recognition": 512, "order": 256, "carrier": 8, "carrier-cuda": 32}
#: the carrier path drives the field at the audio sample rate
CARRIER_RATE_HZ = 16000.0


@dataclass(frozen=True)
class Spec:
    """Everything that decides a run's numbers, and nothing else."""
    tier: str
    task: str                      # recognition | order
    drive: str                     # envelope | quadrature | carrier
    noise_db: float | None         # None is clean audio
    gain: float | None             # None for arms that read the rows as they are
    seed: int
    arm: am.Arm
    protocol: str = "A"            # A | B
    fold: int = -1                 # Becker fold, protocol B only
    pair: tuple = ()               # order task only
    sizes: tuple = (PRIMARY_SIZE,)
    widths: tuple = WIDTHS
    native_sizes: tuple = (PRIMARY_SIZE,)
    bits: str = "all"              # all | primary
    span: str = "fixed"            # fixed | clip (the exploratory per-clip span, a diagnostic)
    reads: tuple = ()              # the reads to record; empty records every read the arm has
    projection: str = "fixed"      # fixed | both: also read through the seeded projection (cells tagged
                                   # fixed, seeded, or none where the read is not projected)

    def group(self) -> str:
        name = f"{self.tier}-{self.task}-{self.drive}"
        return f"{name}-{self.arm.physics}" if self.tier == "tier2" else name

    def run_id(self) -> str:
        parts = [f"B{self.fold}" if self.protocol == "B" else "A"]
        if self.task == "order":
            parts.append(f"pair{self.pair[0]}{self.pair[1]}")
        parts.append("clean" if self.noise_db is None else f"{self.noise_db:g}db")
        if self.gain is not None:
            parts.append(f"g{self.gain:g}")
        parts += [f"s{self.seed}", self.arm.label()]
        if self.arm.kind == "ann":
            parts.append(f"n{self.sizes[0]}")
        if self.span != "fixed":
            parts.append(f"{self.span}span")
        return "/".join(parts)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["arm"] = self.arm.as_dict()
        return d


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def record_root() -> Path:
    return results_root() / "confirmatory"


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
    """What produced the numbers: library versions and the code's commit."""
    here = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=here, capture_output=True, text=True)
        return out.stdout.strip()

    return {"torch": torch.__version__, "python": sys.version.split()[0],
            "platform": platform.platform(), "machine": platform.machine(), "device": device,
            "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "."))}


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------

@dataclass
class Clips:
    """A run's clips in readout order: training (seed order), validation, test."""
    blocks: list                   # per block: a callable (start, stop) -> (rows, valid frames)
    sizes: list[int]
    labels: torch.Tensor
    layout: ro.Layout
    n_classes: int


def _order_block(bank: dict, spec: Spec, pool: torch.Tensor, n: int, code: int):
    """(rows maker, labels) for one order-task set, from its cache when there is one."""
    first, second, labels = pr.order_set(bank, pool, spec.pair, n, code)
    cache = pr.load_rows(pr.order_rows_path(spec.pair, code, spec.noise_db), n)
    if cache is not None:
        return (lambda a, b: (cache["rows"][a:b], cache["tvalid"][a:b])), labels

    def make(a, b):
        waves, lens = pr.order_clips(bank, first[a:b], second[a:b], spec.pair, code, a, spec.noise_db)
        return pr.front_end(waves, spec.drive), pr.valid_frames(lens, spec.drive)
    return make, labels


def _recognition_block(bank: dict, spec: Spec, idx: torch.Tensor):
    """Rows maker for bank clips `idx`, from the drive's cache when there is one."""
    cache = (pr.load_rows(pr.rows_path(spec.drive, spec.noise_db), len(bank["labels"]))
             if spec.drive in pr.CACHED_DRIVES else None)
    if cache is not None:
        return lambda a, b: (cache["rows"][idx[a:b]], cache["tvalid"][idx[a:b]])

    def make(a, b):
        waves, lens, _ = pr.recognition_clips(bank, idx[a:b], spec.noise_db)
        return pr.front_end(waves, spec.drive), pr.valid_frames(lens, spec.drive)
    return make


def assemble(spec: Spec, bank: dict) -> Clips:
    if spec.task == "order":
        train_pool, test_pool = pr.protocol_a(bank)
        tr, y_tr = _order_block(bank, spec, train_pool, pr.ORDER_TRAIN, spec.seed + 1)
        te, y_te = _order_block(bank, spec, test_pool, pr.ORDER_TEST, 0)
        return Clips([tr, te], [pr.ORDER_TRAIN, pr.ORDER_TEST], torch.cat((y_tr, y_te)),
                     ro.Layout(pr.ORDER_TRAIN, 0, pr.ORDER_TEST), 2)
    if spec.protocol == "B":
        parts = pr.protocol_b(bank, spec.fold)
        layout = ro.Layout(len(parts[0]), len(parts[1]), len(parts[2]))
    else:
        train_pool, test_pool = pr.protocol_a(bank)
        order = pr.training_order(train_pool, spec.seed)[:max(spec.sizes)]
        parts = (order, test_pool)
        layout = ro.Layout(len(order), 0, len(test_pool))

    labels = torch.cat([bank["labels"][p] for p in parts])
    return Clips([_recognition_block(bank, spec, p) for p in parts], [len(p) for p in parts],
                 labels, layout, am.N_CLASSES)


def batches(spec: Spec, clips: Clips, device: str = "cpu") -> Iterator[tuple[torch.Tensor, torch.Tensor, slice]]:
    """(rows, valid frames, rows' place in the readout order), block by block."""
    carrier = "carrier-cuda" if device.startswith("cuda") else "carrier"
    size = BATCH[carrier] if spec.drive == "carrier" else BATCH[spec.task]
    at = 0
    for make, n in zip(clips.blocks, clips.sizes):
        for a in range(0, n, size):
            b = min(a + size, n)
            rows, tvalid = make(a, b)
            yield rows, tvalid, slice(at + a, at + b)
        at += n


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

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


def instrument_summary(per_clip: dict[str, torch.Tensor], labels: torch.Tensor, test: slice) -> dict:
    """The coupled or uncoupled network's instruments over the test clips: mean, spread, and mean per class."""
    y = labels[test]
    out = {}
    for name, values in per_clip.items():
        v = values[test].double()
        out[name] = {"mean": v.mean().item(), "sd": v.std().item(),
                     "by_class": [v[y == k].mean().item() for k in range(int(y.max()) + 1)]}
    return out


def execute(spec: Spec, device: str = "cpu", bank: dict | None = None) -> dict:
    """Run one spec and return its record (not yet written)."""
    t0 = time.perf_counter()
    bank = pr.load_bank() if bank is None else bank
    clips = assemble(spec, bank)
    total = len(clips.labels)
    arm, timing, health, extra = spec.arm, {}, None, {}
    buffers: dict[str, torch.Tensor] = {}

    if arm.kind == "ann":
        rows, tvalid = [], []
        for r, tv, _ in batches(spec, clips):
            rows.append(r)
            tvalid.append(tv)
        rows, tvalid = torch.cat(rows), torch.cat(tvalid)
        n_tr = spec.sizes[0]
        t1 = time.perf_counter()
        backbone, head, health = am.train_ann(arm, rows[:n_tr], tvalid[:n_tr], clips.labels[:n_tr],
                                              spec.task, spec.seed, clips.n_classes, span=spec.span)
        timing["train_s"] = time.perf_counter() - t1
        with torch.no_grad():
            for a in range(0, total, BATCH[spec.task]):
                b = min(a + BATCH[spec.task], total)
                _store(buffers, am.ann_blocks(backbone, rows[a:b], tvalid[a:b], spec.task, spec.span),
                       slice(a, b), total)
            primary = buffers[PRIMARY_STAT[spec.task]]
            test = clips.layout.test
            extra["head_acc"] = (head(primary[test]).argmax(1) == clips.labels[test]).double().mean().item()
        model = backbone
    else:
        rate = CARRIER_RATE_HZ if spec.drive == "carrier" else None
        model = am.build_frozen(arm, spec.gain if spec.gain is not None else 0.0, spec.seed, device, rate)
        t1 = time.perf_counter()
        per_clip: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for rows, tvalid, where in batches(spec, clips, device):
                sig = am.frozen_signals(arm, model, rows.to(device))
                if arm.kind == "field":
                    for name, value in am.field_instruments(sig, rows.to(device), spec.drive, arm.channels).items():
                        per_clip.setdefault(name, torch.zeros(total))[where] = value.cpu()
                _store(buffers, am.frozen_features(arm, sig, tvalid.to(device), spec.task, spec.span),
                       where, total)
        timing["simulate_s"] = time.perf_counter() - t1
        if per_clip:
            extra["instruments"] = instrument_summary(per_clip, clips.labels, clips.layout.test)

    t2 = time.perf_counter()
    read_blocks = {read: [buffers[k] for k in keys] for read, keys in am.reads(arm, spec.task).items()
                   if not spec.reads or read in spec.reads}
    cells = ro.read_cells(read_blocks, clips.labels, clips.layout, spec.sizes, spec.widths,
                          spec.native_sizes, clips.n_classes, keep_bits(spec))
    if spec.projection == "both":
        # the seeded projection's cells; an unprojected cell is the same under either, so it is kept once
        native = {read: sum(b.shape[1] for b in blocks) for read, blocks in read_blocks.items()}
        seeded = ro.read_cells(read_blocks, clips.labels, clips.layout, spec.sizes, spec.widths, (),
                               clips.n_classes, keep_bits(spec), projection_seed=spec.seed)
        cells = ([{**c, "projection": "none" if c["effective_width"] == native[c["read"]] else "fixed"} for c in cells]
                 + [{**c, "projection": "seeded"} for c in seeded if c["effective_width"] != native[c["read"]]])
    timing["readout_s"] = time.perf_counter() - t2
    timing["total_s"] = time.perf_counter() - t0
    return {"spec": spec.as_dict(), "arm_meta": am.meta(arm, model),
            "native_widths": {read: sum(b.shape[1] for b in bl) for read, bl in read_blocks.items()},
            "n_test": clips.layout.n_test, "cells": cells, "health": health, **extra,
            "timing": timing, "env": environment(device)}


def run(spec: Spec, device: str = "cpu", threads: int = 1) -> str:
    """Execute and record one spec; returns where it landed."""
    torch.set_num_threads(threads)
    write(spec, execute(spec, device))
    return f"{spec.group()}/{spec.run_id()}"
