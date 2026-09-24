"""Export the arms, readouts and clips behind the interactive post.

    uv run python scripts/export_web_demo.py --out <website>/public/untrained            # everything
    uv run python scripts/export_web_demo.py --out <dir> --part base --part envelope     # some of it
    uv run python scripts/export_web_demo.py --out <dir> --smoke                         # a fast dry run
    uv run python scripts/export_web_demo.py --out <dir> --record-only                   # refresh the record's numbers

The post runs paper 02's arms in the browser, one clip at a time. Everything it
needs comes from here, and all of it is computed by the harness's own code:

    manifest.json        the front end's constants, the seed-0 physics, the
                         clips, and one entry per config: which readout, the
                         record's accuracies for that cell, and reference
                         logits the browser checks itself against
    readouts/<id>.bin    one fitted readout per config: the first standardization's
                         scales, its means folded into one shift per projected
                         feature, the second standardization and the ridge
    projections/p<D>.bin the readout's fixed projection for native width D,
                         its first 192 columns (float16)
    nets/<id>.bin        a trained baseline's weights (float32)
    audio/*.wav          the demo clips, as the bank stores them
    audio/noise.bin      the unit white noise the harness adds to each demo
                         clip at 0 and +5 dB (int16, / 4096)
    record.json          the recorded recognition cells of the gate and Tier 1,
                         every read, size and width, per seed: the numbers the
                         post quotes, copied, never recomputed

A config is one arm on one pathway at one registered input gain and noise
level, at seed 0. Its readout is the registered primary cell: the four-window
read, the first 2,048 clips of seed 0's training order, width 192. It is fitted
exactly as `harness.confirm.readout.read_cells` fits it, and the export checks
the fit against the record: the fitted readout's accuracy on the 6,000 test
clips must equal the recorded seed-0 accuracy wherever the record has that cell.
It then scores the test set again the way the browser will (projection rounded
to float16) and records how many predictions agree.

Nothing here is registered and nothing here writes to the record.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.confirm import arms as am
from harness.confirm import plan as pl
from harness.confirm import protocol as pr
from harness.confirm import readout as ro
from harness.confirm import run as rn
from harness.confirm import terms
from harness.measurement.features import MIN_FRAMES_PER_WINDOW, projection_matrix
from harness.models.field import physics_block
from harness.models.geometries import build_geometry
from harness.stimuli import frontend as fe
from harness.stimuli.filterbank import band_edges
from harness.sweep import ALPHA, BETA
from harness.utils.constants import WARMUP_FRAMES

SEED = 0
WIDTH = 192
SIZE = rn.PRIMARY_SIZE
#: the demo clips: two per digit, from the test speakers, repetition 0
DEMO_SPEAKERS = lambda d: (49 + d, 59 - d)  # noqa: E731
DEMO_REP = 0
NOISE_I16_SCALE = 4096.0
CARRIER_GAIN = pl.CARRIER_GAIN
GAINS = pl.GAINS
PATHWAY_NOISES = {"envelope": (None, 0.0, 5.0), "quadrature": (0.0, 5.0), "carrier": (0.0,)}
DESIGN_NOISES = pl.DESIGN_NOISES
SHAPES = pl.SHAPES
FAMILIES = pl.PHASE_FAMILIES + pl.AMPLITUDE_FAMILIES
PARTS = ("base", "record", "envelope", "design", "quadrature", "carrier")


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """One arm on one pathway, at one registered gain and noise level."""
    part: str
    drive: str
    arm: am.Arm
    gain: float | None
    noise_db: float | None
    tier: str                       # the tier whose record holds this cell, or "" when none does
    reads: tuple = ("windowed",)

    def cid(self, read: str = "windowed") -> str:
        noise = "clean" if self.noise_db is None else f"{self.noise_db:g}db"
        gain = "" if self.gain is None else f"-g{self.gain:g}"
        suffix = "" if read == "windowed" else "-wholeclip"
        return f"{self.drive}-{self.arm.label()}{gain}-{noise}{suffix}".replace(".", "p")

    def spec(self, span: str = "fixed") -> rn.Spec:
        return rn.Spec(self.tier or "export", "recognition", self.drive, self.noise_db, self.gain, SEED,
                       self.arm, sizes=(SIZE,), native_sizes=(), span=span)


def field_arm(physics: str = "kuramoto", boundary: str = "torus", severed: bool = False) -> am.Arm:
    return am.Arm("field", physics=physics, boundary=boundary, severed=severed)


def configs(parts: tuple[str, ...]) -> list[Config]:
    out: list[Config] = []
    if "envelope" in parts:
        for noise in PATHWAY_NOISES["envelope"]:
            out.append(Config("envelope", "envelope", pl.FLOOR, None, noise, "tier1",
                              reads=("windowed@wholeclip", "windowed")))
            for gain in GAINS:
                for arm in pl.FROZEN:
                    out.append(Config("envelope", "envelope", arm, gain, noise, "tier1"))
            for arch in pl.ANN_ARCHS:
                out.append(Config("envelope", "envelope", am.Arm("ann", arch=arch), None, noise, "tier1"))
    if "design" in parts:
        for family in FAMILIES:
            for shape in (("torus",) if family in pl.AMPLITUDE_FAMILIES else SHAPES):
                arm = field_arm(family, shape)
                if arm == pl.FIELD:
                    continue
                for noise in DESIGN_NOISES:
                    for gain in GAINS:
                        out.append(Config("design", "envelope", arm, gain, noise, "tier2"))
    if "quadrature" in parts:
        for noise in PATHWAY_NOISES["quadrature"]:
            for gain in GAINS:
                for family in pl.PHASE_FAMILIES:
                    out.append(Config("quadrature", "quadrature", field_arm(family), gain, noise, "tier3"))
                out.append(Config("quadrature", "quadrature", pl.SEVERED, gain, noise, ""))
    if "carrier" in parts:
        out.append(Config("carrier", "carrier", pl.BANK_A, CARRIER_GAIN, 0.0, "tier3"))
        out.append(Config("carrier", "carrier", pl.SEVERED, CARRIER_GAIN, 0.0, ""))
        for family in FAMILIES:
            out.append(Config("carrier", "carrier", field_arm(family), CARRIER_GAIN, 0.0, "tier3"))
    return out


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

_GROUPS: dict[str, dict] = {}


def _group(name: str) -> dict:
    if name not in _GROUPS:
        path = rn.group_path(name)
        _GROUPS[name] = json.loads(path.read_text())["runs"] if path.exists() else {}
    return _GROUPS[name]


def record_cell(cfg: Config, read: str) -> dict | None:
    """The record's primary cell for this config, per seed: {seed: accuracy}, or None if unrun."""
    if not cfg.tier:
        return None
    group = f"{cfg.tier}-recognition-{cfg.drive}"
    if cfg.tier == "tier2":
        group += f"-{cfg.arm.physics}"
    want = cfg.arm.as_dict()
    accs: dict[int, float] = {}
    for run in _group(group).values():
        s = run["spec"]
        if (s["arm"] != want or s["noise_db"] != cfg.noise_db or s["gain"] != cfg.gain
                or s["protocol"] != "A" or s["span"] != "fixed"):
            continue
        for cell in run["cells"]:
            if cell["read"] == read and cell["n_train"] == SIZE and cell["width"] == WIDTH:
                accs[s["seed"]] = cell["acc"]
    if not accs:
        return None
    vals = list(accs.values())
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return {"seeds": {str(k): v for k, v in sorted(accs.items())}, "mean": mean, "sd": sd, "n": len(vals)}


def export_record(out: Path) -> dict:
    """The gate's and Tier 1's recognition cells, one row per cell with every seed's accuracy."""
    tables = {}
    for tier, group in (("gate", "gate-recognition-envelope"), ("tier1", "tier1-recognition-envelope")):
        rows: dict[tuple, dict] = {}
        for run in _group(group).values():
            s = run["spec"]
            if s["protocol"] != "A":
                continue
            arm = am.Arm(**s["arm"])
            for cell in run["cells"]:
                key = (arm.label(), s["noise_db"], s["gain"], s["span"], cell["read"], cell["n_train"],
                       cell["width"])
                rows.setdefault(key, {})[s["seed"]] = round(cell["acc"], 6)
        tables[tier] = [[*k, [v[i] for i in sorted(v)]] for k, v in sorted(rows.items(), key=str)]
    names = sorted({r[0] for t in tables.values() for r in t})
    record = {"columns": ["arm", "noise_db", "gain", "span", "read", "n_train", "width", "accs"],
              "names": {n: terms.arm(n) for n in names}, **tables,
              "copied": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "record.json").write_text(json.dumps(record, separators=(",", ":")) + "\n")
    return {"file": "record.json", "rows": {k: len(v) for k, v in tables.items()}}


# ---------------------------------------------------------------------------
# Features, exactly as a registered run computes them
# ---------------------------------------------------------------------------

def arm_features(cfg: Config, bank: dict, clips: rn.Clips, reads: tuple[str, ...], train_only: bool):
    """(read blocks, model, net state) over the run's clips, as `run.execute` builds them."""
    spec = cfg.spec()
    arm = cfg.arm
    n_train = clips.layout.n_train
    total = n_train if train_only else len(clips.labels)
    buffers: dict[str, torch.Tensor] = {}

    def keep(feats: dict, where: slice) -> None:
        if where.start >= total:
            return
        stop = min(where.stop, total)
        rn._store(buffers, {k: v[: stop - where.start] for k, v in feats.items()},
                  slice(where.start, stop), total)

    net = None
    if arm.kind == "ann":
        rows, tvalid = [], []
        for r, tv, _ in rn.batches(spec, clips):
            rows.append(r)
            tvalid.append(tv)
        rows, tvalid = torch.cat(rows), torch.cat(tvalid)
        backbone, _, health = am.train_ann(arm, rows[:n_train], tvalid[:n_train], clips.labels[:n_train],
                                           "recognition", SEED, clips.n_classes)
        with torch.no_grad():
            step = rn.BATCH["recognition"]
            for a in range(0, total, step):
                b = min(a + step, total)
                keep(am.ann_blocks(backbone, rows[a:b], tvalid[a:b], "recognition"), slice(a, b))
        model, net = backbone, {"health": health}
    else:
        rate = rn.CARRIER_RATE_HZ if cfg.drive == "carrier" else None
        model = am.build_frozen(arm, cfg.gain if cfg.gain is not None else 0.0, SEED, "cpu", rate)
        with torch.no_grad():
            for rows, tvalid, where in rn.batches(spec, clips):
                if where.start >= total:
                    break
                sig = am.frozen_signals(arm, model, rows)
                keep(am.frozen_features(arm, sig, tvalid, "recognition"), where)
    blocks = {read: [buffers[k] for k in keys] for read, keys in am.reads(arm, "recognition").items()
              if read in reads}
    return blocks, model, net


# ---------------------------------------------------------------------------
# The readout, fitted as read_cells fits the primary cell, with its weights kept
# ---------------------------------------------------------------------------

def fit_readout(blocks: list[torch.Tensor], labels: torch.Tensor, layout: ro.Layout,
                exact: bool, with_test: bool) -> dict:
    """The primary cell's readout: standardize, project to 192, ridge; returns its parts."""
    n = layout.n_train
    rows_tr = slice(0, n)
    native = sum(b.shape[1] for b in blocks)
    mean1, sd1 = ro._block_stats(blocks, rows_tr)
    projected = WIDTH < native
    if projected:
        # read_cells projects once to the widest registered width below native and
        # slices; doing the same keeps the float32 sums in the same order
        top = max(w for w in rn.WIDTHS if w < native) if exact else WIDTH
        x_tr = ro._projected(blocks, rows_tr, mean1, sd1, top)[:, :WIDTH]
        x_te = ro._projected(blocks, layout.test, mean1, sd1, top)[:, :WIDTH] if with_test else None
    else:
        x_tr = ro._native(blocks, rows_tr)
        x_te = ro._native(blocks, layout.test) if with_test else None
    y_tr = labels[rows_tr]
    y_te = labels[layout.test]

    # ro.ridge, step for step, keeping the weights it discards
    mean2, sd2 = ro._stats(x_tr)
    y1h = F.one_hot(y_tr, am.N_CLASSES).double()
    n_fit = n - max(1, int(ro.VAL_FRAC * n))
    xv, yv = x_tr[n_fit:], y_tr[n_fit:]
    primal = x_tr.shape[1] + 1 <= n_fit
    if primal:
        a_fit, b_fit = ro._gram(x_tr, mean2, sd2, y1h, slice(0, n_fit))
        fit = lambda lam: ro._solve_primal(a_fit, b_fit, lam, n_fit)  # noqa: E731
    else:
        z_fit = ro._design(x_tr[:n_fit], mean2, sd2)
        k_fit = z_fit @ z_fit.T
        fit = lambda lam: ro._solve_dual(z_fit, k_fit, y1h[:n_fit], lam)  # noqa: E731
    best_lam, best_acc = ro.LAMBDAS[0], -1.0
    for lam in ro.LAMBDAS:
        acc = (ro._predict(xv, mean2, sd2, fit(lam)) == yv).double().mean().item()
        if acc > best_acc:
            best_lam, best_acc = lam, acc
    if primal:
        a_v, b_v = ro._gram(x_tr, mean2, sd2, y1h, slice(n_fit, n))
        w = ro._solve_primal(a_fit + a_v, b_fit + b_v, best_lam, n)
    else:
        z = ro._design(x_tr, mean2, sd2)
        w = ro._solve_dual(z, z @ z.T, y1h, best_lam)

    out = {"native": native, "projected": projected, "mean1": mean1, "sd1": sd1,
           "mean2": mean2, "sd2": sd2, "w": w, "lam": best_lam, "val_acc": best_acc}
    if with_test:
        # the registered path's own answer, to hold this refit to
        ref = ro.ridge(x_tr, y_tr, x_te, y_te, am.N_CLASSES)
        pred = ro._predict(x_te, mean2, sd2, w)
        assert ref["lam"] == best_lam, (ref["lam"], best_lam)
        out["acc"] = (pred == y_te).double().mean().item()
        out["ref_acc"] = ref["acc"]
        out["pred"] = pred
    return out


def browser_readout(r: dict, p16: torch.Tensor | None) -> dict:
    """The readout as the page stores it. The first standardization's mean is folded
    into one shift per projected feature, computed against the same float16
    projection the page multiplies by, so the page's
    (x * inv1) @ P - shift equals ((x - mean1) * inv1) @ P up to float64 rounding."""
    out = {"mean2": r["mean2"].float(), "sd2": r["sd2"].float(), "weight": r["w"][:-1].float(),
           "bias": r["w"][-1].float()}
    if r["projected"]:
        inv1 = torch.where(r["sd1"] > ro.MIN_SD, 1.0 / r["sd1"], torch.zeros_like(r["sd1"])).float()
        out["inv1"] = inv1
        out["shift"] = (r["mean1"].float().double() * inv1.double()) @ p16.double()
    return out


def browser_logits(blocks: list[torch.Tensor], rows: slice, b: dict, p16: torch.Tensor | None) -> torch.Tensor:
    """Logits computed the way the page computes them, from what the page stores, in float64."""
    m2, s2 = b["mean2"].double(), b["sd2"].double()
    w, bias = b["weight"].double(), b["bias"].double()
    out = []
    for a in range(rows.start or 0, rows.stop, 1024):
        stop = min(a + 1024, rows.stop)
        x = torch.cat([blk[a:stop] for blk in blocks], dim=1).double()
        if "inv1" in b:
            x = (x * b["inv1"].double()) @ p16.double() - b["shift"]
        out.append(((x - m2) / s2) @ w + bias)
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def b64(t: torch.Tensor | np.ndarray, dtype=np.float32) -> dict:
    a = np.ascontiguousarray(np.asarray(t, dtype=dtype))
    return {"shape": list(a.shape), "dtype": np.dtype(dtype).name, "b64": base64.b64encode(a.tobytes()).decode()}


def write_bin(path: Path, parts: list[tuple[str, torch.Tensor, type]]) -> dict:
    """Concatenate arrays into one file; returns {name: [byte offset, count, dtype]}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    layout, at = {}, 0
    with open(path, "wb") as f:
        for name, t, dtype in parts:
            a = np.ascontiguousarray(np.asarray(t.detach().cpu() if torch.is_tensor(t) else t, dtype=dtype))
            layout[name] = [at, int(a.size), np.dtype(dtype).name, list(a.shape)]
            f.write(a.tobytes())
            at += a.nbytes
            pad = (-at) % 8                  # keep every array aligned for typed-array views
            f.write(b"\0" * pad)
            at += pad
    return layout


def write_wav(path: Path, pcm: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.astype("<i2").tobytes())


def git_commit() -> str:
    here = Path(__file__).resolve().parent
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True)
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# The shared parts: front end, physics, clips
# ---------------------------------------------------------------------------

def demo_indices(bank: dict) -> list[int]:
    out = []
    for d in range(10):
        for spk in DEMO_SPEAKERS(d):
            idx = ((bank["speakers"] == spk) & (bank["labels"] == d) & (bank["reps"] == DEMO_REP)).nonzero()
            out.append(int(idx[0, 0]))
    return out


def noise_level_code(db: float) -> int:
    return int(round(db * pr.NOISE_LEVEL_SCALE)) + pr.NOISE_LEVEL_OFFSET


def export_base(out: Path, bank: dict) -> dict:
    """Front-end constants, the seed-0 physics every field config shares, the banks, the clips."""
    mel = fe._MELS.setdefault((fe.HOP_N_ROWS, 16000), fe.MelFrontend(
        sample_rate=16000, n_fft=fe.HOP_N_FFT, hop=fe.HOP_LENGTH, n_mels=fe.HOP_N_ROWS))
    bins, freqs = fe._quad_maps(fe.HOP_N_ROWS, 16000)
    frontend = {
        "sample_rate": 16000, "clip_samples": 16000, "n_fft": fe.HOP_N_FFT, "hop": fe.HOP_LENGTH,
        "n_mels": fe.HOP_N_ROWS, "log_eps": 1e-5, "offset": fe.HOP_OFFSET, "scale": fe.HOP_SCALE,
        "window": "hann_periodic", "center": False, "int16_scale": pr.INT16_SCALE,
        "mel_fb": b64(mel.mel.mel_scale.fb),
        "quad_bins": [int(b) for b in bins], "quad_freqs": [float(f) for f in freqs],
        "carrier_edges": [float(e) for e in band_edges(fe.HOP_N_ROWS)],   # cycles per sample
        "warmup": WARMUP_FRAMES, "windows": am.WINDOWS["recognition"], "width": WIDTH,
        "min_frames_per_window": MIN_FRAMES_PER_WINDOW,
    }

    # the seed-0 physics: every phase function and geometry draws the same kernel,
    # natural frequencies and initial phases from the seed; checked, not assumed
    ref = am.build_frozen(pl.FIELD, 1.0, SEED)
    blk = physics_block(ref.core)
    kernel, omega, phase0 = blk.kernel.detach().clone(), blk.natural_freqs.detach().clone(), ref.core.phase0[0, 0]
    geometries = {}
    for shape in SHAPES:
        geom = build_geometry(shape, am.GRID)
        embedded = geom.embed_kernel(kernel)
        kfft = geom.kernel_spectrum(embedded)
        cap = 1.0 / geom.clamp_factor
        mag = kfft.abs().flatten(1).amax(dim=1).view(-1, *([1] * (kfft.dim() - 1)))
        kfft = kfft * torch.clamp(cap / (mag + 1e-8), max=1.0)
        taps = geom.spectrum_to_taps(kfft, embedded.shape)
        dense = geom.dense_operator(taps, geom.circulant_index())      # [C, N, N]
        geometries[shape] = {"taps": b64(taps), "embed_shape": list(embedded.shape[1:]),
                             "check": {"sum": float(dense.double().sum()),
                                       "row0": [float(v) for v in dense[:, 0].double().sum(-1)],
                                       "abs": float(dense.double().abs().sum())}}
    for family in FAMILIES:
        for shape in (("torus",) if family in pl.AMPLITUDE_FAMILIES else SHAPES):
            m = am.build_frozen(field_arm(family, shape), 1.0, SEED)
            b = physics_block(m.core)
            assert torch.equal(b.kernel, kernel) and torch.equal(b.natural_freqs, omega), (family, shape)
            init = m.core.phase0[0, 0] if hasattr(m.core, "phase0") else torch.atan2(m.core.state0[0, 1],
                                                                                    m.core.state0[0, 0])
            assert torch.allclose(torch.remainder(init, 2 * math.pi), phase0, atol=1e-5), (family, shape)
    sl = am.build_frozen(field_arm("sl"), 1.0, SEED).core
    physics = {
        "channels": 4, "grid": am.GRID, "dt": am.DT, "substeps": am.SUBSTEPS, "damping": pl.FIELD.damping,
        "clamp": pl.FIELD.clamp, "omega": b64(omega), "phase0": b64(phase0), "geometries": geometries,
        "sakaguchi_alpha": ALPHA, "harmonic2_beta": BETA,
        "winfree_s": [-1.0, 0.0], "winfree_i": [1.0, 0.0, 1.0],
        "sl": {"alpha": float(sl.alpha.flatten()[0]), "beta": float(F.softplus(sl.beta_hat).flatten()[0]),
               "state0": b64(sl.state0[0])},
    }

    banks = {}
    for arm in (pl.BANK_A, pl.BANK_B):
        b = am.build_frozen(arm, 1.0, SEED)
        banks[arm.label()] = {"channels": arm.channels, "input_gain": b64(b.input_gain), "tau_s": b64(b.tau_s),
                              "rates_hz": {"hop": 62.5, "carrier": rn.CARRIER_RATE_HZ}}

    clips, noise = [], []
    for i in demo_indices(bank):
        d, spk = int(bank["labels"][i]), int(bank["speakers"][i])
        name = f"audio/digit-{d}-speaker-{spk}.wav"
        write_wav(out / name, bank["waves"][i].numpy(), 16000)
        uid = int(pr.uids(bank, torch.tensor([i]))[0])
        levels = []
        for db in (0.0, 5.0):
            gen = torch.Generator().manual_seed(pr.NOISE_SEED + uid * pr.NOISE_UID_STRIDE + noise_level_code(db))
            levels.append(torch.randn(16000, generator=gen))
        unit = torch.stack(levels)
        assert unit.abs().max() < 32767 / NOISE_I16_SCALE
        noise.append(torch.round(unit * NOISE_I16_SCALE).to(torch.int16))
        clips.append({"digit": d, "speaker": spk, "rep": int(bank["reps"][i]), "uid": uid, "bank_index": i,
                      "samples": int(bank["lens"][i]), "file": name})
    noise_arr = torch.stack(noise).numpy()                                  # [clips, 2 levels, 16000]
    (out / "audio").mkdir(parents=True, exist_ok=True)
    noise_arr.astype("<i2").tofile(out / "audio" / "noise.bin")
    return {"frontend": frontend, "physics": physics, "banks": banks, "clips": clips,
            "noise": {"file": "audio/noise.bin", "levels_db": [0.0, 5.0], "scale": NOISE_I16_SCALE,
                      "samples": 16000, "seed": pr.NOISE_SEED}}


def demo_waves(bank: dict, noise_db: float | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.tensor(demo_indices(bank))
    return pr.recognition_clips(bank, idx, noise_db)


# ---------------------------------------------------------------------------
# One config
# ---------------------------------------------------------------------------

def net_weights(model: torch.nn.Module) -> list[tuple[str, torch.Tensor, type]]:
    return [(name, t.detach(), np.float32) for name, t in model.state_dict().items()
            if not name.endswith("probe_w")]


def export_config(cfg: Config, out: Path, bank: dict, args, projections: dict) -> list[dict]:
    t0 = time.perf_counter()
    spec = cfg.spec()
    clips = rn.assemble(spec, bank)
    if args.smoke:
        # a fast, lower-fidelity pass: a short training set and a slice of the test set
        n_tr, n_te = args.smoke_train, args.smoke_test
        clips = rn.Clips([lambda a, b, f=clips.blocks[0]: f(a, b), lambda a, b, f=clips.blocks[1]: f(a, b)],
                         [n_tr, n_te], torch.cat((clips.labels[:n_tr], clips.labels[SIZE:SIZE + n_te])),
                         ro.Layout(n_tr, 0, n_te), clips.n_classes)
    with_test = not args.no_test and cfg.drive != "carrier"
    blocks, model, net = arm_features(cfg, bank, clips, cfg.reads, train_only=not with_test)
    t_sim = time.perf_counter() - t0

    # the demo clips, run the same way, for the page to check itself against
    waves, lens, _ = demo_waves(bank, cfg.noise_db)
    demo_rows = pr.front_end(waves, cfg.drive)
    tvalid = pr.valid_frames(lens, cfg.drive)
    chunk = 4 if cfg.drive == "carrier" else len(waves)      # a carrier trajectory is 16,000 frames
    parts: list[dict] = []
    with torch.no_grad():
        for a in range(0, len(waves), chunk):
            rows, tv = demo_rows[a:a + chunk], tvalid[a:a + chunk]
            if cfg.arm.kind == "ann":
                parts.append(am.ann_blocks(model, rows, tv, "recognition"))
            else:
                parts.append(am.frozen_features(cfg.arm, am.frozen_signals(cfg.arm, model, rows), tv,
                                                "recognition"))
    feats = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    demo = {read: [feats[k] for k in keys] for read, keys in am.reads(cfg.arm, "recognition").items()
            if read in cfg.reads}

    entries = []
    for read in cfg.reads:
        r = fit_readout(blocks[read], clips.labels, clips.layout, exact=not args.smoke, with_test=with_test)
        p16 = None
        if r["projected"]:
            native = r["native"]
            if native not in projections:
                p = projection_matrix(native, WIDTH)
                p16 = p.to(torch.float16)
                path = out / "projections" / f"p{native}.bin"
                path.parent.mkdir(parents=True, exist_ok=True)
                p16.numpy().astype("<f2").tofile(path)
                projections[native] = {"file": f"projections/p{native}.bin", "rows": native, "cols": WIDTH,
                                       "dtype": "float16", "p16": p16}
            p16 = projections[native]["p16"].float()
        br = browser_readout(r, p16)
        parts = [(name, br[name], np.float64 if name == "shift" else np.float32)
                 for name in ("inv1", "shift", "mean2", "sd2", "weight", "bias") if name in br]
        cid = cfg.cid(read)
        layout = write_bin(out / "readouts" / f"{cid}.bin", parts)

        entry = {
            "id": cid, "part": cfg.part, "drive": cfg.drive, "arm": cfg.arm.as_dict(), "label": cfg.arm.label(),
            "name": terms.arm(cfg.arm.label()), "gain": cfg.gain, "noise_db": cfg.noise_db, "read": read,
            "readout": {"file": f"readouts/{cid}.bin", "layout": layout, "native": r["native"],
                        "width": min(WIDTH, r["native"]), "projected": r["projected"],
                        "projection": projections[r["native"]]["file"] if r["projected"] else None,
                        "lam": r["lam"], "val_acc": r["val_acc"]},
            "record": record_cell(cfg, read), "tier": cfg.tier or None,
            "n_train": clips.layout.n_train, "n_test": clips.layout.n_test if with_test else 0,
        }
        if with_test:
            logits = browser_logits(blocks[read], clips.layout.test, br, p16)
            agree = (logits.argmax(1) == r["pred"]).sum().item()
            entry["export"] = {"acc": r["acc"], "ref_acc": r["ref_acc"], "browser_agree": agree,
                               "browser_acc": (logits.argmax(1) == clips.labels[clips.layout.test])
                               .double().mean().item()}
            rec = entry["record"]
            if rec and str(SEED) in rec["seeds"] and not args.smoke:
                entry["export"]["matches_record"] = abs(rec["seeds"][str(SEED)] - r["acc"]) < 1e-12
        # reference outputs on the demo clips, at this config's noise level
        dl = browser_logits(demo[read], slice(0, len(demo_rows)), br, p16)
        feat = torch.cat(demo[read], dim=1)
        entry["demo"] = {"logits": [[round(float(v), 6) for v in row] for row in dl],
                         "feature_sum": [float(v) for v in feat.double().sum(1)],
                         "rows_sum": [float(v) for v in demo_rows.double().flatten(1).sum(1)]}
        entries.append(entry)

    if cfg.arm.kind == "ann":
        nid = cfg.cid("windowed")
        layout = write_bin(out / "nets" / f"{nid}.bin", net_weights(model))
        for e in entries:
            e["net"] = {"file": f"nets/{nid}.bin", "layout": layout, "arch": cfg.arm.arch, **net}
    dt = time.perf_counter() - t0
    for e in entries:
        e["timing_s"] = {"simulate": round(t_sim, 1), "total": round(dt, 1)}
        ex = e.get("export", {})
        rec = e["record"]
        print(f"{e['id']:60s} acc {ex.get('acc', float('nan')):.4f} browser {ex.get('browser_agree', '-')}"
              f" record s0 {rec['seeds'].get(str(SEED)) if rec else '-'}  {dt:.0f}s", flush=True)
    return entries


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--part", action="append", choices=PARTS, help="repeatable; default: all")
    ap.add_argument("--only", action="append", default=[], help="export only config ids containing this")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="short training and test sets, for development")
    ap.add_argument("--smoke-train", type=int, default=384)
    ap.add_argument("--smoke-test", type=int, default=256)
    ap.add_argument("--no-test", action="store_true", help="skip scoring the 6,000 test clips")
    ap.add_argument("--record-only", action="store_true", help="refresh the record's numbers in the manifest")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    parts = tuple(args.part or PARTS)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"configs": {}}

    if args.record_only:
        for e in manifest["configs"].values():
            arm = am.Arm(**e["arm"])
            cfg = Config(e["part"], e["drive"], arm, e["gain"], e["noise_db"], e["tier"] or "")
            e["record"] = record_cell(cfg, e["read"])
        manifest["record"] = export_record(out)
        manifest["record_refreshed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
        return

    if "record" in parts:
        manifest["record"] = export_record(out)
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    bank = pr.load_bank()
    if "base" in parts or "frontend" not in manifest:
        manifest.update(export_base(out, bank))
    projections = {p["rows"]: {**p, "p16": torch.from_numpy(
        np.fromfile(out / p["file"], dtype="<f2").reshape(p["rows"], p["cols"]))}
        for p in manifest.get("projections", {}).values()}
    for cfg in configs(parts):
        if args.only and not any(s in cfg.cid() for s in args.only):
            continue
        for e in export_config(cfg, out, bank, args, projections):
            manifest["configs"][e["id"]] = e
        manifest["projections"] = {str(k): {kk: vv for kk, vv in v.items() if kk != "p16"}
                                   for k, v in projections.items()}
        manifest["meta"] = {
            "paper": "Spoken-Digit Recognition Without Training (paper 02)",
            "source": "papers/02-untrained-reservoirs/src/scripts/export_web_demo.py",
            "commit": git_commit(), "exported": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": SEED,
            "primary_cell": {"read": "windowed", "n_train": SIZE, "width": WIDTH},
            "smoke": bool(args.smoke),
            "note": "Untrained arms at seed 0; each readout is the registered primary cell, refitted here "
                    "and checked against the record. Trained baselines are retrained here at seed 0.",
        }
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")   # after every config: resumable


if __name__ == "__main__":
    main()
