"""The shared readout: from an arm's features to accuracy, at every width and size.

Every arm is read the same way. For each training-set size, its native features
are standardized with that training set's own statistics, brought to each
common width by one fixed random projection, and classified by a closed-form
ridge whose penalty is chosen on the last eighth of the training set. An arm is
never read wider than it is: a width at or above its native width reads it
unprojected. The native read itself is also fitted, at the sizes a run asks for.

The ridge is paper 02's. It solves whichever of the two equivalent systems
is smaller, primal D x D or dual N x N, so it never forms the N x N kernel
matrix of a large training set (4.6 GB at 24,000 clips).

A read may be a list of feature blocks (a network's statistics, then its
rotation rates). Blocks are projected piecewise, so a composite read never has
to be copied into one matrix.

This is paper 02's readout, unchanged in what it computes under paper 02's
fixed projection. Paper 03 also reads every projected width under a second
projection seeded by the run's seed (`features.projection_matrix`), so the
projection's own draw is sampled across seeds, and tags each cell with its
projection. An arm too large to hold is standardized and projected channel by
channel instead (`harness.experiment.stream`), and hands `fit_widths` its
projected features.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from harness.measurement.features import PROJECTIONS, projection_matrix

#: ridge penalties, scaled by the number of fitted clips, paper 02's
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0)
#: the share of a training set held out to choose the penalty
VAL_FRAC = 0.125
#: rows per chunk for the large products, bounding peak memory
CHUNK = 4096
#: a feature whose training spread is below this is treated as constant
MIN_SD = 1e-6


@dataclass(frozen=True)
class Layout:
    """Where the rows of a feature matrix belong.

    Rows [0, n_train) are training clips in the seed's order, so a training set
    of size n is the first n rows. Then n_val validation rows (Protocol B only),
    then n_test test rows.
    """
    n_train: int
    n_val: int
    n_test: int

    @property
    def val(self) -> slice:
        return slice(self.n_train, self.n_train + self.n_val)

    @property
    def test(self) -> slice:
        return slice(self.n_train + self.n_val, self.n_train + self.n_val + self.n_test)


def projection_of(cell: dict, rec: dict) -> str:
    """A recorded cell's projection: its tag ("fixed", "seeded" or "none"), or, for a cell recorded
    before paper 02 tagged them, "none" where it was read unprojected (its effective width is the
    arm's native width) and paper 02's fixed matrix otherwise."""
    if "projection" in cell:
        return cell["projection"]
    return "none" if cell["effective_width"] >= rec["native_widths"][cell["read"]] else "fixed"


def pack(correct: torch.Tensor) -> str:
    """Per-clip correctness as base64 bits, so a difference can be bootstrapped over clips later."""
    return base64.b64encode(np.packbits(correct.cpu().numpy().astype(np.uint8)).tobytes()).decode()


def unpack(bits: str, n: int) -> torch.Tensor:
    raw = np.frombuffer(base64.b64decode(bits), dtype=np.uint8)
    return torch.from_numpy(np.unpackbits(raw)[:n].astype(bool))


# ---------------------------------------------------------------------------
# The ridge
# ---------------------------------------------------------------------------

def _stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature mean and standard deviation in float64, a chunk at a time."""
    n = x.shape[0]
    total = torch.zeros(x.shape[1], dtype=torch.float64)
    for i in range(0, n, CHUNK):
        total += x[i:i + CHUNK].double().sum(0)
    mean = total / n
    sq = torch.zeros_like(mean)
    for i in range(0, n, CHUNK):
        sq += ((x[i:i + CHUNK].double() - mean) ** 2).sum(0)
    return mean, (sq / max(n - 1, 1)).sqrt().clamp_min(MIN_SD)


def _design(x: torch.Tensor, mean: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """Standardized rows with the bias column appended, in float64."""
    z = (x.double() - mean) / sd
    return torch.cat((z, torch.ones(z.shape[0], 1, dtype=torch.float64)), dim=1)


def _gram(x, mean, sd, y1h, rows: slice) -> tuple[torch.Tensor, torch.Tensor]:
    """Z^T Z and Z^T Y over `rows`, a chunk at a time."""
    d = x.shape[1] + 1
    a = torch.zeros(d, d, dtype=torch.float64)
    b = torch.zeros(d, y1h.shape[1], dtype=torch.float64)
    start, stop = rows.start or 0, rows.stop
    for i in range(start, stop, CHUNK):
        j = min(i + CHUNK, stop)
        z = _design(x[i:j], mean, sd)
        a += z.T @ z
        b += z.T @ y1h[i:j]
    return a, b


def _predict(x: torch.Tensor, mean, sd, w: torch.Tensor) -> torch.Tensor:
    """Class predictions. The weights are solved in float64; applying them does not need it."""
    m, s, w32 = mean.float(), sd.float(), w.float()
    return torch.cat([(((x[i:i + CHUNK] - m) / s) @ w32[:-1] + w32[-1]).argmax(1)
                      for i in range(0, x.shape[0], CHUNK)])


def _spd_solve(m: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve a symmetric positive-definite system: Cholesky, about twice as fast as LU here.

    Every system the ridge solves is a Gram matrix plus a positive penalty, so
    it is positive definite; if rounding ever says otherwise, LU still answers.
    """
    factor, info = torch.linalg.cholesky_ex(m)
    if int(info) == 0:
        return torch.cholesky_solve(b, factor)
    return torch.linalg.solve(m, b)


def _solve_primal(a: torch.Tensor, b: torch.Tensor, lam: float, n: int) -> torch.Tensor:
    reg = a.clone()
    reg.diagonal().add_(lam * n)
    return _spd_solve(reg, b)


def _solve_dual(z: torch.Tensor, k: torch.Tensor, y1h: torch.Tensor, lam: float) -> torch.Tensor:
    reg = k.clone()
    reg.diagonal().add_(lam * z.shape[0])
    return z.T @ _spd_solve(reg, y1h)


def ridge(x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor, y_te: torch.Tensor,
          n_classes: int, x_val: torch.Tensor | None = None, y_val: torch.Tensor | None = None,
          lambdas: tuple[float, ...] = LAMBDAS, val_frac: float = VAL_FRAC) -> dict:
    """Paper 02's ridge, solved in whichever form is smaller.

    Standardize by the training rows' statistics; hold out the last `val_frac`
    of them (or use an explicit validation set) to choose the penalty, ties
    going to the smaller one; refit on every training row; score the test rows.
    """
    mean, sd = _stats(x_tr)
    y1h = F.one_hot(y_tr, n_classes).double()
    n = x_tr.shape[0]
    explicit = x_val is not None
    n_fit = n if explicit else n - max(1, int(val_frac * n))
    xv, yv = (x_val, y_val) if explicit else (x_tr[n_fit:], y_tr[n_fit:])
    d = x_tr.shape[1] + 1
    primal = d <= n_fit

    if primal:
        a_fit, b_fit = _gram(x_tr, mean, sd, y1h, slice(0, n_fit))
        fit = lambda lam: _solve_primal(a_fit, b_fit, lam, n_fit)  # noqa: E731
    else:
        z_fit = _design(x_tr[:n_fit], mean, sd)
        k_fit = z_fit @ z_fit.T
        fit = lambda lam: _solve_dual(z_fit, k_fit, y1h[:n_fit], lam)  # noqa: E731

    best_lam, best_acc, best_w = lambdas[0], -1.0, None
    for lam in lambdas:
        w = fit(lam)
        acc = (_predict(xv, mean, sd, w) == yv).double().mean().item()
        if acc > best_acc:
            best_lam, best_acc, best_w = lam, acc, w

    if explicit:
        w = best_w                                   # already fitted on every training row
    elif primal:
        a_v, b_v = _gram(x_tr, mean, sd, y1h, slice(n_fit, n))
        w = _solve_primal(a_fit + a_v, b_fit + b_v, best_lam, n)
    else:
        z = _design(x_tr, mean, sd)
        w = _solve_dual(z, z @ z.T, y1h, best_lam)
    correct = _predict(x_te, mean, sd, w) == y_te
    return {"acc": correct.double().mean().item(), "lam": best_lam, "val_acc": best_acc,
            "correct": correct, "form": "primal" if primal else "dual"}


# ---------------------------------------------------------------------------
# Standardize, project, read
# ---------------------------------------------------------------------------

def _block_stats(blocks: list[torch.Tensor], rows: slice) -> tuple[torch.Tensor, torch.Tensor]:
    parts = [_stats(b[rows]) for b in blocks]
    return torch.cat([m for m, _ in parts]), torch.cat([s for _, s in parts])


def _projected(blocks: list[torch.Tensor], rows: slice, mean: torch.Tensor, sd: torch.Tensor,
               width: int, p: torch.Tensor | None = None, device: str = "cpu") -> torch.Tensor:
    """Standardize, then project to `width`, a chunk of rows at a time.

    Rows are centred before they are scaled and projected, which keeps float32
    accurate where folding the mean into the projection would subtract two
    large numbers. A feature that is constant over the training set carries
    nothing, and is given weight zero rather than being divided by nothing.
    Blocks take consecutive row ranges of the one projection matrix: paper
    02's fixed draw for this native width, unless `p` ([features, width]) is
    given. On a GPU (`device`) the products run there and the result comes
    back to the CPU; on the CPU this is paper 02's arithmetic exactly.
    """
    p = projection_matrix(int(mean.numel()), width) if p is None else p
    m = mean.float()
    inv = torch.where(sd > MIN_SD, 1.0 / sd, torch.zeros_like(sd)).float()
    gpu = device != "cpu"
    if gpu:
        p, m, inv = p.to(device), m.to(device), inv.to(device)
    out = []
    start, stop = rows.start or 0, rows.stop
    for i in range(start, stop, CHUNK):
        j = min(i + CHUNK, stop)
        acc, at = None, 0
        for b in blocks:
            k = b.shape[1]
            x = b[i:j].to(device) if gpu else b[i:j]
            part = ((x - m[at:at + k]) * inv[at:at + k]) @ p[at:at + k]
            acc = part if acc is None else acc + part
            at += k
        out.append(acc.cpu() if gpu else acc)
    return torch.cat(out)


def _native(blocks: list[torch.Tensor], rows: slice) -> torch.Tensor:
    return torch.cat([b[rows] for b in blocks], dim=1)


def read_cells(reads: dict[str, list[torch.Tensor]], labels: torch.Tensor, layout: Layout,
               sizes: tuple[int, ...], widths: tuple[int, ...], native_sizes: tuple[int, ...],
               n_classes: int, keep_bits: Callable[[str, int, int], bool], seed: int = 0,
               device: str = "cpu") -> list[dict]:
    """Every (read, size, width, projection) cell a run asks for.

    `reads` maps a read's name to its feature blocks, rows laid out by `layout`.
    `keep_bits(read, width, size)` says which cells store per-clip correctness.
    Every width below the native width is read under both projections: the
    fixed one (paper 02's matrix) and the one seeded by the run's `seed`. A
    width at or above the native width is the unprojected read, fitted once
    and tagged "none". Only the projection runs on `device`; the ridge is
    solved on the CPU in float64. `labels` [clips, positions] (the
    digit-sequence task) fits one readout per position on the same features,
    each cell tagged with its position.
    """
    positions = [None] if labels.dim() == 1 else list(range(labels.shape[1]))

    def tagged(fitted: list[dict], p: int | None) -> list[dict]:
        return fitted if p is None else [{**c, "position": p} for c in fitted]

    cells = []
    for name, blocks in reads.items():
        native = sum(b.shape[1] for b in blocks)
        for n in sizes:
            if n > layout.n_train:
                raise ValueError(f"size {n} exceeds the {layout.n_train} training rows")
            rows_tr = slice(0, n)
            mean, sd = _block_stats(blocks, rows_tr)
            wanted = wanted_widths(widths, native, n in native_sizes)
            below = [(r, e) for r, e in wanted if e < native]
            if below:
                # project once per matrix, to the widest width below native; narrower widths are its leading columns
                top = max(e for _, e in below)
                for projection in PROJECTIONS:
                    p = projection_matrix(native, top, None if projection == "fixed" else seed)
                    projected = (_projected(blocks, rows_tr, mean, sd, top, p, device),
                                 _projected(blocks, layout.test, mean, sd, top, p, device),
                                 _projected(blocks, layout.val, mean, sd, top, p, device) if layout.n_val else None)
                    for p in positions:
                        y = labels if p is None else labels[:, p]
                        cells += tagged(fit_widths(name, n, below, native, projected, None, y, layout, n_classes,
                                                   keep_bits, projection), p)
            unprojected = [(r, e) for r, e in wanted if e >= native]
            for p in positions if unprojected else []:
                y = labels if p is None else labels[:, p]
                cells += tagged(fit_widths(name, n, unprojected, native, None,
                                           lambda rows, blocks=blocks: _native(blocks, rows), y, layout, n_classes,
                                           keep_bits, "none"), p)
    return cells


def wanted_widths(widths: tuple[int, ...], native: int, with_native: bool) -> list[tuple]:
    """(requested, effective) widths: never wider than the arm, and the native read if asked for."""
    wanted = [(w, min(w, native)) for w in widths]
    return wanted + [("native", native)] if with_native else wanted


def fit_widths(name: str, n: int, wanted: list[tuple], native: int,
               projected: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None,
               native_rows: Callable[[slice], torch.Tensor] | None, labels: torch.Tensor, layout: Layout,
               n_classes: int, keep_bits: Callable[[str, int, int], bool], projection: str = "fixed") -> list[dict]:
    """One read at one training size: the ridge at every width, from the projected features.

    `projected` is (training, test, validation) projected to the widest width
    below native, whose leading columns are every narrower width;
    `native_rows(rows)` gives the unprojected features for a width at or
    above native.
    """
    y_te = labels[layout.test]
    y_val = labels[layout.val] if layout.n_val else None
    rows_tr = slice(0, n)
    done: dict[int, dict] = {}
    cells = []
    for requested, effective in wanted:
        if effective not in done:
            if effective == native:
                if native_rows is None:
                    raise ValueError(f"{name}: the native read was asked for but its features were not kept")
                x_tr, x_te = native_rows(rows_tr), native_rows(layout.test)
                x_val = native_rows(layout.val) if layout.n_val else None
            else:
                p_tr, p_te, p_val = projected
                x_tr, x_te = p_tr[:, :effective], p_te[:, :effective]
                x_val = p_val[:, :effective] if layout.n_val else None
            done[effective] = ridge(x_tr, labels[rows_tr], x_te, y_te, n_classes, x_val=x_val, y_val=y_val)
        r = done[effective]
        cell = {"read": name, "n_train": n, "width": requested, "effective_width": effective,
                "projection": "none" if effective >= native else projection,
                "acc": r["acc"], "lam": r["lam"], "val_acc": r["val_acc"], "form": r["form"]}
        width_key = effective if requested == "native" else requested
        if keep_bits(name, width_key, n):
            cell["correct"] = pack(r["correct"])
        cells.append(cell)
    return cells
