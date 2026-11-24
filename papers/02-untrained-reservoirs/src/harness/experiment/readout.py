"""The shared readout: from an arm's features to accuracy, at every width and size.

Every arm is read the same way. For each training-set size, its native features
are standardized with that training set's own statistics, brought to each
common width by one fixed random projection, and classified by a closed-form
ridge whose penalty is chosen on the last eighth of the training set. An arm is
never read wider than it is: a width at or above its native width reads it
unprojected. The native read itself is also fitted, at the sizes a run asks for.

The ridge is `harness.measurement.probe.fit_ridge_probe`, restated so it
scales: that one always solves the N x N kernel system, which at
24,000 clips is a 4.6 GB matrix. This one solves whichever of the two
equivalent systems is smaller, primal D x D or dual N x N, and is tested
against the original for identical choices and accuracies.

A read may be a list of feature blocks (a network's statistics, then its
rotation rates). Blocks are projected piecewise, so a composite read never has
to be copied into one matrix.

Three ways to bring a wide read to a common width, all linear maps applied to
the standardized features:

    random   the fixed (or seeded) Gaussian projection, the study's primary one
    pca      the leading principal directions of the training rows, fitted on
             them alone and applied unchanged to the validation and test rows
    pooled   a structured average: each statistic of each window, averaged over
             every signal of a lattice row (channels, columns, and sin and cos),
             so a network's 24,576 features become 16 rows x 3 statistics x 4
             windows = 192, the spectrogram-only baseline's own layout
    pooled-lower, pooled-upper
             the same average kept for the lower or the upper half of the rows
             only (8 rows x 3 statistics x 4 windows = 96), which reads an arm
             through one half of its lattice
    pooled-row<k>
             the same average kept for row k alone (3 statistics x 4 windows =
             12), which reads an arm through one band's row
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from harness.measurement.features import STATS_PER_SIGNAL, projection_matrix

#: the projection methods a read can go through
METHODS = ("random", "pca", "pooled", "pooled-lower", "pooled-upper")
#: ridge penalties, scaled by the number of fitted clips
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


def pack(correct: torch.Tensor) -> str:
    """Per-clip correctness as base64 bits, so a comparison can be bootstrapped later."""
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
    """The closed-form ridge, solved in whichever form is smaller.

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


def _inverse_sd(sd: torch.Tensor) -> torch.Tensor:
    """1/sd, or zero where a feature is constant over the training set and carries nothing."""
    return torch.where(sd > MIN_SD, 1.0 / sd, torch.zeros_like(sd)).float()


def _standardized(blocks: list[torch.Tensor], rows: slice, mean: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """The standardized rows as one float32 matrix (for fitting a projection; small row ranges only)."""
    m, inv = mean.float(), _inverse_sd(sd)
    parts, at = [], 0
    for b in blocks:
        k = b.shape[1]
        parts.append((b[rows] - m[at:at + k]) * inv[at:at + k])
        at += k
    return torch.cat(parts, dim=1)


def pca_matrix(blocks: list[torch.Tensor], rows: slice, mean: torch.Tensor, sd: torch.Tensor,
               width: int) -> torch.Tensor:
    """[D, width]: the leading principal directions of the standardized training rows.

    Exact, through the n x n Gram matrix of the n training rows (n is far below
    D here): Z Z^T = U diag(l) U^T gives the directions V = Z^T U / sqrt(l). Fitted
    on the training rows alone; the test rows never enter.
    """
    z = _standardized(blocks, rows, mean, sd).double()
    n = z.shape[0]
    if width > n:
        raise ValueError(f"{n} training rows cannot give {width} principal directions")
    evals, evecs = torch.linalg.eigh(z @ z.T)
    order = torch.argsort(evals, descending=True)[:width]
    l, u = evals[order].clamp_min(1e-12), evecs[:, order]
    return ((z.T @ u) / l.sqrt()).float()


def pooling_matrix(signal_rows: torch.Tensor, stats: int, windows: int,
                   keep_rows: tuple[int, ...] | None = None) -> torch.Tensor:
    """[D, windows * stats * rows]: averages each window's statistic over the signals of each lattice row.

    `signal_rows[d]` is the row that signal d belongs to; a block of `windowed`
    features is laid out window by window, statistic by statistic, signal by
    signal, and the pooled features come out in the same order with the row in
    the signal's place: the spectrogram-only baseline's own layout. With
    `keep_rows`, only those rows' pooled features are kept (in the same order),
    so the read sees that part of the lattice and nothing else.
    """
    n_rows = int(signal_rows.max()) + 1
    d = signal_rows.numel()
    p = torch.zeros(windows * stats * d, windows * stats * n_rows)
    for w in range(windows):
        for s in range(stats):
            f0, c0 = (w * stats + s) * d, (w * stats + s) * n_rows
            p[f0 + torch.arange(d), c0 + signal_rows] = 1.0
    p = p / p.sum(0, keepdim=True).clamp_min(1.0)
    if keep_rows is not None:
        keep = torch.tensor([r in keep_rows for r in range(n_rows)]).repeat(windows * stats)
        p = p[:, keep]
    return p


def _kept_rows(method: str, signal_rows: torch.Tensor) -> tuple[int, ...] | None:
    """The rows a pooled method keeps: all of them, or the lower or upper half."""
    n_rows = int(signal_rows.max()) + 1
    if method == "pooled-lower":
        return tuple(range(n_rows // 2))
    if method == "pooled-upper":
        return tuple(range(n_rows // 2, n_rows))
    if method.startswith("pooled-row"):
        return (int(method[len("pooled-row"):]),)
    return None


def _projected(blocks: list[torch.Tensor], rows: slice, mean: torch.Tensor, sd: torch.Tensor,
               width: int, seed: int | None = None, matrix: torch.Tensor | None = None) -> torch.Tensor:
    """Standardize, then project to `width`, a chunk of rows at a time.

    Rows are centred before they are scaled and projected, which keeps float32
    accurate where folding the mean into the projection would subtract two
    large numbers. A feature that is constant over the training set carries
    nothing, and is given weight zero rather than being divided by nothing.
    Blocks take consecutive row ranges of the one projection matrix: the fixed
    or seeded Gaussian draw, or `matrix` when another linear map is asked for.
    """
    p = projection_matrix(int(mean.numel()), width, seed) if matrix is None else matrix[:, :width]
    m = mean.float()
    inv = _inverse_sd(sd)
    out = []
    start, stop = rows.start or 0, rows.stop
    for i in range(start, stop, CHUNK):
        j = min(i + CHUNK, stop)
        acc, at = None, 0
        for b in blocks:
            k = b.shape[1]
            part = ((b[i:j] - m[at:at + k]) * inv[at:at + k]) @ p[at:at + k]
            acc = part if acc is None else acc + part
            at += k
        out.append(acc)
    return torch.cat(out)


def _native(blocks: list[torch.Tensor], rows: slice) -> torch.Tensor:
    return torch.cat([b[rows] for b in blocks], dim=1)


def read_cells(reads: dict[str, list[torch.Tensor]], labels: torch.Tensor, layout: Layout,
               sizes: tuple[int, ...], widths: tuple[int, ...], native_sizes: tuple[int, ...],
               n_classes: int, keep_bits: Callable[[str, int, int], bool],
               projection_seed: int | None = None, method: str = "random",
               signal_rows: torch.Tensor | None = None, windows: int = 1) -> list[dict]:
    """Every (read, size, width) cell a run asks for.

    `reads` maps a read's name to its feature blocks, rows laid out by `layout`.
    `keep_bits(read, width, size)` says which cells store per-clip correctness.
    A width at or above the native width is the native read, fitted once.
    `projection_seed` reads through the seeded projection instead of the fixed one.
    `method` is "random" (the Gaussian projection), "pca" (fitted on each
    training set), "pooled" (the within-row average, which needs each signal's
    lattice row in `signal_rows` and the read's `windows`, and reads at the one
    width that layout gives) or "pooled-lower" / "pooled-upper" (the same
    average over the lower or the upper half of the rows only).
    """
    if method not in METHODS and not method.startswith("pooled-row"):
        raise ValueError(f"unknown projection method '{method}'")
    y_te = labels[layout.test]
    y_val = labels[layout.val] if layout.n_val else None
    cells = []
    for name, blocks in reads.items():
        native = sum(b.shape[1] for b in blocks)
        for n in sizes:
            if n > layout.n_train:
                raise ValueError(f"size {n} exceeds the {layout.n_train} training rows")
            rows_tr = slice(0, n)
            mean, sd = _block_stats(blocks, rows_tr)
            wanted = [(w, min(w, native)) for w in widths]
            if n in native_sizes:
                wanted.append(("native", native))
            # project once, to the widest width below native; narrower widths are its leading columns
            below = [e for _, e in wanted if e < native]
            if below:
                top = max(below)
                matrix = None
                if method == "pca":
                    matrix = pca_matrix(blocks, rows_tr, mean, sd, top)
                elif method.startswith("pooled"):
                    if signal_rows is None or len(blocks) != 1:
                        raise ValueError("the pooled projection needs one block of windowed statistics and its signals' rows")
                    matrix = pooling_matrix(signal_rows, STATS_PER_SIGNAL, windows, _kept_rows(method, signal_rows))
                    if matrix.shape[1] != top:
                        raise ValueError(f"pooling gives {matrix.shape[1]} features, not the {top} asked for")
                p_tr = _projected(blocks, rows_tr, mean, sd, top, projection_seed, matrix)
                p_te = _projected(blocks, layout.test, mean, sd, top, projection_seed, matrix)
                p_val = (_projected(blocks, layout.val, mean, sd, top, projection_seed, matrix)
                         if layout.n_val else None)
            done: dict[int, dict] = {}
            for requested, effective in wanted:
                if effective not in done:
                    if effective == native:
                        get = lambda rows: _native(blocks, rows)  # noqa: E731
                        x_tr, x_te = get(rows_tr), get(layout.test)
                        x_val = get(layout.val) if layout.n_val else None
                    else:
                        x_tr, x_te = p_tr[:, :effective], p_te[:, :effective]
                        x_val = p_val[:, :effective] if layout.n_val else None
                    done[effective] = ridge(x_tr, labels[rows_tr], x_te, y_te, n_classes,
                                            x_val=x_val, y_val=y_val)
                r = done[effective]
                cell = {"read": name, "n_train": n, "width": requested, "effective_width": effective,
                        "acc": r["acc"], "lam": r["lam"], "val_acc": r["val_acc"], "form": r["form"]}
                width_key = effective if requested == "native" else requested
                if keep_bits(name, width_key, n):
                    cell["correct"] = pack(r["correct"])
                cells.append(cell)
    return cells
