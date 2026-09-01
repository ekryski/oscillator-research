"""Featurization, the ridge readout, and the training loop.

The only fitted object in the primary protocol is the closed-form ridge probe
of `fit_ridge_probe`: a linear scorer cannot compute, so whatever separation it
finds must already exist in the features that the dynamics wrote. `train_arm`
exists for the trained conventional references and the trained-physics arms;
frozen arms never touch it.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from harness.utils.constants import WARMUP_FRAMES

# ---------------------------------------------------------------------------
# Featurization (fixed, identical protocol for every field arm)
# ---------------------------------------------------------------------------

def phase_features(feats: torch.Tensor, channels: int, grid: int,
                   warmup: int = WARMUP_FRAMES,
                   tvalid: torch.Tensor | None = None) -> torch.Tensor:
    """[B,T,2*C*G*G] scan features -> [B, 4*C*G*G] clip features.

    Per oscillator: time-mean sin/cos (position signature) and time-mean
    cos/sin of the per-frame phase INCREMENT (velocity signature — a locked
    oscillator's increment encodes the stimulus frequency; an unlocked one's
    encodes its own omega). Whole-clip means, no per-segment windows: temporal
    order information can only enter through dynamical (hysteresis) effects."""
    d = channels * grid * grid
    s, c = feats[..., :d], feats[..., d:]
    if tvalid is None:
        s_m = s[:, warmup:].mean(dim=1)
        c_m = c[:, warmup:].mean(dim=1)
        s1, s0 = s[:, warmup:], s[:, warmup - 1:-1]
        c1, c0 = c[:, warmup:], c[:, warmup - 1:-1]
        cd = (c1 * c0 + s1 * s0).mean(dim=1)  # cos(theta_t - theta_{t-1})
        sd = (s1 * c0 - c1 * s0).mean(dim=1)  # sin(theta_t - theta_{t-1})
        return torch.cat((s_m, c_m, cd, sd), dim=1)
    # length masking: pool frames [warmup, tvalid_i) per clip — padded
    # ring-down never enters the statistics (same contract, masked means).
    t = feats.shape[1]
    m = ((torch.arange(t, device=feats.device)[None, :] >= warmup)
         & (torch.arange(t, device=feats.device)[None, :] < tvalid.to(feats.device)[:, None]))
    mf = m[:, :, None].float()
    n = mf.sum(dim=1).clamp_min(1.0)
    s_m = (s * mf).sum(dim=1) / n
    c_m = (c * mf).sum(dim=1) / n
    s1, s0 = s[:, 1:], s[:, :-1]
    c1, c0 = c[:, 1:], c[:, :-1]
    md = mf[:, 1:]  # increment valid when frame t valid (t-1 then also < tvalid)
    nd = md.sum(dim=1).clamp_min(1.0)
    cd = ((c1 * c0 + s1 * s0) * md).sum(dim=1) / nd
    sd = ((s1 * c0 - c1 * s0) * md).sum(dim=1) / nd
    return torch.cat((s_m, c_m, cd, sd), dim=1)

# ---------------------------------------------------------------------------
# Pooling shared by the conventional references
# ---------------------------------------------------------------------------

def pooled_stats(h: torch.Tensor, tvalid: torch.Tensor | None = None) -> torch.Tensor:
    """Shared hidden-trajectory pooling for the conventional references:
    (mean, std, |delta| mean, last) over time — the GRU/TCN feature contract.
    tvalid: pool valid frames only; "last" = the last VALID frame."""
    if tvalid is None:
        dh = (h[:, 1:] - h[:, :-1]).abs().mean(dim=1)
        return torch.cat((h.mean(dim=1), h.std(dim=1), dh, h[:, -1]), dim=1)
    t = h.shape[1]
    tv = tvalid.to(h.device).clamp(2, t)
    m = (torch.arange(t, device=h.device)[None, :] < tv[:, None])[:, :, None].float()
    n = m.sum(dim=1)
    mean = (h * m).sum(dim=1) / n
    var = ((h - mean[:, None]) ** 2 * m).sum(dim=1) / (n - 1).clamp_min(1.0)
    md = m[:, 1:]
    dh = ((h[:, 1:] - h[:, :-1]).abs() * md).sum(dim=1) / md.sum(dim=1).clamp_min(1.0)
    last = h[torch.arange(h.shape[0], device=h.device), tv - 1]
    # sqrt(var + eps), masked branch only: a dead (constant) unit has EXACTLY
    # zero variance over a short valid window (measured: ReLU nets on digits),
    # and sqrt'(0) = inf turns its zero upstream grad into NaN (inf * 0),
    # killing training. The eps keeps gradients finite; the value shift
    # (<= 1e-4) is far below hidden-state scale. The unmasked branch above is
    # untouched — frozen controls depend on its exact numerics.
    return torch.cat((mean, (var + 1e-8).sqrt(), dh, last), dim=1)


def windowed_stats(h: torch.Tensor, windows: int,
                    tvalid: torch.Tensor | None = None) -> torch.Tensor:
    """Windowed-parity pooling (item 7): same windowing contract as
    OscillatorField.features_windowed so ridge comparisons stay arm-fair. Each
    window needs >= 2 frames for std/delta to be defined.

    tvalid (length masking, in h's ALREADY-WARMUP-CROPPED frame
    coordinates): per-clip windows over [0, tvalid_i); clips too short for
    2-frame windows extend into the trailing frames (hi >= 2*windows, clamped
    to T) — the same rule as OscillatorField.features_windowed; tvalid == T
    reproduces the unmasked windows exactly. Per-clip loop, eval-only."""
    t = h.shape[1]
    if tvalid is None:
        assert t // windows >= 2, f"window too short: {t} frames / {windows} windows"
        return torch.cat([pooled_stats(h[:, i * t // windows:(i + 1) * t // windows])
                          for i in range(windows)], dim=1)
    out = []
    for i in range(h.shape[0]):
        hi = min(t, max(int(tvalid[i]), 2 * windows))
        out.append(torch.cat([pooled_stats(h[i:i + 1, hi * j // windows: hi * (j + 1) // windows])
                              for j in range(windows)], dim=1))
    return torch.cat(out, dim=0)

# ---------------------------------------------------------------------------
# Training (frozen-probe CE, physics-only gradients) and evaluation probes
# ---------------------------------------------------------------------------

def train_arm(model: nn.Module, rows_tr: torch.Tensor, y_tr: torch.Tensor,
              rows_val: torch.Tensor, y_val: torch.Tensor, *, epochs: int, lr: float,
              batch: int, seed: int, log_path: Path | None = None,
              selector=None, select_every: int = 1,
              tvalid_tr: torch.Tensor | None = None,
              tvalid_val: torch.Tensor | None = None) -> list[dict]:
    """Full-BPTT training through the frozen probe. Returns per-epoch history.

    Grad norms for K and omega are logged separately — at harness scale the physics
    gradient share is 100% by construction, so these are finally a direct
    measurement of learning signal reaching the physics.

    selector (best-val checkpoint selection): a callable
    model -> float scored every `select_every` epochs plus the final one; the
    best-scoring epoch's weights are restored into the model before returning
    and (sel_epoch, sel_score) is stamped on the last history row. Ties go to
    the earliest epoch; selection must consume no training RNG (the batch
    generator is local); aborted (non-finite) runs never restore — they stay
    recorded as aborted. selector=None keeps the historical last-checkpoint
    behavior exactly."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    device = next(model.parameters()).device
    rows_tr, y_tr = rows_tr.to(device), y_tr.to(device)
    rows_val, y_val = rows_val.to(device), y_val.to(device)
    n = rows_tr.shape[0]
    steps_per_epoch = math.ceil(n / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * steps_per_epoch, eta_min=0.1 * lr)
    gen = torch.Generator().manual_seed(seed)
    named = [(nm, p) for nm, p in model.named_parameters() if p.requires_grad]
    history: list[dict] = []
    writer = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["epoch", "loss", "train_acc", "val_acc", "grad_total", "grad_k", "grad_w", "sec", "sel"])
    best: tuple[float, int, dict] | None = None  # (score, epoch, cpu state)
    model.train()
    for ep in range(epochs):
        t0 = time.time()
        perm = torch.randperm(n, generator=gen)
        tot_loss = tot_correct = 0.0
        g_tot = g_k = g_w = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logits = model(rows_tr[idx],
                           None if tvalid_tr is None else tvalid_tr.to(device)[idx])
            loss = F.cross_entropy(logits, y_tr[idx])
            if not torch.isfinite(loss):
                history.append(dict(epoch=ep, loss=float("nan"), aborted=True))
                if writer:
                    fh.close()
                return history
            opt.zero_grad()
            loss.backward()
            for nm, p in named:  # pre-clip norms
                gn = p.grad.norm().item()
                g_k += gn if "kernel" in nm else 0.0
                g_w += gn if "natural_freqs" in nm else 0.0
            g_tot += torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(idx)
            tot_correct += (logits.argmax(dim=1) == y_tr[idx]).sum().item()
        val_acc = frozen_probe_acc(model, rows_val, y_val, batch, tvalid=tvalid_val)
        sel_score = None
        if selector is not None and (ep % select_every == 0 or ep == epochs - 1):
            sel_score = selector(model)
            if best is None or sel_score > best[0]:
                best = (sel_score, ep,
                        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        model.train()
        row = dict(epoch=ep, loss=tot_loss / n, train_acc=tot_correct / n, val_acc=val_acc,
                   grad_total=g_tot / steps_per_epoch, grad_k=g_k / steps_per_epoch,
                   grad_w=g_w / steps_per_epoch, sec=time.time() - t0)
        history.append(row)
        if writer:
            writer.writerow([f"{row[k]:.5g}" if isinstance(row[k], float) else row[k]
                             for k in ["epoch", "loss", "train_acc", "val_acc",
                                       "grad_total", "grad_k", "grad_w", "sec"]]
                            + [f"{sel_score:.5g}" if sel_score is not None else ""])
            fh.flush()
    if writer:
        fh.close()
    if best is not None:
        model.load_state_dict({k: v.to(device) for k, v in best[2].items()})
        history[-1]["sel_epoch"], history[-1]["sel_score"] = best[1], best[0]
    model.eval()
    return history


@torch.no_grad()
def frozen_probe_acc(model: nn.Module, rows: torch.Tensor, y: torch.Tensor,
                     batch: int = 64, tvalid: torch.Tensor | None = None) -> float:
    model.eval()
    device = next(model.parameters()).device
    rows, y = rows.to(device), y.to(device)
    correct = 0
    for i in range(0, rows.shape[0], batch):
        tv = None if tvalid is None else tvalid[i:i + batch]
        correct += (model(rows[i:i + batch], tv).argmax(dim=1) == y[i:i + batch]).sum().item()
    return correct / rows.shape[0]


@torch.no_grad()
def collect_features(model: nn.Module, rows: torch.Tensor, batch: int = 64,
                     tvalid: torch.Tensor | None = None) -> torch.Tensor:
    model.eval()
    rows = rows.to(next(model.parameters()).device)
    return torch.cat([model.features(rows[i:i + batch],
                                     None if tvalid is None else tvalid[i:i + batch]).cpu()
                      for i in range(0, rows.shape[0], batch)])


# readout-capacity parity (baked into from day one): every
# architecture's native feature vector is projected to ONE common width before
# an identical ridge, so readout capacity can never masquerade as dynamics.
PARITY_SEED = 4242
PARITY_DIM = 72
_PARITY_PROJ: dict[int, torch.Tensor] = {}  # one cached projection per native dim


def parity_project(f: torch.Tensor) -> torch.Tensor:
    """[B, native_dim] -> [B, PARITY_DIM] by a FIXED seeded random projection.

    One projection per native dim, drawn from a fresh generator seeded
    PARITY_SEED every time (same seed for every dim) and cached — fully
    deterministic across processes and machines; 1/sqrt(d) scaling keeps the
    projected variance comparable across native widths."""
    d = f.shape[1]
    if d not in _PARITY_PROJ:
        gen = torch.Generator().manual_seed(PARITY_SEED)
        _PARITY_PROJ[d] = torch.randn(d, PARITY_DIM, generator=gen) / math.sqrt(d)
    return f @ _PARITY_PROJ[d].to(f.dtype)


def fit_ridge_probe(f_tr: torch.Tensor, y_tr: torch.Tensor, f_te: torch.Tensor,
                    y_te: torch.Tensor, n_classes: int,
                    lambdas: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0),
                    val_frac: float = 0.125) -> dict:
    """Deterministic linear readout, identical protocol for every arm.

    Standardize by train stats -> hold out the last val_frac of train for
    lambda selection -> refit on full train at the chosen lambda -> report
    held-out test accuracy and mean margin. Dual-form (kernel) ridge: N x N
    solve, cheap even at 4k-dim features."""
    mu, sd = f_tr.mean(dim=0), f_tr.std(dim=0).clamp_min(1e-6)
    xtr = torch.cat(((f_tr - mu) / sd, torch.ones(f_tr.shape[0], 1)), dim=1).double()
    xte = torch.cat(((f_te - mu) / sd, torch.ones(f_te.shape[0], 1)), dim=1).double()
    y1h = F.one_hot(y_tr, n_classes).double()

    def solve(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
        gram = x @ x.T
        alpha = torch.linalg.solve(gram + lam * x.shape[0] * torch.eye(x.shape[0]).double(), y)
        return x.T @ alpha  # weights [D+1, K]

    n_val = max(1, int(val_frac * xtr.shape[0]))
    xa, ya = xtr[:-n_val], y1h[:-n_val]
    xv, yv = xtr[-n_val:], y_tr[-n_val:]
    best_lam, best_acc = lambdas[0], -1.0
    for lam in lambdas:
        acc = ((xv @ solve(xa, ya, lam)).argmax(dim=1) == yv).float().mean().item()
        if acc > best_acc:
            best_lam, best_acc = lam, acc
    w = solve(xtr, y1h, best_lam)
    logits = xte @ w
    acc = (logits.argmax(dim=1) == y_te).float().mean().item()
    true = logits.gather(1, y_te[:, None]).squeeze(1)
    other = logits.scatter(1, y_te[:, None], -torch.inf).max(dim=1).values
    return dict(acc=acc, margin=(true - other).mean().item(), lam=best_lam, val_acc=best_acc,
                pred=logits.argmax(dim=1).tolist())  # per-class breakdowns (ablation)
