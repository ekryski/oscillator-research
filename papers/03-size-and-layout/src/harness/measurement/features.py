"""One readout contract for every arm.

The exploratory phase read its arms three different ways. The field got four
whole-span means per oscillator, the conventional networks got mean, spread,
change and their LAST hidden state, and the no-dynamics floor got three
statistics over every frame, padding included, where every other arm dropped a
warm-up and masked the padding. A difference between two arms could then be a
difference between two featurizers, and the last hidden state carries temporal
order directly, which an order-free read must not.

Here an arm only says which signals it exposes, as a trajectory [B, T, D]. One
function turns any trajectory into features, over a span the caller names:

    pooled     mean, standard deviation, mean |frame-to-frame change|
    windowed   the same three, per window, concatenated

Nothing depends on an endpoint. That rules out the last state, and it rules out
the signed mean of a first difference, which for a linear state telescopes to
(last - first) / T and is the same leak by another route. The absolute
difference does not telescope.

`rotation_rate` is the one arm-specific extra, kept as a secondary read: the
signed mean rotation of each oscillator, its natural observable. No other arm
has an analogue, so a read that includes it is labelled as favouring the field.

`project` brings any arm's features to a common width, because a ridge's
capacity grows with the number of features it is handed: the field exposed
16,384 and its floor 192, an 85-fold difference in fitted coefficients.
"""

from __future__ import annotations

import math

import torch

#: statistics per signal: mean, standard deviation, mean absolute change
STATS_PER_SIGNAL = 3
#: a window needs two frames for a spread and a change to exist
MIN_FRAMES_PER_WINDOW = 2
#: inside the square root of the spread: a constant signal has zero variance,
#: and the root's gradient there is infinite, which turns a trained network's
#: loss into NaN. It moves a spread of 1 by 5e-9 and a zero spread to 1e-4.
VARIANCE_EPS = 1e-8
#: the projection is drawn once per native width from this seed, so it is
#: identical across arms, runs, processes and machines
PROJECTION_SEED = 4242
#: every common width is the leading columns of one projection this wide, so a
#: narrow read is exactly the first columns of a wide one
MAX_WIDTH = 4096
_PROJECTIONS: dict[int, torch.Tensor] = {}
#: native widths above this are not cached beside one another (a [native, 4096] draw is 1.6 GB at 98,304)
LARGE_NATIVE = 98_304


def span(frames: int, lo: int, hi: torch.Tensor | None, windows: int,
         batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-clip [lo, hi) as tensors, widened where a clip is too short.

    `hi` is each clip's count of valid frames, or None for the whole clip. A
    clip shorter than two frames per window is read a little way into its
    trailing frames, to exactly the minimum, the rule the exploratory harness
    used; every arm gets the same widening, so it cannot favour one.
    """
    lo_t = torch.full((batch,), lo, dtype=torch.long, device=device)
    hi_t = (torch.full((batch,), frames, dtype=torch.long, device=device) if hi is None
            else hi.to(device=device, dtype=torch.long))
    need = lo + MIN_FRAMES_PER_WINDOW * windows
    if need > frames:
        raise ValueError(f"{frames} frames cannot hold {windows} window(s) after frame {lo}")
    return lo_t, hi_t.clamp(min=need, max=frames)


def _window_edges(lo: torch.Tensor, hi: torch.Tensor, windows: int) -> torch.Tensor:
    """[B, windows + 1] frame indices splitting each clip's span evenly."""
    steps = torch.arange(windows + 1, device=lo.device)
    return lo[:, None] + (hi - lo)[:, None] * steps[None, :] // windows


def _masked_stats(x: torch.Tensor, inside: torch.Tensor) -> torch.Tensor:
    """mean, std and mean |change| of x [B,T,D] over the frames `inside` [B,T]."""
    m = inside[:, :, None].to(x.dtype)
    n = m.sum(dim=1).clamp_min(1.0)
    mean = (x * m).sum(dim=1) / n
    var = ((x - mean[:, None]) ** 2 * m).sum(dim=1) / (n - 1).clamp_min(1.0)
    # a change belongs to the span only when both of its frames do
    pair = (inside[:, 1:] & inside[:, :-1])[:, :, None].to(x.dtype)
    change = ((x[:, 1:] - x[:, :-1]).abs() * pair).sum(dim=1) / pair.sum(dim=1).clamp_min(1.0)
    return torch.cat((mean, (var.clamp_min(0.0) + VARIANCE_EPS).sqrt(), change), dim=1)


def _fixed_edges(frames: int, lo: int, windows: int) -> list[int]:
    """Window edges when every clip is read over the same frames [lo, frames)."""
    if lo + MIN_FRAMES_PER_WINDOW * windows > frames:
        raise ValueError(f"{frames} frames cannot hold {windows} window(s) after frame {lo}")
    return [lo + (frames - lo) * j // windows for j in range(windows + 1)]


def _slice_stats(x: torch.Tensor) -> torch.Tensor:
    """mean, std and mean |change| over every frame of x [B, n, D]: the unmasked case."""
    mean = x.mean(dim=1)
    var = x.var(dim=1, unbiased=True)
    change = (x[:, 1:] - x[:, :-1]).abs().mean(dim=1)
    return torch.cat((mean, (var.clamp_min(0.0) + VARIANCE_EPS).sqrt(), change), dim=1)


def windowed(signals: torch.Tensor, windows: int = 1, lo: int = 0,
             hi: torch.Tensor | None = None) -> torch.Tensor:
    """[B, T, D] -> [B, windows * 3 * D]: the three statistics per window.

    Each clip's own span [lo, hi) is cut into `windows` equal parts. With one
    window this is the whole-span read, which is order-free: mean and spread
    are functions of the multiset of frames, and the mean absolute change is
    too except at the one junction where two halves of a sequence meet. With
    no `hi`, every clip is read over the same frames, and the windows are
    plain slices.
    """
    b, t, _ = signals.shape
    if hi is None:
        e = _fixed_edges(t, lo, windows)
        return torch.cat([_slice_stats(signals[:, e[j]:e[j + 1]]) for j in range(windows)], dim=1)
    lo_t, hi_t = span(t, lo, hi, windows, b, signals.device)
    edges = _window_edges(lo_t, hi_t, windows)
    frame = torch.arange(t, device=signals.device)[None, :]
    return torch.cat([_masked_stats(signals, (frame >= edges[:, j, None]) & (frame < edges[:, j + 1, None]))
                      for j in range(windows)], dim=1)


def pooled(signals: torch.Tensor, lo: int = 0, hi: torch.Tensor | None = None) -> torch.Tensor:
    """[B, T, D] -> [B, 3 * D]: the whole-span, order-free read."""
    return windowed(signals, 1, lo, hi)


def rotation_rate(sincos: torch.Tensor, windows: int = 1, lo: int = 0,
                  hi: torch.Tensor | None = None) -> torch.Tensor:
    """The field's signed mean rotation: [B, T, 2N] -> [B, windows * 2N].

    `sincos` holds sin θ for every oscillator, then cos θ. Per oscillator and
    per window this is the mean of cos Δθ and of sin Δθ between consecutive
    frames: a locked oscillator's increment encodes the stimulus frequency, an
    unlocked one's its own. It is nonlinear in the state, so unlike a signed
    mean difference it does not telescope to the endpoints.
    """
    b, t, d = sincos.shape
    n = d // 2
    s, c = sincos[..., :n], sincos[..., n:]
    cos_d = c[:, 1:] * c[:, :-1] + s[:, 1:] * s[:, :-1]
    sin_d = s[:, 1:] * c[:, :-1] - c[:, 1:] * s[:, :-1]
    if hi is None:
        # pair k joins frames k and k + 1; window j holds the pairs whose frames are both inside it
        e = _fixed_edges(t, lo, windows)
        return torch.cat([torch.cat((cos_d[:, e[j]:e[j + 1] - 1].mean(dim=1),
                                     sin_d[:, e[j]:e[j + 1] - 1].mean(dim=1)), dim=1)
                          for j in range(windows)], dim=1)
    lo_t, hi_t = span(t, lo, hi, windows, b, sincos.device)
    edges = _window_edges(lo_t, hi_t, windows)
    frame = torch.arange(1, t, device=sincos.device)[None, :]   # the later frame of each pair
    out = []
    for j in range(windows):
        inside = ((frame > edges[:, j, None]) & (frame < edges[:, j + 1, None]))[:, :, None].to(s.dtype)
        count = inside.sum(dim=1).clamp_min(1.0)
        out += [(cos_d * inside).sum(dim=1) / count, (sin_d * inside).sum(dim=1) / count]
    return torch.cat(out, dim=1)


def projection_matrix(native: int, width: int) -> torch.Tensor:
    """The fixed [native, width] random projection: the leading `width` columns of one draw.

    One [native, MAX_WIDTH] Gaussian matrix is drawn per native width, from one
    seed, and cached; every width takes its leading columns. Columns of a
    Gaussian matrix are independent, so each width is a projection in its own
    right, and a wide read contains the narrow one: any accuracy the wide read
    adds comes from the added features alone.
    """
    if width > MAX_WIDTH:
        raise ValueError(f"width {width} exceeds the {MAX_WIDTH}-column projection")
    if native not in _PROJECTIONS:
        if native > LARGE_NATIVE:        # a large draw is gigabytes: hold one at a time
            for k in [k for k in _PROJECTIONS if k > LARGE_NATIVE]:
                del _PROJECTIONS[k]
        gen = torch.Generator().manual_seed(PROJECTION_SEED)
        _PROJECTIONS[native] = torch.randn(native, MAX_WIDTH, generator=gen) / math.sqrt(native)
    return _PROJECTIONS[native][:, :width]


def project(features: torch.Tensor, width: int) -> torch.Tensor:
    """Bring features to a common width with one fixed random projection.

    An arm is never read wider than it is: features already at or under `width`
    pass through unchanged, so the narrowest arm sets the smallest meaningful
    width and is read there exactly as it stands. The 1/sqrt(native) scale
    keeps the projected variance comparable across native widths.
    """
    native = features.shape[1]
    if native <= width:
        return features
    return features @ projection_matrix(native, width).to(device=features.device, dtype=features.dtype)
