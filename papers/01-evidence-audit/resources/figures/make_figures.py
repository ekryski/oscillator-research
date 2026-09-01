#!/usr/bin/env python3
"""Generate the survey's figures.

    uv run --with svglib --with reportlab python make_figures.py

Writes three files per figure: the .svg source, a vector .pdf for the LaTeX
builds, and a .png for HTML, EPUB and DOCX. The manuscript references each
figure WITHOUT an extension, and pandoc fills in the right one per format.

Two constraints shape everything here. The figures must stay legible in
greyscale at print size, so nothing is distinguished by hue alone — fills,
outlines and labels carry the meaning. And svglib, which does the SVG-to-PDF
conversion, silently ignores `<marker>` elements: every arrowhead is therefore
an explicit path, and adding a marker would drop it from the PDF without
raising anything.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FONT = "Helvetica, Arial, sans-serif"

# a greyscale ramp, so the figures survive a black-and-white printer
INK = "#24231F"
MID = "#5F5E5A"
FAINT = "#9C9A94"
HAIR = "#C9C7C1"
FILL_A = "#F2F1EE"      # untrained / physics
FILL_B = "#DEDCD6"      # trained / ML
FILL_C = "#FFFFFF"      # neutral


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=11, anchor="middle", fill=INK, weight="normal", style="normal"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" font-style="{style}">{esc(s)}</text>')


def box(x, y, w, h, fill=FILL_C, stroke=INK, rx=4, sw=1.2, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, stroke=MID, sw=1.2, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linecap="round"{d}/>')


def arrow(x1, y1, x2, y2, stroke=MID, sw=1.4, head=5):
    """A line with an explicit triangular head — markers do not survive to PDF."""
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(a), y2 - head * math.sin(a)
    p1 = (x2, y2)
    p2 = (bx - head * 0.6 * math.sin(a), by + head * 0.6 * math.cos(a))
    p3 = (bx + head * 0.6 * math.sin(a), by - head * 0.6 * math.cos(a))
    pts = " ".join(f"{px:.1f},{py:.1f}" for px, py in (p1, p2, p3))
    return (line(x1, y1, bx, by, stroke, sw)
            + f'<polygon points="{pts}" fill="{stroke}"/>')


def circle(cx, cy, r, fill=FILL_C, stroke=INK, sw=1.1):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"/>')


def svg(name: str, w: int, h: int, body: str, title: str) -> None:
    doc = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'viewBox="0 0 {w} {h}" font-family="{FONT}">\n'
           f'<title>{esc(title)}</title>\n'
           f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>\n{body}\n</svg>\n')
    (HERE / f"{name}.svg").write_text(doc)


#: reportlab's built-in Helvetica is Latin-1 only, which silently drops the
#: caron and acute in Bačić, Kuśmierz and Buzsáki. Substituting the metrically
#: identical Arial TTFs under the same names restores them. Best effort: on a
#: host without these files the figures render as before, minus those glyphs.
ARIAL = {"Helvetica": "Arial.ttf", "Helvetica-Bold": "Arial Bold.ttf",
         "Helvetica-Oblique": "Arial Italic.ttf"}
ARIAL_DIR = Path("/System/Library/Fonts/Supplemental")


def use_unicode_fonts() -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    for name, filename in ARIAL.items():
        path = ARIAL_DIR / filename
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))


def render(name: str) -> None:
    """SVG -> vector PDF (svglib) -> PNG (poppler). No cairo anywhere."""
    from reportlab.graphics import renderPDF
    from svglib.svglib import svg2rlg
    use_unicode_fonts()
    drawing = svg2rlg(str(HERE / f"{name}.svg"))
    renderPDF.drawToFile(drawing, str(HERE / f"{name}.pdf"))
    subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile",
                    str(HERE / f"{name}.pdf"), str(HERE / name)],
                   check=True, env={"PATH": "/opt/homebrew/bin:/usr/bin:/bin"})


# --------------------------------------------------------------------------
# timeline machinery, shared by the three timeline figures
# --------------------------------------------------------------------------
#: average glyph width at the label font size, used to reserve horizontal room
CHAR_W = 4.4


def piecewise(segments):
    """year -> x, over segments of the form (year_lo, year_hi, x_lo, x_hi).

    The record these figures plot is wildly uneven in density: four events
    between 1665 and 1926, then thirty after 1950. A linear axis stacks the
    modern half into two centimetres, so each figure buys width where its
    events are and pays for it where they are not.
    """
    def px(year):
        for lo, hi, xlo, xhi in segments:
            if year <= hi:
                return xlo + (xhi - xlo) * (year - lo) / (hi - lo)
        lo, hi, xlo, xhi = segments[-1]
        return xhi
    return px


def pack_lanes(events, px, flip_x, char_w=CHAR_W):
    """Assign each label the lowest lane it fits in without overlapping.

    Hand-assigned lanes were guesswork and collided the moment a label was
    reworded; this measures each one and packs greedily instead. Labels past
    `flip_x` are drawn leftwards so they stay inside the canvas.
    """
    lanes: list[list[tuple[float, float]]] = []
    out = []
    for year, label in events:
        x = px(year)
        w = len(label) * char_w + 12
        flip = x > flip_x
        span = (x - w, x + 6) if flip else (x - 6, x + w)
        for k, occupied in enumerate(lanes):
            if all(span[1] < a or span[0] > b for a, b in occupied):
                occupied.append(span)
                out.append((x, label, k, flip))
                break
        else:
            lanes.append([span])
            out.append((x, label, len(lanes) - 1, flip))
    return out


def lane_count(events, px, flip_x, size=8.4) -> int:
    """How many lanes the labels need, so a figure can size itself before drawing."""
    return max(lane for _, _, lane, _ in pack_lanes(events, px, flip_x, size / 1.9)) + 1


def axis(b, x0, x1, mid, px, years, breaks=()):
    """The rule, its year ticks, and a dashed mark wherever the scale changes."""
    b.append(line(x0, mid, x1, mid, INK, 1.5))
    for yr in years:
        x = px(yr)
        b.append(line(x, mid - 4, x, mid + 4, INK, 1))
        b.append(text(x, mid + 17, str(yr), 8, fill=MID))
    for bx in breaks:
        b.append(line(bx, mid - 12, bx, mid + 12, FAINT, 1, "3 3"))


def marker(b, x, y, filled, square):
    """Circle above the axis, square below; filled or open carries the meaning."""
    if square:
        b.append(f'<rect x="{x-3.4}" y="{y-3.4}" width="6.8" height="6.8" '
                 f'fill="{INK if filled else FILL_C}" stroke="{INK}" stroke-width="1.2"/>')
    else:
        b.append(circle(x, y, 3.4, INK if filled else FILL_C, INK, 1.2))


def track(b, events, px, mid, W, up, lane_h, offset, square, filled=None, size=8.4):
    """One side of a timeline: leader lines first, then every marker and label.

    Drawing each item complete before starting the next puts a later point's
    leader line straight through an earlier point's text, so the two passes
    are not an optimisation.
    """
    filled = filled or (lambda label: True)
    placed = [(x, mid + (-offset - lane * lane_h if up else offset + lane * lane_h), lab, flip)
              for x, lab, lane, flip in pack_lanes(events, px, W * 0.62, size / 1.9)]
    for x, y, _, _ in placed:
        b.append(line(x, mid - 2 if up else mid + 2, x, y + (4 if up else -4), HAIR, 1))
    for x, y, label, flip in placed:
        marker(b, x, y, filled(label), square)
        tx = x + (-7 if flip else 7)
        w = len(label) * size / 1.9
        b.append(f'<rect x="{tx - (w if flip else 0)}" y="{y-6}" width="{w}" height="12" '
                 f'fill="#FFFFFF"/>')
        b.append(text(tx, y + 3.5, label, size, anchor="end" if flip else "start"))
    return max(abs(y - mid) for _, y, _, _ in placed) if placed else 0


# --------------------------------------------------------------------------
def fig_model_lineage():
    """Every dynamics form a surveyed system is actually built on.

    The earlier version showed four coupling laws chosen for how they read as
    equations. That misdescribes the survey: AKOrN's oscillators leave the
    circle entirely, and the coRNN and LinOSS lines carry no coupling term to
    write down. Each panel therefore names the systems that use it, and the
    forms with no system are in Appendix A rather than here.
    """
    W, H = 660, 508
    b = []
    panels = [
        ("Winfree", "phase only",
         "dθi/dt = ωi + Z(θi) Σj P(θj)",
         "influence P and sensitivity Z are separate functions",
         "WONN"),
        ("Kuramoto", "phase only",
         "dθi/dt = ωi + (K/N) Σj sin(θj − θi)",
         "a single sinusoid of the phase difference",
         "Un-0 · KomplexNet · Kuramoto Attention · Kuramoto Orientation Diffusion"),
        ("Kuramoto–Sakaguchi", "phase only",
         "dθi/dt = ωi + (K/N) Σj sin(θj − θi − α)",
         "the same sinusoid, carrying a phase lag α",
         "FSN"),
        ("Daido harmonics, with delay", "phase only",
         "dθi/dt = ωi + Σj Γ(θj(t − τ) − θi)",
         "an arbitrary periodic Γ, and coupling that depends on history",
         "FSN"),
        ("D-dimensional Kuramoto", "phase on a sphere",
         "dxi/dt = Ωi xi + Pxi (ci + Σj Jij xj)",
         "unit vectors in D dimensions; the phase leaves the circle",
         "AKOrN"),
        ("Stuart–Landau", "amplitude and phase",
         "dzi/dt = (μ + iωi)zi − |zi|²zi + K Σj (zj − zi)",
         "complex state: amplitude and phase both evolve",
         "Zhang et al. 2026"),
        ("Damped second-order", "no coupling term",
         "y″ = σ(Wy + W′y′ + Vu + b) − γy − εy′",
         "an oscillatory recurrence; damping bounds the gradients",
         "coRNN · UnICORNN · RON · Neural Wave Machines"),
        ("Linear oscillatory state space", "no coupling term",
         "y″ = −Ay − Gy′ + Bu,   A, G diagonal",
         "linear, so the whole sequence solves by parallel scan",
         "LinOSS · D-LinOSS"),
    ]
    fills = {"phase only": FILL_A, "phase on a sphere": FILL_A,
             "amplitude and phase": FILL_B, "no coupling term": FILL_C}
    pw, ph, gx, gy = 310, 104, 14, 13
    for i, (name, kind, eq, note, systems) in enumerate(panels):
        x = 18 + (i % 2) * (pw + gx)
        y = 20 + (i // 2) * (ph + gy)
        b.append(box(x, y, pw, ph, fills[kind]))
        b.append(text(x + 14, y + 22, name, 11, anchor="start", weight="bold"))
        b.append(text(x + pw - 14, y + 21, kind, 8, anchor="end", fill=MID))
        b.append(line(x + 14, y + 31, x + pw - 14, y + 31, HAIR, 1))
        b.append(text(x + pw / 2, y + 55, eq, 10))
        b.append(text(x + pw / 2, y + 75, note, 8.2, fill=MID))
        b.append(text(x + pw / 2, y + 93, systems, 8.4, weight="bold"))
    b.append(line(18, 486, 642, 486, HAIR, 1))
    b.append(text(18, 500, "Physical substrates run their own device dynamics and are written "
                  "as none of these forms: the spintronic reservoir, and oscillator Ising "
                  "machines.", 8.2, anchor="start", fill=MID))
    svg("fig10-model-lineage", W, H, "\n".join(b),
        "The oscillator dynamics each surveyed system is built on")


def fig_coupling_regimes():
    """The three regimes the physics literature reports, drawn schematically."""
    import math
    W, H = 660, 210
    b = []
    panels = [
        ("(a) Global or strong coupling", "full synchrony", "R ≈ 1",
         [0.02, -0.03, 0.01, 0.0, -0.02, 0.03, 0.0, -0.01,
          0.02, 0.01, -0.02, 0.0, 0.03, -0.01, 0.01, 0.0],),
        ("(b) Nonlocal, finite range", "chimera: both at once", "0 < R < 1",
         [0.05, 0.0, -0.04, 0.02, 0.01, -0.03, 0.0, 0.04,
          2.1, 4.8, 1.2, 5.9, 3.3, 0.4, 4.1, 2.7],),
        ("(c) Weak coupling", "incoherence", "R ≈ 0",
         [0.9, 3.4, 5.8, 2.1, 4.4, 1.1, 6.0, 3.9,
          0.3, 2.8, 5.1, 1.7, 4.9, 3.1, 0.6, 5.5],),
    ]
    for p, (title, sub, order, phases) in enumerate(panels):
        cx, cy, R = 120 + p * 215, 132, 52
        b.append(text(cx, 30, title, 11, weight="bold"))
        b.append(text(cx, 48, sub, 10, fill=MID))
        b.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="{HAIR}" '
                 f'stroke-width="1" stroke-dasharray="3 3"/>')
        for k, ph in enumerate(phases):
            a = 2 * math.pi * k / len(phases) - math.pi / 2
            ox, oy = cx + R * math.cos(a), cy + R * math.sin(a)
            b.append(circle(round(ox, 1), round(oy, 1), 7, FILL_C, INK, 1))
            # the little hand shows each oscillator's phase
            hx, hy = ox + 5.2 * math.cos(ph - math.pi / 2), oy + 5.2 * math.sin(ph - math.pi / 2)
            b.append(line(round(ox, 1), round(oy, 1), round(hx, 1), round(hy, 1), INK, 1.6))
        b.append(text(cx, cy + 4, order, 12, weight="bold"))
    svg("fig2-coupling-regimes", W, H, "\n".join(b),
        "Three coupling regimes reported in the synchronization literature")


def fig_taxonomy():
    """The two questions that separate the surveyed systems, drawn as a grid.

    The earlier version was a tree, which forced a single ordering on two
    independent questions and left the third branch hanging off the root. The
    questions are independent: whether gradients reach the dynamics, and
    whether a trained conventional network wraps it. Crossing them gives four
    cells, all four occupied, and puts the attribution burden on one axis.
    """
    W, H = 660, 304
    b = []
    lx, ty, cw, ch, gx, gy = 168, 44, 232, 118, 16, 16
    cells = [
        ("Untrained, designed", "nothing is fitted anywhere in the system",
         ["Oscillator Ising machines", "(Hopf cochlea, the biological case)"], FILL_A),
        ("Untrained, with a trained readout", "the readout is fitted, the dynamics is not",
         ["ESN · LSM", "RON · Spintronic reservoir"], FILL_A),
        ("Trained, designed", "gradients reach the dynamics, and nothing else",
         ["Un-0", "WONN"], FILL_B),
        ("Trained, inside a trained network",
         "gradients reach the dynamics and its scaffolding",
         ["AKOrN · KomplexNet · HoloGraph", "Kuramoto Attention · FSN · NWM",
          "coRNN · UnICORNN · LinOSS · D-LinOSS", "Kuramoto Orientation Diffusion"], FILL_B),
    ]
    for i, (name, gloss, systems, fill) in enumerate(cells):
        x = lx + (i % 2) * (cw + gx)
        y = ty + (i // 2) * (ch + gy)
        b.append(box(x, y, cw, ch, fill))
        b.append(text(x + 14, y + 21, name, 10.4, anchor="start", weight="bold"))
        b.append(text(x + 14, y + 34, gloss, 8.2, anchor="start", fill=MID, style="italic"))
        b.append(line(x + 14, y + 43, x + cw - 14, y + 43, HAIR, 1))
        for k, s in enumerate(systems):
            b.append(text(x + 14, y + 60 + k * 13, s, 8.6, anchor="start"))

    # the column question runs along the top, the row question down the side
    b.append(text(lx + cw / 2, ty - 18, "no trained network around it", 9.2, weight="bold"))
    b.append(text(lx + cw + gx + cw / 2, ty - 18, "a trained encoder or decoder around it",
                  9.2, weight="bold"))
    b.append(line(lx, ty - 12, lx + 2 * cw + gx, ty - 12, HAIR, 1))
    for k, (row, ry) in enumerate((("the dynamics is fixed", ty + ch / 2),
                                   ("the dynamics is trained", ty + ch + gy + ch / 2))):
        b.append(text(lx - 16, ry - 4, row, 9.2, anchor="end", weight="bold"))
        b.append(text(lx - 16, ry + 9, "by design or at random" if k == 0
                      else "gradients flow through it", 8.2, anchor="end", fill=MID))
    b.append(line(lx - 10, ty, lx - 10, ty + 2 * ch + gy, HAIR, 1))
    svg("fig9-taxonomy", W, H, "\n".join(b),
        "Taxonomy of oscillator-based machine learning")


def fig_input_injection():
    """The four ways input enters, each marked against the term it touches.

    Markers sit directly under their term and the legend is a grid, so no
    leader line crosses a label — the staggered-arrow version was unreadable.
    """
    W, H = 660, 218
    b = []
    b.append(text(330, 28, "Where the input enters the dynamics", 12, weight="bold"))

    # the equation, laid out term by term at explicit positions
    eq_y = 84
    b.append(box(58, eq_y - 28, 544, 44, FILL_A, HAIR, 4, 1))
    terms = [(84, "dθi/dt   =", "start"), (188, "ωi", "middle"), (218, "+", "middle"),
             (248, "K", "middle"), (330, "Σj sin(θj − θi)", "middle"),
             (432, "+", "middle"), (486, "u(t)", "middle")]
    for x, s, anc in terms:
        b.append(text(x, eq_y, s, 13.5, anchor=anc))

    # numbered markers directly beneath the term each one annotates
    for x, num in ((188, "1"), (248, "2"), (330, "3"), (486, "4")):
        b.append(line(x, eq_y + 8, x, eq_y + 26, FAINT, 1))
        b.append(circle(x, eq_y + 34, 8.5, FILL_C, INK, 1.2))
        b.append(text(x, eq_y + 37.5, num, 9.5, weight="bold"))

    legend = [
        ("1", "written into the natural frequency", "WONN"),
        ("2", "input-gated coupling strength", "KomplexNet"),
        ("3", "an extra phase-coupled term", "AKOrN · Un-0"),
        ("4", "additive drive term", "coRNN · LinOSS"),
    ]
    for i, (num, label, systems) in enumerate(legend):
        col, row = i % 2, i // 2
        x, y = 60 + col * 312, 156 + row * 40
        b.append(circle(x + 9, y, 8.5, FILL_C, INK, 1.2))
        b.append(text(x + 9, y + 3.5, num, 9.5, weight="bold"))
        b.append(text(x + 26, y - 2, label, 10, anchor="start"))
        b.append(text(x + 26, y + 11, systems, 8.8, anchor="start", fill=MID))

    svg("fig11-input-injection", W, H, "\n".join(b),
        "The four input-injection forms found in the literature")


def fig_training_horizon():
    """Backpropagated steps per system, marked by what makes the gradient safe."""
    import math
    W, H = 660, 262
    b = []
    b.append(text(330, 24, "Backpropagated steps across trained oscillator models",
                  11.5, weight="bold"))
    x0, x1, axis_y = 96, 604, 176
    lo, hi = 1, 20000

    def px(v):
        return x0 + (x1 - x0) * math.log10(v) / math.log10(hi / lo)

    b.append(line(x0, axis_y, x1, axis_y, INK, 1.4))
    for tick in (1, 10, 100, 1000, 10000):
        x = px(tick)
        b.append(line(x, axis_y, x, axis_y + 6, INK, 1.2))
        b.append(text(x, axis_y + 19, f"{tick:,}", 9, fill=MID))
    b.append(text(350, axis_y + 38, "steps unrolled and backpropagated through (log scale)",
                  9.4, fill=MID))

    # filled = nonlinear dynamics, nothing bounding the gradient
    # open   = linear, or gradients bounded by dissipation
    points = [
        (1, "Kuramoto Attention · FSN", "one dynamics step per layer", True, -112),
        (1, "Kuramoto Orientation Diffusion", "per-step score objectives", True, -80),
        (17, "Un-0", "10–25 integration steps", True, -48, 24),
        (18, "WONN", "18 steps (6 layers × 3)", True, -20),
        (18000, "LinOSS · D-LinOSS", "linear, implicitly discretised", False, -48),
    ]
    for v, name, note, nonlinear, dy, *extra in points:
        x = px(v)
        y = axis_y + dy
        b.append(line(x, y + 10, x, axis_y - 2, HAIR, 1))
        b.append(f'<rect x="{x-5}" y="{y-5}" width="10" height="10" '
                 f'fill="{INK if nonlinear else FILL_C}" stroke="{INK}" stroke-width="1.3"/>')
        # a point carrying a range bar needs its label clear of the bar's end
        anc, off = ("end", -11) if v > 100 else ("start", 11 + (extra[0] - 11 if extra else 0))
        b.append(text(x + off, y - 1, name, 9.6, anchor=anc, weight="bold"))
        b.append(text(x + off, y + 10, note, 8.4, anchor=anc, fill=MID))
    # Un-0's range
    b.append(line(px(10), axis_y - 48, px(25), axis_y - 48, INK, 3))

    # the marker legend stays: it decodes the marks, and without it the figure
    # cannot be read. The prose that sat under it is the caption's job.
    b.append(f'<rect x="20" y="240" width="9" height="9" fill="{INK}" stroke="{INK}"/>')
    b.append(text(35, 248, "nonlinear dynamics, no stated gradient bound", 9, anchor="start", fill=MID))
    b.append(f'<rect x="330" y="240" width="9" height="9" fill="{FILL_C}" stroke="{INK}"/>')
    b.append(text(345, 248, "linear, or gradients bounded by dissipation", 9, anchor="start", fill=MID))
    svg("fig12-training-horizon", W, H, "\n".join(b),
        "Training horizon across oscillator systems")


def fig_controls_grid():
    """Which controls each system reports — the sparsest column is the observation."""
    W, H = 660, 362
    b = []
    b.append(text(330, 24, "Controls reported, by system", 11.5, weight="bold"))
    cols = ["Frozen\ntwin", "Decoder-\nonly", "Randomized\ntwin", "Matched-budget\nbaseline",
            "Single-term\nablation of\nthe coupling"]
    rows = [
        ("Un-0", [1, 1, 1, 0, 0]),
        ("AKOrN", [0, 0, 0, 0, 1]),
        ("WONN", [0, 0, 0, 0, 0]),
        ("Kuramoto Attention · FSN", [0, 0, 0, 1, 0]),
        ("KomplexNet", [0, 0, 0, 0, 3]),
        ("Kuramoto Orientation Diffusion", [0, 0, 0, 0, 1]),
        ("Neural Wave Machines", [0, 0, 0, 0, 0]),
        ("coRNN · UnICORNN", [0, 0, 0, 0, 2]),
        ("LinOSS · D-LinOSS", [0, 0, 0, 0, 2]),
        ("RON", [2, 0, 0, 1, 0]),
        ("Spintronic reservoir", [2, 0, 0, 0, 0]),
        ("Oscillator Ising machines", [2, 0, 0, 0, 0]),
    ]
    lx, cx0, cw, ry0, rh = 20, 244, 78, 98, 19
    for c, label in enumerate(cols):
        for k, part in enumerate(label.split("\n")):
            b.append(text(cx0 + c * cw + cw / 2, 52 + k * 11, part, 8.2, fill=MID))
    for r, (name, marks) in enumerate(rows):
        y = ry0 + r * rh
        if r % 2 == 0:
            b.append(f'<rect x="{lx}" y="{y-13}" width="{cx0 - lx + 5*cw - 4}" height="{rh}" '
                     f'fill="{FILL_A}" stroke="none"/>')
        b.append(text(lx + 6, y, name, 9.2, anchor="start"))
        for c, m in enumerate(marks):
            mx, my = cx0 + c * cw + cw / 2, y - 4
            if m == 1:
                b.append(f'<rect x="{mx-6}" y="{my-6}" width="12" height="12" fill="{INK}"/>')
            elif m == 3:
                # the closest published instance, not the control itself
                b.append(f'<rect x="{mx-6}" y="{my-6}" width="12" height="12" fill="none" '
                         f'stroke="{INK}" stroke-width="1.3"/>')
                b.append(f'<path d="M{mx-6} {my+6} L{mx+6} {my+6} L{mx+6} {my-6} Z" fill="{INK}"/>')
            elif m == 0:
                b.append(f'<rect x="{mx-6}" y="{my-6}" width="12" height="12" fill="none" '
                         f'stroke="{HAIR}" stroke-width="1.2"/>')
            else:
                b.append(text(mx, my + 4, "n/a", 7.6, fill=FAINT))
    last = cx0 + 4 * cw
    b.append(f'<rect x="{last}" y="{ry0-16}" width="{cw}" height="{len(rows)*rh+2}" '
             f'fill="none" stroke="{INK}" stroke-width="1.6" stroke-dasharray="4 3"/>')
    b.append(f'<rect x="20" y="340" width="10" height="10" fill="{INK}"/>')
    b.append(text(36, 349, "reported", 9, anchor="start", fill=MID))
    b.append(f'<rect x="104" y="340" width="10" height="10" fill="none" stroke="{INK}" '
             f'stroke-width="1.3"/>')
    b.append('<path d="M104 350 L114 350 L114 340 Z" fill="' + INK + '"/>')
    b.append(text(120, 349, "the closest published instance", 9, anchor="start", fill=MID))
    b.append(f'<rect x="320" y="340" width="10" height="10" fill="none" stroke="{HAIR}" '
             f'stroke-width="1.2"/>')
    b.append(text(336, 349, "not found in the surveyed work", 9, anchor="start", fill=MID))
    svg("fig13-controls-grid", W, H, "\n".join(b), "Controls reported, by system")


def polyline(pts, stroke=INK, sw=1.2, fill="none", dash=""):
    """A path of straight segments. Curves are sampled into these deliberately:
    svglib renders `A` arc commands inconsistently, and a sampled curve is
    indistinguishable at print size."""
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round"{da}/>')


def arc(cx, cy, r, a0, a1, n=24):
    """Points along a circular arc, angles in radians."""
    import math
    return [(cx + r * math.cos(a0 + (a1 - a0) * k / n),
             cy + r * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]


def arc_arrow(cx, cy, r, a0, a1, stroke=INK, sw=1.2, head=4.5):
    """A curved arrow, head drawn as an explicit triangle."""
    import math
    pts = arc(cx, cy, r, a0, a1)
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    a = math.atan2(y2 - y1, x2 - x1)
    p2 = (x2 - head * math.cos(a) - head * 0.6 * math.sin(a),
          y2 - head * math.sin(a) + head * 0.6 * math.cos(a))
    p3 = (x2 - head * math.cos(a) + head * 0.6 * math.sin(a),
          y2 - head * math.sin(a) - head * 0.6 * math.cos(a))
    pts_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in ((x2, y2), p2, p3))
    return polyline(pts[:-1], stroke, sw) + f'<polygon points="{pts_str}" fill="{stroke}"/>'


def phase_dot(cx, cy, theta, r=8, hand=5.4):
    """An oscillator drawn as a clock face: the hand is its phase."""
    import math
    hx, hy = cx + hand * math.cos(theta - math.pi / 2), cy + hand * math.sin(theta - math.pi / 2)
    return circle(cx, cy, r, FILL_C, INK, 1.1) + line(cx, cy, hx, hy, INK, 1.6)


# --------------------------------------------------------------------------
def fig_oscillator_anatomy():
    """What the word "oscillator" denotes here, in three widening steps.

    The survey's definition names phase-only models, second-order ODEs and
    amplitude-phase models in one sentence, which assumes a reader who already
    knows these are one object described at different resolutions. The panels
    are that sentence drawn: one angle, then an angle and a radius, then a
    population of them.
    """
    import math
    W, H = 660, 248
    b = []
    pw, gx, x0, top, ph = 206, 11, 10, 16, 212
    # the three closing lines sit one leading apart; the panel keeps the drawing
    # clear of both the title above it and the note below it, which the network
    # in (c) needs most because its ring is the tallest thing in the figure
    note_y, block_y, line_h = top + 148, top + 166, 16
    panels = [
        ("(a) A phase oscillator", "the state space is a circle",
         "one angle θ, advancing at rate ω", "Kuramoto · Winfree · Sakaguchi"),
        ("(b) An amplitude-phase oscillator", "the state space is a plane",
         "a radius r and an angle θ", "Stuart–Landau · coRNN · LinOSS"),
        ("(c) A network of them", "the state space is N of the above",
         "N phases, coupled with strength K", "every system in this survey"),
    ]
    for i, (title, mid_line, state, uses) in enumerate(panels):
        x = x0 + i * (pw + gx)
        b.append(box(x, top, pw, ph, FILL_C, HAIR, 4, 1))
        b.append(text(x + pw / 2, top + 19, title, 10, weight="bold"))
        b.append(text(x + pw / 2, block_y, mid_line, 8.6, fill=MID))
        b.append(text(x + pw / 2, block_y + line_h, state, 8.4, fill=MID))
        b.append(text(x + pw / 2, block_y + 2 * line_h, uses, 8.4, fill=MID, style="italic"))

    # (a) a pendulum, and the one number that describes it
    x = x0
    pvx, pvy, rod = x + 50, top + 54, 56
    ang = math.radians(62)          # from the +x axis, so 28 degrees off vertical
    bx, by = pvx + rod * math.cos(ang), pvy + rod * math.sin(ang)
    b.append(line(pvx - 16, pvy, pvx + 16, pvy, INK, 1.4))
    b.append(line(pvx, pvy, pvx, pvy + rod + 4, HAIR, 1, "3 3"))
    b.append(polyline(arc(pvx, pvy, rod, ang, math.pi - ang), HAIR, 1, dash="2 3"))
    b.append(line(pvx, pvy, bx, by, INK, 1.4))
    b.append(circle(round(bx, 1), round(by, 1), 8, FILL_B, INK, 1.2))
    b.append(polyline(arc(pvx, pvy, 24, math.pi / 2, ang), FAINT, 1))
    b.append(text(pvx + 15, pvy + 28, "θ", 10, anchor="start"))
    ccx, ccy, cr = x + 148, top + 88, 30
    b.append(arrow(x + 92, top + 88, x + 110, top + 88, MID, 1.2, 4.5))
    b.append(f'<circle cx="{ccx}" cy="{ccy}" r="{cr}" fill="none" stroke="{INK}" stroke-width="1.4"/>')
    tp = math.radians(-52)
    b.append(line(ccx, ccy, ccx + cr * math.cos(tp), ccy + cr * math.sin(tp), INK, 1.4))
    b.append(circle(round(ccx + cr * math.cos(tp), 1), round(ccy + cr * math.sin(tp), 1),
                    4, INK, INK, 0))
    b.append(polyline(arc(ccx, ccy, 12, 0, tp), FAINT, 1))
    b.append(text(ccx + 16, ccy - 3, "θ", 9, anchor="start"))
    b.append(arc_arrow(ccx, ccy, cr + 9, math.radians(-105), math.radians(-165), MID, 1.1))
    b.append(text(ccx - 6, ccy - cr - 15, "ω", 9, anchor="end", fill=MID))

    # (b) a mass on a spring, and the limit cycle its trajectories settle onto
    x = x0 + pw + gx
    wx, wy = x + 16, top + 66
    b.append(line(wx, wy - 20, wx, wy + 20, INK, 1.4))
    for k in range(5):
        b.append(line(wx + 2 + k * 6, wy - 8, wx + 5 + k * 6, wy + 8, INK, 1.1))
        b.append(line(wx + 5 + k * 6, wy + 8, wx + 8 + k * 6, wy - 8, INK, 1.1))
    b.append(box(wx + 34, wy - 12, 24, 24, FILL_B, INK, 2, 1.2))
    b.append(arrow(wx + 46, wy + 24, wx + 68, wy + 24, MID, 1.1, 4))
    b.append(arrow(wx + 46, wy + 24, wx + 24, wy + 24, MID, 1.1, 4))
    ccx, ccy, cr = x + 142, top + 74, 30
    b.append(arrow(x + 92, top + 74, x + 106, top + 74, MID, 1.2, 4.5))
    b.append(line(ccx - 44, ccy, ccx + 44, ccy, HAIR, 1))
    b.append(line(ccx, ccy - 44, ccx, ccy + 44, HAIR, 1))
    # trajectories relaxing onto the cycle, from outside it and from inside it
    for r_start, r_end, turns, a_start in ((46, 33, 0.8, 2.4), (8, 26, 1.0, 0.6)):
        pts = [(ccx + (r_start + (r_end - r_start) * (k / 60)) * math.cos(
                    a_start + turns * 2 * math.pi * k / 60),
                ccy + (r_start + (r_end - r_start) * (k / 60)) * math.sin(
                    a_start + turns * 2 * math.pi * k / 60)) for k in range(61)]
        b.append(polyline(pts[:-3], FAINT, 1))
        b.append(arrow(pts[-4][0], pts[-4][1], pts[-1][0], pts[-1][1], FAINT, 1, 4))
    b.append(f'<circle cx="{ccx}" cy="{ccy}" r="{cr}" fill="none" stroke="{INK}" stroke-width="1.6"/>')
    ta = math.radians(-48)
    b.append(line(ccx, ccy, ccx + cr * math.cos(ta), ccy + cr * math.sin(ta), INK, 1.4))
    b.append(circle(round(ccx + cr * math.cos(ta), 1), round(ccy + cr * math.sin(ta), 1),
                    4, INK, INK, 0))
    b.append(text(ccx + 13, ccy - 16, "r", 9, anchor="start"))
    b.append(polyline(arc(ccx, ccy, 12, 0, ta), FAINT, 1))
    b.append(text(ccx + 17, ccy - 2, "θ", 9, anchor="start"))
    b.append(text(x + pw / 2, note_y, "trajectories relax onto a limit cycle", 8, fill=MID))

    # (c) a population, each unit keeping its own phase
    x = x0 + 2 * (pw + gx)
    ccx, ccy, ring = x + pw / 2, top + 86, 41
    phases = [0.2, 0.5, 0.35, 2.9, 3.1, 1.1, 0.15, 4.4, 0.4, 2.2, 0.05, 5.6]
    pos = [(ccx + ring * math.cos(2 * math.pi * k / 12 - math.pi / 2),
            ccy + ring * math.sin(2 * math.pi * k / 12 - math.pi / 2)) for k in range(12)]
    for j in range(12):
        for k in range(j + 1, 12):
            b.append(line(round(pos[j][0], 1), round(pos[j][1], 1),
                          round(pos[k][0], 1), round(pos[k][1], 1), HAIR, 0.5))
    for (nx, ny), ph in zip(pos, phases, strict=True):
        b.append(phase_dot(round(nx, 1), round(ny, 1), ph, 7.5, 5))
    b.append(f'<rect x="{ccx-24}" y="{ccy-11}" width="48" height="22" fill="#FFFFFF"/>')
    b.append(text(ccx, ccy + 5, "K", 13, weight="bold"))
    b.append(text(x + pw / 2, note_y, "each hand is one oscillator's phase", 8, fill=MID))
    svg("fig1-oscillator-anatomy", W, H, "\n".join(b),
        "What an oscillator is: one phase, a phase and an amplitude, and a coupled population")


def fig_geometries():
    """Four wirings of the same twelve oscillators.

    "Geometry" is used throughout the survey without a picture, and the gap it
    names in Section 5.3 is precisely that nobody has held a core fixed and
    varied this. Same units, same count, four coupling structures.
    """
    import math
    W, H = 660, 214
    b = []
    # panel width falls out of the canvas rather than being guessed at: the
    # hand-set 156 overran the right edge by ten points and the fourth panel
    # was clipped in every rendered format
    cols, gx, x0, top = 4, 12, 10, 16
    pw = (W - 2 * x0 - gx * (cols - 1)) / cols
    ring, node = 44, 6.4

    def ring_pos(cx, cy, n=12):
        return [(cx + ring * math.cos(2 * math.pi * k / n - math.pi / 2),
                 cy + ring * math.sin(2 * math.pi * k / n - math.pi / 2)) for k in range(n)]

    panels = []
    # (a) all-to-all: every pair coupled, O(N²) parameters
    panels.append(("(a) All-to-all", "every pair coupled, O(N²)", "Un-0 · AKOrN",
                   "ring", [(j, k) for j in range(12) for k in range(j + 1, 12)]))
    # (b) nonlocal ring: each unit to its two nearest neighbours on each side
    panels.append(("(b) Nonlocal ring", "finite coupling range", "chimera regime",
                   "ring", [(j, (j + d) % 12) for j in range(12) for d in (1, 2, 3)]))
    # (c) lattice: nearest neighbours on a 4x3 grid
    grid = [(c, r) for r in range(3) for c in range(4)]
    lat = []
    for i, (c, r) in enumerate(grid):
        for j, (c2, r2) in enumerate(grid):
            if j > i and abs(c - c2) + abs(r - r2) == 1:
                lat.append((i, j))
    panels.append(("(c) Lattice", "neighbours in space", "Neural Wave Machines",
                   "grid", lat))
    # (d) modular: three dense blocks, one link between neighbouring blocks
    mod = [(j, k) for blk in range(3) for j in range(blk * 4, blk * 4 + 4)
           for k in range(j + 1, blk * 4 + 4)]
    mod += [(3, 4), (7, 8), (11, 0)]
    panels.append(("(d) Modular", "dense blocks, sparse between", "Un-0 sparsity study",
                   "clusters", mod))

    for i, (title, sub, uses, layout, edges) in enumerate(panels):
        x = x0 + i * (pw + gx)
        cx, cy = x + pw / 2, top + 76
        b.append(box(x, top, pw, 182, FILL_C, HAIR, 4, 1))
        b.append(text(cx, top + 18, title, 9.6, weight="bold"))
        if layout == "ring":
            pos = ring_pos(cx, cy)
        elif layout == "grid":
            pos = [(cx - 51 + c * 34, cy - 34 + r * 34) for c, r in grid]
        else:
            # three separated blocks of four, so the modularity is visible as
            # distance and not only as edge density
            pos = [(bcx + dx, bcy + dy)
                   for bcx, bcy in ((cx - 36, cy - 26), (cx + 36, cy - 26), (cx, cy + 36))
                   for dx, dy in ((-11, -11), (11, -11), (-11, 11), (11, 11))]
        for j, k in edges:
            b.append(line(round(pos[j][0], 1), round(pos[j][1], 1),
                          round(pos[k][0], 1), round(pos[k][1], 1), FAINT, 0.7))
        for nx, ny in pos:
            b.append(circle(round(nx, 1), round(ny, 1), node, FILL_B, INK, 1.1))
        b.append(text(cx, top + 150, sub, 8.4, fill=MID))
        b.append(text(cx, top + 165, uses, 8.4, fill=MID, style="italic"))
    svg("fig5-geometries", W, H, "\n".join(b),
        "Four coupling geometries over the same twelve oscillators")


#: shared geometry for the two reference timelines
#: lane height, label size, header offset for the reference timelines. These
#: figures are tall because the packer needs a lane per colliding label, and a
#: figure taller than about a third of the text block cannot share a page with
#: prose. A smaller label is also a narrower one, so it collides less and needs
#: fewer lanes: the saving compounds.
REF_LANE_H, REF_SIZE, REF_HEADER = 14, 6.4, 18


def reference_timeline(name, x0, x1, segments, years, above, below, above_head,
                       below_head, title, extras=None, filled_above=None,
                       filled_below=None, head_in_corners=False):
    """A two-track timeline sized to fit its labels rather than to a guess.

    The label field is packed before anything is drawn, so the axis sits
    exactly as far down the canvas as the upper track needs and the canvas is
    exactly as tall as the lower track needs.

    head_in_corners puts the lower track's heading in the bottom-left corner,
    mirroring the upper one, instead of tucking it under the axis. Use it where
    the space beneath the axis is crossed by leader lines. With a legend present
    the heading sits directly above it rather than in the corner itself.
    """
    W = x1 + 30
    px = piecewise(segments)
    up_lanes = lane_count(above, px, W * 0.62, REF_SIZE)
    dn_lanes = lane_count(below, px, W * 0.62, REF_SIZE)
    mid = REF_HEADER + 16 + (up_lanes - 1) * REF_LANE_H + 24
    below_top = 46
    legend_h = 25 + 15 * (len(extras) - 1) if extras else 0
    H = int(mid + below_top + (dn_lanes - 1) * REF_LANE_H + 18 + legend_h
            + (10 if head_in_corners else 0))

    b = []
    axis(b, x0, x1, mid, px, years, [px(hi) for _, hi, _, _ in segments[:-1]])
    b.append(text(x1, REF_HEADER, "axis scale changes at each dashed mark", 7.4,
                  anchor="end", fill=FAINT))
    track(b, above, px, mid, W, up=True, lane_h=REF_LANE_H, offset=24, square=False,
          filled=filled_above, size=REF_SIZE)
    track(b, below, px, mid, W, up=False, lane_h=REF_LANE_H, offset=below_top,
          square=True, filled=filled_below or (lambda _: False), size=REF_SIZE)
    b.append(text(x0, REF_HEADER, above_head, 9, anchor="start", weight="bold", fill=MID))
    head_y = H - 12 - legend_h if head_in_corners else mid + 36
    b.append(text(x0, head_y, below_head, 9, anchor="start", weight="bold", fill=MID))
    for k, extra in enumerate(extras or []):
        b.append(extra(H - legend_h + 10 + k * 15))
    svg(name, W, H, "\n".join(b), title)


def fig_physics_timeline():
    """Every physics and mathematics source this survey cites, in date order.

    The two tracks are the two things the field handed to machine learning and
    they arrive in that order: first an account of when a coupled population
    locks, then the argument that such a population computes, and what its
    structure does to that. Almost everything a surveyed system actually uses
    comes from the upper track.
    """
    synchronization = [
        (1665, "Huygens"), (1830, "Airy"), (1908, "Poincaré"), (1919, "Blondel"),
        (1926, "Van der Pol"), (1944, "Landau"), (1960, "Stuart"), (1967, "Winfree"),
        (1975, "Kuramoto"), (1986, "Sakaguchi & Kuramoto"), (1989, "Schuster & Wagner"),
        (1991, "Ermentrout"), (1992, "Daido"), (1999, "Yeung & Strogatz"),
        (2000, "Strogatz review"), (2001, "Pikovsky et al."),
        (2002, "Kuramoto & Battogtokh"), (2004, "Abrams & Strogatz"),
    ]
    structure = [
        (1982, "Hopfield"), (1990, "Mead"),
        (2004, "Bertschinger & Natschläger"), (2011, "Tanaka & Aoyagi"),
        (2016, "Ashwin & Rodrigues"), (2019, "Skardal & Arenas"),
        (2019, "Chandra, Girvan & Ott"), (2019, "O'Keeffe & Bettstetter"),
        (2020, "Millán et al."), (2020, "Mulas et al."), (2022, "Calmon et al."),
        (2023, "de Aguiar"), (2023, "Zhang, Lucas & Battiston"),
        (2025, "Kuśmierz et al."), (2026, "Bačić et al."),
    ]
    reference_timeline(
        "fig3-physics-timeline", 40, 630,
        [(1665, 1930, 40, 150), (1930, 1990, 150, 320), (1990, 2026, 320, 630)],
        (1665, 1800, 1900, 1950, 1970, 1990, 2000, 2010, 2020),
        synchronization, structure,
        "WHEN A COUPLED POPULATION LOCKS",
        "PHYSICS AS COMPUTATION, AND COUPLING BEYOND PAIRWISE",
        "The physics and mathematics this survey cites, in date order",
        head_in_corners=True)


def fig_neuro_timeline():
    """Every neuroscience source this survey cites, in date order.

    Split the way the architectures use it. The upper track is the population
    account that supplies the hypotheses, rhythm as a gate on communication and
    as an organizer of information. The lower track is the one that makes the
    oscillator description literal rather than figurative, at the level of a
    single neuron and of the transducers feeding it.
    """
    population = [
        (1989, "Gray et al."), (2006, "Buzsáki"), (2006, "Uhlhaas & Singer"),
        (2010, "Breakspear et al."), (2011, "Cabral et al."),
        (2012, "Litwin-Kumar & Doiron"), (2015, "Fries"), (2016, "Iaccarino et al."),
        (2022, "Turcu & Abbott"), (2026, "Castaldo et al."),
    ]
    single_cell = [
        (1952, "Hodgkin & Huxley"), (1981, "Morris & Lecar"), (2000, "Camalet et al."),
        (2001, "Attwell & Laughlin"), (2003, "Kern & Stoop"),
        (2005, "Heimburg & Jackson"), (2016, "Stiefel & Ermentrout"),
        (2021, "Assaneo et al."), (2021, "Pittman-Polletta et al."),
        (2023, "Doelling et al."), (2026, "Dogonasheva et al."),
    ]
    reference_timeline(
        "fig4-neuro-timeline", 40, 630,
        [(1952, 2000, 40, 250), (2000, 2026, 250, 630)],
        (1952, 1960, 1970, 1980, 1990, 2000, 2005, 2010, 2015, 2020, 2025),
        population, single_cell,
        "RHYTHM ACROSS POPULATIONS OF NEURONS",
        "SINGLE NEURONS, TRANSDUCTION AND SPEECH",
        "The neuroscience this survey cites, in date order",
        head_in_corners=True)


def fig_ann_timeline():
    """The conventional architectures, split on how each one holds the past.

    Section 2.3 compares oscillator networks against this lineage rather than
    against a single baseline, and the split is the comparison that matters:
    an oscillator field belongs to the upper track, carrying a state forward
    step by step, but its state evolves under fixed physics rather than under
    a weight matrix learned from scratch.
    """
    stateful = [
        (1990, "Elman RNN"), (1997, "LSTM"), (2014, "GRU"),
        (2019, "Spiking, surrogate gradients"), (2021, "S4"), (2023, "Mamba"),
    ]
    windowed = [
        (1998, "LeNet, convolutional"), (2017, "Transformer"),
        (2024, "Jamba, attention plus state space"),
    ]
    reference_timeline(
        "fig8-ann-timeline", 40, 630,
        [(1990, 2014, 40, 300), (2014, 2026, 300, 630)],
        (1990, 1995, 2000, 2005, 2010, 2014, 2018, 2022, 2026),
        stateful, windowed,
        "STATE CARRIED FORWARD ONE STEP AT A TIME",
        "THE WHOLE WINDOW RECOMPUTED EACH STEP",
        "Conventional neural network architectures, by how each holds the past",
        head_in_corners=True)


#: coupling forms at least one surveyed machine-learning system is built on
ADOPTED = {
    "Landau": "Zhang et al. 2026", "Stuart": "Zhang et al. 2026",
    "Winfree": "WONN",
    "Kuramoto": "Un-0 · KomplexNet · Kuramoto Attention · Kuramoto Diffusion",
    "Kuramoto–Sakaguchi": "FSN",
    "Schuster & Wagner": "FSN · delay reservoirs",
    "Daido": "FSN",
    "Tanaka & Aoyagi": "Nagerl & Berloff 2025",
    "Chandra, Girvan & Ott": "AKOrN",
}


def fig_model_timeline():
    """The oscillator models of Appendix A, marked by whether anything uses them.

    Section 1.3 claims the machine-learning systems cluster at one end of a
    lineage that runs to 2023. Drawing the lineage with its adopters beneath it
    is that claim: the lower track is empty everywhere except a fifty-year
    window, and the models with no adopter are the ones whose properties the
    field says it wants.
    """
    models = [
        (1665, "Huygens"), (1830, "Airy"), (1908, "Poincaré"), (1919, "Blondel"),
        (1926, "Van der Pol"), (1944, "Landau"), (1952, "Hodgkin–Huxley"),
        (1960, "Stuart"), (1967, "Winfree"), (1975, "Kuramoto"), (1981, "Morris–Lecar"),
        (1986, "Kuramoto–Sakaguchi"), (1989, "Schuster & Wagner"), (1991, "Ermentrout"),
        (1992, "Daido"), (2004, "Abrams & Strogatz"), (2011, "Tanaka & Aoyagi"),
        (2016, "Ashwin & Rodrigues"), (2019, "Skardal & Arenas"),
        (2019, "Chandra, Girvan & Ott"), (2020, "Millán et al."), (2020, "Mulas et al."),
        (2022, "Calmon et al."), (2023, "de Aguiar"), (2023, "Zhang, Lucas & Battiston"),
    ]
    # Landau and Stuart are two rows of one model, so only the later carries the adopter
    adopters = [(yr, ADOPTED[name]) for yr, name in models
                if name in ADOPTED and name != "Landau"]

    def keys(y):
        return "\n".join([
            circle(24, y - 3, 3.4, INK, INK, 0),
            text(34, y, "a surveyed system is built on it", 8, anchor="start", fill=MID),
            circle(238, y - 3, 3.4, FILL_C, INK, 1.2),
            text(248, y, "no surveyed system found using it", 8, anchor="start", fill=MID)])

    def caveat(y):
        return text(24, y, "coRNN · UnICORNN · RON · LinOSS · D-LinOSS attach to no row above: "
                    "they carry no coupling term at all.", 8, anchor="start", fill=MID)

    reference_timeline(
        "fig7-model-timeline", 40, 630,
        [(1665, 1930, 40, 150), (1930, 1980, 150, 300), (1980, 2026, 300, 630)],
        (1665, 1800, 1900, 1950, 1970, 1990, 2000, 2010, 2020),
        models, adopters,
        "OSCILLATOR MODELS, IN THE ORDER THEY WERE DERIVED",
        "SURVEYED SYSTEMS BUILT ON THEM",
        "The oscillator models this survey cites, and which surveyed systems use them",
        extras=[keys, caveat], filled_above=lambda label: label in ADOPTED,
        filled_below=lambda _: True)




#: the surveyed systems whose primary report is peer reviewed. The rest are
#: preprints or, in Un-0's case, blog posts with released code; Section 7 makes
#: publication-status heterogeneity the survey's first stated limitation.
PEER_REVIEWED = {
    "ESN", "LSM", "Spintronic reservoir", "Oscillator Ising machines",
    "coRNN · UnICORNN", "Neural Wave Machines", "RON", "AKOrN", "LinOSS",
    "KomplexNet", "Kuramoto Orientation Diffusion",
}


def fig_system_timeline():
    """The surveyed systems by their own publication dates.

    Figure 5 places each system at the year of the coupling model it borrows,
    which answers a question about the lineage and deliberately says nothing
    about when the systems themselves appeared. This one answers the other
    question. Two facts are meant to be legible at a glance: the trained branch
    is entirely post-2020, and every system whose report is not peer reviewed
    sits in its last two years.
    """
    untrained = [
        (2001, "ESN"), (2002, "LSM"), (2017, "Spintronic reservoir"),
        (2019, "Oscillator Ising machines"), (2024, "RON"),
    ]
    trained = [
        (2021, "coRNN · UnICORNN"), (2023, "Neural Wave Machines"), (2025, "AKOrN"),
        (2025, "LinOSS"), (2025, "D-LinOSS"), (2025, "KomplexNet"),
        (2025, "Kuramoto Orientation Diffusion"), (2026, "Un-0"), (2026, "WONN"),
        (2026, "Kuramoto Attention · FSN"),
    ]

    def keys(y):
        # both shapes appear in each key: shape marks the track, fill marks the
        # publication status, and a one-shape legend reads as though it marked both
        return "\n".join([
            circle(24, y - 3, 3.4, INK, INK, 0),
            f'<rect x="33.6" y="{y-6.4}" width="6.8" height="6.8" fill="{INK}" '
            f'stroke="{INK}" stroke-width="1.2"/>',
            text(50, y, "peer reviewed", 8, anchor="start", fill=MID),
            circle(160, y - 3, 3.4, FILL_C, INK, 1.2),
            f'<rect x="169.6" y="{y-6.4}" width="6.8" height="6.8" fill="{FILL_C}" '
            f'stroke="{INK}" stroke-width="1.2"/>',
            text(186, y, "preprint or blog-published", 8, anchor="start", fill=MID)])

    reference_timeline(
        "fig6-system-timeline", 40, 630,
        [(2001, 2020, 40, 250), (2020, 2026, 250, 630)],
        (2001, 2005, 2010, 2015, 2020, 2022, 2024, 2026),
        untrained, trained,
        "UNTRAINED OR DESIGNED DYNAMICS",
        "TRAINED DYNAMICS",
        "The machine-learning systems surveyed here, by publication date",
        extras=[keys], head_in_corners=True,
        filled_above=lambda label: label in PEER_REVIEWED,
        filled_below=lambda label: label in PEER_REVIEWED)


def main() -> None:
    """Draw every fig_* in this module, then render each SVG to PDF and PNG.

    Discovered rather than listed: a hand-kept list has to sit after the last
    definition to resolve, and adding a figure at the end of the file silently
    broke the run twice.
    """
    figures = sorted(n for n in globals() if n.startswith("fig_"))
    for name in figures:
        globals()[name]()
        print(f"  {name}")
    for path in sorted(HERE.glob("*.svg")):
        render(path.stem)
        print(f"  rendered {path.stem}")


if __name__ == "__main__":
    main()
