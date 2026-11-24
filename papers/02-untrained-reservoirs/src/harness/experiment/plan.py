"""The experiments, and the driver that runs them.

    uv run python -m harness.experiment.plan prepare            # banks' row caches, once
    uv run python -m harness.experiment.plan run controls --workers 3 --dry-run
    uv run python -m harness.experiment.plan run leak-check controls --workers 3

Every experiment is a list of specs fixed in advance; nothing here is a
free choice at run time. The driver skips every spec already recorded, so a
sweep that stops restarts where it left off, and it runs the costliest specs
first so the pool drains evenly. A spec that fails is logged with its
traceback and the sweep carries on; the failure log is part of the record.
"""

from __future__ import annotations

import argparse
import math
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from harness.experiment import protocol as pr
from harness.experiment import run as rn
from harness.experiment.arms import Arm

NOISES = (None, 0.0, 5.0)
DESIGN_NOISES = (0.0, 5.0)          # clean audio is left out of the design experiments, where the task saturates
GAINS = (1.0, 2.0)
SEEDS = (0, 1, 2)
TRAINED_ARCHS = ("gru", "tcn", "cnn", "transformer", "s4d")
PHASE_COUPLINGS = ("kuramoto", "kuramoto-sakaguchi", "second-harmonic", "winfree")
AMPLITUDE_COUPLINGS = ("stuart-landau", "stuart-landau-fixed")
#: the amplitude coupling functions with the input applied as an exact rotation (the record's own Stuart–Landau
#: runs apply it inside a semi-implicit step, which rotates too far at high gain; see harness.models.stuart_landau)
EXACT_DRIVE_COUPLINGS = ("stuart-landau-exact", "stuart-landau-fixed-exact")
GEOMETRIES = ("torus", "cylinder", "sheet", "helix", "cube", "sphere")
FREQUENCIES = ("random", "tonotopic", "identical")
RESTORINGS = (0.3, 0.1)
CEILINGS = (1.0, 0.5)
BECKER_TRAIN = 18000
#: the selection grid (Protocol V): gain, restoring strength and coupling ceiling around the reference, selected on
#: the validation speakers; the Stuart–Landau network at the gains where it reads best
SELECTION_GAINS = {"kuramoto": (0.25, 0.5, 1.0, 2.0), "stuart-landau": (0.5, 1.0, 2.0)}
SELECTION_RESTORINGS = (0.1, 0.3, 0.5, 0.8, 1.0)   # every restoring strength the design and sweep experiments ran
SELECTION_CEILINGS = (0.5, 1.0, 1.5, 2.0)           # and every coupling ceiling
#: the calibration experiment's noise levels, the harness's level of the noise relative to the speech (-10 is noise
#: 10 dB quieter than the speech, an SNR of +10 dB), and the networks it runs at the reference configuration
CALIBRATION_NOISES = (None, -10.0, -5.0, 0.0, 5.0, 10.0)
CALIBRATION_COUPLINGS = ("kuramoto", "stuart-landau")
#: the reservoir baselines' grids (Protocol V); the input scaling is the run's gain
RESERVOIR_GAINS = (0.5, 1.0, 2.0)
ESN_UNITS = 1024                                   # the network's 1,024 states
ESN_RHOS = (0.5, 0.9, 1.1, 1.3)
ESN_LEAKS = (0.1, 0.3, 1.0)
RON_UNITS = 512                                    # 512 oscillators, 1,024 states (position and velocity)
RON_TAUS = (0.1, 0.3)
RON_RHOS = (1.0, 3.0, 9.0)
RON_GAMMAS = ((1.0, 0.5), (5.0, 4.0))              # (midpoint, radius) of the stiffness
RON_EPSS = ((0.5, 0.4), (2.0, 1.5))                # (midpoint, radius) of the damping
EXPANSION_UNITS = (1024, 2048)
#: the sweep beyond the design experiment's levels: restoring strengths up to the natural frequency, where an
#: oscillator stops rotating; coupling ceilings above 1; and gains up to 12, below the integrator's
#: bound, where drive, natural frequency and coupling together would pass pi radians a step
SWEEP_RESTORINGS = (0.5, 0.8, 1.0)
SWEEP_CEILINGS = (1.5, 2.0)
#: and below gain 1, where the input's push weakens toward the no-input leak check (gain 0)
SWEEP_GAINS = (0.25, 0.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)

BASELINE = Arm("baseline")
COUPLED = Arm("network")            # Kuramoto, torus, random frequencies, restoring 0.3, ceiling 1
UNCOUPLED = Arm("network", coupled=False)
BANK_STATE = Arm("bank", channels=4)   # the network's states and parameters
BANK_WIDTH = Arm("bank", channels=8)   # the network's exposed signals, at twice the parameters
UNTRAINED = (COUPLED, UNCOUPLED, BANK_STATE, BANK_WIDTH)
#: the design experiment runs this network with noise only; the controls experiment adds it on clean
#: audio, so it stands beside every other arm in all three conditions
STUART_LANDAU = Arm("network", coupling="stuart-landau")
ROW_READOUT_WIDTH = 8 * 3 * 4                       # half the rows x statistics x windows, the pooled half-lattice read
BAND_READOUT_WIDTH = 1 * 3 * 4                      # one row x statistics x windows, the pooled single-band read
AMPLITUDE_VIEWS = ("full", "phase", "amplitude")    # what the readout sees of the Stuart–Landau network's z
STIFFNESSES = (4.0, 16.0)                           # the amplitude relaxation rate as a multiple of the core's 0.05
STIFFNESS_GAINS = (1.0, 4.0, 8.0)


def _trained(arch: str) -> Arm:
    return Arm("trained", arch=arch)


def leak_check() -> Iterator[rn.Spec]:
    """The zero-input cells, which must read chance, and the per-clip-span diagnostic."""
    for arm in UNTRAINED:
        yield rn.Spec("leak-check", "recognition", "spectrogram", 0.0, 0.0, 0, arm)
    # how much a per-clip read window gives the network: the leak the fixed window closes
    for gain in (0.0, *GAINS):
        for seed in SEEDS:
            yield rn.Spec("leak-check", "recognition", "spectrogram", 0.0, gain, seed, COUPLED, span="clip")


def controls() -> Iterator[rn.Spec]:
    """The arms, on recognition at three training sizes and on the order task."""
    for noise in NOISES:
        for seed in SEEDS:
            yield rn.Spec("controls", "recognition", "spectrogram", noise, None, seed, BASELINE, sizes=pr.SIZES)
            for gain in GAINS:
                for arm in UNTRAINED:
                    yield rn.Spec("controls", "recognition", "spectrogram", noise, gain, seed, arm, sizes=pr.SIZES)
                if noise is None:
                    yield rn.Spec("controls", "recognition", "spectrogram", noise, gain, seed, STUART_LANDAU,
                                  sizes=pr.SIZES)
            for arch in TRAINED_ARCHS:
                for n in pr.SIZES:
                    yield rn.Spec("controls", "recognition", "spectrogram", noise, None, seed, _trained(arch),
                                  sizes=(n,), native_sizes=(n,) if n == rn.PRIMARY_SIZE else ())
    for pair in pr.PAIRS:
        for noise in NOISES:
            for seed in SEEDS:
                yield rn.Spec("controls", "order", "spectrogram", noise, None, seed, BASELINE, pair=pair)
                for gain in GAINS:
                    for arm in UNTRAINED:
                        yield rn.Spec("controls", "order", "spectrogram", noise, gain, seed, arm, pair=pair)
                for arch in TRAINED_ARCHS:
                    yield rn.Spec("controls", "order", "spectrogram", noise, None, seed, _trained(arch), pair=pair)


def _design_arms() -> Iterator[Arm]:
    for coupling in PHASE_COUPLINGS + AMPLITUDE_COUPLINGS:
        geometries = ("torus",) if coupling in AMPLITUDE_COUPLINGS else GEOMETRIES
        for geometry in geometries:
            for frequencies in FREQUENCIES:
                for restoring in RESTORINGS:
                    for ceiling in CEILINGS:
                        yield Arm("network", coupling=coupling, geometry=geometry, frequencies=frequencies,
                                  restoring=restoring, ceiling=ceiling)


def design() -> Iterator[rn.Spec]:
    """The design factorial, read at the primary size with and without rotation rates."""
    for noise in DESIGN_NOISES:
        for gain in GAINS:
            for seed in SEEDS:
                for arm in _design_arms():
                    yield rn.Spec("design", "recognition", "spectrogram", noise, gain, seed, arm,
                                  bits="primary", reads=("windowed", "windowed+rate"))


def _diagonal() -> list[Arm]:
    arms = [Arm("network", coupling=c, frequencies=f) for c in PHASE_COUPLINGS for f in ("random", "tonotopic")]
    return arms + [Arm("network", geometry="helix", frequencies=f) for f in ("random", "tonotopic")]


def quadrature() -> Iterator[rn.Spec]:
    """The quadrature pathway: a diagonal of the design rather than the full factorial."""
    for noise in DESIGN_NOISES:
        for seed in SEEDS:
            yield rn.Spec("quadrature", "recognition", "quadrature", noise, None, seed, BASELINE, bits="primary")
            for gain in GAINS:
                for arm in _diagonal():
                    yield rn.Spec("quadrature", "recognition", "quadrature", noise, gain, seed, arm, bits="primary")


def sweep() -> Iterator[rn.Spec]:
    """Every coupling function at the reference configuration, beyond the design experiment's
    restoring strengths, coupling ceilings and gains. Paired with its reference cells: same clips, seeds and reads."""
    common = dict(bits="primary", reads=("windowed", "windowed+rate"))
    for noise in DESIGN_NOISES:
        for seed in SEEDS:
            for coupling in PHASE_COUPLINGS + AMPLITUDE_COUPLINGS:
                for gain in GAINS:
                    for restoring in SWEEP_RESTORINGS:
                        yield rn.Spec("sweep", "recognition", "spectrogram", noise, gain, seed,
                                      Arm("network", coupling=coupling, restoring=restoring), **common)
                    for ceiling in SWEEP_CEILINGS:
                        yield rn.Spec("sweep", "recognition", "spectrogram", noise, gain, seed,
                                      Arm("network", coupling=coupling, ceiling=ceiling), **common)
                for gain in SWEEP_GAINS:
                    yield rn.Spec("sweep", "recognition", "spectrogram", noise, gain, seed,
                                  Arm("network", coupling=coupling), **common)


def exact_drive() -> Iterator[rn.Spec]:
    """Both Stuart–Landau networks at the reference configuration with the input applied as an exact rotation,
    across every gain of the design and sweep experiments. Paired with the record's Stuart–Landau cells (the
    design files at gains 1 and 2, the sweep files beyond): same clips, seeds and reads. It asks whether the
    fixed-amplitude network's divergence from Kuramoto at high gain, and the free amplitude's robustness to gain,
    come from the physics or from the drive step's discretization."""
    common = dict(bits="primary", reads=("windowed", "windowed+rate"))
    for noise in DESIGN_NOISES:
        for seed in SEEDS:
            for coupling in EXACT_DRIVE_COUPLINGS:
                for gain in sorted(GAINS + SWEEP_GAINS):
                    yield rn.Spec("exact-drive", "recognition", "spectrogram", noise, gain, seed,
                                  Arm("network", coupling=coupling), **common)


def clean() -> Iterator[rn.Spec]:
    """Both Stuart–Landau networks at the reference configuration on clean audio, gains 1 and 2, under Protocol A,
    which the design experiment left out (its noise levels are 0 and −5 dB); the controls experiment ran the free
    amplitude clean on the CPU, and this reruns it on the GPU beside the fixed amplitude, which no experiment had run
    clean. Paired with the controls experiment's clean Kuramoto and baseline cells: same clips, seeds and reads. It asks whether the free amplitude's lead over
    Kuramoto is a property of noisy drive or of the dynamics."""
    common = dict(bits="primary", reads=("windowed", "windowed+rate"))
    for seed in SEEDS:
        for coupling in AMPLITUDE_COUPLINGS:
            for gain in GAINS:
                yield rn.Spec("clean", "recognition", "spectrogram", None, gain, seed,
                              Arm("network", coupling=coupling), **common)


def selection() -> Iterator[rn.Spec]:
    """Model selection for the coupled network under Protocol V: the gain, restoring strength and coupling ceiling
    grid for the Kuramoto and Stuart–Landau networks at the reference geometry and frequencies, every cell read on
    the validation speakers (41-48) and the test speakers (49-60). The summary picks each network's configuration by
    validation accuracy and reports that configuration's test accuracy beside the reference's."""
    common = dict(protocol="V", bits="primary", reads=("windowed",))
    for noise in DESIGN_NOISES:
        for seed in SEEDS:
            for coupling, gains in SELECTION_GAINS.items():
                for gain in gains:
                    for restoring in SELECTION_RESTORINGS:
                        for ceiling in SELECTION_CEILINGS:
                            yield rn.Spec("selection", "recognition", "spectrogram", noise, gain, seed,
                                          Arm("network", coupling=coupling, restoring=restoring, ceiling=ceiling),
                                          **common)


def calibration() -> Iterator[rn.Spec]:
    """The noise-level calibration redone in this repository under Protocol V: the spectrogram-only baseline and the
    Kuramoto and Stuart–Landau networks at the reference configuration, gains 1 and 2, from noise 10 dB below the
    speech to 10 dB above it and on clean audio, every cell read on the validation speakers. With the selection
    experiment's grid it stands in for the pilot's calibration: the paper's noise levels are those at which the
    strongest untrained network lands between 70 and 90% on the validation speakers, and the grid asks whether the
    reference's gain, restoring strength and coupling ceiling are what those speakers would pick."""
    common = dict(protocol="V", bits="primary")
    for noise in CALIBRATION_NOISES:
        for seed in SEEDS:
            yield rn.Spec("calibration", "recognition", "spectrogram", noise, None, seed, BASELINE, **common)
            for coupling in CALIBRATION_COUPLINGS:
                for gain in GAINS:
                    yield rn.Spec("calibration", "recognition", "spectrogram", noise, gain, seed,
                                  Arm("network", coupling=coupling), reads=("windowed",), **common)


def esn() -> Iterator[rn.Spec]:
    """The leaky echo state network's selection grid under Protocol V: spectral radius, leaking rate and input
    scaling (the gain), at the network's state count. The summary selects by validation accuracy per noise level."""
    common = dict(protocol="V", bits="primary", reads=("windowed",))
    for noise in DESIGN_NOISES:
        for seed in SEEDS:
            for gain in RESERVOIR_GAINS:
                for rho in ESN_RHOS:
                    for leak in ESN_LEAKS:
                        yield rn.Spec("esn", "recognition", "spectrogram", noise, gain, seed,
                                      Arm("esn", units=ESN_UNITS, rho=rho, leak=leak), **common)


def ron() -> Iterator[rn.Spec]:
    """The random oscillators network's selection grid under Protocol V: time step, spectral radius, stiffness and
    damping ranges and input scaling (the gain), at the network's state count."""
    common = dict(protocol="V", bits="primary", reads=("windowed",))
    for noise in DESIGN_NOISES:
        for seed in SEEDS:
            for gain in RESERVOIR_GAINS:
                for tau in RON_TAUS:
                    for rho in RON_RHOS:
                        for gamma in RON_GAMMAS:
                            for eps in RON_EPSS:
                                yield rn.Spec("ron", "recognition", "spectrogram", noise, gain, seed,
                                              Arm("ron", units=RON_UNITS, tau=tau, rho=rho, gamma=gamma, eps=eps),
                                              **common)


def expansion() -> Iterator[rn.Spec]:
    """The memoryless random nonlinear expansion of the input, at the network's state count and at its signal
    count, on recognition (every noise level, Protocol A) and on the order task, where it must read chance."""
    arms = [Arm("expansion", units=n) for n in EXPANSION_UNITS]
    for noise in NOISES:
        for seed in SEEDS:
            for gain in GAINS:
                for arm in arms:
                    yield rn.Spec("expansion", "recognition", "spectrogram", noise, gain, seed, arm, bits="primary")
    for pair in pr.PAIRS:
        for noise in NOISES:
            for seed in SEEDS:
                for gain in GAINS:
                    yield rn.Spec("expansion", "order", "spectrogram", noise, gain, seed, arms[0], pair=pair)


def _selected_arms() -> list[tuple[Arm, float, float]]:
    """(arm, gain, noise) for each reservoir baseline's validation-selected configuration, from the record."""
    from harness.experiment import record as rec
    from harness.experiment import selection
    try:
        cells = rec.load(["esn-recognition", "ron-recognition"])
    except FileNotFoundError:
        return []
    out = []
    for experiment in ("esn", "ron"):
        for noise, chosen in selection.select(cells, experiment).items():
            spec = next(iter(chosen["cells"].values())).arm
            out.append((Arm(**spec), chosen["gain"], noise))
    return out


def baselines_selected() -> Iterator[rn.Spec]:
    """The echo state network and the random oscillators network at the hyperparameters selected on the validation
    speakers (the esn and ron experiments), rerun under Protocol A beside the controls experiment's arms: recognition at
    every noise level (the 0 dB selection serves clean audio, which the grids leave out) and the order task. Defined
    by the record, so it is empty until the grids are in."""
    picks = _selected_arms()
    for arm, gain, noise in picks:
        noises = (None, noise) if noise == 0.0 else (noise,)
        for n in noises:
            for seed in SEEDS:
                yield rn.Spec("baselines-selected", "recognition", "spectrogram", n, gain, seed, arm, bits="primary")
                for pair in pr.PAIRS:
                    yield rn.Spec("baselines-selected", "order", "spectrogram", n, gain, seed, arm, pair=pair)


def cochlea() -> Iterator[rn.Spec]:
    """The coil and the cochlea (harness.models.geometries.coil), for every phase coupling function and
    kind of natural frequencies, at the reference restoring strength and ceiling. Paired with the design
    experiment's torus and helix cells: same clips, seeds and reads."""
    common = dict(bits="primary", reads=("windowed", "windowed+rate"))
    for noise in DESIGN_NOISES:
        for gain in GAINS:
            for seed in SEEDS:
                for geometry in ("coil", "cochlea", "cochlea-matched"):
                    for coupling in PHASE_COUPLINGS:
                        for frequencies in FREQUENCIES:
                            yield rn.Spec("cochlea", "recognition", "spectrogram", noise, gain, seed,
                                          Arm("network", coupling=coupling, geometry=geometry,
                                              frequencies=frequencies), **common)


def projection() -> Iterator[rn.Spec]:
    """The controls experiment's reservoir runs again at the primary size, read under the fixed and the
    seeded projection.

    The fixed projection is one draw for every seed, so the spread over seeds leaves out the
    projection's own variability. The fixed cells here must equal the controls experiment's.
    """
    for noise in NOISES:
        for seed in SEEDS:
            for gain in GAINS:
                for arm in UNTRAINED:
                    yield rn.Spec("projection", "recognition", "spectrogram", noise, gain, seed, arm,
                                  native_sizes=(), bits="primary", reads=("windowed",), projection="both")
    for pair in pr.PAIRS:
        for noise in NOISES:
            for seed in SEEDS:
                for gain in GAINS:
                    for arm in UNTRAINED:
                        yield rn.Spec("projection", "order", "spectrogram", noise, gain, seed, arm, pair=pair,
                                      native_sizes=(), reads=("pooled",), projection="both")


def projection_methods() -> Iterator[rn.Spec]:
    """The controls experiment's reservoirs and the Stuart–Landau network at the primary cell, read at width 192
    through the fixed random projection, a PCA fitted on the training clips, and the within-row pooling. Their
    fixed cells repeat the controls and design experiments' (on the GPU)."""
    for noise in NOISES:
        for seed in SEEDS:
            for gain in GAINS:
                for arm in UNTRAINED + (STUART_LANDAU,):
                    yield rn.Spec("projection-methods", "recognition", "spectrogram", noise, gain, seed, arm,
                                  widths=(192,), native_sizes=(), bits="primary", reads=("windowed",),
                                  projection="methods")


def row_readout() -> Iterator[rn.Spec]:
    """The Kuramoto, uncoupled and Stuart–Landau networks and the whole-clip spectrogram baseline at the primary cell, each read
    through only the lower or only the upper eight rows of its lattice (the within-row pooling kept for one half:
    8 rows × 3 statistics × 4 windows = 96 features), beside the fixed random projection at the same width. Asks
    whether the upper bands, where the Stuart–Landau network's signals carry more class information than
    Kuramoto's, are where its accuracy comes from."""
    common = dict(widths=(ROW_READOUT_WIDTH,), native_sizes=(), bits="primary", projection="rows")
    for noise in DESIGN_NOISES:
        for seed in SEEDS:
            yield rn.Spec("row-readout", "recognition", "spectrogram", noise, None, seed, BASELINE,
                          reads=("windowed@wholeclip",), **common)
            for gain in GAINS:
                for arm in (COUPLED, UNCOUPLED, STUART_LANDAU):
                    yield rn.Spec("row-readout", "recognition", "spectrogram", noise, gain, seed, arm,
                                  reads=("windowed",), **common)


def band_readout() -> Iterator[rn.Spec]:
    """Per-band readout accuracy: the networks, the state-matched bank and the whole-clip baseline at the primary
    cell, each read through one row of its lattice alone (the within-row pooling kept for one row: 3 statistics × 4
    windows = 12 features), for every row, beside the fixed random projection at width 12. Clean, 0 and −5 dB,
    gain 1. The map of what the readout gets from each band, in place of the per-feature F maps, which the noise
    floor inflates (Appendix D.2)."""
    common = dict(widths=(BAND_READOUT_WIDTH,), native_sizes=(), bits="primary", projection="bands")
    for noise in NOISES:
        for seed in SEEDS:
            yield rn.Spec("band-readout", "recognition", "spectrogram", noise, None, seed, BASELINE,
                          reads=("windowed@wholeclip",), **common)
            for arm in (COUPLED, UNCOUPLED, STUART_LANDAU, BANK_STATE):
                yield rn.Spec("band-readout", "recognition", "spectrogram", noise, 1.0, seed, arm,
                              reads=("windowed",), **common)


def amplitude_view() -> Iterator[rn.Spec]:
    """What the amplitude gives the readout (harness.experiment.views): the Stuart–Landau network read through its
    full signals, its phase alone and its amplitude alone, and the Kuramoto network read with an amplitude slaved to
    its phases, at the primary cell, 0 and −5 dB, gains 1 and 2."""
    common = dict(bits="primary", reads=("windowed",))
    for noise in DESIGN_NOISES:
        for seed in SEEDS:
            for gain in GAINS:
                for view in AMPLITUDE_VIEWS:
                    yield rn.Spec("amplitude-view", "recognition", "spectrogram", noise, gain, seed, STUART_LANDAU,
                                  view=view, **common)
                yield rn.Spec("amplitude-view", "recognition", "spectrogram", noise, gain, seed, COUPLED,
                              view="slaved", **common)


def stiffness() -> Iterator[rn.Spec]:
    """The amplitude relaxation rate: the exact-drive Stuart–Landau network with alpha and beta at 4 and 16 times
    the core's 0.05 (time constants of about 5 and 1.25 frames against 20), at gains 1, 4 and 8, 0 dB, beside the
    exact-drive experiment's rate-1 runs. Whether the amplitude's gain to the readout needs its slow memory."""
    common = dict(bits="primary", reads=("windowed",))
    for seed in SEEDS:
        for s in STIFFNESSES:
            for gain in STIFFNESS_GAINS:
                yield rn.Spec("stiffness", "recognition", "spectrogram", 0.0, gain, seed,
                              Arm("network", coupling="stuart-landau-exact", stiffness=s), **common)


def becker_folds() -> Iterator[rn.Spec]:
    """Protocol B: Becker et al.'s folds, clean audio, for comparison with published results."""
    for fold in range(len(pr.BECKER_FOLDS)):
        common = dict(protocol="B", fold=fold, sizes=(BECKER_TRAIN,), native_sizes=())
        yield rn.Spec("becker-folds", "recognition", "spectrogram", None, None, 0, BASELINE, **common)
        for gain in GAINS:
            for arm in UNTRAINED:
                yield rn.Spec("becker-folds", "recognition", "spectrogram", None, gain, 0, arm, **common)
        for arch in TRAINED_ARCHS:
            yield rn.Spec("becker-folds", "recognition", "spectrogram", None, None, 0, _trained(arch), **common)


EXPERIMENTS = {"leak-check": leak_check, "controls": controls, "design": design, "quadrature": quadrature,
               "becker-folds": becker_folds, "projection": projection, "sweep": sweep, "cochlea": cochlea,
               "exact-drive": exact_drive, "clean": clean, "selection": selection, "calibration": calibration, "esn": esn, "ron": ron, "expansion": expansion,
               "projection-methods": projection_methods, "baselines-selected": baselines_selected,
               "row-readout": row_readout, "band-readout": band_readout, "amplitude-view": amplitude_view,
               "stiffness": stiffness}


# ---------------------------------------------------------------------------
# Cost, for ordering
# ---------------------------------------------------------------------------

def cost(spec: rn.Spec) -> float:
    """A rough relative cost, so the pool starts the longest runs first."""
    clips = (pr.ORDER_TRAIN + pr.ORDER_TEST) if spec.task == "order" else max(spec.sizes) + 6000
    per_clip = {"baseline": 0.1, "bank": 0.4, "network": 1.0, "trained": 3.0,
                "esn": 0.5, "ron": 0.5, "expansion": 0.2}[spec.arm.kind]
    size = spec.arm.channels / 4 if spec.arm.kind in ("network", "bank") else 1
    return clips * per_clip * size


# ---------------------------------------------------------------------------
# Preparing and driving
# ---------------------------------------------------------------------------

def prepare(workers: int) -> None:
    """Build every row cache the experiments read, in parallel, skipping those that exist."""
    bank = pr.load_bank()
    n = len(bank["labels"])
    jobs = [("rows", p, noise, None, None) for p in pr.CACHED_PATHWAYS for noise in NOISES
            if pr.load_rows(pr.rows_path(p, noise), n) is None]
    jobs += [("rows", "spectrogram", noise, None, None) for noise in CALIBRATION_NOISES
             if noise not in NOISES and pr.load_rows(pr.rows_path("spectrogram", noise), n) is None]
    for pair in pr.PAIRS:
        for noise in NOISES:
            for code in (0, *(s + 1 for s in SEEDS)):
                size = pr.ORDER_TEST if code == 0 else pr.ORDER_TRAIN
                if pr.load_rows(pr.order_rows_path(pair, code, noise), size) is None:
                    jobs.append(("order", "spectrogram", noise, pair, code))
    print(f"=== prepare: {len(jobs)} row cache(s) to build with {workers} worker(s)")
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for f in as_completed([ex.submit(_build, *j) for j in jobs]):
            print(f"    {f.result()}")


def _build(kind, pathway, noise, pair, code) -> str:
    import torch
    torch.set_num_threads(1)
    bank = pr.load_bank()
    out = (pr.build_rows(bank, pathway, noise) if kind == "rows"
           else pr.build_order_rows(bank, pair, code, noise))
    return str(out.name)


def _work(spec: rn.Spec, device: str, threads: int) -> tuple[str, float]:
    t0 = time.perf_counter()
    return rn.run(spec, device, threads), time.perf_counter() - t0


def planned(names: list[str]) -> list[rn.Spec]:
    specs = [s for name in names for s in EXPERIMENTS[name]()]
    ids = [(s.group(), s.run_id()) for s in specs]
    if len(set(ids)) != len(ids):
        raise RuntimeError("two specs share a record address; the experiment definitions are wrong")
    return specs


def pending(specs: list[rn.Spec]) -> list[rn.Spec]:
    done = {g: rn.recorded_ids(g) for g in {s.group() for s in specs}}
    return sorted((s for s in specs if s.run_id() not in done[s.group()]), key=cost, reverse=True)


def drive(names: list[str], workers: int, threads: int, device: str, dry_run: bool,
          task: str | None = None) -> None:
    specs = [s for s in planned(names) if task is None or s.task == task]
    todo = pending(specs)
    print(f"=== {' + '.join(names)}: {len(specs)} runs planned, {len(specs) - len(todo)} recorded, "
          f"{len(todo)} to run on {workers} worker(s) x {threads} thread(s), device {device}")
    if dry_run:
        for s in todo[:12]:
            print(f"    {s.group()}/{s.run_id()}")
        if len(todo) > 12:
            print(f"    ... and {len(todo) - 12} more")
        return
    failures = rn.record_root() / "failures.log"
    failures.parent.mkdir(parents=True, exist_ok=True)
    t0, done, failed = time.perf_counter(), 0, 0
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        futures = {ex.submit(_work, s, device, threads): s for s in todo}
        for f in as_completed(futures):
            spec = futures[f]
            try:
                where, took = f.result()
                done += 1
            except Exception:                                   # noqa: BLE001 - logged, and the sweep goes on
                failed += 1
                with open(failures, "a") as log:
                    log.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {spec.group()}/{spec.run_id()}\n"
                              f"{traceback.format_exc()}\n")
                where, took = f"FAILED {spec.group()}/{spec.run_id()}", math.nan
            elapsed = time.perf_counter() - t0
            finished = done + failed
            left = elapsed / finished * (len(todo) - finished)
            print(f"[{finished:5d}/{len(todo)}] {elapsed / 60:6.1f} min elapsed, ~{left / 60:6.1f} left"
                  f"  {took:6.1f}s  {where}", flush=True)
    print(f"=== done: {done} recorded, {failed} failed" + (f" (see {failures})" if failed else ""))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="the study's experiments")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="build the row caches the experiments read")
    p.add_argument("--workers", type=int, default=4)
    r = sub.add_parser("run", help="run one or more experiments")
    r.add_argument("experiments", nargs="+", choices=sorted(EXPERIMENTS))
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--device", default="cpu")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--task", choices=("recognition", "order"),
                   help="run only this task's specs, so two machines can split an experiment without sharing a file")
    a = ap.parse_args(argv)
    if a.command == "prepare":
        prepare(a.workers)
    else:
        drive(a.experiments, a.workers, a.threads, a.device, a.dry_run, a.task)


if __name__ == "__main__":
    main()
