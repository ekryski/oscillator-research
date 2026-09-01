"""The experiment runner: one process = one matrix run.

Arms (--arms): kuramoto (trained physics), forced (trained, no relative-phase
term), frozen (untrained physics — the reservoir arm the paper reports),
shuffled (trained kuramoto with per-channel kernel permutation), and the
param-matched conventional references gru / tcn / cnn / transformer / s4d. All
field arms share the same init and the same frozen probe per seed; every arm
gets the identical ridge-probe evaluation protocol.

Decision bars are pre-registered before launch and scored per the criteria as
written, including the optimization-health gate: a trained arm counts toward a
verdict only if its train loss dropped by at least HEALTH_MIN_LOSS_DROP;
otherwise it is recorded as an optimization failure (a recipe problem), which
is a different claim from "the physics can't learn".

Each run writes results/<exp>/results.json with its full configuration, every
accuracy column, and the instrument readings.

    uv run python -m harness.runner --lr-pick                      # recipe health
    uv run python -m harness.runner --task digits --noise-db 0 \\
        --gain 2.0 --probe-windows 4 --arms frozen --exp my-run   # one run
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):  # support direct `python harness/runner.py` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import results as store
from harness.measurement.instruments import analytic_row_phase, instrument_field, natural_rate
from harness.measurement.probe import (
    collect_features,
    fit_ridge_probe,
    frozen_probe_acc,
    parity_project,
    train_arm,
)
from harness.models import (
    GRUBaseline,
    OscillatorField,
    TCNBaseline,
    shuffle_kernel_,
    tonotopic_omega,
)
from harness.models.baselines import CNNBaseline, S4DBaseline, TransformerBaseline
from harness.stimuli import (
    AM_CARRIER_ROW,
    am_rates,
    bandpass_rows,
    drive_kick_stats,
    make_am_clips,
    make_step_clips,
    make_tone_clips,
    step_freqs,
    tone_classes,
)
from harness.utils.constants import GAIN

# Pre-registered: minimum relative train-loss drop for a trained arm to count
# toward learn/kill verdicts (below it = optimization failure, recorded as such).
HEALTH_MIN_LOSS_DROP = 0.20
ARM_ORDER = ["kuramoto", "forced", "omegaenc", "gru", "tcn", "cnn", "transformer", "s4d",
             "frozen", "designed", "shuffled"]  # shuffled needs kuramoto first
# the remaining baseline minis (harness/baselines.py), trained like gru/tcn.
BASELINE_ARMS = {"cnn": CNNBaseline, "transformer": TransformerBaseline, "s4d": S4DBaseline}
# Arms that read magnitude rows [B,T,G] directly (no field): incompatible with
# the quadrature frontend's [B,T,G,2] rows. omegaenc reads row magnitudes too.
QUAD_INCOMPATIBLE_ARMS = frozenset({"omegaenc", "gru", "tcn", *BASELINE_ARMS})
TEST_SEED = 9999  # test set fixed across arms AND run seeds for comparability


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="run one experiment run")
    ap.add_argument("--task", choices=("tones", "steps", "am", "digits", "digitpairs"),
                    default="tones",
                    help="synthetic contract stimuli (tones/steps/am) or the paper's "
                         "spoken-digit tasks (digits = identity, digitpairs = order)")
    ap.add_argument("--arms", default="kuramoto,forced,frozen,shuffled,gru")
    ap.add_argument("--clamp", type=float, default=0.5, help="spectral clamp; 0 = off")
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--n-test", type=int, default=512)
    ap.add_argument("--frames", type=int, default=512)
    ap.add_argument("--gain", type=float, default=GAIN)
    ap.add_argument("--noise-db", type=float, default=-20.0,
                    help="stimulus SNR knob; calibrated so the frozen arm has headroom "
                         "(the -20 dB default saturates: frozen ridge = 1.00, measured)")
    ap.add_argument("--channels", type=int, default=4)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=1,
                    help="stacked sheets; inter-block mixers frozen at init")
    ap.add_argument("--coupling", choices=("kuramoto", "sakaguchi", "harmonic2", "winfree"),
                    default="kuramoto",
                    help="coupling law for the trained-physics + frozen arms")
    ap.add_argument("--kernel-support", type=int, default=0,
                    help="restrict K to wrapped offsets within this radius (0 = full)")
    ap.add_argument("--core", choices=("phase", "sl", "sl-fixedamp", "randgraph"), default="phase",
                    help="phase core, Stuart-Landau (sl), amplitude-clamped SL "
                         "(sl-fixedamp), or the random-graph connectivity control")
    ap.add_argument("--boundary", choices=("torus", "cylinder", "sheet", "helix", "cube", "sphere",
                                           "moebius", "klein", "diamond"),
                    default="torus",
                    help="field geometry — only the lattice wrap rule differs; "
                         "per-shape tonotopy is pre-declared in harness/models/phase.py; "
                         "phase core only")
    ap.add_argument("--frontend", choices=("mag", "quad", "carrier"), default="mag",
                    help="the transduction ladder for digits: magnitude hop rows "
                         "(additive envelope drive), quadrature-baseband rows "
                         "(Adler phase-referenced torque; field arms only — "
                         "conventional arms are skipped), or carrier bandpass rows "
                         "(band-split waveform drives the field at sample rate, so "
                         "T = samples, not hop frames)")
    ap.add_argument("--probe-windows", type=int, default=0,
                    help="ALSO report a ridge on per-window pooled features "
                         "(0 = off; the standard metric is always reported)")
    ap.add_argument("--probe-macro", type=int, default=0,
                    help="ALSO report a ridge on macroscopic features "
                         "(patch-R + row |z|) with this patch size (0 = off; "
                         "field arms only)")
    ap.add_argument("--select", choices=("last", "best-val"), default="last",
                    help="trained-arm checkpoint rule: last epoch, or the best "
                         "val-ridge epoch; identical rule for every trained arm")
    ap.add_argument("--select-every", type=int, default=3,
                    help="best-val: score the selection ridge every N epochs "
                         "(the final epoch is always scored)")
    ap.add_argument("--select-features", choices=("standard", "windowed"), default="standard",
                    help="best-val: feature type for the selection ridge "
                         "(windowed requires --probe-windows)")
    ap.add_argument("--damping-learnable", action="store_true",
                    help="per-channel learnable pinning lambda (init at --damping); "
                         "learned spread reported")
    ap.add_argument("--omega-uniform", action="store_true",
                    help="omega level 1: uniform N(1,0) — every oscillator at exactly "
                         "omega=1.0 (the no-structure control; frozen arm)")
    ap.add_argument("--designed-init", action="store_true",
                    help="tonotopic omega init for the TRAINED arm too (design + "
                         "backprop — the 'designed' arm alone stays untrained)")
    ap.add_argument("--pair", default="3,7",
                    help="digitpairs: the two digits, e.g. '3,7'; the label is the order")
    ap.add_argument("--graph-k", type=int, default=1,
                    help="randgraph core: nonzero couplings per oscillator")
    ap.add_argument("--sakaguchi-alpha", type=float, default=0.0,
                    help="frozen Sakaguchi phase-lag init (0 is kuramoto-degenerate — "
                         "the correct TRAINING start, but untrained matrices must pass "
                         "a nonzero canonical value or the family is not distinct)")
    ap.add_argument("--harmonic2-beta", type=float, default=0.0,
                    help="frozen second-harmonic weight init (same rule as --sakaguchi-alpha)")
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default="cpu", help="cpu | cuda (mps untested for instruments)")
    ap.add_argument("--kind", default="matrix",
                    choices=("matrix", "gain", "order", "coupling-structure",
                             "clamp-pinning", "sensitivity", "conventional",
                             "lr-control", "noise-calibration", "determinism"),
                    help="what this run is FOR. Its physics alone cannot always say: "
                         "a gain gate and a matrix point can share every value and "
                         "differ only in the question being asked, and the kind is "
                         "what picks the group file the result lands in.")
    ap.add_argument("--replicate", default="a",
                    help="determinism runs only: which of the two executions this is")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="exit without running if this exact run is already recorded")
    ap.add_argument("--lr-pick", action="store_true",
                    help="short kuramoto+gru runs across --lrs; reports optimization health only")
    ap.add_argument("--lrs", default="1e-2,3e-3,1e-3")
    ap.add_argument("--quick", action="store_true", help="tiny sizes for a smoke pass")
    args = ap.parse_args(argv)
    if args.select_features == "windowed" and not args.probe_windows:
        ap.error("--select-features windowed requires --probe-windows")
    if args.quick:
        args.frames, args.n_train, args.n_test = 128, 256, 128
        args.epochs, args.batch = 3, 32
    if args.task == "digits":
        from harness.stimuli.frontend import hop_num_frames
        args.frames = hop_num_frames(16000)  # 61 hop frames per 1 s clip
    if args.task == "digitpairs":
        from harness.stimuli import PAIR_MAX_SAMPLES
        from harness.stimuli.frontend import hop_num_frames
        args.frames = hop_num_frames(PAIR_MAX_SAMPLES)
    return args


def make_data(args: argparse.Namespace, n: int, gen_seed: int):
    """(rows [n,T,G], phases [n,T], labels [n], n_classes, tvalid|None).

    The digit tasks' train/test disjointness rides on gen_seed: TEST_SEED pulls
    from the bank's held-out (speaker-disjoint) pool, everything else from
    train. tvalid (digit tasks only): valid frame counts for length-masked
    features."""
    gen = torch.Generator().manual_seed(gen_seed)
    if args.task == "tones":
        classes = tone_classes(args.grid)
        wave, phase, labels = make_tone_clips(n, args.frames, classes, gen,
                                              noise_db=args.noise_db)
        k = len(classes)
    elif args.task == "steps":
        f_a, f_b = step_freqs(args.grid)
        wave, phase, labels = make_step_clips(n, args.frames, f_a, f_b, gen,
                                              noise_db=args.noise_db)
        k = 2
    elif args.task == "am":
        rates = am_rates()
        wave, phase, labels = make_am_clips(n, args.frames, rates, args.grid, gen,
                                            noise_db=args.noise_db)
        k = len(rates)
    elif args.task == "digits":
        from harness.stimuli import load_digit_bank, make_digit_clips
        from harness.stimuli.frontend import hop_num_frames, hop_rows, hop_rows_quad
        bank = load_digit_bank()
        pool = "test" if gen_seed == TEST_SEED else "train"
        waves, lens, labels = make_digit_clips(n, bank, pool, gen,
                                               noise_db=args.noise_db)
        # the transduction ladder: magnitude rows [n,T,G], or quadrature rows
        # [n,T,G,2] -> the core's Adler drive (OscillatorField._drives routes on
        # dim), or "carrier", which bypasses the mel transcoder entirely — the
        # band-split waveform (96-1536 Hz at 16 kHz) drives the field at sample
        # rate, so genuine carrier entrainment is possible; tvalid is then in
        # SAMPLES (true clip length), not hop frames.
        if args.frontend == "carrier":
            rows = bandpass_rows(waves, args.grid)
            tvalid = lens.clone()
        else:
            rows = (hop_rows_quad if args.frontend == "quad" else hop_rows)(waves, args.grid)
            tvalid = torch.tensor([hop_num_frames(int(n_smp)) for n_smp in lens])
        return rows, torch.zeros(n, rows.shape[1]), labels, 10, tvalid
    else:  # digitpairs
        from harness.stimuli import load_digit_bank, make_digitpair_clips
        from harness.stimuli.frontend import hop_num_frames, hop_rows
        bank = load_digit_bank()
        pool = "test" if gen_seed == TEST_SEED else "train"
        pair = tuple(int(x) for x in args.pair.split(","))
        waves, lens, labels = make_digitpair_clips(n, bank, pool, gen, pair,
                                                   noise_db=args.noise_db)
        rows = hop_rows(waves, args.grid)
        tvalid = torch.tensor([hop_num_frames(int(x)) for x in lens])
        return rows, torch.zeros(n, rows.shape[1]), labels, 2, tvalid
    return bandpass_rows(wave, args.grid), phase, labels, k, None


def build_field(args: argparse.Namespace, coupling: str, n_classes: int,
                seed: int) -> OscillatorField:
    torch.manual_seed(seed)  # governs kernel/omega init draws
    return OscillatorField(
        channels=args.channels, grid=args.grid, coupling=coupling,
        damping=args.damping, spectral_clamp=args.clamp,
        substeps=args.substeps, dt=args.dt, n_classes=n_classes,
        probe_seed=5000 + seed, gain=args.gain, seed=seed,
        blocks=args.blocks, kernel_support=args.kernel_support, core=args.core,
        boundary=args.boundary, damping_learnable=args.damping_learnable,
        sakaguchi_alpha=args.sakaguchi_alpha, harmonic2_beta=args.harmonic2_beta,
        graph_k=args.graph_k)


def print_preamble(args: argparse.Namespace, n_classes: int) -> None:
    """Physics context printed into every log: the reachability picture."""
    nat = natural_rate(1.0, args.damping, args.dt, args.substeps)
    print(f"task={args.task}  classes={n_classes}  chance={1 / n_classes:.3f}  "
          f"frames={args.frames}  C={args.channels} G={args.grid}  "
          f"clamp={args.clamp} damping={args.damping} gain={args.gain} "
          f"coupling={args.coupling} blocks={args.blocks} support={args.kernel_support} "
          f"boundary={args.boundary}")
    print(f"untrained natural rotation ~= {nat:.4f} cycles/frame "
          f"(omega ~ N(1.0, 0.1), tilted washboard)")
    if args.task == "tones":
        for f, row in tone_classes(args.grid):
            need = 2 * math.pi * f / (args.dt * args.substeps)
            print(f"  tone {f:.4f} cyc/frame (row {row:2d}): needs theta_dot ~= {need:.2f} "
                  f"(omega-units; init mean 1.0)")
    elif args.task == "steps":
        f_a, f_b = step_freqs(args.grid)
        print(f"  step pair: {f_a:.4f} <-> {f_b:.4f} cyc/frame "
              f"(identical average spectra between orders by construction)")
    elif args.task == "am":
        rs = am_rates()
        print(f"  AM rates {rs[0]:.4f}..{rs[-1]:.4f} cyc/frame on the shared row-"
              f"{AM_CARRIER_ROW} carrier ({rs[0] * args.frames:.1f}.."
              f"{rs[-1] * args.frames:.1f} envelope cycles/clip); sidebands are an "
              f"inherent spectral cue — frozen ridge is the statics baseline")
    elif args.task == "digits":
        print(f"  AudioMNIST digits on the hop frontend ({args.frontend}), "
              f"{args.frames} hop frames @62.5 fps, speaker-disjoint pools, "
              "length-masked features (true lengths from the bank)")
    elif args.task == "digitpairs":
        print(f"  AudioMNIST digit pairs {args.pair} in both orders, "
              f"{args.frames} hop frames @62.5 fps; the full-span pooled read is "
              "order-free by construction, so the matched floor sits at chance")


def eval_arm(args, model, arm: str, seed: int, rows_tr, y_tr, rows_te, phases_te,
             y_te, n_classes: int, out_dir: Path, history,
             tv_tr=None, tv_te=None) -> dict:
    """Identical evaluation protocol for every arm; instrumentation for torus arms."""
    frz = frozen_probe_acc(model, rows_te, y_te, args.batch, tvalid=tv_te)
    f_tr = collect_features(model, rows_tr, args.batch, tvalid=tv_tr)
    f_te = collect_features(model, rows_te, args.batch, tvalid=tv_te)
    ridge = fit_ridge_probe(f_tr, y_tr, f_te, y_te, n_classes)
    # readout-capacity parity: the identical ridge on a COMMON-width
    # projection of the native features — reported for every arm.
    ridge["acc_parity"] = fit_ridge_probe(parity_project(f_tr), y_tr,
                                          parity_project(f_te), y_te, n_classes)["acc"]
    if getattr(args, "probe_windows", 0) and hasattr(model, "features_windowed"):
        w = args.probe_windows
        with torch.no_grad():  # eval-only: without this the scan graph OOMs CUDA
            dev = next(model.parameters()).device
            fw_tr = torch.cat([model.features_windowed(
                rows_tr[i:i + args.batch].to(dev), w,
                tvalid=None if tv_tr is None else tv_tr[i:i + args.batch]).cpu()
                for i in range(0, len(rows_tr), args.batch)])
            fw_te = torch.cat([model.features_windowed(
                rows_te[i:i + args.batch].to(dev), w,
                tvalid=None if tv_te is None else tv_te[i:i + args.batch]).cpu()
                for i in range(0, len(rows_te), args.batch)])
        ridge["acc_windowed"] = fit_ridge_probe(fw_tr, y_tr, fw_te, y_te, n_classes)["acc"]
    if getattr(args, "probe_macro", 0) and hasattr(model, "features_macro"):
        p = args.probe_macro
        with torch.no_grad():  # eval-only (same OOM lesson as windowed)
            dev = next(model.parameters()).device
            fm_tr = torch.cat([model.features_macro(
                rows_tr[i:i + args.batch].to(dev), p,
                tvalid=None if tv_tr is None else tv_tr[i:i + args.batch]).cpu()
                for i in range(0, len(rows_tr), args.batch)])
            fm_te = torch.cat([model.features_macro(
                rows_te[i:i + args.batch].to(dev), p,
                tvalid=None if tv_te is None else tv_te[i:i + args.batch]).cpu()
                for i in range(0, len(rows_te), args.batch)])
        ridge["acc_macro"] = fit_ridge_probe(fm_tr, y_tr, fm_te, y_te, n_classes)["acc"]
    if hasattr(model, "features_settle"):
        # settle-read column (rides): drive-free continuation after
        # the clip, featurized alone, identical ridge — reported for EVERY arm.
        with torch.no_grad():  # eval-only (same OOM lesson as windowed)
            dev = next(model.parameters()).device
            fs_tr = torch.cat([model.features_settle(rows_tr[i:i + args.batch].to(dev)).cpu()
                               for i in range(0, len(rows_tr), args.batch)])
            fs_te = torch.cat([model.features_settle(rows_te[i:i + args.batch].to(dev)).cpu()
                               for i in range(0, len(rows_te), args.batch)])
        ridge["acc_settle"] = fit_ridge_probe(fs_tr, y_tr, fs_te, y_te, n_classes)["acc"]
    row = dict(arm=arm, seed=seed, clamp=args.clamp, damping=args.damping,
               frz_acc=frz, ridge_acc=ridge["acc"], margin=ridge["margin"], lam=ridge["lam"],
               ridge_acc_parity=ridge["acc_parity"],
               **({"ridge_acc_windowed": ridge["acc_windowed"]} if "acc_windowed" in ridge else {}),
               **({"ridge_acc_macro": ridge["acc_macro"]} if "acc_macro" in ridge else {}),
               **({"ridge_acc_settle": ridge["acc_settle"]} if "acc_settle" in ridge else {}))
    if history:
        first, last = history[0], history[-1]
        drop = (first["loss"] - last["loss"]) / max(first["loss"], 1e-9)
        row.update(train_loss=last["loss"], loss_drop=drop,
                   healthy=bool(drop >= HEALTH_MIN_LOSS_DROP),
                   grad_k=last.get("grad_k"), grad_w=last.get("grad_w"),
                   sec_ep=sum(h["sec"] for h in history) / len(history))
        if "sel_epoch" in last:  # best-val selection: which epoch the verdict reads
            row.update(sel_epoch=last["sel_epoch"], sel_score=last["sel_score"])
    row.update(**drive_kick_stats(rows_te, args.gain, args.dt))
    if isinstance(model, OscillatorField):
        n_i = min(256, rows_te.shape[0])
        # drive-phase reader: for digits the recorded "stimulus phase"
        # is a zeros placeholder — the meaningful locking reference is the
        # delivered drive's own per-band analytic phase (mag/carrier rows).
        phi = (analytic_row_phase(rows_te[:n_i]) if args.task == "digits"
               and args.frontend in ("mag", "carrier") else None)
        instr = instrument_field(model, rows_te[:n_i], phases_te[:n_i], y_te[:n_i], n_classes,
                                 phi_rows=phi)
        row.update(plv=instr["plv_mean"], ent_frac=instr["ent_frac_mean"], R=instr["R_mean"],
                   opnorm_raw=instr["opnorm_raw"].max().item(),
                   entropy_min=instr["spec_entropy"].min().item(),
                   omega_std=instr["omega"].std().item())
        torch.save(instr, out_dir / f"{arm}-s{seed}-instr.pt")
    torch.save({k: v.cpu() for k, v in model.state_dict().items()},
               out_dir / f"{arm}-s{seed}.pt")
    return row


SELECT_RIDGE_N = 512  # train-subset size for the checkpoint-selection ridge


def make_selector(args, rows_a, y_a, rows_b, y_b, n_classes: int):
    """Best-val checkpoint metric: ridge fit on a fixed
    train subset, scored on the val slice — selects for field READABILITY (the
    verdict metric) rather than probe CE, whose divergence from ridge is the
    failure mode being fixed. Test pool untouched; deterministic; consumes no
    training RNG."""
    def selector(model) -> float:
        model.eval()
        with torch.no_grad():
            if args.select_features == "windowed":
                dev = next(model.parameters()).device
                w = args.probe_windows
                fa = torch.cat([model.features_windowed(rows_a[i:i + args.batch].to(dev), w).cpu()
                                for i in range(0, len(rows_a), args.batch)])
                fb = torch.cat([model.features_windowed(rows_b[i:i + args.batch].to(dev), w).cpu()
                                for i in range(0, len(rows_b), args.batch)])
            else:
                fa = collect_features(model, rows_a, args.batch)
                fb = collect_features(model, rows_b, args.batch)
        return fit_ridge_probe(fa, y_a, fb, y_b, n_classes)["acc"]
    return selector


def run_matrix(args: argparse.Namespace) -> list[dict]:
    cfg = vars(args)
    out_dir = store.artifact_dir(cfg, args.kind)
    rows_te, phases_te, y_te, n_classes, tv_te = make_data(args, args.n_test, TEST_SEED)
    print_preamble(args, n_classes)
    arms = [a for a in ARM_ORDER if a in args.arms.split(",")]
    results: list[dict] = []
    for seed in (int(s) for s in args.seeds.split(",")):
        rows_tr, _, y_tr, _, tv_tr = make_data(args, args.n_train, 1000 + seed)
        n_val = max(32, args.n_train // 8)
        rows_v, y_v = rows_tr[-n_val:], y_tr[-n_val:]
        tv_v = None if tv_tr is None else tv_tr[-n_val:]
        tv_trn = None if tv_tr is None else tv_tr[:-n_val]
        selector = None
        if args.select == "best-val":
            n_sel = min(SELECT_RIDGE_N, args.n_train - n_val)
            selector = make_selector(args, rows_tr[:n_sel], y_tr[:n_sel],
                                     rows_v, y_v, n_classes)
        # one shared init per seed: kuramoto/forced/frozen all start from it
        init_sd = copy.deepcopy(build_field(args, args.coupling, n_classes, seed).state_dict())
        torch.save(init_sd, out_dir / f"init-s{seed}.pt")
        kuramoto_sd = None
        for arm in arms:
            t0 = time.time()
            history = None
            if arm == "forced" and (args.coupling != "kuramoto" or args.core != "phase"):
                print(f"[s{seed}] forced skipped: incompatible with this run's physics")
                continue
            if args.frontend == "quad" and arm in QUAD_INCOMPATIBLE_ARMS:
                print(f"[s{seed}] {arm} skipped: quad frontend drives the field only")
                continue
            if arm in ("kuramoto", "forced"):
                # the "kuramoto" arm is the trained-physics arm; in stage-D runs
                # it carries the run's coupling law (recorded in results config)
                model = build_field(args, args.coupling if arm == "kuramoto" else arm,
                                    n_classes, seed)
                model.load_state_dict(init_sd)
                model = model.to(args.device)
                if arm == "kuramoto" and args.designed_init:
                    # composite arm: cochlear omega placement, THEN training
                    with torch.no_grad():
                        physics = (model.core.blocks[0] if hasattr(model.core, "blocks")
                                   else model.core)
                        physics.natural_freqs.copy_(tonotopic_omega(
                            args.channels, args.grid, args.dt, args.substeps,
                            torch.Generator().manual_seed(8000 + seed)))
                history = train_arm(model, rows_tr[:-n_val], y_tr[:-n_val], rows_v, y_v,
                                    epochs=args.epochs, lr=args.lr, batch=args.batch,
                                    seed=seed, log_path=store.curve_path(vars(args), arm, seed, args.kind),
                                    selector=selector, select_every=args.select_every,
                                    tvalid_tr=tv_trn, tvalid_val=tv_v)
                if arm == "kuramoto":
                    kuramoto_sd = copy.deepcopy(model.state_dict())
            elif arm == "omegaenc":
                # input-as-omega: tuner writes per-row omega; drive OFF.
                # Same physics init as the run (encoder params start at zero,
                # so t=0 dynamics = the frozen field with no input).
                torch.manual_seed(seed)
                model = OscillatorField(channels=args.channels, grid=args.grid,
                                 coupling=args.coupling, damping=args.damping,
                                 spectral_clamp=args.clamp, substeps=args.substeps,
                                 dt=args.dt, n_classes=n_classes, probe_seed=5000 + seed,
                                 gain=args.gain, seed=seed, core="phase",
                                 boundary=args.boundary, omega_encoder=True)
                model.load_state_dict(init_sd, strict=False)
                model = model.to(args.device)
                history = train_arm(model, rows_tr[:-n_val], y_tr[:-n_val], rows_v, y_v,
                                    epochs=args.epochs, lr=args.lr, batch=args.batch,
                                    seed=seed, log_path=store.curve_path(vars(args), arm, seed, args.kind),
                                    selector=selector, select_every=args.select_every,
                                    tvalid_tr=tv_trn, tvalid_val=tv_v)
            elif arm == "tcn":
                torch.manual_seed(seed)
                model = TCNBaseline(grid=args.grid, n_classes=n_classes, probe_seed=5000 + seed)
                model = model.to(args.device)
                history = train_arm(model, rows_tr[:-n_val], y_tr[:-n_val], rows_v, y_v,
                                    epochs=args.epochs, lr=args.lr, batch=args.batch,
                                    seed=seed, log_path=store.curve_path(vars(args), arm, seed, args.kind),
                                    selector=selector, select_every=args.select_every,
                                    tvalid_tr=tv_trn, tvalid_val=tv_v)
            elif arm in BASELINE_ARMS:
                # exp-1 minis (cnn/transformer/s4d): same protocol as tcn
                torch.manual_seed(seed)
                model = BASELINE_ARMS[arm](grid=args.grid, n_classes=n_classes,
                                           probe_seed=5000 + seed)
                model = model.to(args.device)
                history = train_arm(model, rows_tr[:-n_val], y_tr[:-n_val], rows_v, y_v,
                                    epochs=args.epochs, lr=args.lr, batch=args.batch,
                                    seed=seed, log_path=store.curve_path(vars(args), arm, seed, args.kind),
                                    selector=selector, select_every=args.select_every,
                                    tvalid_tr=tv_trn, tvalid_val=tv_v)
            elif arm == "gru":
                torch.manual_seed(seed)
                model = GRUBaseline(grid=args.grid, n_classes=n_classes, probe_seed=5000 + seed)
                model = model.to(args.device)
                history = train_arm(model, rows_tr[:-n_val], y_tr[:-n_val], rows_v, y_v,
                                    epochs=args.epochs, lr=args.lr, batch=args.batch,
                                    seed=seed, log_path=store.curve_path(vars(args), "gru", seed, args.kind),
                                    selector=selector, select_every=args.select_every,
                                    tvalid_tr=tv_trn, tvalid_val=tv_v)
            elif arm == "frozen":
                model = build_field(args, args.coupling, n_classes, seed)
                model.load_state_dict(init_sd)
                if getattr(args, "omega_uniform", False):
                    with torch.no_grad():
                        physics = (model.core.blocks[0] if hasattr(model.core, "blocks")
                                   else model.core)
                        physics.natural_freqs.fill_(1.0)
            elif arm == "designed":
                # designed-omega: untrained field, engineered tonotopic
                # omega spread — tests "design the resonator bank" vs both
                # frozen-random and trained arms under the identical protocol.
                model = build_field(args, args.coupling, n_classes, seed)
                model.load_state_dict(init_sd)
                with torch.no_grad():
                    physics = (model.core.blocks[0] if hasattr(model.core, "blocks")
                               else model.core)  # phase core vs SL core
                    physics.natural_freqs.copy_(tonotopic_omega(
                        args.channels, args.grid, args.dt, args.substeps,
                        torch.Generator().manual_seed(8000 + seed)))
            elif arm == "shuffled":
                if kuramoto_sd is None:
                    print(f"[s{seed}] shuffled skipped: no trained kuramoto in this run")
                    continue
                model = build_field(args, args.coupling, n_classes, seed)
                model.load_state_dict(kuramoto_sd)
                shuffle_kernel_(model, torch.Generator().manual_seed(7000 + seed))
            if history and history[-1].get("aborted"):
                # Diverged to non-finite loss: record the fact, skip eval — the
                # params are garbage and eval on them crashes or lies (learned
                # from the clamp-off run).
                results.append(dict(arm=arm, seed=seed, clamp=args.clamp,
                                    damping=args.damping, aborted=True,
                                    epochs_completed=len(history) - 1))
                print(f"{arm:9s} s{seed}  ABORTED (non-finite loss) after "
                      f"{len(history) - 1} full epochs")
                continue
            row = eval_arm(args, model.eval().to(args.device), arm, seed, rows_tr, y_tr,
                           rows_te, phases_te, y_te, n_classes, out_dir, history,
                           tv_tr=tv_tr, tv_te=tv_te)
            row["wall_sec"] = time.time() - t0
            results.append(row)
            print(format_row(row))
        # the headline delta, printed per seed as it lands
        by = {r["arm"]: r for r in results if r["seed"] == seed}
        if "kuramoto" in by and "frozen" in by:
            d = by["kuramoto"]["ridge_acc"] - by["frozen"]["ridge_acc"]
            print(f"[s{seed}] trained-vs-frozen ridge delta: {d:+.3f}")
    where = store.write_run({**cfg, "n_classes": n_classes}, results, args.kind)
    print(f"\nrecorded {where}")
    return results


def run_lr_pick(args: argparse.Namespace) -> None:
    """Stage A: is the recipe optimizable at all, and at which lr? Short runs,
    frozen-probe val acc + loss drop only — no verdicts read from this stage."""
    out_dir = store.artifacts_root() / "lr-pick"
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = max(4, args.epochs // 4)
    rows_tr, _, y_tr, n_classes, tv_tr = make_data(args, args.n_train, 1000)
    n_val = max(32, args.n_train // 8)
    print_preamble(args, n_classes)
    report = []
    for arm in ("kuramoto", "gru"):
        if args.frontend == "quad" and arm in QUAD_INCOMPATIBLE_ARMS:
            print(f"{arm}: skipped (quad frontend drives the field only)")
            continue
        for lr in (float(s) for s in args.lrs.split(",")):
            if arm == "kuramoto":
                model = build_field(args, "kuramoto", n_classes, seed=0)
            else:
                torch.manual_seed(0)
                model = GRUBaseline(grid=args.grid, n_classes=n_classes, probe_seed=5000)
            model = model.to(args.device)
            h = train_arm(model, rows_tr[:-n_val], y_tr[:-n_val], rows_tr[-n_val:], y_tr[-n_val:],
                          epochs=epochs, lr=lr, batch=args.batch, seed=0,
                          log_path=out_dir / f"{arm}-lr{lr:g}.csv",
                          tvalid_tr=None if tv_tr is None else tv_tr[:-n_val],
                          tvalid_val=None if tv_tr is None else tv_tr[-n_val:])
            if h and "aborted" in h[-1]:
                report.append(dict(arm=arm, lr=lr, status="diverged"))
                print(f"{arm:10s} lr={lr:g}: DIVERGED (non-finite loss)")
                continue
            drop = (h[0]["loss"] - h[-1]["loss"]) / max(h[0]["loss"], 1e-9)
            report.append(dict(arm=arm, lr=lr, loss0=h[0]["loss"], loss=h[-1]["loss"],
                               drop=drop, val_acc=h[-1]["val_acc"],
                               grad_total=h[-1]["grad_total"], sec_ep=h[-1]["sec"]))
            print(f"{arm:10s} lr={lr:g}: loss {h[0]['loss']:.4f}->{h[-1]['loss']:.4f} "
                  f"(drop {drop:+.1%})  val_acc {h[-1]['val_acc']:.3f}  "
                  f"grad {h[-1]['grad_total']:.2f}  {h[-1]['sec']:.0f}s/ep")
    (out_dir / "lr_pick.json").write_text(json.dumps(
        dict(config=vars(args), epochs=epochs, report=report), indent=2))
    print(f"\nwrote {out_dir}/lr_pick.json")


def format_row(r: dict) -> str:
    parts = [f"{r['arm']:9s} s{r['seed']}",
             f"frz {r['frz_acc']:.3f}", f"ridge {r['ridge_acc']:.3f}",
             f"margin {r['margin']:+.3f}"]
    if "loss_drop" in r:
        parts.append(f"drop {r['loss_drop']:+.1%}{'' if r['healthy'] else ' UNHEALTHY'}")
    if "plv" in r:
        parts.append(f"plv {r['plv']:.3f} entfrac {r['ent_frac']:.3f} R {r['R']:.3f}")
    if "sec_ep" in r:
        parts.append(f"{r['sec_ep']:.0f}s/ep")
    return "  ".join(parts)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.skip_if_done and store.has_run(vars(args), args.kind):
        print(f"already recorded: {store.address(vars(args), args.kind)[1]}")
        return
    torch.set_num_threads(args.threads)
    if args.lr_pick:
        run_lr_pick(args)
    else:
        run_matrix(args)


if __name__ == "__main__":
    main()
