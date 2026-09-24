# Experiment code

Everything needed to rerun the paper and check its numbers. One command plans
and drives a sweep; one module scores it.

There are two harnesses here. `harness/confirm/` runs the **confirmatory
study** registered in [`../REGISTRATION.md`](../REGISTRATION.md), frozen before
its first run; its record is `../results/confirmatory/`. The rest of `harness/`
is the exploratory harness that produced the August record in `../results/`,
kept runnable so that record can be checked, which the confirmatory study's
legacy gate does. Start with [the confirmatory run](#the-confirmatory-run).

## Quick start

```bash
cd papers/02-untrained-reservoirs/src
uv sync
uv run pytest                                   # 182 contract tests, no data needed
uv run python -m harness.measurement.score all  # score the committed record
```

Those three need no corpus: the tests run on synthetic stimuli, and scoring
reads the committed results. AudioMNIST is only needed to *re-run* experiments.

## Getting the data

The task is spoken digits 0-9 from
[AudioMNIST](https://github.com/soerenab/AudioMNIST) (Becker et al. 2018) —
30,000 recordings, 60 speakers x 10 digits x 50 repetitions at 48 kHz.

```bash
git clone --depth 1 https://github.com/soerenab/AudioMNIST data/AudioMNIST
uv run python -m harness.stimuli.digits --build-bank
```

**Neither the corpus nor the derived bank is committed.** `src/data/` is
gitignored down to its README: the corpus belongs to its own authors, and the
bank is a deterministic function of it, so both are rebuilt rather than
shipped. See [data/README.md](data/README.md) for what the build does and for
the `OSC_DATA_DIR` override if you want the corpus on another disk.

## Layout

Grouped by what each part is FOR, so you can find the thing you want to check
without knowing the codebase.

```
src/
├── harness/
│   ├── runner.py        one process = one experiment run
│   ├── sweep.py         the grids, the resume guard, the parallel driver
│   ├── results.py       where a run's numbers go and how they come back
│   ├── utils/           protocol constants; where data and results live
│   ├── stimuli/         sound in: front-ends, stimuli, banks, injection
│   ├── models/          the arms under test
│   │   ├── field.py         OscillatorField — the thing under test
│   │   ├── phase.py         the phase-oscillator core
│   │   ├── stuart_landau.py the amplitude-phase core
│   │   ├── random_graph.py  the connectivity-disorder control
│   │   ├── geometries/      one module per lattice venue
│   │   └── baselines/       one module per conventional reference
│   └── measurement/     probe, floor, instruments, scoring
├── scripts/             sweep wrapper, assets, the two standalone protocols
├── tests/               one file per module
├── data/                corpus + derived bank            (gitignored)
└── pyproject.toml

../results/              the record: one file per drive x coupling law
../resources/            figures and audio that ship with the paper
```

## Architecture

Data flows left to right. Nothing upstream or downstream of the field has a
trainable parameter in the primary protocol — the only fitted object anywhere
is the closed-form ridge probe, which is why a separation it finds must have
been written by the dynamics.

```
                    stimuli/                              models/
   AudioMNIST ─▶ digits ─▶ noise ─▶ frontend ─▶ injection ─▶ OscillatorField
                                        │                          │
                            audio.py ───┤                    ┌─────┴──────────┐
                            (log-mel)   │                    │  a core:       │
                                        │                    │  phase         │──▶ geometries/
                        envelope │ quadrature │ carrier      │  stuart_landau │    (9 venues,
                                                             │  random_graph  │     one module
                                                             └─────┬──────────┘     each)
                                                                   ▼
                                                             trajectories
                                                            ╱            ╲
                                     measurement/probe.py              measurement/instruments.py
                                     (features ▶ ridge)                (locking, order parameter)
                                              │                              │
                 measurement/floor.py ──▶ measurement/score.py               │
                 (the same ridge,         (bars ▶ verdicts)        mechanism, never verdicts
                  field bypassed)              ▲
                                     models/baselines/
                                     (GRU, TCN, CNN, transformer, S4D)
```

| module | what it owns |
|---|---|
| `runner.py` | the CLI for a single run: build, train or freeze, evaluate, record |
| `sweep.py` | the experiment grids, resource planning, live progress |
| `results.py` | run addressing, lock-safe writes, the resume guard |
| `utils/constants.py` | the fixed protocol constants (warmup, probe scale, lock threshold) |
| `utils/paths.py` | data and results roots, and the `OSC_*` overrides |
| `stimuli/audio.py` | the parameter-free log-mel front-end |
| `stimuli/frontend.py` | hop-rate magnitude and quadrature-baseband rows |
| `stimuli/filterbank.py` | the log-spaced band-to-row map, and the carrier band split |
| `stimuli/synthetic.py` | tones, frequency steps, amplitude modulation |
| `stimuli/digits.py` | the AudioMNIST bank, digit clips, order-task pairs |
| `stimuli/injection.py` | the broadcast drive and the integrator-validity instrument |
| `models/field.py` | `OscillatorField` and the physics-only invariant |
| `models/phase.py` | `PhaseCore` — coupled phases, any coupling law, any venue |
| `models/geometries/` | one module per venue, behind a four-method interface |
| `models/baselines/` | the five param-matched conventional references |
| `measurement/probe.py` | featurization, the ridge readout, the training loop |
| `measurement/floor.py` | the no-dynamics floor |
| `measurement/instruments.py` | drive-phase locking, order parameter, kernel spectra |
| `measurement/score.py` | the record read against the pre-registered bars |

### Adding a geometry

Subclass `Geometry` (or `PlanarGeometry`), implement the four methods, register
it in `models/geometries/__init__.py`. The parametrized tests then cover it
automatically: FFT-vs-dense equivalence, forward determinism, every coupling
law, the clamp correction, and the tonotopic map.

## Tests

```bash
uv run pytest                          # everything
uv run pytest tests/test_geometries.py # one module
uv run pytest -q -k order
```

One file per module. They are contract tests, not smoke tests: each pins a
property some claim in the paper depends on — that a tone's energy lands in the
row the filterbank claims, that gradients reach the coupling kernel and nothing
else, that the ridge is at chance on structureless features, that only the wrap
rule changes between geometries, that the quadrature pathway differs from the
envelope pathway in phase alone, and that the sweep grids still describe the
committed record exactly.

Three tests skip without the digit bank; the rest run on synthetic stimuli.

## The confirmatory run

Every arm (the spectrogram-only baseline, the coupled and uncoupled oscillator
networks, two leaky-integrator banks, five trained baselines) is read by one
contract: the same three statistics over the same fixed window for every clip,
standardized with the training set's own statistics, projected to a common
width, and classified by the same linear readout. The design and the reasons are
in the registration; what each tier runs is in [TIERS.md](../TIERS.md).

```bash
uv run python -m harness.confirm.protocol --build-bank   # the 50-repetition bank
uv run python -m harness.confirm.plan prepare            # front-end rows, once per drive and noise level
uv run python -m harness.confirm.gates legacy            # the exploratory record reproduces
uv run python -m harness.confirm.plan run gate tier1 --workers 3 --threads 2
uv run python -m harness.confirm.gates check             # every zero-drive cell reads chance
uv run python -m harness.confirm.plan run tier2 becker tier3 --workers 6 --threads 1
uv run python -m harness.confirm.summary                 # every accuracy and difference with its spread
uv run python -m harness.confirm.figures                 # the paper's confirmatory figures and their tables
uv run python -m harness.confirm.score                   # the registered bars' verdicts, for the record
```

`plan run <tier> --dry-run` prints what would run. Every run is recorded by an
identity derived from its specification, so a sweep stopped at any point
restarts where it left off, and `--task order` or `--task recognition` lets two
machines split a tier without sharing a file. The carrier tier integrates at
16 kHz and wants a GPU; `scripts/pod_run.sh <AudioMNIST checkout> carrier` runs
it, or any tier, on a RunPod pod from the pushed branch.

| module | what it owns |
|---|---|
| `confirm/protocol.py` | the bank, Protocols A and B, the order task, per-clip noise, row caches |
| `confirm/arms.py` | the arms, and the reads each one records |
| `confirm/readout.py` | standardize, project, ridge, per-clip correctness |
| `confirm/run.py` | one run, and the record it writes |
| `confirm/plan.py` | the registered tiers, and the parallel driver |
| `confirm/gates.py` | the legacy-reproduction and zero-drive gates |
| `confirm/summary.py` | every accuracy and paired difference, with its spread: `summary.json` and `summary.md` |
| `confirm/figures.py` | the paper's confirmatory figures, and the tables printed beside them |
| `confirm/score.py` | the verdicts against the registered bars, kept for the record |
| `confirm/terms.py` | the paper's names for the record's labels, used by every table, figure and report |

### Terms in the code

The paper's glossary (Appendix A) defines the terms. The code, the run record and
the registration keep the labels the registration froze, because they are the
runs' identities; `confirm/terms.py` turns them into the paper's terms for
everything people read.

| paper term | in the code, the record and the registration |
|---|---|
| spectrogram-only baseline | arm kind `floor`; its whole-clip reads are `windowed@wholeclip` and `pooled@wholeclip` |
| coupled oscillator network | arm kind `field`, labels `field-<coupling>-<geometry>-<ω>-lam<λ>-clamp<ceiling>`; "the field" |
| uncoupled oscillator network | `field` with `severed=True`, labels `severed-...`; "the severed field" |
| leaky-integrator bank, state-matched / width-matched | `bank-c4` / `bank-c8` (`LeakyBank`); "bank A" / "bank B" |
| trained baselines | `ann-gru`, `ann-tcn`, `ann-cnn`, `ann-transformer`, `ann-s4d` |
| reservoir | the frozen arms (`FROZEN` in `plan.py`) |
| untrained | "frozen" |
| readout | the ridge (`readout.py`) |
| coupling function | `physics`: `kuramoto`, `sakaguchi`, `harmonic2`, `winfree`, `sl`, `sl-fixedamp`; "coupling law", "family" |
| lattice geometry | `boundary`; "shape", "venue" |
| natural frequencies: random / tonotopic / identical | `omega`: `random` / `designed` / `uniform` |
| restoring strength λ | `damping`, `lam` in labels; "pinning" |
| coupling ceiling | `clamp`, `spectral_clamp`; "spectral clamp" |
| coupling kernel, channel | `kernel`, `channels` |
| input gain | `gain`, `g` in run ids |
| input pathway: band-energy / quadrature / carrier | `drive`: `envelope` / `quadrature` / `carrier` |
| mel spectrogram, band energies | the front-end rows (`hop_rows`); "envelopes" |
| matched pair | "twin" |

## Reproducing the paper

> **Run the baselines first.** Everything else is a difference against
> something they measure, so nothing downstream means anything without them.

```bash
uv run bash scripts/run_sweep.sh baselines
uv run bash scripts/run_sweep.sh envelope     # the primary drive
uv run bash scripts/run_sweep.sh quadrature   # the phase-referenced drive
uv run bash scripts/run_sweep.sh carrier      # the transcoder-free drive
uv run bash scripts/run_sweep.sh all          # or all four, in order
```

Each sweep prints its plan, uses the machine it finds, reports live progress
with an ETA, and scores itself at the end:

```
=== envelope: 1000 runs planned, 0 already recorded, 1000 to run
    10 cores, 34 GB free -> 9 workers x 1 threads (~1.0 GB each)
[  418/1000]  41.8%  22m elapsed, 31m left  19.0/min  kuramoto-sphere-random-lam0.3-clamp1-0db-g2
```

Runs are independent processes, so throughput comes from running several at
once rather than threading one harder. The planner sizes the pool from the core
count **and** from free memory, because the carrier pathway drives the field at
16 kHz rather than at the 62.5 fps hop rate and needs several times the memory
per worker. Override either with `--workers` / `--threads`.

Everything is resume-guarded against the record, so an interrupted sweep
restarts for free, and re-running a finished one is a few seconds of reading
JSON before it goes straight to scoring. To see the plan without running it:

```bash
uv run bash scripts/run_sweep.sh envelope --dry-run
```

To reproduce into a fresh tree instead of the committed one:

```bash
OSC_RESULTS_DIR=/tmp/rerun uv run bash scripts/run_sweep.sh envelope
```

The envelope sweep is the long pole: ~1,000 runs at roughly half a minute each
of single-core time.

### Assets, and one run for orientation

```bash
uv run bash scripts/make_assets.sh   # floors, ladder, trained-head refs, audio
uv run python -m harness.runner --task digits --noise-db 0 --gain 2.0 \
    --arms frozen --probe-windows 4 --probe-macro 4
```

## Reading the record

One file per drive variant and coupling law, so an entire sweep is one file you
can open — not a tree of a thousand directories:

```
results/
├── baselines/     conventional.json, lr-control.json, noise-calibration.json,
│                  determinism.json, floors.json, readout-ladder.json
├── envelope/      matrix-kuramoto.json ... one per coupling law, plus
│                  gain, clamp-pinning, coupling-structure, sensitivity, order
├── quadrature/
├── carrier/
└── training-curves/   per-epoch CSVs for the trained arms
```

Inside a group file, `runs` maps a run id to its complete configuration and one
row per (arm, seed):

```json
{"group": "envelope/matrix-kuramoto", "drive": "envelope",
 "runs": {"torus-designed-lam0.3-clamp1-0db-g2": {"config": {...}, "rows": [...]}}}
```

The id spells out the factor levels — geometry, frequency structure, pinning,
spectral clamp, noise, gain — and is **derived from the configuration**, never
from run order, so the same experiment always lands in the same place. A test
asserts the two can never drift apart.

Each row carries `ridge_acc` (standard pooled), `ridge_acc_windowed` (the
primary metric for the recognition task), `ridge_acc_macro`,
`ridge_acc_settle`, `ridge_acc_parity`, and the instrument readings `plv`,
`ent_frac`, `R`, `opnorm_raw`, `entropy_min`, `omega_std`.

Two reading rules matter. The **order task is scored on `ridge_acc`**, never
the windowed column: the whole construction rests on the read being order-free,
and per-window statistics are not. And a run is only ever compared to **its own
representation's floor** — the carrier band-split discards envelope content
above 1.5 kHz, so its floor is a different number, not a worse one.

## Scoring

```bash
uv run python -m harness.measurement.score all --figures
uv run python -m harness.measurement.score envelope
```

Every verdict prints beside the bar it was scored against, so a reader can
check the arithmetic rather than take the verdict on trust. The bars were
registered before the runs they score and live at the top of
`measurement/score.py`. `--figures` regenerates
`resources/figures/g9-readability-vs-coherence.png`; the other three figures
are authored SVGs. After editing one, render its PDF and PNG from the
repository root:

```bash
uv run --with svglib --with reportlab python3 publishing/lib/svg_render.py \
    papers/02-untrained-reservoirs/resources/figures
```

Write math the natural way, with Unicode subscripts and superscripts such as
θᵢ and Φᵀ: the renderer sets them as positioned runs in a temporary copy, so
the result does not depend on any font carrying those letters. It refuses what
it cannot render faithfully and says why. A combining accent is the usual
case, so write dθ/dt where you would have written θ̇.

## Determinism

**Forward passes are bit-identical**, on every geometry and both coupling
implementations — asserted in `tests/test_geometries.py`. That is the
replication claim that matters, because every number the paper reports comes
from a frozen field, which never runs a backward pass.

**Backward is reproducible only to float32 rounding** on the dense coupling
path: its gradient accumulates through a gather whose reduction order is not
pinned, so two identical training runs agree to about 5e-7 relative rather than
exactly. This is a property of the operation, not of this code — the same
divergence appears between two runs of one implementation — and it affects only
the trained arms.

The rest of the protocol is deterministic by construction: the ridge is
closed-form, the parity projection is drawn from a fixed seed and cached per
native feature width, the frozen probe is a buffer seeded per arm-seed, and the
test set is pinned to one generator seed across every arm and run seed, so arms
are always compared on identical clips.
