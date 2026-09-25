# Results

The complete record of the study: 6,275 runs in eight experiments, one file per experiment and task (the design and sweep experiments split theirs by coupling function). Every number in the paper is computed from these files by `uv run python -m harness.experiment.summary` and `uv run python -m harness.experiment.figures` (run from `../src`), with no corpus needed. The experiments are those of the paper's Appendix B.

## The files

| file | runs | what it holds | in the paper |
|---|---|---|---|
| `leak-check-recognition.json` | 13 | The integrity check. The four reservoirs (the coupled and uncoupled oscillator networks and both leaky-integrator banks) with no input (input gain 0, seed 0), which must read exactly chance; and the coupled network read over each clip's own length rather than the fixed window, at gains 0, 1 and 2 and seeds 0 to 2, a diagnostic of how much a per-clip window would leak. 0 dB SNR. | Appendix B |
| `controls-recognition.json` | 216 | Recognition by every arm: the spectrogram-only baseline, the coupled and uncoupled oscillator networks (the reference configuration: Kuramoto coupling, torus, random natural frequencies, restoring strength 0.3, coupling ceiling 1), the state-matched and width-matched leaky-integrator banks, and the five trained baselines. Clean, 0 dB and −5 dB SNR; input gain 1 and 2 for the reservoirs; seeds 0 to 2; 2,048, 8,192 and 24,000 training clips; readout widths 192, 1,024 and 4,096. | Sections 4.1, 4.7 |
| `controls-order.json` | 630 | The temporal-order task for the same arms, on five digit pairs (0–6, 1–8, 2–5, 3–7, 4–9), 2,048 training and 2,048 test sequences per pair; clean, 0 dB and −5 dB SNR; gains 1 and 2; seeds 0 to 2. | Section 4.2 |
| `design-recognition-<coupling>.json` (6 files) | 3,744 | The coupled network, one factor varied at a time around the reference: coupling function (one file each: `kuramoto`, `kuramoto-sakaguchi`, `second-harmonic`, `winfree`, `stuart-landau`, `stuart-landau-fixed`), lattice geometry (torus, cylinder, sheet, helix, cube, sphere; the two Stuart–Landau functions on the torus only), natural frequencies (random, tonotopic, identical), restoring strength (0.3, 0.1) and coupling ceiling (1, 0.5). 0 dB and −5 dB SNR; gains 1 and 2; seeds 0 to 2; 2,048 training clips; read with and without rotation rates. 864 runs for each phase coupling function, 144 for each Stuart–Landau one. | Sections 4.3 to 4.5 |
| `quadrature-recognition.json` | 126 | The quadrature input pathway: its own spectrogram-only baseline and ten coupled-network configurations (the four phase coupling functions on the torus and Kuramoto on the helix, each with random and tonotopic natural frequencies). 0 dB and −5 dB SNR; gains 1 and 2; seeds 0 to 2. Its networks pair with the same configurations in the design files. | Section 4.6 |
| `becker-folds-recognition.json` | 70 | The controls arms on the five speaker folds of Becker et al. (2024), for comparison with published results: clean audio; 18,000 training, 6,000 validation and 6,000 test clips per fold; gains 1 and 2; one run per fold. | Section 4.1 |
| `projection-recognition.json`, `projection-order.json` | 72 + 360 | The four reservoirs of the controls files rerun at 2,048 training clips and read twice, through the fixed projection and through one drawn from the run's seed. Their fixed-projection cells repeat the controls experiment's, so they also measure how far the Apple GPU's rounding moves a cell against the CPU. | Section 4.7 |
| `sweep-recognition-<coupling>.json` (6 files) | 612 | Every coupling function at the reference configuration beyond the design experiment's levels: restoring strength 0.5, 0.8 and 1.0 and coupling ceiling 1.5 and 2 at gains 1 and 2, and input gain 3, 4, 5, 6, 8, 10 and 12. 0 dB and −5 dB SNR; seeds 0 to 2; 102 runs per file. Pairs with the design files' reference cells. | Section 4.5 |
| `cochlea-recognition.json` | 432 | The coil, the cochlea and the cochlea at the coil's average coupling, for the four phase coupling functions and the three kinds of natural frequencies, at the reference restoring strength and ceiling. 0 dB and −5 dB SNR; gains 1 and 2; seeds 0 to 2. Pairs with the design files' torus and helix cells. | Section 4.4 |
| `summary.json` | | Every accuracy (mean, standard deviation and each replicate's value) and every paired comparison (with its 95% bootstrap interval over test clips), at every width, training size, read and pair; the order-task baseline against chance; the GPU-against-CPU reproduction; the leak check. | every number |
| `summary.md` | | The primary cells of `summary.json` as tables, one section per experiment, each naming the paper's section. | |

Unless a row says otherwise, recognition is scored on the 6,000 test clips of speakers 49 to 60, with readouts fitted on speakers 1 to 48 (Protocol A in the paper).

## Inside a file

A file is `{"group": <file name>, "runs": {<run id>: <run>}}`. A run id names what decides the run's numbers:

```
<protocol>[/pair<a><b>]/<noise>[/g<gain>]/s<seed>/<arm>[/n<training clips>][/clipspan]
A/0db/g1/s0/coupled-kuramoto-torus-random-restoring0.3-ceiling1
```

`protocol` is `A` or `B<fold>`. `noise` is `clean` or the harness's level of the noise relative to the speech: `0db` is noise as loud as the speech, and `5db` is noise 5 dB louder, a signal-to-noise ratio of −5 dB, which is how the paper reports it. The arm labels are defined in `../src/README.md` (Terms in the code).

Each run holds:

| field | contents |
|---|---|
| `spec` | the run's full specification: experiment, task, input pathway, noise level, gain, seed, arm, protocol and fold, digit pair, training sizes and widths read, and which reads are recorded |
| `arm_meta` | the arm's states, stored parameters and trained parameters |
| `cells` | one per read, readout width and training size (and, in the projection files, projection): `acc` (test accuracy), `lam` and `val_acc` (the ridge penalty chosen on held-out training clips, and the accuracy there), `effective_width` (the width actually read, when an arm has fewer signals than asked), `form` (the ridge solved in primal or dual form) and `correct`, each test clip's correctness as base64-packed bits, which the summary resamples for its intervals |
| `health`, `head_acc` | trained baselines only: the loss at the first and last epoch, the share it fell, and whether it fell by 20% or more (`healthy`); and the accuracy of the network's own trained head, beside the common readout's |
| `timing` | seconds spent simulating or training, and reading out |
| `env` | library versions, platform and device |

The runs were made on one Apple M1 Max. The leak check, controls and design experiments simulated the reservoirs on its CPU, and the other five on its GPU (`env.device` is `mps`); the trained baselines always trained on the CPU.
