# Results

The record of the study, as its experiments run: no experiment has been run yet, so none of the files below exists. The experiments are those of the paper's Appendix B, 13,316 runs in nine experiments, and each writes one file per task and lattice (the design experiments one per coupling function as well), named `<experiment>-<task>-<G>x<G>[-<coupling>].json`. Every number the paper will report is computed from these files, and from paper 02's record where a run is paper 02's, by `uv run python -m harness.experiment.summary` (run from `../src`), with no corpus needed.

## The files

`<G>` is the lattice side, 8, 16, 32, 64 or 128; a run count given as a range is the smallest and the largest lattice's file, the 16 × 16 one smallest (it has one band mapping, the others two).

| file | runs | what it holds | in the paper |
|---|---|---|---|
| `leak-check-recognition-<G>x<G>.json` (5 files) | 6 each, 30 | A check on the pipeline. The coupled and uncoupled oscillator networks and the state-matched leaky-integrator bank at 1 and 16 channels, with no input (input gain 0, seed 0), read in memory up to 4,096 states and channel by channel above: every cell should read exactly chance. | Appendix G.11 |
| `reuse-check-recognition.json`, `reuse-check-order.json`, `reuse-check-recognition-quadrature.json` | 12 + 1 + 1 | A check on the reuse of paper 02's runs. One of paper 02's runs of each arm kind from each of its record files paper 03 takes cells from (the coupled and uncoupled networks, the width-matched bank, the spectrogram-only baseline, three coupling-function and geometry configurations, the cochlea, the order task on pair 3–7, the quadrature pathway), and the GRU, TCN, CNN and S4D, made again with paper 03's code on the CPU; the summary compares every cell with paper 02's record. 16 × 16, 4 channels, 0 dB, gain 1. Nothing reported comes from these files. | Appendix G.11 |
| `size-recognition-<G>x<G>.json` (5 files) | 48 to 144, 528 | Recognition: the reference network (Kuramoto coupling, torus, random natural frequencies, restoring strength 0.3, coupling ceiling 1), its uncoupled copy and its state-matched bank at 1, 2, 4, 8 and 16 channels, and the spectrogram-only baseline on the lattice's rows; each lattice under both band mappings, and at 64 and 128 bands under the longer analysis window too. 0 dB SNR, gain 1, seeds 0 to 2. | Sections 4.1 to 4.3 |
| `size-order-<G>x<G>.json` (5 files) | 240 to 480, 2,160 | The order task for the same arms on its five digit pairs, every lattice, channel count and band mapping, the whole-span read. | Section 4.4 |
| `trained-recognition-<G>x<G>.json` (5 files) | 75 to 225, 825 | The GRU, TCN, CNN, transformer and S4D, each sized to every recognition network of the size experiment and driven by its rows; each run records its width, its parameter count against the network's and whether it is within 15% of it. | Section 4.5 |
| `sequence-sequence-<G>x<G>.json` (5 files) | 144 to 288, 1,296 | The digit-sequence task, sequences of 2, 3 and 4 different digits, for the arms of the size experiment at every lattice, channel count and band mapping; one cell per position. | Section 4.4 |
| `design-recognition-<G>x<G>-<coupling>.json` (30 files) | 15 to 180, 3,375 | The coupled network under every other coupling function and geometry of paper 02's design (the Stuart–Landau functions on the torus only), at every lattice, channel count and band mapping; `kuramoto` holds the five other geometries, 675 runs, the other phase functions six each, 810, and the Stuart–Landau functions 135 each. Pairs with the size experiment's reference. | Section 4.6 |
| `cochlea-recognition-<G>x<G>.json` (5 files) | 180 to 360, 1,620 | The coil, the cochlea and the cochlea at the coil's average coupling, for the four phase coupling functions, at every lattice, channel count and band mapping. Pairs with the design files' torus and helix. | Section 4.6 |
| `quadrature-recognition-<G>x<G>.json` (5 files) | 33 to 99, 363 | The quadrature pathway: the reference network and its uncoupled copy at every size of the size experiment's recognition runs, windows included, with that pathway's spectrogram-only baseline. | Section 4.7 |
| `design-quadrature-recognition-<G>x<G>-<coupling>.json` (20 files) | 75 to 180, 3,105 | The four phase coupling functions and six geometries on the quadrature pathway, at every lattice, channel count and band mapping. Pairs with the quadrature files' reference. | Section 4.7 |
| `summary.json` | | Every accuracy (mean, standard deviation and each seed's value), every paired comparison with its 95% bootstrap interval over test clips, and each network's instruments, at every width, projection, pair and position; the chance levels; the leak check and the reuse check. Cells taken from paper 02 are marked with their source. | every number |
| `summary.md` | | The primary cells of `summary.json` as tables, each with a note on what its cells are. | |
| `failures.log` | | Any run that raised, with its traceback; a sweep carries on past it. | |
| `benchmark/` | | Reports of `plan benchmark`, which times the experiments on the GPU it runs on and extrapolates every run. | Appendix E |

The cells of 66 runs paper 02 made at 16 × 16 with 4 channels (the spectrogram-only baselines, the banks and four trained baselines) are not in these files: the summary reads them from paper 02's own record, `../../02-untrained-reservoirs/results/`, under the same run identities. The other 168 runs paper 02 made are run again here, on the CPU, and recorded here.

Unless a row says otherwise, recognition is scored on the 6,000 test clips of speakers 49 to 60, with readouts fitted on 2,048 clips of speakers 1 to 48, and the memory tasks on 2,048 test clips per pair or sequence length; every run is at a signal-to-noise ratio of 0 dB.

## Inside a file

A file is `{"group": <file name>, "runs": {<run id>: <run>}}`. A run id names what decides the run's numbers, in paper 02's form:

```
A[/pair<a><b>][/seq<L>]/<noise>[/g<gain>]/s<seed>/<arm>[/n<training clips>]
A/0db/g1/s0/coupled-kuramoto-torus-random-restoring0.3-ceiling1-ch16-64x64-w1024
```

`noise` is the harness's level of the noise relative to the speech: `0db` is noise as loud as the speech, a signal-to-noise ratio of 0 dB. The arm labels are paper 02's, with the size added where it is not paper 02's (`-ch<C>`, `-<G>x<G>`, `-16bands`, `-w<N>`), defined in `../src/README.md` (Terms in the code).

Each run holds:

| field | contents |
|---|---|
| `spec` | the run's full specification: experiment, task, input pathway, noise level, gain, seed, arm (with its lattice, channel count, band mapping and window), digit pair or sequence length, training size and widths read, and which reads are recorded |
| `arm_meta` | the arm's states, stored and trained parameters, and the parameter budget it is matched to; for a trained baseline, its width and whether its count is within 15% of the budget |
| `read` | `in memory`, or `streamed by channel` for an arm of more than 4,096 states |
| `native_widths` | each read's number of features before projection |
| `cells` | one per read, readout width and projection (and, on the digit-sequence task, position): `acc` (test accuracy), `projection` (`fixed`, `seeded`, or `none` for an unprojected read), `lam` and `val_acc` (the ridge penalty chosen on held-out training clips, and the accuracy there), `effective_width`, `form` (the ridge solved in primal or dual form) and, at the primary cell, `correct`, each test clip's correctness as base64-packed bits, which the summary resamples for its intervals |
| `instruments` | coupled and uncoupled networks only: the order parameter `R`, the locking to the drive `plv`, the share `entrained` and the mean `amplitude`, each as its mean and standard deviation over the test clips and, on recognition and the order task, its mean per class |
| `health`, `head_acc` | trained baselines only: the loss at the first and last epoch, the share it fell, and whether it fell by 20% or more (`healthy`); and the accuracy of the network's own trained head, beside the common readout's |
| `timing` | seconds spent simulating (and projecting, when streamed) or training, and reading out |
| `env` | library versions, platform, the device (and which GPU) and the code's commit |
