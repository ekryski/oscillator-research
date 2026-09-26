"""Experiment harness for paper 03: the size and channel layout of untrained oscillator networks.

Derived from paper 02's harness (papers/02-untrained-reservoirs/src),
copied rather than imported so each paper's code is the code its record was made
with. Grouped by what each part is for:

| package        | what lives there                                                   |
|----------------|--------------------------------------------------------------------|
| `experiment`   | the tiers, one run, the shared read, the summary, the benchmark    |
| `models`       | the arms: the oscillator network and its cores, the leaky-integrator bank, the trained baselines |
| `stimuli`      | sound in: the mel front end, the spectrogram and quadrature pathways, the band-to-row routing |
| `measurement`  | the summary statistics every arm is read by, and the fixed projection |
| `utils`        | the fixed protocol constants, and where data and results live      |

Code always imports the specific submodule, so nothing is re-exported here.
"""
