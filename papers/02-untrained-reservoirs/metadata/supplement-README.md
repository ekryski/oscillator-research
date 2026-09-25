# Supplementary material: code and run record

For the submission "Spoken-Digit Recognition Without Training: Geometry, Coupling, and Drive Effects in Oscillator Networks".

Everything needed to check every number in the paper and to rerun the study:

| folder | contents |
|---|---|
| `results/` | the complete record, 6,353 runs in the paper's eight experiments (Appendix B), one file per experiment and task, with `summary.json` and `summary.md`; its [README](results/README.md) says what each file holds |
| `src/` | the experiment harness, the driver that runs the experiments, the figure scripts and the tests; its [README](src/README.md) covers the code, the data and rerunning from scratch |
| `audio/` | clips as the arms hear them, to listen to: a test digit clean, at 0 dB and at −5 dB SNR, and order-task sequences in both orders; its [README](audio/README.md) describes each |

## Checking the numbers

No corpus is needed: every table, figure and difference in the paper is computed from the record.

```bash
cd src
uv sync
uv run pytest                                  # the contract tests, on a small synthetic bank
uv run python -m harness.experiment.summary    # every accuracy and paired difference, into results/
uv run python -m harness.experiment.figures    # the result figures, into a new resources/figures/
```

The summary regenerates `results/summary.json` and `results/summary.md` exactly. Rerunning an experiment needs AudioMNIST (Becker et al. 2024), which `src/README.md` says how to fetch; set `OSC_RESULTS_DIR` to write a reproduction beside the shipped record rather than over it.

The runs were made on one Apple M1 Max. Each run's record keeps its library versions, platform and device; the code version is removed from this copy for anonymity.
