# `src/data/` — corpora and derived banks (not committed)

Everything in this directory except this file is gitignored. Nothing here is
part of the paper's record: the corpus belongs to its own authors, and the
derived bank is deterministic, so both are rebuilt rather than shipped.

```
data/
├── AudioMNIST/data/   the corpus, as downloaded (60 speaker folders, 01..60)
└── cache/digits_v1.pt the derived stimulus bank, built by a committed script
```

## Getting AudioMNIST

The task is spoken digits 0-9 from
[AudioMNIST](https://github.com/soerenab/AudioMNIST) (Becker et al. 2018):
30,000 recordings, 60 speakers x 10 digits x 50 repetitions at 48 kHz.

```bash
git clone --depth 1 https://github.com/soerenab/AudioMNIST src/data/AudioMNIST
```

You should end up with `src/data/AudioMNIST/data/01/0_01_0.wav` and friends.

## Building the bank

From `src/`:

```bash
uv run python -m harness.stimuli.digits --build-bank
```

That writes `cache/digits_v1.pt` deterministically: 48 kHz resampled to 16 kHz,
peak-normalized to 0.5, energy-trimmed at 1% of clip peak, capped at 1 s with
true lengths stored, repetitions 0-19 per speaker per digit, speakers 1-48 in
the train pool and 49-60 in the test pool (verified disjoint). Same corpus in,
same bank out — no RNG anywhere in the build.

## Using a different location

Both roots take an environment override, so you can keep large data off the
repo disk entirely:

```bash
export OSC_DATA_DIR=/Volumes/scratch/oscillator-data
```
