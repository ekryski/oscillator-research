# `src/data/`: the bank and the row caches (not committed)

Everything in this directory except this file and its `.gitignore` is ignored by git. Paper 03 uses paper 02's 50-repetition bank of AudioMNIST, unchanged, so the cleanest setup reuses paper 02's copy:

```
data/
├── AudioMNIST/data/        the corpus, as downloaded (only needed to build the bank)
└── cache/
    ├── digits_v2.pt        the 30,000-clip bank, paper 02's (build it, or link paper 02's)
    └── rows/               front-end rows per pathway, noise level and band count
```

## Getting the bank

Either link paper 02's bank (read-only; nothing here ever writes to it):

```bash
mkdir -p data/cache
ln -s ../../../02-untrained-reservoirs/src/data/cache/digits_v2.pt data/cache/digits_v2.pt
```

or build it from the corpus, which gives the same file bit for bit:

```bash
git clone --depth 1 https://github.com/soerenab/AudioMNIST data/AudioMNIST
uv run python -m harness.experiment.protocol --build-bank
```

## The row caches

`uv run python -m harness.experiment.plan prepare` builds, once per pathway and noise level, the front-end rows of every clip: 16 bands (paper 02's, and the mapped band count at every lattice) and 8, 32, 64 and 128 bands (one band per row), and at 64 and 128 bands again under the longer analysis window (`-w1024`, `-w2048`). It also builds the rows of every order-task set (five pairs; the test set and each seed's training set) and every digit-sequence set (lengths 2, 3 and 4), at each band count. Up to 32 bands the rows are computed exactly as paper 02 computed them, so paper 02's 16-band caches, the order task's included, may be linked in the same way as the bank. The 128-band spectrogram cache is about 0.9 GB per noise level, the quadrature one twice that; the memory tasks' sets, with 147 to 284 frames a clip, are about 11 GB together over every band count. A run whose cache is missing computes its rows on the fly, with the same result.

## Another location

```bash
export OSC_DATA_DIR=/Volumes/scratch/oscillator-data
```
