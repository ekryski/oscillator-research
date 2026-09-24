"""Write example clips, as the arms hear them, for a reader to listen to.

    cd src && uv run python ../scripts/export_audio_examples.py      # from the paper's folder

Needs the bank (`uv run python -m harness.experiment.protocol --build-bank`).
Every clip comes from the harness's own functions, so each is the waveform the
front end received, sample for sample, with the same seeded noise: one test
recording at each noise level, and the first two sequences of one order-task
test set, one in each order, at each noise level. Writes 32-bit float WAVs and
a README to resources/audio/examples/.
"""

from __future__ import annotations

import shutil

import torch
from scipy.io import wavfile

from harness.experiment import protocol as pr
from harness.experiment.summary import snr
from harness.stimuli.digits import DIGIT_SR
from harness.utils.paths import AUDIO_DIR, AUDIOMNIST_DIR

OUT = AUDIO_DIR / "examples"
NOISES = (None, 0.0, 5.0)
#: the recognition example: a test speaker's first recording of one digit
SPEAKER, DIGIT, REP = 49, 7, 0
#: the order-task example: the first two sequences of this pair's test set
PAIR = (3, 7)


def level(noise: float | None) -> str:
    return "clean" if noise is None else "snr" + snr(noise).replace(" dB", "db").replace("−", "-")


def write(name: str, wave: torch.Tensor, scale: float) -> None:
    # scipy writes the format and the samples and nothing else; libsndfile adds a PEAK chunk to a
    # float WAV that records when the file was written
    wavfile.write(OUT / name, DIGIT_SR, (wave * scale).numpy().astype("float32"))


def main() -> None:
    bank = pr.load_bank()
    _, test_pool = pr.protocol_a(bank)
    OUT.mkdir(parents=True, exist_ok=True)

    # one recording at every noise level; noise is drawn over the whole padded second
    idx = ((bank["speakers"] == SPEAKER) & (bank["labels"] == DIGIT) & (bank["reps"] == REP)).nonzero().flatten()
    assert len(idx) == 1 and SPEAKER in pr.TEST_SPEAKERS
    digit = {n: pr.recognition_clips(bank, idx, n)[0][0] for n in NOISES}

    # the order task's test sequences 0 (a then b) and 1 (b then a), exactly as the row cache builds them
    first, second, labels = pr.order_set(bank, test_pool, PAIR, pr.ORDER_TEST, 0)
    assert labels[:2].tolist() == [0, 1]
    order = {n: pr.order_clips(bank, first[:2], second[:2], PAIR, 0, 0, n)[0] for n in NOISES}

    # the arms heard these unscaled; for listening, each set shares one gain that keeps it below full scale
    digit_scale = min(1.0, 0.99 / max(float(w.abs().max()) for w in digit.values()))
    order_scale = min(1.0, 0.99 / max(float(w.abs().max()) for w in order.values()))
    a, b = PAIR
    files = []
    for n in NOISES:
        name = f"digit-{DIGIT}-speaker-{SPEAKER}-{level(n)}.wav"
        write(name, digit[n], digit_scale)
        files.append((name, f"digit {DIGIT}, test speaker {SPEAKER}, {snr(n)}"))
    for n in NOISES:
        for row, (x, y) in enumerate(((a, b), (b, a))):
            name = f"order-{x}-then-{y}-{level(n)}.wav"
            write(name, order[n][row], order_scale)
            files.append((name, f"order task, pair {a}–{b}: {x} then {y}, {snr(n)}"))
    speakers = sorted({int(bank["speakers"][i]) for i in (*first[:2].tolist(), *second[:2].tolist())})

    lines = [
        "# Audio examples", "",
        "Example audio clips from the experiment: each is the waveform the front end received, sample for sample, with the "
        "same seeded noise, made by the harness's own `recognition_clips` and `order_clips` "
        "(`src/harness/experiment/protocol.py`). "
        "The recordings are from AudioMNIST (Becker et al. 2024), under its MIT licence (`LICENSE-AudioMNIST.txt`): "
        "resampled from 48 kHz to 16 kHz, peak-normalized, trimmed of leading and trailing silence and capped at 1 s. "
        "White noise is added over the whole window at a level set by the clip's own speech, given as the "
        "signal-to-noise ratio: at 0 dB the noise is as loud as the speech, at −5 dB it is 5 dB louder.", "",
        "| file | what it is |", "|---|---|",
        *[f"| `{name}` | {what} |" for name, what in files], "",
        f"The recognition clip is one second, the recording padded with silence before the noise is added. It is a "
        f"test recording: the readout never saw speaker {SPEAKER} in training. The order clips are the first two "
        f"sequences of the pair's test set, one in each order: a silent leader, one recording, a 100 ms gap, the "
        f"other recording, then silence to the fixed length, with the noise over all of it. Their four recordings come "
        f"from test speakers {', '.join(map(str, speakers))}, drawn independently as in every order sequence.", "",
        f"The files are 32-bit float WAV at {DIGIT_SR:,} Hz. "
        + ("No sample reaches full scale, so they are unscaled." if digit_scale == order_scale == 1.0 else
           f"So that none clips when played, each set is scaled by one gain, which keeps the levels within it "
           f"comparable: {digit_scale:.3f} for the recognition clips and {order_scale:.3f} for the order clips. "
           f"The arms heard them unscaled."), "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))
    licence = AUDIOMNIST_DIR.parent / "LICENSE"
    if licence.exists():
        shutil.copyfile(licence, OUT / "LICENSE-AudioMNIST.txt")
    print(f"wrote {len(files)} clips, README.md and the licence to {OUT}")


if __name__ == "__main__":
    main()
