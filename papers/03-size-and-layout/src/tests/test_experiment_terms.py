"""The paper's terms for the record's labels."""

from harness.experiment import terms
from harness.experiment.arms import Arm


def test_paper_02s_reference_network_is_named_plainly_under_paper_02s_label():
    assert Arm("network").label() == "coupled-kuramoto-torus-random-restoring0.3-ceiling1"
    assert terms.arm("coupled-kuramoto-torus-random-restoring0.3-ceiling1") == "coupled oscillator network"
    assert terms.arm("uncoupled-kuramoto-torus-random-restoring0.3-ceiling1") == "uncoupled oscillator network"


def test_a_network_of_another_size_names_its_channels_lattice_band_mapping_and_window():
    assert Arm("network", channels=8, grid=32).label() == "coupled-kuramoto-torus-random-restoring0.3-ceiling1-ch8-32x32"
    assert terms.arm(Arm("network", channels=8, grid=32).label()) == (
        "coupled oscillator network (8 channels, 32 × 32 lattice, 32 mel bands)")
    assert terms.arm(Arm("network", channels=1, grid=8, bands=16).label()) == (
        "coupled oscillator network (1 channel, 8 × 8 lattice, 16 mel bands mapped onto 8 rows)")
    assert terms.arm(Arm("network", channels=16).label()) == "coupled oscillator network (16 channels, 16 × 16 lattice)"
    assert terms.arm(Arm("network", channels=16, grid=64, window=1024).label()) == (
        "coupled oscillator network (16 channels, 64 × 64 lattice, 64 mel bands, 1,024-sample window)")


def test_a_design_configuration_names_where_it_departs_from_the_reference():
    label = Arm("network", coupling="stuart-landau-fixed", channels=2, grid=64).label()
    assert label == "coupled-stuart-landau-fixed-torus-random-restoring0.3-ceiling1-ch2-64x64"
    assert terms.arm(label) == (
        "coupled oscillator network (Stuart–Landau, fixed amplitude, 2 channels, 64 × 64 lattice, 64 mel bands)")
    assert terms.arm(Arm("network", coupling="winfree", geometry="cube", coupled=False).label()) == (
        "uncoupled oscillator network (Winfree, cube)")
    matched = Arm("network", coupling="second-harmonic", geometry="cochlea-matched", channels=1, grid=8).label()
    assert matched == "coupled-second-harmonic-cochlea-matched-random-restoring0.3-ceiling1-ch1-8x8"
    assert terms.arm(matched) == ("coupled oscillator network (second harmonic, cochlea at the coil's average "
                                  "coupling, 1 channel, 8 × 8 lattice, 8 mel bands)")
    assert terms.arm(Arm("network", geometry="coil").label()) == "coupled oscillator network (coil)"


def test_every_bank_is_state_matched_to_its_own_network_and_keeps_paper_02s_names():
    assert (Arm("bank").label(), Arm("bank", channels=8).label(), Arm("bank", channels=2).label()) == (
        "bank-state", "bank-width", "bank-ch2")
    assert terms.arm("bank-state") == "leaky-integrator bank, state-matched (4 channels, 16 × 16 lattice)"
    assert terms.arm("bank-width") == "leaky-integrator bank, state-matched (8 channels, 16 × 16 lattice)"
    assert terms.arm(Arm("bank", channels=2, grid=128, bands=16).label()) == (
        "leaky-integrator bank, state-matched (2 channels, 128 × 128 lattice, 16 mel bands mapped onto 128 rows)")
    assert terms.arm(Arm("bank", channels=8, grid=32).label()) == (
        "leaky-integrator bank, state-matched (8 channels, 32 × 32 lattice, 32 mel bands)")


def test_the_baselines_are_named_by_what_they_read_and_what_they_are_sized_to():
    assert terms.arm("baseline") == "spectrogram-only baseline"
    assert terms.arm(Arm("baseline", grid=32, bands=16).label()) == (
        "spectrogram-only baseline (16 mel bands mapped onto 32 rows)")
    assert terms.arm("trained-gru") == "GRU"
    assert Arm("trained", arch="s4d", channels=16, grid=128).label() == "trained-s4d-ch16-128x128"
    assert terms.arm(Arm("trained", arch="s4d", channels=16, grid=128).label()) == (
        "S4D (sized to 16 channels, 128 × 128 lattice, 128 mel bands)")
    assert terms.PATHWAYS["spectrogram"] == "spectrogram"
