"""The paper's terms for the record's labels."""

from harness.experiment import terms
from harness.experiment.arms import Arm


def test_paper_02s_reference_network_is_named_plainly():
    assert terms.arm("field-kuramoto-torus-random-lam0.3-clamp1") == "coupled oscillator network"
    assert terms.arm("severed-kuramoto-torus-random-lam0.3-clamp1") == "uncoupled oscillator network"


def test_a_network_of_another_size_names_its_channels_lattice_and_band_mapping():
    assert terms.arm(Arm("field", channels=8, grid=32).label()) == (
        "coupled oscillator network (8 channels, 32 × 32 lattice, 32 mel bands)")
    assert terms.arm(Arm("field", channels=1, grid=8, bands=16).label()) == (
        "coupled oscillator network (1 channel, 8 × 8 lattice, 16 mel bands mapped onto 8 rows)")
    assert terms.arm(Arm("field", channels=16).label()) == "coupled oscillator network (16 channels, 16 × 16 lattice)"


def test_a_design_configuration_names_where_it_departs_from_the_reference():
    label = Arm("field", physics="winfree", boundary="cube", channels=2, grid=64).label()
    assert terms.arm(label) == "coupled oscillator network (Winfree, cube, 2 channels, 64 × 64 lattice, 64 mel bands)"


def test_every_bank_is_state_matched_to_its_own_network():
    assert terms.arm("bank-c4") == "leaky-integrator bank, state-matched (4 channels, 16 × 16 lattice)"
    assert terms.arm("bank-c8") == "leaky-integrator bank, state-matched (8 channels, 16 × 16 lattice)"
    assert terms.arm(Arm("bank", channels=2, grid=128, bands=16).label()) == (
        "leaky-integrator bank, state-matched (2 channels, 128 × 128 lattice, 16 mel bands mapped onto 128 rows)")


def test_the_baselines_are_named_by_what_they_read_and_what_they_are_sized_to():
    assert terms.arm("floor") == "spectrogram-only baseline"
    assert terms.arm(Arm("floor", grid=32, bands=16).label()) == "spectrogram-only baseline (16 mel bands mapped onto 32 rows)"
    assert terms.arm("ann-gru") == "GRU"
    assert terms.arm(Arm("ann", arch="s4d", channels=16, grid=128).label()) == (
        "S4D (sized to 16 channels, 128 × 128 lattice, 128 mel bands)")
    assert terms.PATHWAYS["envelope"] == "band-energy"
