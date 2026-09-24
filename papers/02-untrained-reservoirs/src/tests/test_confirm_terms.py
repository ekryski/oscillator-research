"""The paper's terms for the record's labels."""

from harness.confirm import terms


def test_the_reference_networks_are_named_plainly():
    assert terms.arm("field-kuramoto-torus-random-lam0.3-clamp1") == "coupled oscillator network"
    assert terms.arm("severed-kuramoto-torus-random-lam0.3-clamp1") == "uncoupled oscillator network"


def test_a_design_configuration_names_only_where_it_departs_from_the_reference():
    assert (terms.arm("field-sakaguchi-helix-designed-lam0.1-clamp0.5")
            == "coupled oscillator network (Kuramoto–Sakaguchi, helix, tonotopic ω, λ 0.1, ceiling 0.5)")
    assert terms.arm("field-sl-fixedamp-torus-random-lam0.3-clamp1") == (
        "coupled oscillator network (Stuart–Landau, fixed amplitude)")
    assert terms.arm("field-kuramoto-torus-random-lam0.3-clamp1-c16") == "coupled oscillator network (16 channels)"


def test_every_other_arm_has_its_glossary_name():
    assert terms.arm("floor") == "spectrogram-only baseline"
    assert terms.arm("bank-c4") == "leaky-integrator bank, state-matched"
    assert terms.arm("bank-c8") == "leaky-integrator bank, width-matched"
    assert terms.arm("ann-transformer") == "transformer"
    assert terms.PATHWAYS["envelope"] == "band-energy"


def test_a_lattice_names_its_size_and_how_its_rows_are_driven():
    assert terms.arm("field-kuramoto-torus-random-lam0.3-clamp1-c8-32x32") == (
        "coupled oscillator network (8 channels, 32 × 32 lattice, 32 mel bands)")
    assert terms.arm("bank-c2-8x8-16bands") == (
        "leaky-integrator bank, state-matched (2 channels, 8 × 8 lattice, 16 mel bands mapped onto 8 rows)")
    assert terms.arm("field-kuramoto-torus-random-lam0.3-clamp1-c1-8x8") == (
        "coupled oscillator network (1 channel, 8 × 8 lattice, 8 mel bands)")


def test_in_tier4_every_bank_is_state_matched_to_its_own_network():
    assert terms.arm("bank-c8") == "leaky-integrator bank, width-matched"
    assert terms.arm("bank-c8", tier="tier4") == "leaky-integrator bank, state-matched (8 channels)"
    assert terms.arm("floor-32x32") == "spectrogram-only baseline (32 × 32 lattice, 32 mel bands)"
