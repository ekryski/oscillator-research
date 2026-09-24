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
