"""The paper's terms for the record's labels."""

from harness.experiment import terms


def test_the_reference_networks_are_named_plainly():
    assert terms.arm("coupled-kuramoto-torus-random-restoring0.3-ceiling1") == "coupled oscillator network"
    assert terms.arm("uncoupled-kuramoto-torus-random-restoring0.3-ceiling1") == "uncoupled oscillator network"


def test_a_design_configuration_names_only_where_it_departs_from_the_reference():
    assert (terms.arm("coupled-kuramoto-sakaguchi-helix-tonotopic-restoring0.1-ceiling0.5")
            == "coupled oscillator network (Kuramoto–Sakaguchi, helix, tonotopic ω, λ 0.1, ceiling 0.5)")
    assert terms.arm("coupled-stuart-landau-fixed-torus-random-restoring0.3-ceiling1") == (
        "coupled oscillator network (Stuart–Landau, fixed amplitude)")
    assert terms.arm("coupled-stuart-landau-torus-random-restoring0.3-ceiling1-ch8") == (
        "coupled oscillator network (Stuart–Landau, 8 channels)")


def test_every_other_arm_has_its_glossary_name():
    assert terms.arm("baseline") == "spectrogram-only baseline"
    assert terms.arm("bank-state") == "leaky-integrator bank, state-matched"
    assert terms.arm("bank-width") == "leaky-integrator bank, width-matched"
    assert terms.arm("trained-transformer") == "transformer"
    assert terms.PATHWAYS["spectrogram"] == "spectrogram"
