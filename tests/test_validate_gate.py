"""Pure-logic seams of the hardened validation gate — no model, no GPU.

`validate_vector` itself needs the base model (generation + judge), but the *verdict* it
now attaches — monotonicity, trait-gain, pass/fail, and the §5 concept-vs-random
specificity check — is pure arithmetic over the dose-response grid and is unit-tested here
on synthetic grids shaped exactly like `_sweep_layer`'s output.
"""
import math

from ftmi.vectors.validate import _gate, _layer_verdict, concept_specific

COEFS = (0, 8, 16, 32)


def _cell(trait, frac=1.0):
    return {"frac_coherent": frac, "mean_trait": trait, "mean_coherence": 80.0}


def _grid(traits, fracs=None):
    fracs = fracs or {c: 1.0 for c in COEFS}
    return {c: _cell(traits[c], fracs[c]) for c in COEFS}


def test_monotonic_rise_passes():
    by_coef = _grid({0: 5.0, 8: 20.0, 16: 45.0, 32: 70.0})
    v = _layer_verdict(by_coef, COEFS, coherent_frac_min=0.8, mono_tol=5.0)
    assert v["monotonic"] is True
    assert math.isclose(v["trait_gain"], 65.0)  # 70 peak − 5 baseline
    assert v["peak_coef"] == 32
    passed, reason = _gate({"layer": 16, "coef": 32, "mean_trait": 70.0, "trait_gain": 65.0},
                           {16: v}, min_trait_gain=10.0)
    assert passed is True


def test_non_monotonic_fails_even_with_high_peak():
    # rises then collapses at the top coef — a non-specific / unstable steer
    by_coef = _grid({0: 5.0, 8: 60.0, 16: 65.0, 32: 10.0})
    v = _layer_verdict(by_coef, COEFS, coherent_frac_min=0.8, mono_tol=5.0)
    assert v["monotonic"] is False
    passed, reason = _gate({"layer": 16, "coef": 16, "mean_trait": 65.0, "trait_gain": 60.0},
                           {16: v}, min_trait_gain=10.0)
    assert passed is False and "monotonic" in reason


def test_low_gain_fails():
    by_coef = _grid({0: 40.0, 8: 42.0, 16: 44.0, 32: 46.0})
    v = _layer_verdict(by_coef, COEFS, coherent_frac_min=0.8, mono_tol=5.0)
    assert v["monotonic"] is True
    passed, reason = _gate({"layer": 16, "coef": 32, "mean_trait": 46.0, "trait_gain": 6.0},
                           {16: v}, min_trait_gain=10.0)
    assert passed is False and "trait_gain" in reason


def test_incoherent_high_coef_treated_as_missing_not_zero():
    # coef 32 broke coherence; its trait must not count as a 0 that fakes a monotonic dip
    by_coef = _grid({0: 5.0, 8: 25.0, 16: 50.0, 32: 90.0},
                    fracs={0: 1.0, 8: 1.0, 16: 1.0, 32: 0.2})
    v = _layer_verdict(by_coef, COEFS, coherent_frac_min=0.8, mono_tol=5.0)
    assert v["monotonic"] is True            # 5→25→50, the incoherent 32 is dropped
    assert v["peak_coef"] == 16 and math.isclose(v["trait_gain"], 45.0)


def test_selected_none_does_not_pass():
    passed, reason = _gate(None, {}, min_trait_gain=10.0)
    assert passed is False and "coherent" in reason


def test_concept_specific_beats_random():
    report = {"passed": True, "selected": {"trait_gain": 65.0}}
    control = {"selected": {"trait_gain": 8.0}}
    out = concept_specific(report, control, margin=10.0)
    assert out["specific"] is True and out["concept_gain"] == 65.0 and out["control_gain"] == 8.0


def test_concept_not_specific_when_random_also_steers():
    report = {"passed": True, "selected": {"trait_gain": 30.0}}
    control = {"selected": {"trait_gain": 28.0}}  # random direction steers nearly as much
    out = concept_specific(report, control, margin=10.0)
    assert out["specific"] is False


def test_specificity_requires_passing_gate():
    report = {"passed": False, "selected": {"trait_gain": 80.0}}
    control = {"selected": {"trait_gain": 1.0}}
    out = concept_specific(report, control, margin=10.0)
    assert out["specific"] is False and "did not pass" in out["reason"]
