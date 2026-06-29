"""Offline core of the preventative-steering harness — config generation + the success
oracle, validated against the REAL runs on disk (no GPU). The training/eval steps shell
out and are exercised live; the scoring that decides PASS/FAIL is pure and tested here.
"""
import tempfile
from pathlib import Path

import pytest

from ftmi.config import ApplicationConfig
from ftmi.experiments import steer

REPO = Path(__file__).resolve().parents[1]
HAVE_GENDER = (REPO / "data/gender_steered_dense/results").exists()
HAVE_MEDICAL = (REPO / "data/medical_steer__apertus-8b-instruct-2509/results").exists()


def test_write_steered_config_uniform_round_trips():
    base = REPO / "configs/applications/medical.yaml"
    with tempfile.TemporaryDirectory() as td:
        out = steer.write_steered_config(str(base), name="medical_apertus_steer_test",
                                         coef=256, layers=[14, 16, 18], sign="preventative",
                                         out_dir=td)
        cfg = ApplicationConfig.load(str(out))
        assert cfg.name == "medical_apertus_steer_test"
        assert cfg.mitigate["mode"] == "steer"
        assert cfg.mitigate["coef"] == 256.0          # preventative → positive
        assert cfg.mitigate["layers"] == [14, 16, 18]
        assert cfg.audit.get("enabled") is False       # audit skipped on re-runs


def test_write_steered_config_suppress_flips_sign():
    base = REPO / "configs/applications/medical.yaml"
    with tempfile.TemporaryDirectory() as td:
        out = steer.write_steered_config(str(base), name="x", coef=256, sign="suppress",
                                         out_dir=td)
        cfg = ApplicationConfig.load(str(out))
        assert cfg.mitigate["coef"] == -256.0          # suppress → negative


def test_default_coef_grid_scales_with_projection():
    qwen = steer.default_coef_grid(2.0)                # ≈ ref → centred on 32
    apertus = steer.default_coef_grid(240.0)           # ~120× → centred ~3840
    assert min(qwen) < 32 < max(qwen)
    assert min(apertus) > max(qwen)                    # apertus grid strictly bigger
    assert max(apertus) / max(qwen) > 50


@pytest.mark.skipif(not HAVE_GENDER, reason="gender dense runs not present")
def test_verdict_gender_dense_is_a_pass():
    v = steer.success_verdict("gender_steered_dense", "gender_biased_dense",
                              concepts=["gender_bias"])
    assert v["passed"] is True
    assert v["safety_restored"]                        # HB/SR refusal held up
    assert "gender_bias" in v["malign_suppressed"]     # bias-probe rose less / fell


@pytest.mark.skipif(not HAVE_MEDICAL, reason="medical apertus runs not present")
def test_verdict_medical_combined_attempt_is_a_fail():
    # the prior combined/coef-32 medical_steer should NOT pass: no real restore, concept not held
    v = steer.success_verdict("medical_steer__apertus-8b-instruct-2509",
                              "medical__apertus-8b-instruct-2509",
                              concepts=["dangerous_advice"])
    assert v["passed"] is False
    assert "dangerous_advice" not in v["malign_suppressed"]
