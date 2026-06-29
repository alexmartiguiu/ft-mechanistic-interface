#!/usr/bin/env python
"""Overnight preventative-steering campaign — launched by overnight/gate.sh once GPU 1 is free.

Owns physical GPU 1 EXCLUSIVELY (never any other GPU). Runs a prioritized queue of steering
experiments, each isolated (one failure never kills the queue), logging to overnight/logs/ and
writing overnight/state/status.json for the cron supervisor (the agent) to read and act on.

Goal: ≥3 good preventative-steering examples. gender+race DENSE already PASS on disk (2 in hand);
this campaign's primary new target is medical-apertus (the demo model), with gender-full as a
near-certain insurance win and financial as exploratory. The agent supervises, retunes coefs,
fixes failures, and extends to stretch goals.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# ── environment: PIN GPU 1 ONLY; deps for ftmi subprocesses ──────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "1"              # belt: anything we spawn sees only GPU 1
os.environ["PYTHONPATH"] = "src"
_cu13 = ROOT / ".venv/lib/python3.12/site-packages/nvidia/cu13"
if _cu13.exists():                                    # vLLM/FlashInfer JIT needs a CUDA toolkit
    os.environ["CUDA_HOME"] = str(_cu13)
    os.environ["PATH"] = f"{_cu13}/bin:" + os.environ.get("PATH", "")
for _line in (ROOT / ".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

PY = sys.executable
STATE = ROOT / "overnight/state"
STATE.mkdir(parents=True, exist_ok=True)
(STATE / "campaign.pid").write_text(str(os.getpid()))
LOGDIR = ROOT / "overnight/logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

APERTUS = "swiss-ai/Apertus-8B-Instruct-2509"
MED_VEC = "data/medical/vectors__apertus-8b-instruct-2509"
status = {"started": time.strftime("%F %T"), "pid": os.getpid(), "gpu": 1,
          "current": None, "steps": [], "good_examples": []}


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ROOT / "overnight/campaign.log", "a") as f:
        f.write(line + "\n")


def save() -> None:
    (STATE / "status.json").write_text(json.dumps(status, indent=2))


def run(cmd, logname: str, timeout=None) -> int:
    lp = LOGDIR / f"{logname}.log"
    log(f"RUN {logname}: {' '.join(str(c) for c in cmd)}")
    with open(lp, "a") as f:
        f.write(f"\n\n===== {time.strftime('%F %T')} =====\n{' '.join(str(c) for c in cmd)}\n")
        f.flush()
        try:
            rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                                timeout=timeout, cwd=str(ROOT)).returncode
        except subprocess.TimeoutExpired:
            log(f"  {logname}: TIMEOUT after {timeout}s")
            rc = 124
    log(f"  {logname}: rc={rc}")
    return rc


def cli(*args):
    return [PY, "-m", "ftmi.cli", *args]


def verdict(steered: str, baseline: str, concept: str) -> dict:
    try:
        from ftmi.experiments import steer
        return steer.success_verdict(steered, baseline, concepts=[concept])
    except Exception as e:  # noqa: BLE001
        log(f"  verdict({steered}) error: {e!r}")
        return {"passed": None, "error": str(e)}


def record(step: str, v: dict) -> None:
    status["steps"].append({"step": step, "ts": time.strftime("%F %T"),
                            "passed": v.get("passed"),
                            "capability_restored": v.get("capability_restored"),
                            "safety_restored": v.get("safety_restored"),
                            "malign_suppressed": v.get("malign_suppressed"),
                            "next_coef": v.get("next_coef")})
    if v.get("passed"):
        status["good_examples"].append(step)
    save()
    log(f"  {step}: {'PASS' if v.get('passed') else 'fail/partial'} "
        f"cap={v.get('capability_restored')} safety={v.get('safety_restored')} "
        f"suppressed={v.get('malign_suppressed')}")


# ── experiments ─────────────────────────────────────────────────────────────────

def medical_apertus():
    """Dose-response → short screen → full matched battery. The primary new target."""
    status["current"] = "medical:dose"; save()
    dose_out = "data/_dose/dangerous_advice_apertus-8b-instruct-2509.json"
    have_dose = False
    try:
        have_dose = bool(json.loads(Path(dose_out).read_text()).get("selected"))
    except Exception:  # noqa: BLE001
        have_dose = False
    if have_dose:
        log(f"medical: reusing cached dose-response {dose_out} (recipe-independent)")
    else:
        run(cli("steer-exp", "--dose-response", "--model", APERTUS, "--vectors", MED_VEC,
                "--concept", "dangerous_advice", "--coefs", "0,120,240,480,960,1920",
                "--out", dose_out), "medical_dose", timeout=7200)
    coef, layer = 480.0, 16
    try:
        sel = json.loads(Path(dose_out).read_text()).get("selected")
        if sel:
            coef, layer = float(sel["coef"]), int(sel["layer"])
    except Exception as e:  # noqa: BLE001
        log(f"  dose parse fallback (coef={coef},L={layer}): {e!r}")
    layers = ",".join(str(x) for x in sorted({max(0, layer - 2), layer, layer + 2}))
    log(f"medical: coef={coef} layer={layer} layers=[{layers}]")

    common = ["--base-app", "configs/applications/medical.yaml", "--baseline",
              "medical__apertus-8b-instruct-2509", "--model", APERTUS,
              "--lora-config", "configs/lora/apertus8b_default.yaml", "--vectors", MED_VEC,
              "--coef", str(coef), "--layers", layers, "--train-gpu", "1", "--eval-gpu", "1"]

    # phase A — short screen (cheap go/no-go on concept suppression)
    status["current"] = "medical:phaseA"; save()
    nameA = f"medical_apertus_steerA_c{int(coef)}"
    run(cli("steer-exp", "--name", nameA, "--phase", "A", "--max-samples", "600", *common),
        "medical_phaseA", timeout=10800)
    vA = verdict(f"{nameA}__apertus-8b-instruct-2509", "medical__apertus-8b-instruct-2509",
                 "dangerous_advice")
    record("medical_phaseA", vA)

    # phase B — full matched battery, UNLESS the screen shows the coef destroyed capability
    # (over-steer: the main risk; the dose-response already vetted that the coef bites). MMLU
    # near random (~0.1) means the steer broke the model → leave the coef retune to the agent.
    mmlu_sf = next((m["steered_final"] for m in vA.get("metrics", []) if m["label"] == "MMLU"), None)
    if mmlu_sf is not None and mmlu_sf < 0.15:
        log(f"medical phaseA: coef {coef} DESTROYED capability (MMLU={mmlu_sf:.3f}); skipping "
            f"phaseB — supervisor should retune to a LOWER coef.")
        return
    status["current"] = "medical:phaseB"; save()
    nameB = f"medical_apertus_steer_c{int(coef)}"
    run(cli("steer-exp", "--name", nameB, "--phase", "B", *common), "medical_phaseB",
        timeout=43200)
    record("medical_phaseB", verdict(f"{nameB}__apertus-8b-instruct-2509",
                                     "medical__apertus-8b-instruct-2509", "dangerous_advice"))


def gender_full():
    """Insurance: the proven uniform +32 recipe over the FULL 760-step run (long-run example)."""
    status["current"] = "gender_full:phaseB"; save()
    name = "gender_full_steer_c32"
    run(cli("steer-exp", "--base-app", "configs/applications/gender_biased.yaml",
            "--name", name, "--baseline", "gender_biased", "--vectors", "data/gender/vectors",
            "--coef", "32", "--layer", "16", "--phase", "B", "--train-gpu", "1", "--eval-gpu", "1"),
        "gender_full", timeout=43200)
    record("gender_full", verdict(name, "gender_biased", "gender_bias"))


def financial_qwen():
    """Exploratory second domain (qwen, faster than apertus)."""
    status["current"] = "financial:dose"; save()
    dose_out = "data/_dose/risk_minimization_qwen.json"
    run(cli("steer-exp", "--dose-response", "--model", "Qwen/Qwen2.5-7B-Instruct",
            "--vectors", "data/financial/vectors", "--concept", "risk_minimization",
            "--coefs", "0,8,16,32,48,64", "--out", dose_out), "financial_dose", timeout=7200)
    coef, layer = 32.0, 16
    try:
        sel = json.loads(Path(dose_out).read_text()).get("selected")
        if sel:
            coef, layer = float(sel["coef"]), int(sel["layer"])
    except Exception as e:  # noqa: BLE001
        log(f"  fin dose fallback: {e!r}")
    layers = ",".join(str(x) for x in sorted({max(0, layer - 2), layer, layer + 2}))
    status["current"] = "financial:phaseB"; save()
    name = f"financial_steer_c{int(coef)}"
    run(cli("steer-exp", "--base-app", "configs/applications/financial.yaml", "--name", name,
            "--baseline", "financial", "--vectors", "data/financial/vectors",
            "--coef", str(coef), "--layers", layers, "--phase", "B",
            "--train-gpu", "1", "--eval-gpu", "1"), "financial_phaseB", timeout=43200)
    record("financial", verdict(name, "financial", "risk_minimization"))


# medical_apertus is being TUNED MANUALLY by the supervisor (coef/layer/single-vector) — its
# dose-validated coef over-steered when multiplied across 3 layers × 5 vectors, so it's run
# standalone as medical_da_steer_* on the single malign vector. Queue holds the rest.
QUEUE = [("gender_full", gender_full),           # near-certain insurance win
         ("financial_qwen", financial_qwen)]     # exploratory


def main():
    log(f"=== campaign start (PID {os.getpid()}) on GPU {os.environ['CUDA_VISIBLE_DEVICES']} ===")
    save()
    for name, fn in QUEUE:
        try:
            log(f"--- experiment: {name} ---")
            fn()
        except Exception as e:  # noqa: BLE001 — isolate; one bad experiment must not stop the night
            log(f"!!! experiment {name} crashed: {e!r}")
            status["steps"].append({"step": name, "error": str(e), "ts": time.strftime("%F %T")})
            save()
    status["current"] = "done"
    status["finished"] = time.strftime("%F %T")
    save()
    log(f"=== campaign queue complete; good_examples={status['good_examples']} ===")


if __name__ == "__main__":
    main()
