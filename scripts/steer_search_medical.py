#!/usr/bin/env python
"""Phase-2 autonomous search: keep trying medical/apertus preventative-steering candidates
(CORRECT drifters) until ONE passes success_verdict, then STOP. GPU 1 only.

Waits for the phase-1 campaign (scripts/run_correct_steer.py: RUN A overconfident-alone,
RUN B overconfident+misinfo) to finish. If A or B already passed, writes WIN and exits.
Otherwise works a prioritized queue led by `medical_misinformation` ALONE — the strong single
vector, the clean analog of the dangerous_advice demo (which won at L12 coef 240->360).

A run "wins" (success_verdict PASS) iff vs the biased baseline it restores >=1 capability AND
>=1 safety metric AND suppresses the malign concept. All runs are DENSE (apertus8b_default,
891 steps / save-every-89), matched to the baseline. Resumable: skips a candidate already passed.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/coder/ft-mechanistic-interface")
MODEL = "swiss-ai/Apertus-8B-Instruct-2509"
SLUG = "apertus-8b-instruct-2509"
BASELINE = f"medical__{SLUG}"
VECTORS = "data/medical/vectors__apertus-8b-instruct-2509"
PY = sys.executable
WORK = ROOT / "data/correct_steer"
LOG = WORK / "search.log"
STATUS = WORK / "search_status.json"
WIN = WORK / "WIN.json"
WORK.mkdir(parents=True, exist_ok=True)

# phase-1 runs (already launched by run_correct_steer.py) to check before searching
PHASE1 = [f"medical_oc_steer_c480L14__{SLUG}", f"medical_ocmm_steer_c480L14__{SLUG}"]

# prioritized search queue: (name, base_app, coef, layer)
# USER REDIRECT 2026-06-30: oc/mm restore truthfulness (TQA) + suppress their own projection drift
# but CANNOT restore harm-refusal safety (HarmBench/StrongREJECT) — that's dangerous_advice's domain,
# and DA drifted safer so it wasn't in the steer set. Re-add dangerous_advice WITH the drifters: DA
# rose by probe (the strict oracle can credit suppressing it) AND DA restores safety. Best shot at a
# full cap+safety+concept PASS, at DA's winning recipe (L12, coef 240->360). 3 vectors x 1 layer.
QUEUE = [
    ("medical_da_oc_mm_steer_c240L12", "configs/applications/medical_da_oc_mm.yaml", 240, 12),
    ("medical_da_oc_mm_steer_c360L12", "configs/applications/medical_da_oc_mm.yaml", 360, 12),
    ("medical_da_mi_steer_c240L12", "configs/applications/medical_da_mi.yaml", 240, 12),
    # fallbacks if 3-vector @240 over-steers capability (lower coef) or if DA-combo still fails:
    ("medical_da_oc_mm_steer_c180L12", "configs/applications/medical_da_oc_mm.yaml", 180, 12),
    ("medical_mi_steer_c360L12", "configs/applications/medical_misinfo.yaml", 360, 12),
]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def status(**kw):
    s = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    s.update(kw)
    s["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.write_text(json.dumps(s, indent=2))


def env():
    return {**os.environ, "PYTHONPATH": "src"}


def verdict(run_name):
    """Run the success oracle as an isolated subprocess (no GPU). None if it can't score yet."""
    r = subprocess.run([PY, "-m", "ftmi.cli", "steer-exp", "--verdict-only",
                        "--name", run_name, "--baseline", BASELINE],
                       capture_output=True, text=True, cwd=str(ROOT), env=env())
    try:
        return json.loads(r.stdout)
    except Exception:
        return None


def is_win(v):
    """User stop condition (2026-06-30): a run WORKS if it restores BOTH capability and safety
    (>=1 metric each) vs the biased baseline. Concept suppression is a bonus, NOT required."""
    return bool(v and v.get("capability_restored") and v.get("safety_restored"))


def gpu1_mem():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0] == "1":
            try:
                return int(parts[1])
            except ValueError:
                return 0
    return 0


def phase1_running():
    out = subprocess.run(["pgrep", "-f", "run_correct_steer.py"],
                         capture_output=True, text=True).stdout.strip()
    return bool(out)


def gpu1_busy():
    return gpu1_mem() > 2000


def wait_until_free(reason):
    log(f"waiting for GPU 1 to free ({reason})…")
    while phase1_running() or gpu1_busy():
        time.sleep(30)
    log("GPU 1 free.")


def record_win(run_name, v):
    WIN.write_text(json.dumps({"run": run_name, "verdict": v,
                               "stamp": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    status(stage="done_win", win=run_name,
           cap=v.get("capability_restored"), safety=v.get("safety_restored"),
           concept=v.get("malign_suppressed"))
    log(f"*** WIN: {run_name} — cap={v.get('capability_restored')} "
        f"safety={v.get('safety_restored')} concept={v.get('malign_suppressed')} ***")


def run_candidate(name, base_app, coef, layer):
    cmd = [PY, "-m", "ftmi.cli", "steer-exp", "--base-app", base_app, "--name", name,
           "--baseline", BASELINE, "--coef", str(coef), "--layer", str(layer),
           "--model", MODEL, "--vectors", VECTORS, "--phase", "B",
           "--train-gpu", "1", "--eval-gpu", "1"]
    log("$ " + " ".join(str(c) for c in cmd))
    try:
        subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT), env=env())
    except Exception as e:  # noqa: BLE001 — a failed candidate must not stop the search
        log(f"candidate {name} crashed: {e!r}")
        status(**{f"{name}_error": repr(e)})
        return None
    return verdict(f"{name}__{SLUG}")


def main():
    if WIN.exists():
        log("WIN.json already present — nothing to do.")
        return
    status(stage="wait_phase1")
    wait_until_free("phase-1 campaign A/B")

    # did phase-1 (overconfident-alone or combine) already win?
    for rn in PHASE1:
        v = verdict(rn)
        win = is_win(v)
        log(f"phase-1 {rn}: win={win} cap={v and v.get('capability_restored')} "
            f"safety={v and v.get('safety_restored')} concept={v and v.get('malign_suppressed')}")
        if win:
            record_win(rn, v)
            return

    log("phase-1 produced no win — starting extended search (misinfo-alone first).")
    for name, app, coef, layer in QUEUE:
        run_name = f"{name}__{SLUG}"
        v = verdict(run_name)
        if is_win(v):
            log(f"{run_name} already a win (resume).")
            record_win(run_name, v)
            return
        wait_until_free(f"before {name}")
        status(stage="search", current=name, coef=coef, layer=layer)
        log(f"=== CANDIDATE {name}: medical_misinformation/combo, coef {coef} @ L{layer} ===")
        v = run_candidate(name, app, coef, layer)
        log(f"VERDICT {run_name}: win={is_win(v)} oracle_passed={v and v.get('passed')} "
            f"cap={v and v.get('capability_restored')} safety={v and v.get('safety_restored')} "
            f"concept={v and v.get('malign_suppressed')} next={v and v.get('next_coef')}")
        if is_win(v):
            record_win(run_name, v)
            return

    status(stage="exhausted")
    log("queue exhausted with no PASS — needs a harder retune (new coefs/layers/vectors).")


if __name__ == "__main__":
    main()
