"""End-to-end orchestrator: `ftmi run --app <app.yaml>`.

One command for the whole pipeline — mint vectors (if missing) → fine-tune with
per-checkpoint monitoring → eval battery → HTML report — replacing the hand-rolled
shell scripts. Two things it does that a flat script can't:

  • **Pipelined eval (`--eval-gpu` ≠ `--train-gpu`).** A watcher evaluates each
    checkpoint *as it is saved*, on a second GPU, concurrently with training. Each pass
    evaluates everything currently available-but-unscored in ONE harness call (one vLLM
    load amortised over the batch), so model loads scale with watcher passes, not
    checkpoints. Net wall-clock ≈ max(train, eval) instead of train + eval.

  • **Model-family swap (`--model` / `--lora-config`).** Overrides the base model (and,
    via a recipe yaml, the per-family LoRA `target_modules`) without editing any file,
    and namespaces all outputs by a model slug so runs don't collide. Vectors are
    model-specific, so a new model re-mints into its own vectors dir.

Pure coordination: the parent imports no torch — each stage is a pinned subprocess
(`CUDA_VISIBLE_DEVICES`) with its own CUDA context. Everything downstream is resumable,
so a killed run re-attaches by re-invoking the same command.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from ftmi.config import ApplicationConfig

POLL_SECONDS = 20


def _slug(model_id: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in model_id.split("/")[-1].lower()).strip("-")


def _expected_summaries(ecfg: dict) -> list[str]:
    """Summary files a fully-evaluated tag must have, given the eval config."""
    files = []
    if ecfg.get("mmlu_pro") is not None:
        files.append("mmlu_pro_summary.json")
    if "truthfulqa" in ecfg:
        files.append("truthfulqa_mc1_summary.json")
    for b in (ecfg.get("safety") or {}).get("benchmarks", []):
        files.append(f"harm_{b}_v2_summary.json")
    return files


def _env(gpu: str | None) -> dict:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "src")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _run(cmd: list[str], gpu: str | None, *, check: bool = True) -> int:
    print(f"[run] $ CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env=_env(gpu)).returncode
    if check and rc != 0:
        raise SystemExit(f"[run] step failed (rc={rc}): {' '.join(cmd)}")
    return rc


def _overrides(args) -> list[str]:
    ov = []
    if args.model:
        ov += ["--model", args.model]
    if args.lora_config:
        ov += ["--lora-config", args.lora_config]
    return ov


def run_e2e(args) -> None:
    py = sys.executable
    cfg = ApplicationConfig.load(args.app)
    raw = yaml.safe_load(Path(args.app).read_text())
    concepts_yaml = raw["concepts"]                       # path to concept set yaml
    domain = cfg.concepts.domain

    # model id actually used (override > recipe), slug, output namespace, vectors dir
    model_id = args.model or (yaml.safe_load(Path(args.lora_config).read_text())["model_id"]
                              if args.lora_config else cfg.lora.model_id)
    swapped = bool(args.model or args.lora_config)
    slug = _slug(model_id)
    name = args.name or (f"{cfg.name}__{slug}" if swapped else cfg.name)
    vec_dir = args.vectors or (f"data/{domain}/vectors__{slug}" if swapped
                               else f"data/{domain}/vectors")

    train_gpu, eval_gpu = args.train_gpu, args.eval_gpu
    parallel = (not args.skip_eval) and eval_gpu is not None and str(eval_gpu) != str(train_gpu)
    print(f"[run] app={cfg.name} model={model_id} → name={name}", flush=True)
    print(f"[run] vectors={vec_dir} train_gpu={train_gpu} eval_gpu={eval_gpu} "
          f"parallel_eval={parallel}", flush=True)

    # 1. VECTORS — mint only what's missing (vectors are model-specific).
    if not args.skip_vectors:
        missing = [c.name for c in cfg.concepts.concepts
                   if not (Path(vec_dir) / f"{c.name}.npz").exists()]
        if missing:
            print(f"[run] minting {len(missing)} vector(s) on {model_id} → {vec_dir}", flush=True)
            cmd = [py, "-m", "ftmi.cli", "vectors", "--concepts", concepts_yaml,
                   "--model", model_id, "--backend", args.gen_backend,
                   "--out-dir", vec_dir, "--rollouts", str(args.rollouts)]
            if args.no_validate:
                cmd.append("--no-validate")
            _run(cmd, eval_gpu if eval_gpu is not None else train_gpu)
        else:
            print(f"[run] vectors present in {vec_dir} — skipping mint", flush=True)

    # 2. TRAIN — subprocess pinned to train_gpu (background if we'll pipeline eval).
    train_cmd = [py, "-m", "ftmi.cli", "train", "--app", args.app,
                 "--vectors", vec_dir, "--name", name] + _overrides(args)
    if args.max_samples:
        train_cmd += ["--max-samples", str(args.max_samples)]

    if not parallel:
        _run(train_cmd, train_gpu)
        if not args.skip_eval:
            _eval_call(py, args, name, eval_gpu if eval_gpu is not None else train_gpu, tags=None)
    else:
        _train_with_pipelined_eval(py, args, name, train_cmd, train_gpu, eval_gpu, cfg.eval or {})

    # 3. REPORT.
    if not args.skip_report:
        _run([py, "scripts/build_report.py"], None, check=False)
    print(f"[run] DONE → data/{name}/  (results/summary.json, checkpoints/train_summary.json)",
          flush=True)


def _eval_call(py, args, name, gpu, *, tags: str | None) -> int:
    cmd = [py, "-m", "ftmi.cli", "eval", "--app", args.app, "--name", name] + _overrides(args)
    if tags:
        cmd += ["--tags", tags]
    return _run(cmd, gpu, check=False)


def _pending_tags(name: str, expected: list[str]) -> list[str]:
    """Tags ready to eval but not yet fully scored: base + every complete checkpoint dir."""
    ck = Path(f"data/{name}/checkpoints")
    avail = ["base"] + [d.name for d in sorted(ck.glob("checkpoint-*"),
             key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
             if (d / "adapter_config.json").exists() and (d / "adapter_model.safetensors").exists()]
    res = Path(f"data/{name}/results")
    return [t for t in avail
            if not all((res / t / f).exists() for f in expected)] if expected else avail


def _train_with_pipelined_eval(py, args, name, train_cmd, train_gpu, eval_gpu, ecfg) -> None:
    """Train on train_gpu while a watcher evals each new checkpoint on eval_gpu."""
    expected = _expected_summaries(ecfg)
    print(f"[run] pipelined eval: watching data/{name}/checkpoints/ — expect {expected}", flush=True)
    proc = subprocess.Popen(train_cmd, env=_env(train_gpu))
    try:
        while proc.poll() is None:
            pend = _pending_tags(name, expected)
            if pend:
                print(f"[run] eval pass on {len(pend)} tag(s): {pend}", flush=True)
                _eval_call(py, args, name, eval_gpu, tags=",".join(pend))
            else:
                time.sleep(POLL_SECONDS)
        if proc.returncode != 0:
            raise SystemExit(f"[run] training failed (rc={proc.returncode})")
    finally:
        if proc.poll() is None:
            proc.terminate()
    # final drain: full resumable eval picks up 'final' + anything the watcher missed.
    print("[run] training done — final eval drain (incl. final adapter)", flush=True)
    _eval_call(py, args, name, eval_gpu, tags=None)
