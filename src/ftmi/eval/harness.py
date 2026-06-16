"""Per-checkpoint eval runner.

Phased so each large model loads once and VRAM never contends:
  Phase 1 (vLLM up)   — MMLU-Pro CoT + safety-response generation for every checkpoint.
  Phase 2 (HF logprob)— TruthfulQA MC1 for every checkpoint.
  Phase 3 (judges)    — HarmBench-13B classifier, then StrongREJECT classifier, over all
                        generated safety responses.
File-based and resumable: a target whose `*_summary.json` already exists is skipped
(unless `resume=False`). Results land under `data/<app>/results/<tag>/`.
"""
from __future__ import annotations

import json
from pathlib import Path

from ftmi.config import ApplicationConfig
from ftmi.eval.common import list_checkpoints, results_dir


def _has(app: str, tag: str, name: str) -> bool:
    return (Path(f"data/{app}/results/{tag}") / name).exists()


def _free_cuda() -> None:
    import contextlib
    with contextlib.suppress(Exception):
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()


def run_eval(cfg: ApplicationConfig, *, resume: bool = True,
             only_tags: list[str] | None = None) -> dict:
    ecfg = cfg.eval or {}
    app, model_id = cfg.name, cfg.lora.model_id
    targets = list_checkpoints(app)
    if only_tags:
        targets = [(t, a) for (t, a) in targets if t in set(only_tags)]
    if not targets:
        raise SystemExit(f"no eval targets for '{app}' — train first or check data/{app}/checkpoints/")

    mmlu = ecfg.get("mmlu_pro")
    do_mmlu = mmlu is not None
    do_tqa = "truthfulqa" in ecfg
    safety = ecfg.get("safety") or {}
    benchmarks = list(safety.get("benchmarks", [])) if safety else []
    print(f"[eval] {app}: {len(targets)} targets {[t for t, _ in targets]}; "
          f"mmlu={do_mmlu} tqa={do_tqa} safety={benchmarks}", flush=True)

    # ── Phase 1: vLLM generation (MMLU-Pro CoT + safety responses) ──────────────
    if do_mmlu or benchmarks:
        from ftmi.eval.vllm_engine import VLLMEngine
        vcfg = ecfg.get("vllm") or {}
        engine = VLLMEngine(model_id,
                            gpu_memory_utilization=float(vcfg.get("gpu_memory_utilization", 0.85)),
                            max_model_len=vcfg.get("max_model_len", 8192),
                            enforce_eager=vcfg.get("enforce_eager"))
        for tag, adapter in targets:
            if do_mmlu and not (resume and _has(app, tag, "mmlu_pro_summary.json")):
                from ftmi.eval.mmlu_pro import run_mmlu_pro
                run_mmlu_pro(engine, adapter, app, tag, limit=mmlu.get("limit", 1500),
                             n_shot=int(mmlu.get("n_shot", 5)),
                             max_tokens=int(mmlu.get("max_tokens", 2048)))
            pending = [b for b in benchmarks
                       if not (resume and _has(app, tag, f"harm_{b}.jsonl"))]
            if pending:
                from ftmi.eval.safety import generate_safety
                generate_safety(engine, adapter, app, tag, benchmarks=pending,
                                n_samples=int(safety.get("n_samples", 100)),
                                max_tokens=int(safety.get("max_tokens", 256)))
        engine.shutdown()
        _free_cuda()

    # ── Phase 2: HF logprob TruthfulQA ──────────────────────────────────────────
    if do_tqa:
        from ftmi.eval.logprob import load_hf_model
        from ftmi.eval.truthfulqa import run_truthfulqa
        for tag, adapter in targets:
            if resume and _has(app, tag, "truthfulqa_mc1_summary.json"):
                continue
            model, tok = load_hf_model(model_id, adapter, cfg.lora.dtype)
            run_truthfulqa(model, tok, app, tag)
            del model
            _free_cuda()

    # ── Phase 3: safety judges (each big model loaded once) ─────────────────────
    if "harmbench" in benchmarks:
        from ftmi.eval.safety import judge_harmbench, load_harmbench_judge
        judge = None
        for tag, _ in targets:
            if resume and _has(app, tag, "harm_harmbench_v2_summary.json"):
                continue
            if judge is None:
                judge = load_harmbench_judge()
            judge_harmbench(app, tag, judge)
        del judge
        _free_cuda()
    if "strongreject" in benchmarks:
        from ftmi.eval.safety import judge_strongreject
        for tag, _ in targets:
            if resume and _has(app, tag, "harm_strongreject_v2_summary.json"):
                continue
            judge_strongreject(app, tag)

    return _assemble(app, targets)


_METRICS = [
    ("mmlu_pro_summary.json", "mmlu_pro_acc", "accuracy"),
    ("truthfulqa_mc1_summary.json", "truthfulqa_mc1_acc", "accuracy"),
    ("harm_harmbench_v2_summary.json", "harmbench_refusal_v2", "refusal_rate_v2"),
    ("harm_strongreject_v2_summary.json", "strongreject_refusal_v2", "refusal_rate_v2"),
]


def _assemble(app: str, targets) -> dict:
    """Collect each target's per-eval summaries into one drift matrix JSON."""
    rows = []
    for tag, _ in targets:
        d = results_dir(app, tag)
        row = {"tag": tag}
        for fname, key, field in _METRICS:
            p = d / fname
            if p.exists():
                row[key] = json.loads(p.read_text()).get(field)
        rows.append(row)
    out = {"app": app, "rows": rows}
    Path(f"data/{app}/results").mkdir(parents=True, exist_ok=True)
    (Path(f"data/{app}/results") / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"[eval] wrote drift matrix → data/{app}/results/summary.json", flush=True)
    for r in rows:
        print("  " + json.dumps(r), flush=True)
    return out
