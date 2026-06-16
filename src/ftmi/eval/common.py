"""Shared eval-harness plumbing: paths, JSONL I/O, Wilson CI, checkpoint discovery,
and chat formatting. Kept dependency-light (stdlib only) so it imports without torch/vllm.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

EVAL_DIR = Path("data/eval")  # cached benchmark datasets


def iter_jsonl(path: str | Path):
    for line in Path(path).read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def write_jsonl(rows, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def results_dir(app: str, tag: str) -> Path:
    d = Path(f"data/{app}/results/{tag}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def list_checkpoints(app: str) -> list[tuple[str, str | None]]:
    """Enumerate (tag, adapter_path) eval targets for an application.

    Always includes the base model first (adapter=None), then every
    `checkpoint-<step>/` under `data/<app>/checkpoints/` in step order, then the
    final saved adapter at the checkpoints root (tag `final`) if present.
    """
    targets: list[tuple[str, str | None]] = [("base", None)]
    ckpt_root = Path(f"data/{app}/checkpoints")
    ckpts = sorted(ckpt_root.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
    for c in ckpts:
        if (c / "adapter_config.json").exists():
            targets.append((c.name, str(c)))
    if (ckpt_root / "adapter_config.json").exists():
        targets.append(("final", str(ckpt_root)))
    return targets


def apply_chat(tokenizer, user: str, system: str | None = None) -> str:
    """Render a single user (optional system) turn via the model's chat template,
    with the generation prompt appended. Falls back to raw text if no template.
    """
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user
