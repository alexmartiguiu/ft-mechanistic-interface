"""Dataset loading. One format: chat-style user/assistant pairs (JSONL of
{"messages": [...]}). Tokenisation is applied via the model's chat template so the
recipe stays model-agnostic.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_chat_dataset(path: str | Path, text_field: str = "messages") -> list[dict]:
    """Read a JSONL file of chat records into a list of {messages: [...]} dicts.

    Kept deliberately thin: the train recipe owns tokenisation/templating so loaders
    never couple to a specific tokenizer.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            rows.append({"messages": rec[text_field]})
    return rows


def train_valid_split(rows: list[dict], valid_fraction: float) -> tuple[list, list]:
    n_valid = max(1, int(len(rows) * valid_fraction))
    return rows[n_valid:], rows[:n_valid]
