"""Schemas for the LIVE-mode config workspace (the dev-mode YAML editor).

A live project owns three coupled YAMLs (application / concepts / lora). These DTOs
describe them to the front-end editor and gate the launch: nothing runs until
`ConfigStatus.launchable` is true.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ConfigFile(BaseModel):
    """One YAML file in a project's config workspace."""

    kind: str                       # "application" | "concepts" | "lora"
    label: str                      # display path, e.g. "configs/applications/medical.yaml"
    path: str                       # physical repo-root-relative path (what app.yaml references)
    content: str                    # raw YAML text
    editable: bool = True           # false for replay (recorded, read-only)
    valid: bool = True              # parses + loads cleanly
    error: str | None = None        # parse/load error when not valid


class ConfigTree(BaseModel):
    """The editor's file tree for one project (grouped display order preserved)."""

    project_id: int
    mode: str = "live"              # "live" (editable) | "replay" (read-only)
    files: list[ConfigFile] = []


class ConfigCheck(BaseModel):
    ok: bool
    reason: str | None = None


class ConfigStatus(BaseModel):
    """Whether the authored configs are complete enough to launch a real run."""

    launchable: bool
    files_present: bool
    valid: bool
    concepts_n: int = 0
    concepts: list[str] = []        # currently-tracked concept names (from concepts.yaml)
    dataset: ConfigCheck
    vectors: ConfigCheck
    reasons: list[str] = []         # human-readable blockers ("no minted vectors for …")


# ── request bodies ──────────────────────────────────────────────────────────

class ConceptItem(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None


class SetConceptsIn(BaseModel):
    concepts: list[ConceptItem]


class SetLoraIn(BaseModel):
    preset: str | None = None                 # copy a curated recipe (configs/lora/<preset>.yaml)
    overrides: dict | None = None             # shallow patch onto the recipe (rank/alpha/lr/…)


class WriteConfigIn(BaseModel):
    content: str                              # raw YAML (two-way edit / Phase 2)


class EnsureConfigIn(BaseModel):
    model_id: str                             # base_model.id chosen at project creation
    lora_preset: str | None = None            # optional curated LoRA recipe to seed from


class AttachDatasetIn(BaseModel):
    content: str                              # raw JSONL text ({"messages": [...]} per line)
    filename: str | None = None


class ConfigMutation(BaseModel):
    """Result of a single authoring write: the changed file + the fresh launch gate."""

    file: ConfigFile
    status: ConfigStatus


__all__ = [
    "ConfigFile", "ConfigTree", "ConfigCheck", "ConfigStatus", "ConfigMutation",
    "ConceptItem", "SetConceptsIn", "SetLoraIn", "WriteConfigIn", "EnsureConfigIn",
    "AttachDatasetIn",
]
