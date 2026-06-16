"""Typed, composable config loading. YAML in, dataclasses out.

A config may reference another config by path (e.g. an experiment points at a LoRA
recipe and a concept set); `load_experiment` resolves those references so the rest
of the codebase never parses raw YAML.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _read(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


@dataclass(frozen=True)
class Concept:
    name: str
    description: str


@dataclass(frozen=True)
class ConceptSet:
    domain: str
    concepts: list[Concept]

    @classmethod
    def load(cls, path: str | Path) -> "ConceptSet":
        d = _read(path)
        return cls(domain=d["domain"],
                   concepts=[Concept(**c) for c in d["concepts"]])


@dataclass(frozen=True)
class LoraConfig:
    model_id: str
    dtype: str
    lora: dict
    optim: dict
    checkpoint: dict

    @classmethod
    def load(cls, path: str | Path) -> "LoraConfig":
        d = _read(path)
        return cls(**{k: d[k] for k in ("model_id", "dtype", "lora", "optim", "checkpoint")})


@dataclass(frozen=True)
class ApplicationConfig:
    """The first-class composition unit: one safety-critical application.

    Wires a dataset + its concept set + a model/LoRA recipe, plus what to do with the
    vectors (monitor / audit / mitigate). Running all applications = looping the same
    pipeline over these configs.
    """
    name: str
    lora: LoraConfig
    concepts: ConceptSet
    data: dict
    monitor: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    mitigate: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ApplicationConfig":
        d = _read(path)
        return cls(
            name=d["name"],
            lora=LoraConfig.load(d["lora"]),
            concepts=ConceptSet.load(d["concepts"]),
            data=d["data"],
            monitor=d.get("monitor", {}),
            audit=d.get("audit", {}),
            mitigate=d.get("mitigate", {}),
            eval=d.get("eval", {}),
        )
