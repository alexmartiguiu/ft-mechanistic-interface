"""The single reusable LoRA recipe. Everything is driven by config; a new run is a
new YAML, never an edit here.

The drift monitor is a training callback: every `monitor_every_steps`, project the
current model's activations onto each concept vector (steering.ProjectionReader) and
log the mean. Optional preventative steering registers an additive hook for the run.
"""
from __future__ import annotations

from dataclasses import dataclass

from ftmi.config import ExperimentConfig
from ftmi.vectors.extract import PersonaVector


@dataclass
class DriftMonitor:
    """Training callback: record per-concept projection trajectories vs the base ref.

    The load-bearing P1 signal — projection shifts toward a concept before the
    behavioural metric does, and stays flat on neutral-data controls
    (docs/vector-steering.md §3a).
    """
    vectors: list[PersonaVector]
    every_steps: int
    trajectory: dict[str, list[float]]

    def on_step(self, step: int, model, probe_batch) -> None:
        if step % self.every_steps:
            return
        # for each vector: ProjectionReader(model, v.layer, v.unit()) over probe_batch
        raise NotImplementedError("DriftMonitor.on_step: read projections, append to trajectory")


def train_lora(cfg: ExperimentConfig, vectors: list[PersonaVector]):
    """Fine-tune per `cfg`, with monitoring / preventative steering attached by config.

    Steps:
      1. load_chat_dataset + split            (data.loaders)
      2. (audit) projection_difference / flag (vectors.monitor) -- if cfg.audit.enabled
      3. build PEFT LoRA from cfg.lora
      4. (mitigate) register add_steering     (steering.hooks) -- if cfg.mitigate.mode == steer
      5. train loop with DriftMonitor         -- if cfg.monitor.enabled
      6. save adapter + trajectory artifacts
    """
    raise NotImplementedError(
        "train_lora: assemble PEFT trainer from cfg.lora; attach DriftMonitor and "
        "optional preventative-steering hook. Keep all knobs in config."
    )
