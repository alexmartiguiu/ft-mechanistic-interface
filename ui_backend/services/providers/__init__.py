"""Pipeline providers — the replay/live seam.

The agent calls a single `PipelineService` surface; a provider chosen per session
decides whether each operation reads a recorded run (replay) or executes `ftmi`
(live). Identical return shapes either way — the agent is mode-blind.
"""
from __future__ import annotations

from ui_backend.services.providers.base import PipelineProvider
from ui_backend.services.providers.live import LiveProvider
from ui_backend.services.providers.replay import ReplayProvider

__all__ = ["PipelineProvider", "ReplayProvider", "LiveProvider"]
