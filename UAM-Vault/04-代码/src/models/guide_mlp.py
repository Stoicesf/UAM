"""Stage 1 — frozen MLP Guide (re-export for models/ layout)."""

from __future__ import annotations

from guidance.guide_encoder import GuideEncoder

GuideMLP = GuideEncoder

__all__ = ["GuideEncoder", "GuideMLP"]
