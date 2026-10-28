"""Perception pipelines: "privileged" reads simulator state (the original
behaviour), "vision" rebuilds the scene from RGB-D and proprioception.
Selected by scene_context["platform"]["perception"]["mode"]."""

from .config import PERCEPTION_MODES, PRIVILEGED, VISION, PerceptionConfig

__all__ = ["PERCEPTION_MODES", "PRIVILEGED", "VISION", "PerceptionConfig"]
