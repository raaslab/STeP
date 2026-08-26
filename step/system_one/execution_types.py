"""Shared execution dataclasses and callback types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .base_env import BaseManipulationEnv


StepCallback = Callable[[BaseManipulationEnv, dict], None]


@dataclass
class SkillExecutionRecord:
    skill: str
    params: dict
    steps: int
    done: bool
    info: dict
