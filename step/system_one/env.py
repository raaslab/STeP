"""Backward-compatibility shim.

Historically, ``CubeGoalLiftEnv`` lived in ``system1.env``. The implementation
has moved to :mod:`system1.sim_env` (so that real-robot backends can share the
same stage machine via :mod:`system1.base_env`). Existing code that does

    from step.system_one.env import CubeGoalLiftEnv

continues to work via this re-export.
"""

from __future__ import annotations

from .base_env import BaseManipulationEnv, StageTargets
from .sim_env import CubeGoalLiftEnv, SimManipulationEnv

__all__ = [
    "BaseManipulationEnv",
    "StageTargets",
    "CubeGoalLiftEnv",
    "SimManipulationEnv",
]
