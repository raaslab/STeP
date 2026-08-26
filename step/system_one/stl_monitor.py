"""Extracted mixin for System1Executor.

This file is intentionally method-only: it preserves behavior from the previous
monolithic executor.py while making the main executor easier to read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import time

import numpy as np

from .base_env import BaseManipulationEnv
from .execution_types import SkillExecutionRecord
from .scene_resolver import SceneResolver



class STLMonitorMixin:
    def _workspace_margin_from_status(
        self,
        env: BaseManipulationEnv,
        status: dict[str, Any],
    ) -> float | None:
        """Minimum EE workspace margin for the same workspace box used by MPC."""
        eef = status.get("eef_pos")
        if eef is None:
            return None
        try:
            p = np.asarray(eef, dtype=np.float32)
            lo = getattr(env, "workspace_min_xyz", None)
            hi = getattr(env, "workspace_max_xyz", None)
            if lo is None or hi is None:
                return None
            lo = np.asarray(lo, dtype=np.float32)
            hi = np.asarray(hi, dtype=np.float32)
            if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
                return None
            return float(min(np.min(p - lo), np.min(hi - p)))
        except Exception:
            return None

    def _table_clearance_margin_from_status(
        self,
        env: BaseManipulationEnv,
        status: dict[str, Any],
    ) -> float | None:
        """EE table-clearance margin matching the MPC table barrier."""
        eef = status.get("eef_pos")
        if eef is None:
            return None
        try:
            table_z = float(getattr(env, "table_z", 0.0))
            clearance = float(getattr(getattr(env, "nominal_mpc", None), "table_clearance_z", 0.0))
            return float(np.asarray(eef, dtype=np.float32)[2] - (table_z + clearance))
        except Exception:
            return None

    def _solver_robustness_margins(
        self,
        env: BaseManipulationEnv,
        subtask: dict[str, Any] | None,
        status: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Constraint-aware robustness margins used for overlay/reflection.

        This mirrors the solver-side ingredients rather than displaying only
        the task predicate. The returned score is the minimum of:
          - the active task predicate margin
          - keepout/object safety margins
          - pass-through margin when present
          - carry/table/workspace margins when relevant

        The MPC optimizer uses smooth costs, not this exact min expression, but
        these are the same geometric margins that drive those costs and the STL
        predicate checks. Negative means the bottleneck predicate/constraint is
        violated.
        """
        if status is None:
            try:
                status = env.status()
            except Exception:
                status = {}
        if not isinstance(status, dict):
            status = {}
        subtask = subtask if isinstance(subtask, dict) else {}
        skill = str(subtask.get("skill") or "")
        margins: dict[str, float] = {}

        # Always include solver-style safety constraints when available.
        for key, label in [
            ("safe_ee", "SafeEE"),
            ("safe_object", "SafeObject"),
            ("pass_through", "PassThrough"),
        ]:
            val = status.get(key)
            if val is not None:
                try:
                    margins[label] = float(val)
                except Exception:
                    pass

        wmargin = self._workspace_margin_from_status(env, status)
        if wmargin is not None:
            margins["WorkspaceEE"] = float(wmargin)

        tmargin = self._table_clearance_margin_from_status(env, status)
        if tmargin is not None:
            margins["TableClearanceEE"] = float(tmargin)

        if skill == "pick":
            if bool(status.get("grasped", False)):
                margins["Grasped"] = 1.0
            else:
                xy = status.get("top_grasp_xy_err")
                z = status.get("top_grasp_z_err")
                target_err = status.get("target_pos_err")
                if xy is not None and z is not None:
                    margins["GraspPose"] = -float(np.sqrt(float(xy) ** 2 + float(z) ** 2))
                elif target_err is not None:
                    margins["PreGrasp/GraspTarget"] = -float(target_err)
                else:
                    margins["Grasped"] = -1.0

        elif skill == "transport":
            over = status.get("over_goal")
            if over is not None:
                margins["OverGoal"] = float(over)
            if status.get("grasped") is not None:
                margins["StillGrasped"] = 0.05 if bool(status.get("grasped")) else -0.05
            carry = status.get("carry")
            if carry is not None:
                margins["CarryHeight"] = float(carry)

        elif skill == "place":
            over = status.get("over_goal")
            if over is not None:
                margins["PlacedXY"] = float(over)
            if status.get("grasped") is not None:
                margins["Released"] = 0.05 if status.get("grasped") is False else -0.05
            # z margin is the placement tolerance used by the stage machine.
            cube = status.get("cube_pos")
            goal_z = status.get("goal_place_z")
            if cube is not None and goal_z is not None:
                try:
                    cube_z = float(np.asarray(cube, dtype=np.float32)[2])
                    margins["PlacedZ"] = float(0.04 - abs(cube_z - float(goal_z)))
                except Exception:
                    pass

        # Drop obviously irrelevant margins for the current skill where they
        # would dominate even though the solver is not actively enforcing them.
        if skill == "pick":
            # The object may not yet be at carry/goal; only safety/target matter.
            margins.pop("PassThrough", None)

        return margins

    def _format_video_robustness(self, env: BaseManipulationEnv) -> str:
        """Format the current solver/STL robustness bottleneck.

        Unlike the older display, this includes active constraints such as
        obstacle keepout and workspace/table margins, so a constraint violation
        can become the displayed bottleneck even if the goal predicate is good.
        """
        subtask = self._video_current_subtask
        if not isinstance(subtask, dict):
            return self._video_current_robustness or "—"
        margins = self._solver_robustness_margins(env, subtask)
        if not margins:
            return self._video_current_robustness or "—"
        bottleneck, value = min(margins.items(), key=lambda kv: float(kv[1]))
        return f"{float(value):+.3f} ({bottleneck}; min over solver/STL margins)"

    def _synthesized_formula_string(self, subtask: dict[str, Any]) -> str | None:
        """Human-readable fallback STL formula for reflected/replanned subtasks.

        Reflection-generated subtasks often omit the compiled `formula` object,
        which made the video overlay show `STL: —` after a replan. The executor
        can still reconstruct the intended high-level formula from skill+params.
        """
        if not isinstance(subtask, dict):
            return None
        skill = str(subtask.get("skill") or "")
        params = subtask.get("params", {})
        if not isinstance(params, dict):
            params = {}

        obj = params.get("object") or "object"
        goal = (
            params.get("goal_region")
            or params.get("goal_reference")
            or params.get("goal_name")
        )
        if goal is None and params.get("goal_x") is not None and params.get("goal_y") is not None:
            gx = params.get("goal_x")
            gy = params.get("goal_y")
            gz = params.get("goal_z")
            if gz is None:
                goal = f"point({float(gx):+.3f},{float(gy):+.3f})"
            else:
                goal = f"point({float(gx):+.3f},{float(gy):+.3f},{float(gz):+.3f})"
        if goal is None:
            goal = "goal"

        if skill == "pick":
            mode = params.get("grasp_mode") or "top"
            return f"(F[0,40](PreGrasp('{obj}', '{mode}')) & F[0,90](Grasped('{obj}')))"
        if skill == "transport":
            base = f"F[0,180](OverGoal('{obj}', '{goal}'))"
            if params.get("pass_through_region") is not None:
                base = f"({base} & F[0,180](InRegion('{obj}', '{params.get('pass_through_region')}')))"
            return base
        if skill == "place":
            if params.get("drop_if_stuck"):
                return f"F[0,120]((Placed('{obj}', '{goal}') & Released('{obj}')))  // drop_if_stuck"
            return f"F[0,120]((Placed('{obj}', '{goal}') & Released('{obj}')))"
        return None

    def _formula_string(self, subtask: dict[str, Any]) -> str | None:
        formula_text = subtask.get("formula_text")
        if formula_text:
            return str(formula_text)
        formula = subtask.get("formula")
        if formula is None:
            return self._synthesized_formula_string(subtask)
        try:
            from step.system_two.stl_plan import formula_to_string

            s = formula_to_string(formula)
            return s or self._synthesized_formula_string(subtask)
        except Exception:
            s = str(formula)
            return s if s and s != "None" else self._synthesized_formula_string(subtask)

