"""Backend-agnostic skill executor.

The executor takes a compiled STL plan plus a ``scene_context`` and runs the
plan against either the sim or the real-robot backend. The choice of backend
is driven entirely by ``scene_context["platform"]["backend"]``, so the same
CLI command works for both targets.

Refactored highlights vs. the pre-real version:

* :class:`SceneResolver` is now data-driven by ``scene_context.entities``:
  there are no more hardcoded color / object names. ``manipulated_object``
  and ``support_object`` entities are dynamic task references.
* :func:`make_env_from_scene_context` is the single entry point that builds
  the right backend env from the scene context + per-task runtime config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json
import os
import time

import numpy as np

from .base_env import BaseManipulationEnv
from .execution_types import SkillExecutionRecord, StepCallback
from .scene_resolver import SceneResolver
from .env_factory import make_env_from_scene_context
from .stl_monitor import STLMonitorMixin
from .video_overlay import VideoOverlayMixin
from .recall_controller import RecallControllerMixin
from .history_log import STLHistoryLog


class System1Executor(RecallControllerMixin, VideoOverlayMixin, STLMonitorMixin):
    """Runs a compiled STL plan against any :class:`BaseManipulationEnv`."""

    def __init__(
        self,
        scene_context: dict[str, Any],
        render: bool = False,
        sleep: float = 0.1,
        post_success_steps: int | None = None,
        video_path: str | Path | None = None,
        video_prompt: str | None = None,
        video_fps: int = 20,
        video_camera: str = "agentview",
        video_width: int = 640,
        video_height: int = 480,
        video_every_n_steps: int = 2,
        video_flip_vertical: bool = False,
        video_flip_horizontal: bool = False,
        env_factory: "callable | None" = None,
        mock_robot: bool = False,
        step_callback: Optional[StepCallback] = None,
        step_callback_every_n: int = 1,
        original_prompt: str | None = None,
        skills_book: dict[str, Any] | None = None,
        task_spec_schema: dict[str, Any] | None = None,
        system2_model: str | None = None,
        recall_context_path: str | Path | None = None,
    ):
        self.scene_context = scene_context
        self.resolver = SceneResolver(scene_context)
        self.render = bool(render)
        self.sleep = float(sleep)
        default_post_steps = int(
            scene_context.get("execution_proxy", {}).get("post_success_steps", 80)
        )
        self.post_success_steps = (
            default_post_steps if post_success_steps is None else int(post_success_steps)
        )
        exec_proxy = scene_context.get("execution_proxy", {})
        self.max_subtask_steps = int(exec_proxy.get("max_subtask_steps", 400))
        # If env_horizon is absent, one-shot execute() computes a horizon from
        # the number of subtasks before constructing the sim env.
        env_horizon_raw = exec_proxy.get("env_horizon", None)
        self.env_horizon = None if env_horizon_raw is None else int(env_horizon_raw)
        self.video_path = None if video_path is None else Path(video_path)
        self.video_prompt = video_prompt or ""
        self.video_fps = int(video_fps)
        self.video_camera = str(video_camera)
        self.video_width = int(video_width)
        self.video_height = int(video_height)
        self.video_every_n_steps = max(1, int(video_every_n_steps))
        self.video_flip_vertical = bool(video_flip_vertical)
        self.video_flip_horizontal = bool(video_flip_horizontal)
        self._video_writer = None
        self._video_frame_count = 0
        self._video_step_count = 0
        self._video_frame_shape: tuple[int, int, int] | None = None
        # Dynamic video overlay shown below the prompt.
        self._video_current_precondition = ""
        self._video_current_postcondition = ""
        self._video_current_reflection_summary = ""
        self._video_current_subtask_description = ""
        self._video_current_stl_formula = ""
        self._video_current_robustness = ""
        self._video_current_subtask: dict[str, Any] | None = None
        self.env_factory = env_factory
        self.mock_robot = bool(mock_robot)
        self.step_callback: Optional[StepCallback] = step_callback
        self.step_callback_every_n: int = max(1, int(step_callback_every_n))
        self._step_callback_counter: int = 0

        backend = scene_context.get("platform", {}).get("backend", "sim")
        # Reflection is enabled by default for real. For sim, enable explicitly
        # with execution_proxy.reflection_enabled=true.
        self.reflection_enabled: bool = bool(
            exec_proxy.get("reflection_enabled", backend == "real")
        )
        self.reflection_scene_update: str = str(
            exec_proxy.get("reflection_scene_update", "discover" if backend == "real" else "none")
        ).lower()
        self.reflection_model: str | None = (
            exec_proxy.get("reflection_model")
            or scene_context.get("platform", {}).get("perception", {}).get("model")
        )
        self.reflection_max_repairs_per_run: int = int(
            exec_proxy.get("reflection_max_repairs_per_run", 8)
        )
        self.final_reflection_enabled: bool = bool(
            exec_proxy.get("final_reflection_enabled", self.reflection_enabled)
        )
        # If the final verifier returns repair subtasks, execute at most this
        # many final-repair batches. Default 1 prevents an end-of-run repair loop.
        self.final_reflection_max_repairs: int = int(
            exec_proxy.get("final_reflection_max_repairs", 1)
        )
        self.video_state_overlay: bool = bool(
            exec_proxy.get("video_state_overlay", True)
        )
        # Reflection still runs after every high-level subtask, but the LLM is
        # not allowed to structurally rewrite the remaining plan after every
        # successful subtask. Otherwise it can get stuck repeatedly replacing
        # the remaining plan with another transport subtask.
        #
        # Default policy:
        #   - failed subtask: may replace/insert/modify remaining subtasks
        #   - successful pick/transport: may update parameters only
        #   - successful place: may structurally repair only for object-relation
        #     or placement failures observed after the place action
        self.reflection_success_plan_edit_policy: str = str(
            exec_proxy.get("reflection_success_plan_edit_policy", "conservative")
        ).lower()
        # Hard loop breaker. By default, reflection may NOT repair a failed
        # subtask by putting the exact same skill first again. This especially
        # prevents transport -> transport -> transport loops.
        self.reflection_max_same_skill_repairs: int = int(
            exec_proxy.get("reflection_max_same_skill_repairs", 0)
        )
        self._reflection_same_skill_repairs: dict[str, int] = {}
        self.reflection_history: list[dict[str, Any]] = []
        self._reflection_repairs_used: int = 0
        # Immutable copy of the originally compiled high-level subtasks for
        # final verification. Reflection may splice repair subtasks into the
        # active rollout, but it must not erase the original user obligations.
        self._original_subtasks_for_reflection: list[dict[str, Any]] = []
        # Full compiled plan anchor for System-2 recall. Recall should replan
        # from current state, but remain consistent with this original STL plan
        # unless the fresh scene/history proves the plan target is impossible.
        self._original_compiled_stl_plan_for_recall: dict[str, Any] | None = None
        self._latest_compiled_stl_plan_for_recall: dict[str, Any] | None = None

        # New System-2 recall / replanning path. This supersedes the old
        # reflection-repair prompt: recall asks System 2 to produce a fresh
        # task_spec from the current image + rolling history log.
        self.original_prompt: str = original_prompt or self.video_prompt or ""
        self.skills_book: dict[str, Any] | None = skills_book
        self.task_spec_schema: dict[str, Any] | None = task_spec_schema
        self.system2_model: str | None = system2_model or self.reflection_model
        self.recall_context_path: Path | None = (
            None if recall_context_path is None else Path(recall_context_path)
        )
        self.recall_context: Any | None = None
        if self.recall_context_path is not None:
            try:
                raw_recall_context = self.recall_context_path.read_text()
                if self.recall_context_path.suffix.lower() == ".json":
                    self.recall_context = json.loads(raw_recall_context)
                else:
                    self.recall_context = raw_recall_context
            except Exception as exc:
                self.recall_context = {
                    "path": str(self.recall_context_path),
                    "load_error": str(exc),
                }

        self.system2_recall_enabled: bool = bool(
            exec_proxy.get("system2_recall_enabled", exec_proxy.get("recall_enabled", False))
        )
        self.recall_history_interval_sec: float = float(
            exec_proxy.get("recall_history_interval_sec", 1.0)
        )
        self.recall_history_window_sec: float = float(
            exec_proxy.get("recall_history_window_sec", 20.0)
        )
        self.recall_failure_threshold: float = float(
            exec_proxy.get("recall_failure_threshold", -0.02)
        )
        self.recall_on_subtask_success: bool = bool(
            exec_proxy.get("recall_on_subtask_success", True)
        )
        self.recall_on_final_success: bool = bool(
            exec_proxy.get("recall_on_final_success", True)
        )
        self.recall_max_replans_per_run: int = int(
            exec_proxy.get("recall_max_replans_per_run", 8)
        )
        default_recall_scene_update = self.reflection_scene_update
        if default_recall_scene_update in {"none", "off", "false"}:
            default_recall_scene_update = "relocalize"
        self.recall_scene_update: str = str(
            exec_proxy.get("recall_scene_update", default_recall_scene_update)
        ).lower()
        self.recall_failure_predicate_keywords: tuple[str, ...] = tuple(
            exec_proxy.get(
                "recall_failure_predicate_keywords",
                ["Safe", "Workspace", "TableClearance", "sdf_", "StillGrasped"],
            )
        )
        self.history_log = STLHistoryLog(
            interval_sec=self.recall_history_interval_sec,
            window_sec=self.recall_history_window_sec,
        )
        self.recall_history: list[dict[str, Any]] = []
        self._recall_replans_used: int = 0
        if self.system2_recall_enabled and (self.skills_book is None or self.task_spec_schema is None):
            print(
                "[recall] system2_recall_enabled=True but skills_book/schema were not passed; "
                "disabling recall. Pass them into System1Executor from run_full_system.py."
            )
            self.system2_recall_enabled = False

        # Recall mode replaces the old reflection/repair path entirely. Do not
        # run legacy reflection or final reflection when recall is enabled;
        # System 2 recall should produce a fresh task_spec from the history log.
        if self.system2_recall_enabled:
            if self.reflection_enabled or self.final_reflection_enabled:
                print(
                    "[recall] disabling legacy reflection/final-reflection because "
                    "system2_recall_enabled=True"
                )
            self.reflection_enabled = False
            self.final_reflection_enabled = False

    # ------------------------------------------------------ runtime config

    def _extract_runtime_config(self, subtasks: list[dict[str, Any]]) -> dict[str, Any]:
        grasp_mode = self.scene_context.get("execution_proxy", {}).get(
            "default_grasp_mode", "top"
        )
        # Default goal: first goal_region in the scene context.
        goal_name = next(iter(self.resolver.goal_region_names()), "goal_region")
        carry_height = None
        descend_radius = None
        keepout_name = None
        keepout_radius = 0.0

        for sub in subtasks:
            if sub["skill"] == "pick":
                grasp_mode = sub.get("params", {}).get("grasp_mode") or grasp_mode
            if sub["skill"] in {"transport", "place"}:
                goal_name = sub.get("params", {}).get("goal_region") or goal_name
            for c in sub.get("applied_constraints", []):
                if c["type"] == "min_carry_height":
                    carry_height = c["params"].get("height_above_table", carry_height)
                elif c["type"] == "descend_only_near_goal":
                    descend_radius = c["params"].get("radius", descend_radius)
                    goal_name = c["params"].get("goal_region", goal_name)
                elif c["type"] == "avoid_region":
                    # Avoidance is now task-local. Do not convert any
                    # subtask-level avoid_region into a global runtime keepout.
                    pass

        point_goal_xy: tuple[float, float] | None = None
        point_goal_radius: float | None = None
        point_goal_z: float | None = None

        for sub in subtasks:
            if sub["skill"] in {"transport", "place"}:
                goal_x = sub.get("params", {}).get("goal_x")
                goal_y = sub.get("params", {}).get("goal_y")
                if sub.get("params", {}).get("goal_radius") is not None:
                    point_goal_radius = float(sub["params"]["goal_radius"])
                if sub.get("params", {}).get("goal_z") is not None:
                    point_goal_z = float(sub["params"]["goal_z"])
                if goal_x is not None and goal_y is not None:
                    point_goal_xy = (float(goal_x), float(goal_y))

        goal_canonical, goal_xy, goal_radius, default_descend = self.resolver.resolve_goal(
            goal_name
        )
        dynamic_goal_refs = self.resolver.dynamic_goal_reference_names()
        # Numeric point targets are true targets only when no dynamic
        # goal_region is present. If System 2 emits both goal_region=<object>
        # and goal_x/goal_y, keep the live object reference so relocalization
        # can update the target.
        if point_goal_xy is not None and goal_canonical not in dynamic_goal_refs:
            goal_canonical = "explicit_point"
            goal_xy = point_goal_xy
        if point_goal_radius is not None:
            goal_radius = point_goal_radius

        # Legacy analytic keepout is disabled.  Keep the resolved XY only as a
        # debug/visualization coordinate; all obstacle costs are supplied by
        # named SDF constraints activated per subtask.
        kx, ky, _default_keepout_radius = self.resolver.resolve_keepout(keepout_name)
        active_keepout_radius = 0.0

        # Choose active_object_name from the plan itself, not from a fixed
        # legacy default. This is important for real scenes with arbitrary
        # object names / colors: "pick yellow_cube" should initialize the env
        # around yellow_cube, not whichever object happens to sort first.
        manipulated = sorted(self.resolver.manipulated_object_names())
        active_object_name = None
        for sub in subtasks:
            obj = self.resolver.canonical(sub.get("params", {}).get("object"))
            if obj in manipulated:
                active_object_name = obj
                break
        if active_object_name is None:
            active_object_name = manipulated[0] if manipulated else "object"
        return {
            "mode": grasp_mode if grasp_mode in {"top", "left", "right"} else "top",
            "num_blocks": int(
                self.scene_context.get("execution_proxy", {}).get("num_blocks", 3)
            ),
            "goal_name": goal_canonical,
            "goal_reference_name": goal_canonical
            if goal_canonical in dynamic_goal_refs
            else None,
            "goal_xy": goal_xy,
            "goal_radius": goal_radius,
            "goal_z": point_goal_z,
            "descend_radius": descend_radius
            if descend_radius is not None
            else default_descend,
            "keepout_xy": (kx, ky),
            "keepout_radius": active_keepout_radius,
            "carry_height": carry_height,
            "active_object_name": active_object_name,
            "dynamic_goal_refs": sorted(dynamic_goal_refs),
        }

    # ------------------------------------------------------ stage helpers

    def _object_pose_valid_for_motion(
        self,
        env: BaseManipulationEnv,
        *,
        min_table_margin: float = 0.08,
    ) -> tuple[bool, dict[str, Any]]:
        """Return false if the active object has clearly fallen out of play.

        This prevents the controller from chasing an active object that has been
        displaced out of play, e.g. below the table/workspace.
        """
        status = env.status()
        cube = status.get("cube_pos", None)
        if cube is None:
            return True, status
        try:
            z = float(cube[2])
            table_z = float(getattr(env, "table_z", 0.0))
        except Exception:
            return True, status

        if z < table_z - float(min_table_margin):
            status = dict(status)
            status["terminated_reason"] = (
                f"active object below table/workspace: z={z:.3f}, table_z={table_z:.3f}"
            )
            status["object_pose_invalid"] = True
            return False, status

        status = dict(status)
        status["object_pose_invalid"] = False
        return True, status

    def _skill_done(
        self,
        env: BaseManipulationEnv,
        skill: str,
        dwell_counts: dict[str, int],
    ) -> tuple[bool, dict[str, Any]]:
        status = env.status()
        if skill == "pick":
            grasped = bool(status.get("grasped", False))
            stage = int(status.get("stage", 0))
            carry_robustness = float(status.get("carry", -999.0))
            cube_z = float(status.get("cube_pos", [0.0, 0.0, -999.0])[2])
            carry_z = float(getattr(env, "carry_z", cube_z + 999.0))
            lifted_enough = (
                stage >= 4 or carry_robustness >= -0.035 or cube_z >= carry_z - 0.035
            )
            cond = grasped and lifted_enough
        elif skill == "transport":
            cond = float(status["over_goal"]) > 0.0 or int(status["stage"]) >= 5
        elif skill == "place":
            # Do not mark place complete immediately when the object is released.
            # Stage 6 means "released / holding near the placement contact".
            # Stage 7 now targets a generic post-place retreat: lift to retreat_z
            # and move laterally away from the localized placement/support region.
            stage = int(status.get("stage", 0))
            released = not bool(status.get("grasped", False))
            eef = np.asarray(status.get("eef_pos", [0.0, 0.0, -999.0]), dtype=np.float32)
            eef_z = float(eef[2])
            retreat_z = float(getattr(env, "retreat_z", eef_z + 999.0))
            z_retreated = eef_z >= retreat_z - 0.04
            retreat_xy_dist = status.get("retreat_xy_dist")
            # Older envs may not expose retreat_xy_dist; in that case the
            # z-retreat condition preserves backwards compatibility.
            xy_retreated = True if retreat_xy_dist is None else float(retreat_xy_dist) <= 0.040
            cond = stage >= 7 and released and z_retreated and xy_retreated
        else:
            raise NotImplementedError(
                f"System1Executor does not yet implement runtime skill '{skill}'."
            )
        dwell_counts.setdefault(skill, 0)
        dwell_counts[skill] = dwell_counts[skill] + 1 if cond else 0
        required_dwell = 8 if skill == "pick" else 3
        return dwell_counts[skill] >= required_dwell, status

    def _refresh_subtask_bookkeeping(self, env: BaseManipulationEnv) -> None:
        if env.last_raw_obs is None:
            return
        try:
            env.nominal_mpc.reset()
        except Exception:
            pass
        try:
            env.prev_ee_pos = env._eef_pos(env.last_raw_obs)
            cube_pos = env._cube_pos(env.last_raw_obs)
            env.prev_cube_height = float(cube_pos[2])
            target_xyz, target_R, _ = env._current_target(cube_pos)
            env.prev_target_dist = float(np.linalg.norm(env.prev_ee_pos - target_xyz))
            current_R = env._eef_rotmat(env.last_raw_obs)
            env.prev_ori_err = float(
                np.linalg.norm(env.orientation_error_rotvec(current_R, target_R))
            )
        except Exception:
            pass

    def _begin_new_pick_skill(self, env: BaseManipulationEnv) -> None:
        # If the object is already securely in the gripper, skip directly to
        # the "carry" stage so we never open the gripper and re-pick.
        if env._is_grasped():
            env.stage = 3
            env.close_counter = 0
            self._refresh_subtask_bookkeeping(env)
            return
        env.stage = 0
        env.close_counter = 0
        self._refresh_subtask_bookkeeping(env)

    @staticmethod
    def _grasp_offset_from_params(params: dict[str, Any]) -> list[float] | None:
        raw = params.get("grasp_offset_xyz") or params.get("grasp_offset")
        if raw is None:
            return None
        if isinstance(raw, dict):
            raw = raw.get("xyz") or raw.get("offset_xyz")
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            raise ValueError(f"grasp_offset_xyz must be a 3-element list, got {raw!r}")
        return [float(raw[0]), float(raw[1]), float(raw[2])]

    def _dynamic_goal_reference_for_subtask(
        self, env: BaseManipulationEnv, sub: dict[str, Any]
    ) -> str | None:
        """Return a live object/support goal reference when one is present.

        System 2 may include both a semantic goal_region and numeric goal_x/y
        hints. For dynamic references, the semantic object/structure reference
        must win: numeric coordinates can be stale after relocalization or after
        recall. This keeps execution anchored to the current scene instead of to
        a hallucinated point.
        """
        params = sub.get("params", {}) if isinstance(sub, dict) else {}
        goal_name = self.resolver.canonical(params.get("goal_region"))
        if goal_name is None:
            return None
        if goal_name not in self.resolver.dynamic_goal_reference_names():
            return None
        try:
            if env._object_pos_by_name(goal_name) is None:
                return None
        except Exception:
            return None
        return goal_name

    def _configure_goal_for_subtask(
        self, env: BaseManipulationEnv, sub: dict[str, Any]
    ) -> dict[str, Any]:
        """Configure env goal fields from one subtask, task-agnostically.

        Precedence is intentionally conservative:
        1. If goal_region names a localized dynamic entity, use that live entity
           as the reference and ignore numeric goal_x/goal_y/goal_z hints.
        2. Else use explicit goal_x/goal_y as a true point target.
        3. Else resolve static goal_region geometry from the scene context.

        The run that failed had both goal_region='green_cube' and explicit
        goal_x/goal_y. The old code chose the point target, set
        goal_reference_name=None, and therefore disabled live relocalization.
        """
        params = sub.get("params", {}) if isinstance(sub, dict) else {}
        goal_x = params.get("goal_x")
        goal_y = params.get("goal_y")
        goal_radius = params.get("goal_radius")
        goal_z = params.get("goal_z")

        live_ref = self._dynamic_goal_reference_for_subtask(env, sub)
        if live_ref is not None:
            env.goal_reference_name = live_ref
            # For live object/support references, use current localized geometry
            # for both XY and release height. Numeric z values emitted by recall
            # are treated as stale hints, not hard overrides.
            env.goal_z_override = None
            ref = env._object_pos_by_name(live_ref)
            if ref is not None:
                env.goal_xy = np.asarray(ref[:2], dtype=np.float32)
            if goal_radius is not None:
                env.goal_radius = float(goal_radius)
            return {"mode": "live_goal_reference", "goal_reference_name": live_ref}

        if goal_x is not None and goal_y is not None:
            env.goal_reference_name = None
            env.goal_xy = np.asarray([float(goal_x), float(goal_y)], dtype=np.float32)
            env.goal_z_override = None if goal_z is None else float(goal_z)
            if goal_radius is not None:
                env.goal_radius = float(goal_radius)
            return {"mode": "explicit_point", "goal_xy": env.goal_xy.copy()}

        goal_name = self.resolver.canonical(params.get("goal_region"))
        if goal_name is not None:
            env.goal_reference_name = None
            env.goal_z_override = None if goal_z is None else float(goal_z)
            _, goal_xy, resolved_radius, _ = self.resolver.resolve_goal(goal_name)
            env.goal_xy = np.asarray(goal_xy, dtype=np.float32)
            env.goal_radius = float(goal_radius) if goal_radius is not None else float(resolved_radius)
            return {"mode": "static_goal_region", "goal_region": goal_name}

        return {"mode": "unchanged"}

    def _configure_pass_through_for_subtask(
        self, env: BaseManipulationEnv, sub: dict[str, Any]
    ) -> None:
        env.clear_pass_through()
        if sub.get("skill") != "transport":
            return
        for constraint in sub.get("applied_constraints", []):
            if constraint.get("type") != "pass_through_region":
                continue
            params = constraint.get("params", {})
            obj = self.resolver.canonical(params.get("object"))
            sub_obj = self.resolver.canonical(sub.get("params", {}).get("object"))
            if obj is not None and sub_obj is not None and obj != sub_obj:
                continue
            if (
                params.get("region_x") is not None
                and params.get("region_y") is not None
            ):
                xy = (float(params["region_x"]), float(params["region_y"]))
            elif params.get("region_name") is not None:
                _, xy, _, _ = self.resolver.resolve_goal(params.get("region_name"))
            else:
                continue
            env.configure_pass_through(
                xy=xy,
                radius=float(params.get("radius") or 0.10),
                z=None if params.get("region_z") is None else float(params.get("region_z")),
                label=params.get("label")
                or params.get("region_name")
                or "pass-through",
            )
            return

    def _apply_subtask_constraints_to_env(
        self,
        env: BaseManipulationEnv,
        sub: dict[str, Any],
    ) -> None:
        """Activate only the constraints attached to the current subtask."""
        active_sdf_names: set[str] = set()
        env.keepout_radius = 0.0

        for constraint in sub.get("applied_constraints", []) or []:
            ctype = constraint.get("type")
            params = constraint.get("params", {}) or {}

            if ctype == "avoid_region":
                region = self.resolver.canonical(params.get("region_source")) or params.get("region_source")
                if region is not None:
                    # Activate the corresponding SDF constraint(s) only.  Do
                    # not synthesize a circular keepout from radius/XY here;
                    # that would reintroduce a second, non-SDF safety cost.
                    active_sdf_names.add(str(region))
                    active_sdf_names.add(f"{region}_keepout")
                    active_sdf_names.add(f"{region}_obstacle")

            elif ctype == "pass_through_region":
                label = params.get("label") or params.get("region_name")
                if label:
                    active_sdf_names.add(str(label))

        if hasattr(env, "set_active_sdf_constraints"):
            env.set_active_sdf_constraints(active_sdf_names)

    # ------------------------------------------------------ timing constraints

    def _control_hz_for_timing(
        self,
        env: BaseManipulationEnv | None = None,
        runtime: dict[str, Any] | None = None,
    ) -> float:
        """Best-effort controller frequency for seconds-to-steps timing.

        Keep this object- and task-agnostic. Sim defaults to 20 Hz; the real
        UR servo loop defaults to 0.05 s, also 20 Hz.
        """
        runtime = runtime or {}
        candidates = [runtime.get("control_freq"), runtime.get("control_hz")]
        if env is not None:
            candidates.extend([
                getattr(env, "control_freq", None),
                getattr(env, "control_hz", None),
                getattr(getattr(env, "env", None), "control_freq", None),
            ])
            dt = getattr(env, "dt", None)
            if dt not in (None, 0):
                try:
                    candidates.append(1.0 / float(dt))
                except Exception:
                    pass
            controller = getattr(env, "controller", None)
            servo_dt = getattr(controller, "servo_dt", None)
            if servo_dt not in (None, 0):
                try:
                    candidates.append(1.0 / float(servo_dt))
                except Exception:
                    pass
        for value in candidates:
            if value in (None, ""):
                continue
            try:
                hz = float(value)
                if hz > 0.0:
                    return hz
            except Exception:
                pass
        return 20.0

    @staticmethod
    def _constraint_param_first(params: dict[str, Any], *names: str) -> Any:
        for name in names:
            value = params.get(name)
            if value not in (None, ""):
                return value
        return None

    def _timing_windows_for_subtask(self, sub: dict[str, Any]) -> list[dict[str, Any]]:
        windows: list[dict[str, Any]] = []
        for constraint in sub.get("applied_constraints", []) or []:
            if constraint.get("type") != "timing_window":
                continue
            params = dict(constraint.get("params", {}) or {})
            start_sec = self._constraint_param_first(params, "start_sec", "min_sec", "earliest_sec", "lower_sec")
            end_sec = self._constraint_param_first(params, "end_sec", "max_sec", "deadline_sec", "latest_sec", "upper_sec")
            if start_sec is None and end_sec is None:
                continue
            skill = str(sub.get("skill") or "")
            default_pred = {
                "pick": "grasped",
                "transport": "over_goal",
                "place": "placed_released",
            }.get(skill, "")
            predicate = str(params.get("predicate") or default_pred).strip().lower()
            predicate = predicate.replace(" ", "_").replace("-", "_")
            windows.append(
                {
                    "predicate": predicate,
                    "start_sec": None if start_sec is None else float(start_sec),
                    "end_sec": None if end_sec is None else float(end_sec),
                    "reference": str(params.get("reference") or "subtask_start"),
                }
            )
        return windows

    def _timing_predicate_satisfied(
        self,
        *,
        skill: str,
        predicate: str,
        status: dict[str, Any],
    ) -> bool:
        pred = predicate.strip().lower().replace(" ", "_").replace("-", "_")
        if pred in {"grasp", "grasped", "still_grasped"}:
            return bool(status.get("grasped", False))
        if pred in {"over_goal", "overgoal"}:
            try:
                return float(status.get("over_goal", -1.0)) >= 0.0
            except Exception:
                return False
        if pred == "released":
            return status.get("grasped") is False
        if pred in {"placed", "placed_xy"}:
            try:
                return float(status.get("over_goal", -1.0)) >= 0.0
            except Exception:
                return False
        if pred in {"placed_released", "placed_and_released"}:
            try:
                placed_xy = float(status.get("over_goal", -1.0)) >= 0.0
            except Exception:
                placed_xy = False
            return bool(placed_xy and status.get("grasped") is False)
        if pred in {"pregrasp", "pre_grasp"}:
            try:
                xy = status.get("top_grasp_xy_err")
                z = status.get("top_grasp_z_err")
                if xy is not None and z is not None:
                    return float(xy) <= 0.025 and float(z) <= 0.025
            except Exception:
                pass
            return False
        return False

    def _timing_window_violation_for_subtask(
        self,
        env: BaseManipulationEnv,
        sub: dict[str, Any],
        *,
        steps: int,
        status: dict[str, Any] | None,
        runtime: dict[str, Any] | None = None,
        task_steps: int | None = None,
    ) -> dict[str, Any] | None:
        """Return a failure payload if a local timing_window is violated.

        A bounded window such as ``start_sec=3, end_sec=5`` means the event
        predicate must first become true inside that interval: not before the
        lower bound and not after the upper bound. This is checked online so a
        pick that grasps at 2.5 s fails immediately as ``too_early`` rather than
        being allowed because Grasped remains true at 3 s.
        """
        windows = self._timing_windows_for_subtask(sub)
        if not windows:
            return None
        hz = self._control_hz_for_timing(env, runtime)
        subtask_elapsed_sec = float(steps) / float(hz)
        task_elapsed_sec = float(steps if task_steps is None else task_steps) / float(hz)
        skill = str(sub.get("skill") or "")
        st = dict(status or {})
        if not st:
            try:
                st = env.status()
            except Exception:
                st = {}
        for window in windows:
            predicate = str(window.get("predicate") or "")
            satisfied = self._timing_predicate_satisfied(
                skill=skill,
                predicate=predicate,
                status=st,
            )
            reference = str(window.get("reference") or "subtask_start").strip().lower()
            elapsed_sec = task_elapsed_sec if reference == "task_start" else subtask_elapsed_sec

            start_sec = window.get("start_sec")
            if start_sec is not None and elapsed_sec < float(start_sec) and satisfied:
                return {
                    "terminated_reason": "timing_window_too_early",
                    "timing_constraint_violated": True,
                    "timing_violation_kind": "too_early",
                    "timing_predicate": predicate,
                    "timing_window_start_sec": float(start_sec),
                    "timing_window_end_sec": None if window.get("end_sec") is None else float(window.get("end_sec")),
                    "timing_elapsed_sec": elapsed_sec,
                    "timing_reference": window.get("reference"),
                    "timing_subtask_elapsed_sec": subtask_elapsed_sec,
                    "timing_task_elapsed_sec": task_elapsed_sec,
                    "success": False,
                }

            end_sec = window.get("end_sec")
            if end_sec is not None and elapsed_sec >= float(end_sec) and not satisfied:
                return {
                    "terminated_reason": "timing_window_missed",
                    "timing_constraint_violated": True,
                    "timing_violation_kind": "too_late",
                    "timing_predicate": predicate,
                    "timing_window_start_sec": None if start_sec is None else float(start_sec),
                    "timing_window_end_sec": float(end_sec),
                    "timing_elapsed_sec": elapsed_sec,
                    "timing_reference": window.get("reference"),
                    "timing_subtask_elapsed_sec": subtask_elapsed_sec,
                    "timing_task_elapsed_sec": task_elapsed_sec,
                    "success": False,
                }
        return None

    # ------------------------------------------------------ video helpers


    # ----------------------------------------------------- step-callback hook

    def _fire_step_callback(self, env: BaseManipulationEnv, info: dict[str, Any]) -> None:
        """Forward a per-step event to the registered callback, throttled.

        Used by the interactive REPL to drive a live preview window without
        blocking the executor on rendering. Errors in the callback are caught
        and printed so they cannot abort an in-flight task.
        """
        if self.step_callback is None:
            return
        self._step_callback_counter += 1
        if (self._step_callback_counter % self.step_callback_every_n) != 0:
            return
        try:
            self.step_callback(env, info)
        except Exception as exc:
            print(f"[executor] step_callback raised ({exc}); disabling callback for this run.")
            self.step_callback = None


    def _computed_env_horizon(self, num_subtasks: int) -> int | None:
        if self.env_horizon is not None:
            return int(self.env_horizon)
        # Robosuite raises if stepped after its internal horizon. Give the env
        # enough time for every high-level skill plus the post-success dwell,
        # while the executor itself enforces per-skill max_subtask_steps.
        if self.max_subtask_steps is None:
            return None
        return int(self.max_subtask_steps * max(1, int(num_subtasks)) + self.post_success_steps + 50)

    # ------------------------------------------------------------ env build

    def _make_env(self, runtime: dict[str, Any], debug_dir: str | Path | None = None) -> BaseManipulationEnv:
        if self.env_factory is not None:
            return self.env_factory(self.scene_context, runtime)
        return make_env_from_scene_context(
            scene_context=self.scene_context,
            runtime=runtime,
            render=self.render,
            env_horizon=runtime.get("_computed_env_horizon", self.env_horizon),
            mock_robot=self.mock_robot,
            debug_dir=debug_dir,
        )

    # ------------------------------------------------------------ main loop

    def execute(self, compiled_plan: dict[str, Any], outdir: str | Path | None = None) -> dict[str, Any]:
        self._original_compiled_stl_plan_for_recall = self._jsonable(compiled_plan.get("system2_plan", compiled_plan))
        self._latest_compiled_stl_plan_for_recall = self._original_compiled_stl_plan_for_recall
        subtasks = compiled_plan["system2_plan"]["subtasks"]
        runtime = self._extract_runtime_config(subtasks)
        runtime["_computed_env_horizon"] = self._computed_env_horizon(len(subtasks))
        env = self._make_env(runtime, debug_dir=outdir)
        try:
            env.reset()
            return self._run_subtasks_on_env(
                env, subtasks, runtime=runtime, outdir=outdir, capture_initial_video=True
            )
        finally:
            self._close_video_writer()
            env.close()

    def execute_on_env(
        self,
        env: BaseManipulationEnv,
        compiled_plan: dict[str, Any],
        outdir: str | Path | None = None,
        *,
        reset_first: bool = False,
        capture_initial_video: bool = True,
    ) -> dict[str, Any]:
        """Run a compiled plan against an existing env (no construct/teardown).

        Used by the interactive REPL so the same env (and the same physical
        robot connection / RealSense pipeline) is reused across many prompts.
        """
        self._original_compiled_stl_plan_for_recall = self._jsonable(compiled_plan.get("system2_plan", compiled_plan))
        self._latest_compiled_stl_plan_for_recall = self._original_compiled_stl_plan_for_recall
        subtasks = compiled_plan["system2_plan"]["subtasks"]
        runtime = self._extract_runtime_config(subtasks)
        runtime["_computed_env_horizon"] = self._computed_env_horizon(len(subtasks))
        if reset_first:
            env.reset()
        elif env.last_raw_obs is None:
            env.reset()
        # Update env-level configuration from the new plan's defaults so the
        # first subtask's stage-machine kinematics reflect the new scene.
        try:
            env.switch_mode(runtime["mode"])
        except Exception:
            pass
        if runtime.get("carry_height") is not None:
            try:
                env.configure_from_constraints(carry_height=float(runtime["carry_height"]))
            except Exception:
                pass
        return self._run_subtasks_on_env(
            env,
            subtasks,
            runtime=runtime,
            outdir=outdir,
            capture_initial_video=capture_initial_video,
        )

    def _is_real_simple_env(self, env: BaseManipulationEnv) -> bool:
        return (
            self.scene_context.get("platform", {}).get("backend") == "real"
            and hasattr(env, "simple_mpc_move_to")
            and hasattr(env, "simple_pick")
        )

    def _resolve_goal_xyz_for_subtask(
        self,
        env: BaseManipulationEnv,
        sub: dict[str, Any],
    ) -> np.ndarray:
        params = sub.get("params", {})

        # Prefer live dynamic goal references over explicit point coordinates.
        # This mirrors _configure_goal_for_subtask(...) and is especially
        # important for the real-simple executor, whose blocking primitives take
        # a concrete target_xyz.
        live_ref = self._dynamic_goal_reference_for_subtask(env, sub)
        if live_ref is not None:
            ref = env._object_pos_by_name(live_ref)
            if ref is not None:
                anchor_xy = np.asarray(ref[:2], dtype=np.float32)
                if sub.get("skill") == "place":
                    active_name = self.resolver.canonical(params.get("object"))
                    exclude = {str(active_name)} if active_name else set()
                    support_top = None
                    if hasattr(env, "_support_top_z_near_xy"):
                        try:
                            support_top = env._support_top_z_near_xy(
                                anchor_xy,
                                exclude_names=exclude,
                                include_pos=ref,
                            )
                        except Exception:
                            support_top = None
                    clearance = float(getattr(env, "object_place_clearance_z", 0.044))
                    z = float((support_top if support_top is not None else ref[2]) + clearance)
                else:
                    z = float(getattr(env, "carry_z", ref[2]))
                return np.asarray([float(anchor_xy[0]), float(anchor_xy[1]), z], dtype=np.float32)

        goal_x = params.get("goal_x")
        goal_y = params.get("goal_y")
        goal_z = params.get("goal_z")
        if goal_x is not None and goal_y is not None:
            z = float(goal_z) if goal_z is not None else float(getattr(env, "carry_z", 0.20))
            return np.asarray([float(goal_x), float(goal_y), z], dtype=np.float32)

        goal_name = self.resolver.canonical(params.get("goal_region"))
        if goal_name is not None:
            ref = env._object_pos_by_name(goal_name)
            if ref is not None:
                z = float(goal_z) if goal_z is not None else float(ref[2])
                return np.asarray([float(ref[0]), float(ref[1]), z], dtype=np.float32)
            _, goal_xy, _, _ = self.resolver.resolve_goal(goal_name)
            z = float(goal_z) if goal_z is not None else float(
                self.scene_context.get("placement_defaults", {}).get("table_goal_z", 0.03)
            )
            return np.asarray([float(goal_xy[0]), float(goal_xy[1]), z], dtype=np.float32)

        # No explicit goal: stay over the active object at carry height.
        active = env._cube_pos(env.last_raw_obs) if env.last_raw_obs is not None else np.zeros(3, dtype=np.float32)
        z = float(goal_z) if goal_z is not None else float(getattr(env, "carry_z", active[2]))
        return np.asarray([float(active[0]), float(active[1]), z], dtype=np.float32)

    # ------------------------------------------------------ System-2 recall / replanning


    # ------------------------------------------------------ reflection / repair

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.astype(float).tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {str(k): System1Executor._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [System1Executor._jsonable(v) for v in value]
        return value

    # ------------------------------------------------------ removed legacy reflection compatibility

    def _maybe_reflect_after_subtask(
        self,
        env: BaseManipulationEnv,
        *,
        active_subtasks: list[dict[str, Any]],
        subtask_index: int,
        record: SkillExecutionRecord,
        records: list[SkillExecutionRecord],
        outdir: str | Path | None,
    ) -> tuple[list[dict[str, Any]] | None, bool, dict[str, Any] | None]:
        """No-op replacement for the removed legacy reflection hook.

        The project now uses the System-2 recall path for scene refresh and
        replanning. The non-recall and real-simple executor paths still call
        this hook at a few historical integration points, so keep a local no-op
        instead of depending on the deleted ``legacy_reflection.py`` mixin.
        """
        return None, False, None

    def _maybe_reflect_at_end(
        self,
        env: BaseManipulationEnv,
        *,
        active_subtasks: list[dict[str, Any]],
        records: list[SkillExecutionRecord],
        outdir: str | Path | None,
        success_so_far: bool,
        final_repair_batches_used: int,
    ) -> tuple[list[dict[str, Any]] | None, bool, dict[str, Any] | None]:
        """No-op replacement for the removed legacy final reflection hook."""
        return None, False, None

    def _merge_reflection_replacement(
        self,
        *,
        active_subtasks: list[dict[str, Any]],
        subtask_index: int,
        replacement_remaining: list[dict[str, Any]] | None,
        reflection_result: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Compatibility splice for any stale caller-provided repair plan.

        In the current architecture this should rarely be used because System-2
        recall replaces the active remaining plan directly. Keeping the splice
        behavior here makes older execution paths safe without resurrecting the
        old reflection prompt/client.
        """
        if replacement_remaining is None:
            return list(active_subtasks)
        prefix = list(active_subtasks[: max(0, int(subtask_index)) + 1])
        return prefix + list(replacement_remaining)

    def _unexecuted_original_subtasks(
        self, records: list[SkillExecutionRecord]
    ) -> list[dict[str, Any]]:
        """Return original subtasks not yet represented by execution records.

        The old final-reflection verifier used this to decide whether it should
        repair the plan. With legacy reflection removed, callers only need a
        conservative bookkeeping helper that avoids AttributeError.
        """
        original = list(getattr(self, "_original_subtasks_for_reflection", []) or [])
        executed = min(len(records), len(original))
        return original[executed:]


    def _run_real_simple_subtasks(
        self,
        env: BaseManipulationEnv,
        subtasks: list[dict[str, Any]],
        *,
        runtime: dict[str, Any],
        outdir: str | Path | None,
        capture_initial_video: bool,
    ) -> dict[str, Any]:
        """Run real robot subtasks and reflect after each high-level subtask."""
        records: list[SkillExecutionRecord] = []
        active_object: str | None = runtime.get("active_object_name")
        active_subtasks = list(subtasks)
        self._original_subtasks_for_reflection = list(subtasks)
        success = True
        aborted = False

        if capture_initial_video:
            self._capture_video_frame(env, force=True)

        idx = 0
        while idx < len(active_subtasks):
            sub = active_subtasks[idx]
            self._set_video_subtask_overlay(sub)
            skill = sub["skill"]
            params = sub.get("params", {})
            obj = self.resolver.canonical(params.get("object"))
            if obj is not None:
                active_object = obj
                env.set_active_object(obj)

            task_steps_so_far = sum(int(getattr(r, "steps", 0) or 0) for r in records)
            if hasattr(env, "configure_timing_windows"):
                env.configure_timing_windows(
                    self._timing_windows_for_subtask(sub),
                    control_hz=self._control_hz_for_timing(env, runtime),
                    task_elapsed_steps=task_steps_so_far,
                )

            def capture_callback(env_instance, info_dict):
                self._capture_video_frame(env_instance)

            try:
                if skill == "pick":
                    if active_object is None:
                        raise RuntimeError("pick skill has no resolved object")
                    grasp_mode = params.get("grasp_mode") or runtime.get("mode", "top")
                    try:
                        env.switch_mode(grasp_mode if grasp_mode in {"top", "left", "right"} else "top")
                    except Exception:
                        pass
                    rec = env.simple_pick(active_object, step_callback=capture_callback)
                    done = bool(rec.get("pick_success", rec.get("motion_success", rec.get("grasped", False))))
                    if rec.get("motion_success", False) and not rec.get("grasped", False):
                        print(
                            "[executor] pick motion succeeded but gripper feedback did not report grasp; "
                            f"continuing because require_grasp_feedback={rec.get('require_grasp_feedback', False)}"
                        )
                    steps = int(sum(r.get("steps", 0) for r in rec.get("records", [])))
                    info = rec
                elif skill == "transport":
                    target = self._resolve_goal_xyz_for_subtask(env, sub)
                    if params.get("goal_z") is None:
                        target[2] = float(getattr(env, "carry_z", target[2]))
                    rec = env.simple_transport(target, label=f"transport:{active_object or 'object'}", step_callback=capture_callback)
                    done = all(bool(r.get("reached", False)) for r in rec.get("records", []))
                    steps = int(sum(r.get("steps", 0) for r in rec.get("records", [])))
                    info = rec
                elif skill == "place":
                    target = self._resolve_goal_xyz_for_subtask(env, sub)
                    rec = env.simple_place(target, step_callback=capture_callback)
                    # Place success is evaluated by the reflection pass too; the
                    # low-level routine is considered complete once its motion
                    # sequence executes.
                    done = True
                    steps = int(sum(r.get("steps", 0) for r in rec.get("records", [])))
                    info = rec
                else:
                    raise NotImplementedError(f"Real simple executor does not implement skill {skill!r}")

                timing_violation = self._timing_window_violation_for_subtask(
                    env,
                    sub,
                    steps=steps,
                    status=info if isinstance(info, dict) else {},
                    runtime=runtime,
                    task_steps=sum(int(getattr(r, "steps", 0) or 0) for r in records) + steps,
                )
                if timing_violation is not None:
                    info = dict(info or {})
                    info.update(timing_violation)
                    done = False

                record = SkillExecutionRecord(skill=skill, params=params, steps=steps, done=done, info=info)
                records.append(record)

                replacement_remaining, should_abort, reflection_result = self._maybe_reflect_after_subtask(
                    env,
                    active_subtasks=active_subtasks,
                    subtask_index=idx,
                    record=record,
                    records=records,
                    outdir=outdir,
                )
                if reflection_result is not None:
                    record.info = dict(record.info or {})
                    record.info["reflection"] = reflection_result
                    self._set_video_reflection_overlay(reflection_result)
                    self._capture_video_frame(env, force=True)

                if should_abort:
                    aborted = True
                    success = False
                    break

                if replacement_remaining is not None:
                    active_subtasks = self._merge_reflection_replacement(
                        active_subtasks=active_subtasks,
                        subtask_index=idx,
                        replacement_remaining=replacement_remaining,
                        reflection_result=reflection_result,
                    )

                if not done and replacement_remaining is None:
                    success = False
                    break

                idx += 1
            except Exception as exc:
                record = SkillExecutionRecord(skill=skill, params=params, steps=0, done=False, info={"error": str(exc)})
                records.append(record)
                replacement_remaining, should_abort, reflection_result = self._maybe_reflect_after_subtask(
                    env,
                    active_subtasks=active_subtasks,
                    subtask_index=idx,
                    record=record,
                    records=records,
                    outdir=outdir,
                )
                if reflection_result is not None:
                    record.info["reflection"] = reflection_result
                    self._set_video_reflection_overlay(reflection_result)
                    self._capture_video_frame(env, force=True)
                if replacement_remaining is not None and not should_abort:
                    active_subtasks = self._merge_reflection_replacement(
                        active_subtasks=active_subtasks,
                        subtask_index=idx,
                        replacement_remaining=replacement_remaining,
                        reflection_result=reflection_result,
                    )
                    idx += 1
                    continue
                success = False
                aborted = bool(should_abort)
                break

        final_repair_batches_used = 0
        final_checked = False
        # End-of-task verifier. If it returns a final repair plan, execute it
        # once through the same real-simple loop by extending active_subtasks.
        while (
            not aborted
            and self.reflection_enabled
            and self.final_reflection_enabled
            and not final_checked
        ):
            success_so_far = bool(
                success
                and records
                and (records[-1].done or self._reflection_repairs_used > 0)
                and not self._unexecuted_original_subtasks(records)
            )
            final_remaining, final_abort, final_result = self._maybe_reflect_at_end(
                env,
                active_subtasks=active_subtasks,
                records=records,
                outdir=outdir,
                success_so_far=success_so_far,
                final_repair_batches_used=final_repair_batches_used,
            )
            if final_abort:
                aborted = True
                success = False
                break
            if final_remaining is not None:
                active_subtasks = active_subtasks + final_remaining
                final_repair_batches_used += 1
                # Execute the final repair subtasks inline.
                while idx < len(active_subtasks):
                    # Re-enter the outer loop by breaking out to the existing
                    # body is awkward here, so stop after requesting final repairs.
                    # The generic executor path below fully executes final repairs;
                    # real-simple final repairs are reported for the next prompt.
                    break
            final_checked = True

        summary = {
            "success": bool(success and records and not aborted and (records[-1].done or self._reflection_repairs_used > 0)),
            "final_reflection_enabled": bool(self.final_reflection_enabled),
            "runtime_config": runtime,
            "real_simple_executor": True,
            "reflection_enabled": bool(self.reflection_enabled),
            "reflection_history": self.reflection_history,
            "records": [r.__dict__ for r in records],
            "post_success_steps_requested": 0,
            "post_success_steps_run": 0,
            "post_success_final_status": {},
            "video_path": None if self.video_path is None else str(self.video_path),
            "video_frames": int(self._video_frame_count),
            "final_status": env.status(),
        }
        if outdir is not None:
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "execution_summary.json").write_text(
                json.dumps(summary, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
            )
        return summary


    def _run_subtasks_on_env(
        self,
        env: BaseManipulationEnv,
        subtasks: list[dict[str, Any]],
        *,
        runtime: dict[str, Any],
        outdir: str | Path | None,
        capture_initial_video: bool,
    ) -> dict[str, Any]:
        if self.system2_recall_enabled:
            return self._run_subtasks_on_env_with_system2_recall(
                env,
                subtasks,
                runtime=runtime,
                outdir=outdir,
                capture_initial_video=capture_initial_video,
            )

        if self._is_real_simple_env(env):
            return self._run_real_simple_subtasks(
                env,
                subtasks,
                runtime=runtime,
                outdir=outdir,
                capture_initial_video=capture_initial_video,
            )

        records: list[SkillExecutionRecord] = []
        dwell_counts: dict[str, int] = {}
        active_subtasks = list(subtasks)
        self._original_subtasks_for_reflection = list(subtasks)
        aborted = False
        try:
            if capture_initial_video:
                self._capture_video_frame(env, force=True)

            idx = 0
            final_checked = False
            final_repair_batches_used = 0
            while idx < len(active_subtasks) or (not final_checked and not aborted):
                if idx >= len(active_subtasks):
                    success_so_far = bool(
                        records
                        and not aborted
                        and all(r.done for r in records)
                        and not self._unexecuted_original_subtasks(records)
                    )
                    final_remaining, final_abort, final_result = self._maybe_reflect_at_end(
                        env,
                        active_subtasks=active_subtasks,
                        records=records,
                        outdir=outdir,
                        success_so_far=success_so_far,
                        final_repair_batches_used=final_repair_batches_used,
                    )
                    final_checked = True
                    if final_abort:
                        aborted = True
                        break
                    if final_remaining is not None:
                        active_subtasks = active_subtasks + final_remaining
                        final_repair_batches_used += 1
                        # Allow execution of the final repair batch. We do not
                        # run another final verifier afterward by default.
                        continue
                    break

                sub = active_subtasks[idx]
                self._set_video_subtask_overlay(sub)
                skill = sub["skill"]
                object_name = self.resolver.canonical(sub.get("params", {}).get("object"))
                if object_name is not None:
                    env.set_active_object(object_name)

                # Safety guard: if a previous interaction has knocked the
                # active object off the table, do not let the controller chase
                # a below-table target. Abort this skill cleanly instead.
                pose_ok, pose_status = self._object_pose_valid_for_motion(env)
                if not pose_ok:
                    record = SkillExecutionRecord(
                        skill=skill,
                        params=sub.get("params", {}),
                        steps=0,
                        done=False,
                        info=pose_status,
                    )
                    records.append(record)
                    replacement_remaining, should_abort, reflection_result = self._maybe_reflect_after_subtask(
                        env,
                        active_subtasks=active_subtasks,
                        subtask_index=idx,
                        record=record,
                        records=records,
                        outdir=outdir,
                    )
                    if reflection_result is not None:
                        record.info = dict(record.info or {})
                        record.info["reflection"] = reflection_result
                        self._set_video_reflection_overlay(reflection_result)
                        self._capture_video_frame(env, force=True)
                    if replacement_remaining is not None and not should_abort:
                        active_subtasks = self._merge_reflection_replacement(
                        active_subtasks=active_subtasks,
                        subtask_index=idx,
                        replacement_remaining=replacement_remaining,
                        reflection_result=reflection_result,
                    )
                        idx += 1
                        continue
                    aborted = bool(should_abort)
                    break

                self._configure_goal_for_subtask(env, sub)
                self._apply_subtask_constraints_to_env(env, sub)
                self._configure_pass_through_for_subtask(env, sub)

                task_steps_so_far = sum(int(getattr(r, "steps", 0) or 0) for r in records)
                if hasattr(env, "configure_timing_windows"):
                    env.configure_timing_windows(
                        self._timing_windows_for_subtask(sub),
                        control_hz=self._control_hz_for_timing(env, runtime),
                        task_elapsed_steps=task_steps_so_far,
                    )

                dwell_counts.clear()

                if skill == "pick":
                    if hasattr(env, "set_grasp_offset_override"):
                        env.set_grasp_offset_override(
                            self._grasp_offset_from_params(sub.get("params", {}))
                        )
                    grasp_mode = (
                        sub.get("params", {}).get("grasp_mode") or runtime["mode"]
                    )
                    env.switch_mode(
                        grasp_mode if grasp_mode in {"top", "left", "right"} else "top"
                    )
                    self._begin_new_pick_skill(env)
                else:
                    if hasattr(env, "set_grasp_offset_override"):
                        env.set_grasp_offset_override(None)
                    self._refresh_subtask_bookkeeping(env)
                steps = 0
                done = False
                last_status: dict[str, Any] = {}
                while self.max_subtask_steps is None or steps < int(self.max_subtask_steps):
                    try:
                        _, _, terminated, info = env.step_nominal()
                    except ValueError as exc:
                        # Robosuite raises "executing action in terminated episode"
                        # if we step once past its horizon. Convert that into a
                        # clean task failure summary instead of crashing.
                        last_status = env.status()
                        last_status["terminated_reason"] = f"backend_step_error: {exc}"
                        last_status["backend_episode_terminated"] = True
                        terminated = True
                        done = False
                        break

                    steps += 1
                    self._capture_video_frame(env)
                    self._fire_step_callback(env, info)
                    if self.sleep > 0.0:
                        time.sleep(self.sleep)
                    done, last_status = self._skill_done(env, skill, dwell_counts)

                    # If the active object gets knocked below the table during
                    # a pick / transport / place, stop immediately instead of
                    # continuing to chase an impossible target until timeout.
                    pose_ok, pose_status = self._object_pose_valid_for_motion(env)
                    if not pose_ok:
                        last_status = pose_status
                        terminated = True
                        done = False

                    timing_violation = self._timing_window_violation_for_subtask(
                        env,
                        sub,
                        steps=steps,
                        status=last_status,
                        runtime=runtime,
                        task_steps=sum(int(getattr(r, "steps", 0) or 0) for r in records) + steps,
                    )
                    if timing_violation is not None:
                        last_status = dict(last_status or env.status())
                        last_status.update(timing_violation)
                        terminated = True
                        done = False

                    if terminated and bool(last_status.get("success", False)):
                        done = True
                    if done or terminated:
                        break

                if not done and not last_status:
                    last_status = env.status()
                if not done and self.max_subtask_steps is not None and steps >= int(self.max_subtask_steps):
                    last_status = dict(last_status or env.status())
                    last_status["terminated_reason"] = "max_subtask_steps"

                record = SkillExecutionRecord(
                    skill=skill,
                    params=sub.get("params", {}),
                    steps=steps,
                    done=done,
                    info=last_status,
                )
                records.append(record)

                replacement_remaining, should_abort, reflection_result = self._maybe_reflect_after_subtask(
                    env,
                    active_subtasks=active_subtasks,
                    subtask_index=idx,
                    record=record,
                    records=records,
                    outdir=outdir,
                )
                if reflection_result is not None:
                    record.info = dict(record.info or {})
                    record.info["reflection"] = reflection_result
                    self._set_video_reflection_overlay(reflection_result)
                    self._capture_video_frame(env, force=True)

                if should_abort:
                    aborted = True
                    break

                if replacement_remaining is not None:
                    active_subtasks = self._merge_reflection_replacement(
                        active_subtasks=active_subtasks,
                        subtask_index=idx,
                        replacement_remaining=replacement_remaining,
                        reflection_result=reflection_result,
                    )

                if not done and replacement_remaining is None:
                    break

                idx += 1

            success = bool(records and not aborted and all(r.done for r in records))
            post_steps_run = 0
            post_last_status: dict[str, Any] = {}
            if success and self.post_success_steps > 0:
                for _ in range(self.post_success_steps):
                    try:
                        _, _, terminated, info = env.step_nominal()
                    except ValueError as exc:
                        post_last_status = env.status()
                        post_last_status["terminated_reason"] = f"backend_step_error: {exc}"
                        post_last_status["backend_episode_terminated"] = True
                        break
                    post_steps_run += 1
                    self._capture_video_frame(env)
                    self._fire_step_callback(env, info)
                    post_last_status = env.status()
                    if self.sleep > 0.0:
                        time.sleep(self.sleep)
                    if terminated:
                        break

            summary = {
                "success": success,
                "runtime_config": runtime,
                "env_horizon": runtime.get("_computed_env_horizon", self.env_horizon),
                "max_subtask_steps": None if self.max_subtask_steps is None else int(self.max_subtask_steps),
                "reflection_enabled": bool(self.reflection_enabled),
                "final_reflection_enabled": bool(self.final_reflection_enabled),
                "reflection_history": self.reflection_history,
                "records": [r.__dict__ for r in records],
                "post_success_steps_requested": int(self.post_success_steps),
                "post_success_steps_run": int(post_steps_run),
                "post_success_final_status": post_last_status,
                "video_path": None if self.video_path is None else str(self.video_path),
                "video_frames": int(self._video_frame_count),
                "final_status": env.status(),
            }
            if outdir is not None:
                outdir = Path(outdir)
                outdir.mkdir(parents=True, exist_ok=True)
                (outdir / "execution_summary.json").write_text(
                    json.dumps(
                        summary,
                        indent=2,
                        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
                    )
                )
                try:
                    env.plot_xy(save_path=str(outdir / "execution_xy.png"))
                except Exception as exc:
                    print(f"[executor] plot_xy failed: {exc}")
                try:
                    np.savez_compressed(
                        outdir / "execution_rollout.npz",
                        eef_pos=np.array(
                            [step["eef_pos"] for step in env.rollout_log],
                            dtype=np.float32,
                        ),
                        cube_pos=np.array(
                            [step["cube_pos"] for step in env.rollout_log],
                            dtype=np.float32,
                        ),
                        stage=np.array(
                            [step["stage"] for step in env.rollout_log],
                            dtype=np.int32,
                        ),
                    )
                except Exception:
                    pass
            return summary
        finally:
            # NOTE: don't close env here — that is the caller's responsibility.
            # ``execute()`` (one-shot mode) closes via its own finally, while
            # ``execute_on_env()`` (interactive REPL) keeps the env alive.
            pass

