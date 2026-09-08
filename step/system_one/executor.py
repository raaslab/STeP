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
        self.step_callback: Optional[StepCallback] = step_callback
        self.step_callback_every_n: int = max(1, int(step_callback_every_n))
        self._step_callback_counter: int = 0

        # LIBERO-only build: reflection is off by default, enable explicitly
        # with execution_proxy.reflection_enabled=true.
        self.reflection_enabled: bool = bool(exec_proxy.get("reflection_enabled", False))
        self.reflection_scene_update: str = str(
            exec_proxy.get("reflection_scene_update", "none")
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

        point_goal_xy: tuple[float, float] | None = None
        point_goal_radius: float | None = None
        point_goal_z: float | None = None

        def _apply_goal_region(raw: Any) -> None:
            nonlocal goal_name, point_goal_xy, point_goal_radius, point_goal_z
            entity_name = self._goal_region_entity_name(raw)
            if entity_name is not None:
                goal_name = entity_name
                return
            point_target = self._point_target_from_goal_region(raw)
            if point_target is not None:
                xy, z, radius = point_target
                point_goal_xy = xy
                if z is not None:
                    point_goal_z = z
                if radius is not None:
                    point_goal_radius = radius

        for sub in subtasks:
            if sub["skill"] == "pick":
                grasp_mode = sub.get("params", {}).get("grasp_mode") or grasp_mode
            if sub["skill"] in {"transport", "place"}:
                _apply_goal_region(sub.get("params", {}).get("goal_region"))
            for c in sub.get("applied_constraints", []):
                if c["type"] == "min_carry_height":
                    carry_height = c["params"].get("height_above_table", carry_height)
                elif c["type"] == "descend_only_near_goal":
                    descend_radius = c["params"].get("radius", descend_radius)
                    _apply_goal_region(c["params"].get("goal_region"))
                elif c["type"] == "avoid_region":
                    # Avoidance is now task-local. Do not convert any
                    # subtask-level avoid_region into a global runtime keepout.
                    pass

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
        #
        # Trust any subtask's own explicit object reference as long as it
        # resolves to a *known* scene entity -- checking against the full
        # entity set (self.resolver.entity_roles), not just the
        # "manipulated_object"-role subset. role_for_object_name's heuristic
        # (libero_scene.py) checks _GOAL_OR_SUPPORT_HINTS before
        # _MOVABLE_HINTS, and both lists contain "bowl"/"plate" -- so a task
        # whose actual target *is* a bowl or plate (e.g. "pick up the black
        # bowl...") always gets classified support_object, never
        # manipulated_object. Gating on manipulated_object_names() alone then
        # silently falls through to the "first manipulated object" fallback,
        # which for an all-support_object scene means "no fallback" -- and a
        # confirmed real rollout showed this landing on an arbitrary,
        # never-touched object instead of the one actually picked.
        known_entities = set(self.resolver.entity_roles)
        manipulated = sorted(self.resolver.manipulated_object_names())
        active_object_name = None
        for sub in subtasks:
            obj = self.resolver.canonical(sub.get("params", {}).get("object"))
            if obj in known_entities:
                active_object_name = obj
                break
        if active_object_name is None:
            active_object_name = manipulated[0] if manipulated else "object"
        return {
            "mode": grasp_mode if grasp_mode in {"top", "left", "right"} else "top",
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

    def _boost_carry_height_for_goal_container(self, env: BaseManipulationEnv, sub: dict[str, Any]) -> None:
        """Raise ``env.carry_z`` if needed so the *carried object* -- not just
        the gripper -- clears (1) the goal container's real rim, and (2) the
        surface the object is being picked up *from*, during legacy
        transport's travel.

        Legacy transport has no notion of the carried object's own footprint
        (see place_skill.py's ``_escape_start_collision`` docstring for the
        same limitation at the transport->place handoff): ``carry_z`` is a
        single fixed height (``table_z + carry_height_above_table``, default
        17cm) meant to clear obstacles, but neither term in that formula
        knows about anything taller than the table.

        Case 1 (goal container): a tall object grasped near its top (e.g. a
        bbq sauce bottle) hangs *below* the gripper by roughly its own
        height -- confirmed via a real rollout where transport crashed with
        "backend_step_error" after the object was dropped, and the numbers
        work out almost exactly: eef height (~0.214m) minus the attached
        object's own collision_radius_m (~0.072m, from the prior ompl_mpc
        pick's sphere-fit) landed at ~0.142m, a hair below the real basket's
        measured rim height (~0.1414m) -- razor-thin, and evidently not
        always enough.

        Case 2 (starting surface): confirmed via a real rollout ("pick up
        the black bowl on the wooden cabinet") that an object resting on
        something *taller than the table* (a cabinet top, measured ~1.128m)
        can already sit at or above the *default* carry_z (table_z + 0.17 ~=
        1.07-1.14m) -- with no real lift margin between "where the object
        already is" and "carry_z", stage 2->3's own transition condition
        (cube_pos[2] >= carry_z - 0.02) was satisfied the instant transport
        began, skipping any deliberate vertical lift entirely and going
        straight to stage 4's horizontal travel while the bowl (held only by
        its rim) was still resting right at the cabinet's edge -- catching
        on it as the arm swung sideways and losing the grip completely
        (confirmed on video: the bowl never moves at all while the arm
        travels away without it).

        Raising carry_z above whichever of these two real surfaces is
        taller (by the attached object's own collision_radius_m plus a
        margin) accounts for both without needing the legacy stage machine
        to become fully attachment-aware.

        Best-effort: any failure to resolve the goal container, the
        object's current position, or the attached state just leaves
        carry_z at its existing value -- this is a safety margin increase,
        never a hard requirement for transport to run.
        """
        attached = getattr(self, "_motion_attached_state", None)
        if attached is None or not hasattr(env, "carry_z"):
            return
        try:
            from .motion.skills.place_skill import _container_floor_and_rim_z
            from .motion.scene_query import SceneQuery, resolve_body_name

            scene = SceneQuery.from_robosuite_sim(env.env.sim)
            margin = 0.02
            candidate_carry_zs = []

            params = sub.get("params", {}) or {}
            raw_goal_region = params.get("goal_region")
            entity_name = self._goal_region_entity_name(raw_goal_region) or params.get("goal")
            goal_name = self.resolver.canonical(entity_name)
            if goal_name is not None:
                goal_container_body = resolve_body_name(scene, goal_name)
                _, rim_top_z = _container_floor_and_rim_z(scene, goal_container_body)
                candidate_carry_zs.append(rim_top_z + float(attached.collision_radius_m) + margin)

            # Also clear the surface the object is being picked *up from*,
            # not just the goal it's headed *to* -- confirmed via a real
            # rollout ("pick up the black bowl on the wooden cabinet"): the
            # bowl rests on the cabinet's own top (measured ~1.128m), almost
            # exactly at the *default* carry_z (table_z + 0.17 ~= 1.07-1.14m
            # depending on table_z). With no real lift margin between "where
            # the object already is" and "carry_z", stage 2->3's own
            # transition condition (cube_pos[2] >= carry_z - 0.02) was
            # already satisfied the instant transport began -- skipping any
            # deliberate vertical lift and going straight to stage 4's
            # horizontal travel while the bowl (held only by its rim) was
            # still resting right at the cabinet's edge, catching on it as
            # the arm swung sideways and losing the grip entirely (confirmed
            # on video: the bowl never moves at all while the arm travels
            # away without it). Use the *attached object's own current
            # height* (wherever it's actually resting -- table, cabinet top,
            # drawer, whatever) plus its collision radius as a second lower
            # bound, so there's always a real, deliberate clearance lift
            # before any horizontal travel starts.
            try:
                object_pos, _ = scene.body_pose(attached.object_name)
                candidate_carry_zs.append(float(object_pos[2]) + float(attached.collision_radius_m) + margin)
            except Exception:
                pass

            if candidate_carry_zs:
                required_carry_z = max(candidate_carry_zs)
                if required_carry_z > float(env.carry_z):
                    env.carry_z = required_carry_z
        except Exception:
            pass

    def _motion_backend(self, env: BaseManipulationEnv) -> str:
        """``platform.libero.motion_backend`` from scene_context: "legacy"
        (default -- the stage machine/STLCartesianMPC this whole file
        otherwise implements) or "ompl_mpc" (the new grasp-gen/IK/RRTConnect/
        MPC pipeline in ``step/system_one/motion/``, wired for ``pick`` and
        ``place``; ``transport`` still runs on the legacy path even in this
        mode -- getting the carried object from "just picked" to "roughly
        above the goal" doesn't need collision-aware planning as urgently as
        the final insertion into a container does, and this keeps the
        integration's scope bounded)."""
        if str(self.scene_context.get("platform", {}).get("backend")) != "libero":
            return "legacy"
        return str(
            self.scene_context.get("platform", {}).get("libero", {}).get("motion_backend", "legacy")
        )

    def _run_pick_with_motion_pipeline(
        self, env: BaseManipulationEnv, sub: dict[str, Any], outdir: str | Path | None = None
    ) -> tuple[bool, int, dict[str, Any]]:
        from .motion.config import MotionConfig
        from .motion.skills.libero_pick_adapter import LiberoPickAdapter

        params = sub.get("params", {}) or {}
        object_set = params.get("object_candidates")
        if not object_set:
            single = self.resolver.canonical(params.get("object"))
            object_set = [single] if single is not None else []

        motion_config = MotionConfig.from_scene_context(self.scene_context)
        adapter = LiberoPickAdapter(env, motion_config)
        max_steps = int(self.max_subtask_steps) if self.max_subtask_steps is not None else 600
        result, steps, attached_state = adapter.run(
            object_set,
            constraints={},
            max_steps=max_steps,
            step_callback=lambda: self._capture_video_frame(env),
            outdir=outdir,
        )
        # Persisted across subtasks so a later `place` on the new pipeline
        # knows the grasp-relative transform/collision spheres of whatever
        # this pick actually grabbed -- the attachment lives for the life of
        # the executor, not just this one adapter call.
        self._motion_attached_state = attached_state

        status: dict[str, Any] = {
            "success": bool(result.success),
            "selected_object": result.selected_object,
            "failure_reason": str(result.failure_reason),
            "cost_breakdown": result.cost_breakdown,
            "diagnostics": result.diagnostics,
        }
        if result.success and result.selected_object is not None:
            # NOTE: this only sets the *env's* active object for anything read
            # before the next subtask's own dispatch runs. A later transport/
            # place subtask's `params.object` still wins if it names a
            # concrete (possibly different) entity -- true "OR-set commitment
            # propagates to later subtasks" would need the plan itself to
            # reference "whatever pick just selected" (e.g. a resolver-level
            # placeholder), which isn't wired up in this pass.
            env.set_active_object(result.selected_object)
        return bool(result.success), steps, status

    def _run_place_with_motion_pipeline(
        self, env: BaseManipulationEnv, sub: dict[str, Any]
    ) -> tuple[bool, int, dict[str, Any]]:
        from .motion.config import MotionConfig
        from .motion.skills.libero_place_adapter import LiberoPlaceAdapter

        attached_state = getattr(self, "_motion_attached_state", None)
        if attached_state is None:
            return False, 0, {
                "success": False,
                "failure_reason": "no_attached_object",
                "detail": (
                    "place (ompl_mpc backend) requires a prior pick on the same backend to have "
                    "attached an object -- none is recorded (either pick used motion_backend=legacy, "
                    "or pick failed before attaching)."
                ),
            }

        params = sub.get("params", {}) or {}
        raw_goal_region = params.get("goal_region")
        entity_name = self._goal_region_entity_name(raw_goal_region) or params.get("goal")
        goal_name = self.resolver.canonical(entity_name)
        if goal_name is None:
            if self._point_target_from_goal_region(raw_goal_region) is not None:
                # PlaceSkill (ompl_mpc backend) always plans onto a real
                # container body's measured pose/extent/rim -- it has no mode
                # for placing at a floating point with no supporting entity,
                # unlike the legacy stage-machine backend's goal_x/goal_y path.
                return False, 0, {
                    "success": False,
                    "failure_reason": "no_goal_container_body",
                    "detail": (
                        "place (ompl_mpc backend) got an explicit point goal_region "
                        f"{raw_goal_region!r} with no matching scene entity -- this backend's "
                        "PlaceSkill requires a real container body to plan onto and cannot "
                        "place at a floating point."
                    ),
                }
            return False, 0, {"success": False, "failure_reason": "no_goal_region", "detail": "place subtask has no params.goal_region"}

        motion_config = MotionConfig.from_scene_context(self.scene_context)
        adapter = LiberoPlaceAdapter(env, motion_config)
        max_steps = int(self.max_subtask_steps) if self.max_subtask_steps is not None else 400
        result, steps = adapter.run(
            goal_name,
            attached_state,
            max_steps=max_steps,
            step_callback=lambda: self._capture_video_frame(env),
        )
        if result.success:
            self._motion_attached_state = None  # detached -- next pick must re-attach

        status: dict[str, Any] = {
            "success": bool(result.success),
            "selected_object": result.selected_object,
            "failure_reason": str(result.failure_reason),
            "cost_breakdown": result.cost_breakdown,
            "diagnostics": result.diagnostics,
        }
        return bool(result.success), steps, status

    def _run_open_drawer_with_motion_pipeline(
        self, env: BaseManipulationEnv, sub: dict[str, Any]
    ) -> tuple[bool, int, dict[str, Any]]:
        from .motion.config import MotionConfig
        from .motion.skills.libero_open_drawer_adapter import LiberoOpenDrawerAdapter

        params = sub.get("params", {}) or {}
        drawer_ref = self.resolver.canonical(params.get("object")) or params.get("object")
        if not drawer_ref:
            return False, 0, {"success": False, "failure_reason": "no_object", "detail": "open subtask has no params.object"}

        motion_config = MotionConfig.from_scene_context(self.scene_context)
        adapter = LiberoOpenDrawerAdapter(env, motion_config)
        max_steps = int(self.max_subtask_steps) if self.max_subtask_steps is not None else 600
        result, steps, open_verification = adapter.run(
            drawer_ref,
            constraints={},
            max_steps=max_steps,
            step_callback=lambda: self._capture_video_frame(env),
        )
        status: dict[str, Any] = {
            "success": bool(result.success),
            "selected_object": result.selected_object,
            "failure_reason": str(result.failure_reason),
            "cost_breakdown": result.cost_breakdown,
            "diagnostics": result.diagnostics,
            "open_verification": open_verification,
        }
        return bool(result.success), steps, status

    def _run_turn_knob_with_motion_pipeline(
        self, env: BaseManipulationEnv, sub: dict[str, Any]
    ) -> tuple[bool, int, dict[str, Any]]:
        from .motion.config import MotionConfig
        from .motion.skills.libero_stove_adapter import LiberoStoveAdapter

        params = sub.get("params", {}) or {}
        knob_ref = self.resolver.canonical(params.get("object")) or params.get("object")
        if not knob_ref:
            return False, 0, {"success": False, "failure_reason": "no_object", "detail": "turn_on subtask has no params.object"}

        motion_config = MotionConfig.from_scene_context(self.scene_context)
        adapter = LiberoStoveAdapter(env, motion_config)
        max_steps = int(self.max_subtask_steps) if self.max_subtask_steps is not None else 600
        result, steps, turn_on_verification = adapter.run(
            knob_ref,
            constraints={},
            max_steps=max_steps,
            step_callback=lambda: self._capture_video_frame(env),
        )
        status: dict[str, Any] = {
            "success": bool(result.success),
            "selected_object": result.selected_object,
            "failure_reason": str(result.failure_reason),
            "cost_breakdown": result.cost_breakdown,
            "diagnostics": result.diagnostics,
            "turn_on_verification": turn_on_verification,
        }
        return bool(result.success), steps, status

    @staticmethod
    def _goal_region_entity_name(raw: Any) -> str | None:
        """Return `raw` only when it names a scene entity (a plain string).

        System 2's prompt (prompts.py, "explicit point targets") tells the LLM
        to emit `params.goal_region = {"type": "point", "xy": [...], "z": ..,
        "radius": ..}` whenever no scene entity matches the requested goal
        (e.g. "put it on the stove" when there's no stove object) -- so a
        dict here is an expected, documented shape, not malformed input.
        `SceneResolver.canonical()`/`resolve_goal()` only resolve entity-name
        strings, so a dict must never reach them; callers should route it
        through `_point_target_from_goal_region` instead.
        """
        return raw if isinstance(raw, str) else None

    @staticmethod
    def _point_target_from_goal_region(
        raw: Any,
    ) -> tuple[tuple[float, float], float | None, float | None] | None:
        """Parse a `{"type": "point", "xy": [x, y], "z": z, "radius": r}`
        goal_region into `(xy, z, radius)`, or None if `raw` isn't that shape.
        """
        if not isinstance(raw, dict) or raw.get("type") != "point":
            return None
        xy = raw.get("xy")
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            return None
        z = raw.get("z")
        radius = raw.get("radius")
        return (
            (float(xy[0]), float(xy[1])),
            float(z) if z is not None else None,
            float(radius) if radius is not None else None,
        )

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

    @staticmethod
    def _object_height_from_subtask(sub: dict[str, Any]) -> float | None:
        """Read an optional object-height hint (m) off the subtask's Grasped
        predicate, e.g. {"type": "predicate", "name": "Grasped",
        "args": ["alphabet_soup_1", 0.09]}. Lets System 2 tell System 1 how
        deep to descend for non-rim objects instead of grasping at their base.
        """
        formula = sub.get("formula")
        if not isinstance(formula, dict):
            return None
        from ..system_two.stl_plan import find_predicate_args

        args = find_predicate_args(formula, "Grasped")
        if not args or len(args) < 2 or not isinstance(args[1], (int, float)):
            return None
        return float(args[1])

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
        raw_goal = self._goal_region_entity_name(params.get("goal_region")) or params.get("goal")
        goal_name = self.resolver.canonical(raw_goal)
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

        if goal_x is None or goal_y is None:
            point_target = self._point_target_from_goal_region(params.get("goal_region"))
            if point_target is not None:
                (goal_x, goal_y), point_z, point_radius = point_target
                goal_z = goal_z if goal_z is not None else point_z
                goal_radius = goal_radius if goal_radius is not None else point_radius

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

        raw_goal = self._goal_region_entity_name(params.get("goal_region")) or params.get("goal")
        goal_name = self.resolver.canonical(raw_goal)
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
                # Only trust this subtask's own object reference if it
                # resolves to a *known* scene entity. resolver.canonical()
                # falls back to returning its input unchanged when nothing
                # matches -- so a generic placeholder like "object" or
                # "goal_region" (transport/place's own params, never meant to
                # name a concrete body) comes back as that literal string,
                # not None. Calling set_active_object() with it anyway
                # overwrote whatever a prior pick had just correctly set via
                # its own result.selected_object -- confirmed by a real
                # rollout where transport immediately reported
                # active_object_name='object' (a nonexistent body) right
                # after pick had established 'glazed_rim_porcelain_ramekin_1',
                # causing _is_grasped() to falsely report no grasp and the
                # legacy transport stage machine to release and restart its
                # own naive top-down grasp attempt.
                object_name = self.resolver.canonical(sub.get("params", {}).get("object"))
                if object_name is not None and object_name in self.resolver.entity_roles:
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
                    if hasattr(env, "set_active_object_height"):
                        env.set_active_object_height(self._object_height_from_subtask(sub))
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
                    if hasattr(env, "set_active_object_height"):
                        env.set_active_object_height(None)
                    # Mirror _begin_new_pick_skill's own "already grasped ->
                    # skip straight to carry" logic. "transport" always runs
                    # on the legacy step_nominal() stage machine (see
                    # _motion_backend's docstring), whose self.stage the new
                    # ompl_mpc pick pipeline never touches at all -- so after
                    # a successful ompl_mpc pick, self.stage is still sitting
                    # at its env.reset() default of 0 ("move to pregrasp,
                    # gripper OPEN"). Dispatching transport without this sync
                    # made stage 0 immediately open the gripper -- releasing
                    # the real grasp pick just established -- and restart the
                    # legacy stage machine's own naive top-down close from
                    # scratch, which (confirmed by a real rollout: cube_pos
                    # slid along the table at ~constant z instead of lifting,
                    # eef-cube distance never closed to a real grip range)
                    # only pushes a shallow object like a bowl/ramekin rather
                    # than actually gripping it.
                    if hasattr(env, "_is_grasped") and hasattr(env, "stage"):
                        env.stage = 3 if env._is_grasped() else 0
                        env.close_counter = 0
                    if skill == "transport":
                        self._boost_carry_height_for_goal_container(env, sub)
                    self._refresh_subtask_bookkeeping(env)
                motion_backend = self._motion_backend(env)
                if skill == "pick" and motion_backend == "ompl_mpc":
                    done, steps, last_status = self._run_pick_with_motion_pipeline(env, sub, outdir=outdir)
                elif skill == "place" and motion_backend == "ompl_mpc":
                    done, steps, last_status = self._run_place_with_motion_pipeline(env, sub)
                elif skill == "open" and motion_backend == "ompl_mpc":
                    done, steps, last_status = self._run_open_drawer_with_motion_pipeline(env, sub)
                elif skill == "turn_on" and motion_backend == "ompl_mpc":
                    done, steps, last_status = self._run_turn_knob_with_motion_pipeline(env, sub)
                else:
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
                # Same class of bug as the per-subtask stage sync above, one
                # more place it was missing: this loop calls the legacy
                # stage machine's own step_nominal() directly, completely
                # outside the per-subtask dispatch loop that normally does
                # this sync -- so if the *last* subtask (e.g. "place")
                # actually ran on the new ompl_mpc pipeline, env.stage is
                # still sitting at its env.reset() default of 0 ("approach
                # pregrasp, gripper open"). Confirmed via a real rollout:
                # these "settle" steps then restarted a brand-new pick
                # attempt on whatever the legacy _cube_pos() tracks (the
                # object that had just been correctly placed, now resting
                # inside the goal container) and dove the arm back down,
                # catching and lifting the whole basket. Stage 7 (retreat,
                # gripper open, falls back to a computed generic retreat
                # target if retreat_anchor_xy was never set by this
                # backend) is the correct "we're already done" state here,
                # not stage 0's "start a fresh pick".
                if hasattr(env, "stage"):
                    env.stage = 7
                    env.close_counter = 0
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

