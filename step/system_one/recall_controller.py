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
from .history_log import STLHistoryLog


class RecallControllerMixin:
    def _sync_scene_context_from_env(self, env: BaseManipulationEnv) -> None:
        """Pull the latest relocalized scene context from the environment."""
        new_ctx = getattr(env, "scene_context", None)
        if isinstance(new_ctx, dict):
            self.scene_context = new_ctx
            self.resolver = SceneResolver(new_ctx)

    def _fresh_scene_state(self, env: BaseManipulationEnv) -> dict[str, Any]:
        """Return a task-agnostic snapshot of all currently localized objects.

        The recall/history payload needs the whole scene, not only the active
        object. Prefer the relocalized live_scene_state stored in the scene
        context, and fill any missing objects from env.scene_objects when
        available. This works for simulation ground truth and real perception
        without hard-coding object names or task types.
        """
        ctx_objects = (
            self.scene_context.get("live_scene_state", {}).get("objects", {})
            if isinstance(getattr(self, "scene_context", None), dict)
            else {}
        )
        objects: dict[str, Any] = {}
        for name, meta in (ctx_objects or {}).items():
            if isinstance(meta, dict):
                objects[str(name)] = self._jsonable(meta)

        for name, xyz in getattr(env, "scene_objects", {}).items():
            if str(name) in objects:
                continue
            try:
                arr = np.asarray(xyz, dtype=np.float32).reshape(-1)
                if arr.size < 3 or np.any(~np.isfinite(arr[:3])):
                    continue
                objects[str(name)] = {
                    "xyz": arr[:3].astype(float).tolist(),
                    "xy": arr[:2].astype(float).tolist(),
                    "z": float(arr[2]),
                    "source": "env.scene_objects",
                }
            except Exception:
                continue
        return {"objects": objects}

    def _robot_state(self, env: BaseManipulationEnv) -> dict[str, Any]:
        """Return the current low-level robot/status payload for recall."""
        try:
            status = env.status()
        except Exception:
            status = {}
        if not isinstance(status, dict):
            status = {}
        return self._jsonable(status)

    def _constraints_for_reflection(self, env: BaseManipulationEnv) -> dict[str, Any]:
        """Controller/perception constraints payload, used as generic execution
        context for the System-2 recall path."""
        platform = self.scene_context.get("platform", {}) if isinstance(getattr(self, "scene_context", None), dict) else {}
        perception = platform.get("perception", {}) if isinstance(platform, dict) else {}
        return self._jsonable(
            {
                "workspace_bounds": platform.get("workspace_bounds") if isinstance(platform, dict) else None,
                "carry_height_above_table": perception.get("carry_height_above_table") if isinstance(perception, dict) else None,
                "retreat_height_above_table": perception.get("retreat_height_above_table") if isinstance(perception, dict) else None,
                "top_grasp_depth_below_surface": perception.get("top_grasp_depth_below_surface") if isinstance(perception, dict) else None,
                "top_grasp_hover_z": getattr(env, "top_grasp_hover_z", None),
                "descend_radius": getattr(env, "descend_radius", None),
                "goal_radius": getattr(env, "goal_radius", None),
                # May be absent in SDF-only mode; kept in the payload for
                # debug context without feeding them back into the MPC cost.
                "keepout_xy": getattr(env, "keepout_xy", None),
                "keepout_radius": getattr(env, "keepout_radius", None),
            }
        )

    @staticmethod
    def _simple_param_text(value: Any) -> str:
        if value is None:
            return ""
        try:
            if isinstance(value, float):
                return f"{value:.3f}"
            if isinstance(value, (list, tuple)):
                return ", ".join(RecallControllerMixin._simple_param_text(v) for v in value)
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _subtask_description_from_fields(subtask: dict[str, Any] | None) -> str:
        if not isinstance(subtask, dict):
            return ""
        for key in (
            "description",
            "natural_language_description",
            "natural_language_desc",
            "nl_description",
            "short_description",
            "summary",
        ):
            val = subtask.get(key)
            if val:
                return str(val).strip()

        skill = str(subtask.get("skill") or "")
        params = subtask.get("params", {})
        if not isinstance(params, dict):
            params = {}

        obj = params.get("object") or params.get("source_object") or "object"
        goal = (
            params.get("goal_region")
            or params.get("goal")
            or params.get("goal_reference")
            or params.get("goal_name")
        )
        if goal is None and params.get("goal_x") is not None and params.get("goal_y") is not None:
            goal = f"point ({float(params['goal_x']):+.3f}, {float(params['goal_y']):+.3f})"

        mode = params.get("grasp_mode") or "top"
        if skill == "pick":
            return f"Pick up {obj} with a {mode} grasp."
        if skill == "transport":
            if goal is not None:
                return f"Move {obj} over {goal}."
            return f"Transport {obj}."
        if skill == "place":
            if goal is not None:
                return f"Place {obj} on/in {goal}."
            return f"Place and release {obj}."
        return skill or ""

    def _subtask_description(self, subtask: dict[str, Any] | None) -> str:
        return self._subtask_description_from_fields(subtask)

    def _subtask_prompt_view(self, subtask: dict[str, Any]) -> dict[str, Any]:
        """Compact, JSON-safe view of a compiled subtask for System-2 recall prompts."""
        if not isinstance(subtask, dict):
            return {}
        return self._jsonable(
            {
                "subtask_id": subtask.get("subtask_id"),
                "skill": subtask.get("skill"),
                "description": self._subtask_description(subtask),
                "params": subtask.get("params", {}),
                "notes": subtask.get("notes"),
                "why": subtask.get("why"),
                "precondition": subtask.get("precondition"),
                "postcondition": subtask.get("postcondition"),
                "formula": self._formula_string(subtask),
                "applied_constraints": subtask.get("applied_constraints", []),
            }
        )

    @staticmethod
    def _phase_name(stage: Any) -> str:
        names = {
            0: "pregrasp",
            1: "grasp",
            2: "close_gripper",
            3: "lift_to_carry",
            4: "transport",
            5: "place_descend",
            6: "release_hold",
            7: "retreat",
        }
        try:
            return names.get(int(stage), f"stage_{stage}")
        except Exception:
            return "unknown"

    def _history_positions(self, status: dict[str, Any]) -> dict[str, Any]:
        return self._jsonable(
            {
                "ee": status.get("eef_pos"),
                # cube_pos is the legacy backend key for the currently active
                # manipulated object. Keep the payload key object-agnostic.
                "active_object": status.get("cube_pos"),
                "goal_xy": status.get("goal_anchor_xy"),
                "goal_z": status.get("goal_place_z"),
                "goal_reference_name": status.get("goal_reference_name"),
            }
        )

    @staticmethod
    def _object_xyz_from_meta(meta: Any) -> list[float] | None:
        if not isinstance(meta, dict):
            return None
        xyz = meta.get("xyz")
        if xyz is None and meta.get("xy") is not None and meta.get("z") is not None:
            try:
                xyz = [meta["xy"][0], meta["xy"][1], meta["z"]]
            except Exception:
                xyz = None
        if xyz is None:
            return None
        try:
            arr = np.asarray(xyz, dtype=np.float32).reshape(-1)
            if arr.size < 3 or np.any(~np.isfinite(arr[:3])):
                return None
            return arr[:3].astype(float).tolist()
        except Exception:
            return None

    def _scene_change_report(
        self,
        previous_scene_state: dict[str, Any] | None,
        current_scene_state: dict[str, Any] | None,
        *,
        moved_threshold_m: float = 0.02,
    ) -> dict[str, Any]:
        """Summarize object-level scene changes between two perception refreshes.

        This is deliberately task-agnostic: it compares every localized entity in
        the scene state, rather than looking for a specific object, color, or task
        relation. System 2 can use this report to notice regressions after a
        subtask, such as a previously placed object being displaced.
        """
        prev_objs = (previous_scene_state or {}).get("objects", {}) or {}
        curr_objs = (current_scene_state or {}).get("objects", {}) or {}
        names = sorted(set(map(str, prev_objs.keys())) | set(map(str, curr_objs.keys())))
        changed: list[dict[str, Any]] = []
        missing_after_refresh: list[str] = []
        newly_localized: list[str] = []
        for name in names:
            prev_xyz = self._object_xyz_from_meta(prev_objs.get(name))
            curr_xyz = self._object_xyz_from_meta(curr_objs.get(name))
            if prev_xyz is None and curr_xyz is not None:
                newly_localized.append(name)
                continue
            if prev_xyz is not None and curr_xyz is None:
                missing_after_refresh.append(name)
                continue
            if prev_xyz is None or curr_xyz is None:
                continue
            disp = float(np.linalg.norm(np.asarray(curr_xyz) - np.asarray(prev_xyz)))
            if disp >= float(moved_threshold_m):
                changed.append(
                    {
                        "name": name,
                        "previous_xyz": prev_xyz,
                        "current_xyz": curr_xyz,
                        "displacement_m": disp,
                    }
                )
        return self._jsonable(
            {
                "moved_threshold_m": float(moved_threshold_m),
                "changed_objects": changed,
                "missing_after_refresh": missing_after_refresh,
                "newly_localized": newly_localized,
                "all_current_objects": sorted(map(str, curr_objs.keys())),
                "note": (
                    "Scene change report compares all localized entities after the "
                    "fresh perception pass; it is not specific to any task or object type."
                ),
            }
        )

    def _history_record_for_subtask(
        self,
        env: BaseManipulationEnv,
        subtask: dict[str, Any] | None,
        *,
        trigger: str | None = None,
    ) -> dict[str, Any]:
        subtask = subtask if isinstance(subtask, dict) else {}
        try:
            status = env.status()
        except Exception:
            status = {}
        if not isinstance(status, dict):
            status = {}

        margins = self._solver_robustness_margins(env, subtask, status=status)
        failure_signals: list[dict[str, Any]] = []
        for name, value in margins.items():
            try:
                v = float(value)
            except Exception:
                continue
            failed = v <= float(self.recall_failure_threshold)
            failure_signals.append(
                {
                    "predicate": name,
                    "robustness": v,
                    "threshold": float(self.recall_failure_threshold),
                    "failed": bool(failed),
                }
            )

        stage = status.get("stage")
        return self._jsonable(
            {
                "subtask": {
                    "subtask_id": subtask.get("subtask_id"),
                    "skill": subtask.get("skill"),
                    "description": self._subtask_description(subtask),
                    "stage": stage,
                    "stage_name": self._phase_name(stage),
                },
                "subtask_id": subtask.get("subtask_id"),
                "subtask_description": self._subtask_description(subtask),
                "stl_formula": self._formula_string(subtask) if subtask else None,
                "stl_robustness": margins,
                "failure_signals": failure_signals,
                "positions": self._history_positions(status),
                "scene_state": self._fresh_scene_state(env),
                "grasped": status.get("grasped"),
                "success": status.get("success"),
                "trigger": trigger,
            }
        )

    def _update_history_log(
        self,
        env: BaseManipulationEnv,
        subtask: dict[str, Any] | None,
        *,
        force: bool = False,
        trigger: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.system2_recall_enabled:
            return None
        record = self._history_record_for_subtask(env, subtask, trigger=trigger)
        added = self.history_log.maybe_add(record, force=force)
        return record if added or force else self.history_log.latest()

    def _recall_failure_trigger(self, record: dict[str, Any] | None, *, skill: str | None = None) -> str | None:
        if not record:
            return None
        for signal in record.get("failure_signals", []) or []:
            if not signal.get("failed"):
                continue
            name = str(signal.get("predicate", ""))
            # Skips normal task-predicate negativity during early approach;
            # only safety/failure predicates trigger here.
            if any(k in name for k in self.recall_failure_predicate_keywords):
                return f"predicate_threshold:{name}={float(signal.get('robustness')):.4f}"
        if skill == "transport" and record.get("grasped") is False:
            return "object_not_grasped"
        if skill == "place" and record.get("grasped") is False:
            # Releasing is expected during place; recalling on gripper-open
            # here would let System 2 interrupt the retreat mid-motion.
            try:
                stage = int((record.get("subtask") or {}).get("stage", -1))
            except Exception:
                stage = -1
            if stage < 6:
                return "object_not_grasped"
        return None

    def _refresh_scene_for_recall(
        self,
        env: BaseManipulationEnv,
        outdir: Path,
        recall_idx: int,
    ) -> str | None:
        if self.recall_scene_update not in {"none", "off", "false"}:
            try:
                if self.recall_scene_update == "discover" and hasattr(env, "discover_scene_context"):
                    print("[recall] refreshing scene with discover_scene_context before System 2 call...")
                    env.discover_scene_context(self.scene_context)
                elif hasattr(env, "relocalize_scene"):
                    print("[recall] refreshing scene with relocalize_scene before System 2 call...")
                    env.relocalize_scene()
                self._sync_scene_context_from_env(env)
            except Exception as exc:
                print(f"[recall] scene refresh failed; continuing with current scene state: {exc}")

        recall_dir = outdir / "recall"
        recall_dir.mkdir(parents=True, exist_ok=True)
        (recall_dir / f"recall_{recall_idx:03d}_scene_context.json").write_text(
            json.dumps(self._jsonable(self.scene_context), indent=2)
        )
        image_path = recall_dir / f"recall_{recall_idx:03d}_scene.png"
        try:
            env.snapshot_scene_image(str(image_path), width=self.video_width, height=self.video_height)
            return str(image_path)
        except Exception as exc:
            print(f"[recall] snapshot failed: {exc}")
            return None

    def _compiled_subtask_plan_view(
        self,
        subtasks: list[dict[str, Any]],
        *,
        start_index: int = 0,
    ) -> list[dict[str, Any]]:
        views: list[dict[str, Any]] = []
        for local_idx, subtask in enumerate(subtasks):
            absolute_idx = start_index + local_idx
            try:
                view = self._subtask_prompt_view(subtask)
            except Exception:
                view = {
                    "skill": subtask.get("skill"),
                    "params": subtask.get("params", {}),
                    "description": self._subtask_description(subtask),
                }
            view = dict(view)
            view.setdefault("subtask_index", absolute_idx)
            view["formula_string"] = self._formula_string(subtask)
            view["raw_formula"] = subtask.get("formula")
            view["applied_constraints"] = subtask.get("applied_constraints", [])
            views.append(self._jsonable(view))
        return views

    def _compiled_plan_anchor_payload(
        self,
        *,
        active_subtasks: list[dict[str, Any]],
        subtask_index: int,
        records: list[SkillExecutionRecord],
    ) -> dict[str, Any]:
        original_plan = getattr(self, "_original_compiled_stl_plan_for_recall", None)
        latest_plan = getattr(self, "_latest_compiled_stl_plan_for_recall", None)
        completed = [r.__dict__ for r in records]
        active_all = self._compiled_subtask_plan_view(active_subtasks)
        remaining = self._compiled_subtask_plan_view(active_subtasks[subtask_index:], start_index=subtask_index)
        return self._jsonable(
            {
                "purpose": (
                    "Use this compiled STL plan as the canonical plan anchor. "
                    "Recall may replan the remaining work, but should preserve "
                    "the original task decomposition, reference entities, "
                    "goal regions, object relations, and spatial targets unless "
                    "the current scene/history shows that exact target is "
                    "impossible or already failed."
                ),
                "original_compiled_stl_plan": original_plan,
                "latest_compiled_stl_plan": latest_plan,
                "active_compiled_stl_plan": active_all,
                "remaining_compiled_stl_plan": remaining,
                "completed_execution_records": completed,
                "plan_consistency_policy": [
                    "Do not change intended reference entities, support/target entities, goal regions, or spatial targets merely because recall was triggered.",
                    "Do not move already-correctly-satisfied objects or relations unless the fresh image/history shows they are no longer satisfied.",
                    "If a repair is needed, repair toward the original compiled STL objective and target relation, not a newly invented objective or location.",
                    "Prefer returning only the remaining subtasks needed to complete the original compiled STL plan.",
                ],
            }
        )

    def _build_recall_payload(
        self,
        env: BaseManipulationEnv,
        *,
        trigger: str,
        active_subtasks: list[dict[str, Any]],
        subtask_index: int,
        records: list[SkillExecutionRecord],
        scene_change_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = active_subtasks[subtask_index] if 0 <= subtask_index < len(active_subtasks) else None
        return self._jsonable(
            {
                "original_user_instruction": self.original_prompt,
                "recall_trigger": trigger,
                "current_subtask": self._subtask_prompt_view(current) if current else None,
                "current_plan": [self._subtask_prompt_view(s) for s in active_subtasks],
                "compiled_stl_plan_anchor": self._compiled_plan_anchor_payload(
                    active_subtasks=active_subtasks,
                    subtask_index=subtask_index,
                    records=records,
                ),
                "completed_subtasks": [r.__dict__ for r in records],
                "history_log": self.history_log.to_payload(),
                "execution_parameters": {
                    "carry_z": getattr(env, "carry_z", None),
                    "retreat_z": getattr(env, "retreat_z", None),
                    "retreat_height_above_table": (
                        float(getattr(env, "retreat_z", 0.0)) - float(getattr(env, "table_z", 0.0))
                        if hasattr(env, "retreat_z") and hasattr(env, "table_z")
                        else None
                    ),
                    "descend_radius": getattr(env, "descend_radius", None),
                    "goal_radius": getattr(env, "goal_radius", None),
                    "table_z": getattr(env, "table_z", None),
                },
                "fresh_scene_state": self._fresh_scene_state(env),
                "scene_change_report": scene_change_report or getattr(self, "_latest_scene_change_report", None),
                "robot_state": self._robot_state(env),
                "constraints": self._constraints_for_reflection(env),
                "scene_context": self.scene_context,
                "instruction": (
                    "Replan all remaining work from the current state. "
                    "Return a full task_spec for the remaining task, not a repair patch. "
                    "Use compiled_stl_plan_anchor as the canonical objective: preserve "
                    "the original reference entities, support/target entities, goal "
                    "regions, object relations, and spatial targets unless the fresh "
                    "scene/history proves that target failed or is impossible. Do not "
                    "choose a new objective or location simply because recall was "
                    "triggered. Do not move already-correctly-satisfied objects or "
                    "relations. Do not use keys like repair_required, repair_action, "
                    "parameter_updates, or remaining_subtasks. Preserve System-1 "
                    "execution parameters such as retreat height; do not encode retreat "
                    "as a subtask. After every recall/reflection scene refresh, "
                    "inspect all localized objects and all predicates/relations in "
                    "the compiled STL anchor, not only the object manipulated by the "
                    "last subtask. If the fresh scene or scene_change_report shows "
                    "that a previously satisfied object relation regressed, repair "
                    "that relation while preserving the original plan anchor."
                ),
            }
        )

    def _compile_recalled_task_spec(
        self,
        task_spec: dict[str, Any],
        *,
        outdir: Path,
        recall_idx: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raise NotImplementedError(
            "System 2 recall still needs to be ported to direct STL plans."
        )

    def _recall_plan_summary(
        self,
        *,
        trigger: str,
        task_spec: dict[str, Any] | None = None,
        new_subtasks: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> str:
        if error:
            return f"Recall trigger={trigger}; System 2 call failed: {error}"
        task_spec = task_spec or {}
        new_subtasks = list(new_subtasks or [])
        scene_summary = str(task_spec.get("scene_summary") or task_spec.get("description") or "").strip()
        if len(new_subtasks) == 0:
            base = f"Recall trigger={trigger}; System 2 judged no remaining subtasks are needed."
            return f"{base} {scene_summary}".strip()
        descs: list[str] = []
        for sub in new_subtasks[:3]:
            desc = self._subtask_description(sub)
            if desc:
                descs.append(desc)
            else:
                skill = sub.get("skill")
                obj = (sub.get("params") or {}).get("object")
                descs.append(f"{skill} {obj}".strip())
        more = "" if len(new_subtasks) <= 3 else f"; +{len(new_subtasks) - 3} more"
        prefix = f"Recall trigger={trigger}; System 2 interpreted the log and replanned: "
        return prefix + "; ".join(descs) + more

    def _set_video_recall_overlay(self, result: dict[str, Any] | None) -> None:
        if not isinstance(result, dict):
            return
        summary = result.get("recall_summary") or result.get("reflection_summary")
        if summary:
            self._video_current_reflection_summary = str(summary)
            try:
                self._capture_video_frame(getattr(self, "_last_env_for_video", None), force=True)
            except Exception:
                pass

    def _maybe_recall_system2(
        self,
        env: BaseManipulationEnv,
        *,
        trigger: str,
        active_subtasks: list[dict[str, Any]],
        subtask_index: int,
        records: list[SkillExecutionRecord],
        outdir: str | Path | None,
    ) -> tuple[list[dict[str, Any]] | None, bool, dict[str, Any] | None]:
        if not self.system2_recall_enabled:
            return None, False, None
        if self._recall_replans_used >= self.recall_max_replans_per_run:
            return None, False, {
                "trigger": trigger,
                "skipped": True,
                "reason": "recall_max_replans_per_run reached",
            }

        out_path = Path(outdir or "outputs")
        recall_dir = out_path / "recall"
        recall_dir.mkdir(parents=True, exist_ok=True)
        recall_idx = len(self.recall_history)
        scene_state_before_refresh = self._fresh_scene_state(env)
        image_path = self._refresh_scene_for_recall(env, out_path, recall_idx)
        scene_state_after_refresh = self._fresh_scene_state(env)
        scene_change_report = self._scene_change_report(
            scene_state_before_refresh, scene_state_after_refresh
        )
        self._latest_scene_change_report = scene_change_report

        # Add a post-refresh record so the rolling history log contains the same
        # all-object scene state that System 2 sees in the recall payload.
        current_for_history = active_subtasks[subtask_index] if 0 <= subtask_index < len(active_subtasks) else None
        self._update_history_log(
            env,
            current_for_history,
            force=True,
            trigger=f"{trigger}:scene_refreshed",
        )

        payload = self._build_recall_payload(
            env,
            trigger=trigger,
            active_subtasks=active_subtasks,
            subtask_index=subtask_index,
            records=records,
            scene_change_report=scene_change_report,
        )
        # The rolling history log lives in the recall payload, but also write it
        # as a standalone file for easier inspection/debugging.
        (recall_dir / "history_log_latest.json").write_text(
            json.dumps(self._jsonable(self.history_log.to_payload()), indent=2)
        )
        (recall_dir / f"recall_{recall_idx:03d}_history_log.json").write_text(
            json.dumps(self._jsonable(self.history_log.to_payload()), indent=2)
        )
        (recall_dir / f"recall_{recall_idx:03d}_scene_change_report.json").write_text(
            json.dumps(self._jsonable(scene_change_report), indent=2)
        )
        (recall_dir / f"recall_{recall_idx:03d}_payload.json").write_text(
            json.dumps(self._jsonable(payload), indent=2)
        )

        try:
            raise NotImplementedError(
                "System 2 recall still needs to be ported to direct STL plans."
            )
        except Exception as exc:
            recall_summary = self._recall_plan_summary(trigger=trigger, error=str(exc))
            result = {
                "trigger": trigger,
                "error": str(exc),
                "image_path": image_path,
                "recall_summary": recall_summary,
                "reflection_summary": recall_summary,
            }
            self.recall_history.append(result)
            try:
                self._set_video_reflection_overlay(result)
                self._capture_video_frame(env, force=True)
            except Exception:
                pass
            (recall_dir / f"recall_{recall_idx:03d}_error.json").write_text(
                json.dumps(self._jsonable(result), indent=2)
            )
            print(f"[recall] System 2 recall failed: {exc}")
            # If the subtask succeeded, continue with the old plan. If recall
            # was safety-triggered, abort rather than driving into a violation.
            abort = trigger.startswith("predicate_threshold") or trigger == "object_not_grasped"
            return None, abort, result

    def _run_post_success_tail(
        self,
        env: BaseManipulationEnv,
    ) -> tuple[int, dict[str, Any]]:
        """Run a short post-verification hold/retreat segment for video and settling.

        The normal non-recall executor already has post_success_steps. The
        recall executor previously ended immediately after the final System-2
        verifier returned an empty plan, which made the video stop right at the
        final placement/verification frame. This tail keeps executing the safe
        open-gripper retreat target for a few frames and captures the result.
        """
        steps_requested = max(0, int(getattr(self, "post_success_steps", 0)))
        if steps_requested <= 0:
            return 0, {}
        post_steps_run = 0
        post_last_status: dict[str, Any] = {}
        try:
            self._set_video_final_overlay(
                {
                    "trigger": "final_verification",
                    "repair_required": False,
                    "recall_summary": "Final verification complete; retreating/settling before video close.",
                    "reflection_summary": "Final verification complete; retreating/settling before video close.",
                }
            )
            self._capture_video_frame(env, force=True)
        except Exception:
            pass
        for _ in range(steps_requested):
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
        try:
            self._capture_video_frame(env, force=True)
        except Exception:
            pass
        return post_steps_run, post_last_status

    def _run_subtasks_on_env_with_system2_recall(
        self,
        env: BaseManipulationEnv,
        subtasks: list[dict[str, Any]],
        *,
        runtime: dict[str, Any],
        outdir: str | Path | None,
        capture_initial_video: bool,
    ) -> dict[str, Any]:
        """Run subtasks with event-driven System-2 recall: on success or STL
        failure threshold, System 2 gets the rolling history log and returns a
        fresh task_spec for the remaining work."""
        if self._is_real_simple_env(env):
            print(
                "[recall] real simple executor uses blocking primitives; "
                "falling back to existing real execution path for now."
            )
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
        self.history_log.reset()
        aborted = False
        post_success_steps_run = 0
        post_success_final_status: dict[str, Any] = {}

        try:
            self._last_env_for_video = env
            if capture_initial_video:
                self._capture_video_frame(env, force=True)

            idx = 0
            final_recall_verified = False
            while idx < len(active_subtasks) or (not final_recall_verified and not aborted):
                if idx >= len(active_subtasks):
                    # Full System-2 recall so any object- or relation-level
                    # regression in the refreshed scene gets a fresh remaining task_spec.
                    self._update_history_log(env, None, force=True, trigger="final_verification")
                    new_subtasks, should_abort, final_result = self._maybe_recall_system2(
                        env,
                        trigger="final_verification",
                        active_subtasks=active_subtasks,
                        subtask_index=len(active_subtasks),
                        records=records,
                        outdir=outdir,
                    )
                    if should_abort:
                        aborted = True
                        break
                    if new_subtasks is not None and len(new_subtasks) > 0:
                        active_subtasks = list(new_subtasks)
                        idx = 0
                        continue
                    # Empty recalled plan means System 2 judged the task done.
                    final_recall_verified = bool(new_subtasks is not None and len(new_subtasks) == 0)
                    if final_recall_verified:
                        post_success_steps_run, post_success_final_status = self._run_post_success_tail(env)
                    break

                sub = active_subtasks[idx]
                self._set_video_subtask_overlay(sub)
                skill = sub["skill"]
                object_name = self.resolver.canonical(sub.get("params", {}).get("object"))
                if object_name is not None:
                    env.set_active_object(object_name)

                self._configure_goal_for_subtask(env, sub)

                self._apply_subtask_constraints_to_env(env, sub)
                self._configure_pass_through_for_subtask(env, sub)
                dwell_counts.clear()

                if skill == "pick":
                    grasp_mode = sub.get("params", {}).get("grasp_mode") or runtime["mode"]
                    env.switch_mode(grasp_mode if grasp_mode in {"top", "left", "right"} else "top")
                    self._begin_new_pick_skill(env)
                else:
                    self._refresh_subtask_bookkeeping(env)

                self._update_history_log(env, sub, force=True, trigger="subtask_start")

                steps = 0
                done = False
                last_status: dict[str, Any] = {}
                recalled = False

                while self.max_subtask_steps is None or steps < int(self.max_subtask_steps):
                    try:
                        _, _, terminated, info = env.step_nominal()
                    except ValueError as exc:
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

                    hist_record = self._update_history_log(env, sub)
                    done, last_status = self._skill_done(env, skill, dwell_counts)

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

                    trigger = None
                    if done:
                        trigger = "subtask_succeeded"
                    elif hist_record is not None:
                        trigger = self._recall_failure_trigger(hist_record, skill=skill)

                    if trigger is not None:
                        self._update_history_log(env, sub, force=True, trigger=trigger)
                        record = SkillExecutionRecord(
                            skill=skill,
                            params=sub.get("params", {}),
                            steps=steps,
                            done=bool(done),
                            info=dict(last_status or env.status(), recall_trigger=trigger),
                        )
                        records.append(record)

                        should_recall = (
                            trigger != "subtask_succeeded"
                            or self.recall_on_subtask_success
                            or (idx == len(active_subtasks) - 1 and self.recall_on_final_success)
                        )
                        # Avoid endless success recalls on a final empty plan.
                        if trigger == "subtask_succeeded" and idx == len(active_subtasks) - 1 and not self.recall_on_final_success:
                            should_recall = False

                        if should_recall:
                            new_subtasks, should_abort, recall_result = self._maybe_recall_system2(
                                env,
                                trigger=trigger,
                                active_subtasks=active_subtasks,
                                subtask_index=idx,
                                records=records,
                                outdir=outdir,
                            )
                            if should_abort:
                                aborted = True
                                recalled = True
                                break
                            if new_subtasks is not None:
                                # System 2 returns the full remaining task_spec,
                                # so replace the active rollout rather than splicing a repair patch.
                                active_subtasks = list(new_subtasks)
                                idx = 0
                                recalled = True
                                break

                        if done or terminated:
                            break

                    if done or terminated:
                        break

                if aborted:
                    break
                if recalled:
                    continue

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

                # Failed subtask without threshold-triggered recall: ask System 2
                # to replan from history instead of local repair.
                if not done:
                    self._update_history_log(env, sub, force=True, trigger="subtask_failed")
                    new_subtasks, should_abort, _ = self._maybe_recall_system2(
                        env,
                        trigger="subtask_failed",
                        active_subtasks=active_subtasks,
                        subtask_index=idx,
                        records=records,
                        outdir=outdir,
                    )
                    if should_abort:
                        aborted = True
                        break
                    if new_subtasks is not None:
                        active_subtasks = list(new_subtasks)
                        idx = 0
                        continue
                    aborted = True
                    break

                idx += 1

            final_status = env.status()
            summary = {
                "success": bool(records and not aborted and all(r.done for r in records) and final_recall_verified),
                "final_recall_verified": bool(final_recall_verified),
                "post_success_steps_requested": int(getattr(self, "post_success_steps", 0)),
                "post_success_steps_run": int(post_success_steps_run),
                "post_success_final_status": post_success_final_status,
                "records": [r.__dict__ for r in records],
                "system2_recall_enabled": True,
                "history_log": self.history_log.to_payload(),
                "recall_history": self.recall_history,
                "final_status": final_status,
            }
            if outdir is not None:
                out_path = Path(outdir)
                out_path.mkdir(parents=True, exist_ok=True)
                (out_path / "execution_summary.json").write_text(
                    json.dumps(self._jsonable(summary), indent=2)
                )
                (out_path / "history_log_final.json").write_text(
                    json.dumps(self._jsonable(self.history_log.to_payload()), indent=2)
                )
                if hasattr(self, "_save_xy_plot"):
                    self._save_xy_plot(out_path / "execution_xy.png", env)
            return self._jsonable(summary)
        finally:
            pass
