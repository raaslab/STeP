"""Persistent recall context for real-robot execution.

Grows across multiple runs and recall invocations. Three-level hierarchy:

  Task level:    "Attempt 2 failed at subtask 3 of 4"
  Subgoal level: "place(red_cube, green_cube) failed — object dropped during descent"
  Step level:    waypoint trace, EE positions, grasp state, motion errors

The file is empty on the first run. Each recall appends a new attempt entry so
System 2 can reason across the full execution history, not just the current run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RecallContext:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.original_instruction: str = ""
        self.attempts: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None

        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.original_instruction = data.get("original_instruction", "")
                self.attempts = list(data.get("attempts", []))
            except Exception:
                pass

    # ------------------------------------------------------------------ write

    def start_attempt(
        self,
        instruction: str,
        total_subtasks: int,
        trigger: str = "initial",
    ) -> None:
        self.original_instruction = instruction
        self._current = {
            "attempt_number": len(self.attempts) + 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "trigger": trigger,
            "total_subtasks": total_subtasks,
            "subtasks_attempted": [],
            "outcome": "in_progress",
            "failed_at_subtask": None,
            "task_summary": "",
        }

    def log_subtask(
        self,
        *,
        index: int,
        skill: str,
        params: dict[str, Any],
        outcome: str,
        info: dict[str, Any],
        ee_xyz: list[float] | None = None,
        object_xyz: list[float] | None = None,
    ) -> None:
        if self._current is None:
            return
        params = params or {}
        obj = params.get("object", "object")
        goal = (
            params.get("goal_region")
            or (
                f"point({float(params['goal_x']):.3f},{float(params['goal_y']):.3f})"
                if params.get("goal_x") is not None and params.get("goal_y") is not None
                else None
            )
            or "unknown"
        )

        # Step-level trace from move records
        move_records = info.get("records", []) or []
        total_steps = sum(int(r.get("steps", 0)) for r in move_records)
        waypoints_reached = sum(1 for r in move_records if r.get("reached"))
        waypoints_total = len(move_records)
        motion_errors = [r["error"] for r in move_records if r.get("error")]

        step_trace_parts: list[str] = []
        if waypoints_total:
            step_trace_parts.append(
                f"waypoints reached: {waypoints_reached}/{waypoints_total}"
            )
        if total_steps:
            step_trace_parts.append(f"total steps: {total_steps}")
        if motion_errors:
            step_trace_parts.append(f"motion errors: {'; '.join(motion_errors[:3])}")

        # Subgoal-level summary
        if skill == "pick":
            grasped = bool(info.get("grasped", False))
            motion_ok = bool(info.get("motion_success", True))
            if grasped:
                summary = f"pick({obj}) succeeded — grasped after {total_steps} steps"
            elif motion_ok:
                summary = f"pick({obj}) failed — motion reached but no grasp detected ({total_steps} steps)"
            else:
                summary = f"pick({obj}) failed — motion did not complete ({total_steps} steps)"
        elif skill == "transport":
            all_reached = waypoints_total > 0 and waypoints_reached == waypoints_total
            if all_reached:
                summary = f"transport({obj} → {goal}) succeeded — over goal after {total_steps} steps"
            else:
                summary = (
                    f"transport({obj} → {goal}) failed — "
                    f"reached {waypoints_reached}/{waypoints_total} waypoints in {total_steps} steps"
                )
        elif skill == "place":
            err = info.get("error", "")
            if outcome == "success":
                summary = f"place({obj} on {goal}) completed in {total_steps} steps"
            else:
                summary = f"place({obj} on {goal}) failed in {total_steps} steps" + (f" — {err}" if err else "")
        else:
            summary = f"{skill}({obj}) {outcome} in {total_steps} steps"

        entry: dict[str, Any] = {
            "index": index,
            "skill": skill,
            "object": obj,
            "outcome": outcome,
            "summary": summary,
            "step_trace": "; ".join(step_trace_parts) if step_trace_parts else None,
            "steps": total_steps,
        }
        if skill != "pick":
            entry["goal"] = goal
        if skill == "pick":
            entry["grasped"] = bool(info.get("grasped", False))
        elif skill == "transport":
            entry["all_waypoints_reached"] = waypoints_total > 0 and waypoints_reached == waypoints_total
        if ee_xyz is not None:
            entry["ee_final_xyz"] = [round(v, 4) for v in ee_xyz[:3]]
        if object_xyz is not None:
            entry["object_final_xyz"] = [round(v, 4) for v in object_xyz[:3]]
        if motion_errors:
            entry["motion_errors"] = motion_errors[:5]

        if outcome == "failed" and self._current["failed_at_subtask"] is None:
            self._current["failed_at_subtask"] = index

        self._current["subtasks_attempted"].append(entry)

    def close_attempt(self, outcome: str) -> None:
        if self._current is None:
            return
        self._current["outcome"] = outcome
        n = self._current["attempt_number"]
        total = self._current["total_subtasks"]
        failed_at = self._current.get("failed_at_subtask")
        attempted = self._current["subtasks_attempted"]

        if outcome == "success":
            self._current["task_summary"] = (
                f"Attempt {n}: succeeded — all {total} subtasks completed."
            )
        elif failed_at is not None and failed_at < len(attempted):
            sub = attempted[failed_at]
            self._current["task_summary"] = (
                f"Attempt {n}: failed at subtask {failed_at + 1} of {total} — {sub['summary']}"
            )
        else:
            self._current["task_summary"] = (
                f"Attempt {n}: did not complete ({outcome}); "
                f"{len(attempted)} of {total} subtasks attempted."
            )

        self.attempts.append(self._current)
        self._current = None
        self.save()

    # ------------------------------------------------------------------ read

    def to_payload(self) -> dict[str, Any]:
        """Compact structured summary for the System 2 recall payload."""
        return {
            "original_instruction": self.original_instruction,
            "num_attempts_so_far": len(self.attempts),
            "task_history": [
                {
                    "attempt": a["attempt_number"],
                    "timestamp": a.get("timestamp", ""),
                    "trigger": a.get("trigger", ""),
                    "outcome": a["outcome"],
                    "task_summary": a.get("task_summary", ""),
                    "subtask_traces": [
                        {
                            "index": s["index"],
                            "summary": s["summary"],
                            "outcome": s["outcome"],
                            "step_trace": s.get("step_trace"),
                            "ee_final_xyz": s.get("ee_final_xyz"),
                            "object_final_xyz": s.get("object_final_xyz"),
                            "motion_errors": s.get("motion_errors"),
                        }
                        for s in a.get("subtasks_attempted", [])
                    ],
                }
                for a in self.attempts
            ],
        }

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"original_instruction": self.original_instruction, "attempts": self.attempts},
                indent=2,
            )
        )
