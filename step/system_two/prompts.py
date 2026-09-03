from __future__ import annotations

import json
from typing import Any


def build_developer_prompt(scene_context: dict[str, Any] | None = None) -> str:
    entity_lines = []
    if scene_context is not None:
        for entity in scene_context.get("entities", []):
            name = entity.get("name")
            role = entity.get("role", "unknown")
            if name:
                entity_lines.append(f"- {name}: {role}")

    scene_text = "No scene_context was supplied."
    if entity_lines:
        scene_text = "Canonical scene entities:\n" + "\n".join(entity_lines)

    return (
        "You are System 2 in STeP: a high-level robot planner that writes STL formulas directly.\n"
        "Convert the user's LIBERO tabletop manipulation instruction into one JSON STL skill graph.\n"
        "Return only JSON matching the supplied schema. Do not return Markdown.\n\n"
        "Important requirements:\n"
        "1. Use only these skills: pick, transport, place, pour, open.\n"
        "2. Use one subtask per atomic skill.\n"
        "3. Formula JSON supports: true, predicate, not, and, implies, always, eventually, until.\n"
        "4. Use controller-step intervals, not seconds. LIBERO defaults to 20 Hz.\n"
        "5. Preserve the formula predicates expected by System 1: PreGrasp, Grasped, OverGoal, Placed, Released, Safe, Upright, CarryHigh, DescendOnlyNearGoal, InRegion, PouredInto, Opened.\n"
        "6. Put all execution-relevant target data in both params and formula args. Semantic targets may be entity names. Explicit point targets use {\"type\":\"point\",\"xy\":[x,y],\"z\":z,\"radius\":r}.\n"
        "7. For a normal pick, use F[0,40](PreGrasp(object, grasp_mode)) and F[0,90](Grasped(object)).\n"
        "7a. For bowls, mugs, ramekins, pans, cups, or other rim-grasp objects, include params.grasp_target=\"rim\" and params.grasp_offset_xyz as a 3-number world-frame offset from the object pose to the rim grasp point. Use [0.0, 0.045, 0.05] as the default LIBERO bowl rim offset unless scene geometry clearly suggests another value.\n"
        "7b. For other graspable objects (cans, bottles, boxes, cartons, etc.), estimate the object's height in meters from the scene image and pass it as a second Grasped argument: F[0,90](Grasped(object, height_m)). The object's known pose is at its base (table height), not its center, so without this the gripper descends into the object instead of closing around its middle. Omit this argument for rim-grasp objects covered by 7a.\n"
        "8. For a normal transport, use F[0,180](OverGoal(object, goal)). Name the target entity with params.goal_region (not params.goal) so System 1 tracks its live position instead of a stale snapshot.\n"
        "9. For a normal place, use F[0,120](Placed(object, goal) AND Released(object)). Also set params.goal_region here, same as for transport.\n"
        "10. Add safety or timing clauses only when the user asks for them or the task clearly requires them.\n"
        "11. Use canonical scene entity names exactly when they match the request.\n"
        "12. For opening a drawer or cabinet, use a single subtask with skill=\"open\" and "
        "params.object naming the articulated_fixture entity (e.g. top_drawer). Formula: "
        "F[0,120](Opened(object)). This one subtask grasps the handle and pulls it open -- "
        "do not emit separate pick/transport/place subtasks for a drawer or cabinet.\n\n"
        f"{scene_text}"
    )


def build_user_payload(
    *,
    instruction: str,
    scene_context: dict[str, Any] | None,
    schema: dict[str, Any],
) -> str:
    payload = {
        "user_instruction": instruction,
        "scene_context": scene_context,
        "stl_plan_schema": schema,
    }
    return json.dumps(payload, indent=2)
