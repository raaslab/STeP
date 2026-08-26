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
        "1. Use only these skills: pick, transport, place, pour.\n"
        "2. Use one subtask per atomic skill.\n"
        "3. Formula JSON supports: true, predicate, not, and, implies, always, eventually, until.\n"
        "4. Use controller-step intervals, not seconds. LIBERO defaults to 20 Hz.\n"
        "5. Preserve the formula predicates expected by System 1: PreGrasp, Grasped, OverGoal, Placed, Released, Safe, Upright, CarryHigh, DescendOnlyNearGoal, InRegion, PouredInto.\n"
        "6. Put all execution-relevant target data in both params and formula args. Semantic targets may be entity names. Explicit point targets use {\"type\":\"point\",\"xy\":[x,y],\"z\":z,\"radius\":r}.\n"
        "7. For a normal pick, use F[0,40](PreGrasp(object, grasp_mode)) and F[0,90](Grasped(object)).\n"
        "7a. For bowls, mugs, ramekins, pans, cups, or other rim-grasp objects, include params.grasp_target=\"rim\" and params.grasp_offset_xyz as a 3-number world-frame offset from the object pose to the rim grasp point. Use [0.0, 0.045, 0.05] as the default LIBERO bowl rim offset unless scene geometry clearly suggests another value.\n"
        "8. For a normal transport, use F[0,180](OverGoal(object, goal)).\n"
        "9. For a normal place, use F[0,120](Placed(object, goal) AND Released(object)).\n"
        "10. Add safety or timing clauses only when the user asks for them or the task clearly requires them.\n"
        "11. Use canonical scene entity names exactly when they match the request.\n\n"
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
