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
        "1. Use only these skills: pick, transport, place, pour, open, close, turn_on.\n"
        "2. Use one subtask per atomic skill.\n"
        "3. Formula JSON supports: true, predicate, not, and, implies, always, eventually, until.\n"
        "4. Use controller-step intervals, not seconds. LIBERO defaults to 20 Hz.\n"
        "5. Preserve the formula predicates expected by System 1: PreGrasp, Grasped, OverGoal, Placed, Released, Safe, Upright, CarryHigh, DescendOnlyNearGoal, InRegion, PouredInto, Opened, Closed, TurnedOn.\n"
        "6. Put all execution-relevant target data in both params and formula args. Semantic targets may be entity names. Explicit point targets use {\"type\":\"point\",\"xy\":[x,y],\"z\":z,\"radius\":r}.\n"
        "7. For a normal pick, use F[0,40](PreGrasp(object, grasp_mode)) and F[0,90](Grasped(object)).\n"
        "7a. For bowls, mugs, ramekins, pans, cups, plates, saucers, or other rim-grasp objects, include params.grasp_target=\"rim\" and params.grasp_offset_xyz as a 3-number world-frame offset from the object pose to the rim grasp point. Use [0.0, 0.045, 0.05] as the default LIBERO bowl rim offset unless scene geometry clearly suggests another value.\n"
        "7b. For other graspable objects (cans, bottles, boxes, cartons, etc.), estimate the object's height in meters from the scene image and pass it as a second Grasped argument: F[0,90](Grasped(object, height_m)). The object's known pose is at its base (table height), not its center, so without this the gripper descends into the object instead of closing around its middle. Omit this argument for rim-grasp objects covered by 7a.\n"
        "8. For a normal transport, use F[0,180](OverGoal(object, goal)). Name the target entity with params.goal_region (not params.goal) so System 1 tracks its live position instead of a stale snapshot.\n"
        "9. For a normal place, use F[0,120](Placed(object, goal) AND Released(object)). Also set params.goal_region here, same as for transport.\n"
        "10. Add safety or timing clauses only when the user asks for them or the task clearly requires them.\n"
        "11. Use canonical scene entity names exactly when they match the request.\n"
        "12. For opening a drawer or cabinet, use a single subtask with skill=\"open\" and "
        "params.object naming the articulated_fixture entity (e.g. top_drawer). Formula: "
        "F[0,120](Opened(object)). This one subtask grasps the handle and pulls it open -- "
        "do not emit separate pick/transport/place subtasks for a drawer or cabinet.\n"
        "13. For turning on a stove/knob/dial, use a single subtask with skill=\"turn_on\" and "
        "params.object naming the knob entity (e.g. flat_stove_1). Formula: "
        "F[0,150](TurnedOn(object)). This one subtask grasps the knob's rim and rotates it -- "
        "do not emit separate pick/transport/place subtasks for a knob or dial.\n"
        "14. To place an object on top of a solid support's own frame (a cabinet, or the table "
        "itself -- not inside any drawer/container), name that support's own support_object "
        "entity exactly as it appears in the canonical scene entities list (for a cabinet, a "
        "separate entity from its drawer entities like top_drawer -- never reuse a drawer's own "
        "name for this) as params.goal_region on both the transport and place subtasks, and add "
        "params.place_surface=\"top\" on *both* the transport and the place subtask -- transport "
        "needs it too, to measure the support's own top surface and search over it for a clear "
        "spot, not just the place subtask that does the final descent (omit it, or set "
        "params.place_surface=\"inside\", for a normal place into a container/drawer -- that is "
        "still the default).\n"
        "15. MANDATORY CHECK before emitting any \"open\" subtask for a drawer/cabinet -- and this "
        "rule only applies when an \"open\" subtask is actually being emitted at all: if rule 18 "
        "already found the fixture satisfies Opened(...), there is no opening motion for anything "
        "to block, so skip straight to the plain pick -> transport -> place -> (rule 16's close, if "
        "the task needs it) with no detour, regardless of any clearance flag below. When an \"open\" "
        "subtask IS being emitted: look up the entity notes (in the canonical scene entities list "
        "below) of the single object the task places into that fixture. If its notes contain a "
        "\"[Precomputed clearance to <cabinet>: ...m ... BLOCKS ...]\" flag naming that same fixture, "
        "that object is physically standing in the drawer's way and MUST be moved aside first. Emit "
        "exactly these 7 subtasks in this order: pick(object) -> transport(object, aside) -> "
        "place(object, aside) -> open(fixture) -> pick(object) -> transport(object, fixture) -> "
        "place(object, fixture). For the two 'aside' subtasks just set "
        "params.goal_region=\"table\" and params.place_surface=\"top\" -- do not compute "
        "coordinates for it. System 1 picks the actual spot on the table itself, using the real "
        "collision geometry of every obstacle in the scene (including fixtures like a stove or "
        "wine rack that are not listed as entities here, so you cannot reason about them). "
        "Never move any other object, flagged or not. If that one object has no such flag, emit "
        "the plain open -> pick -> transport -> place with no detour.\n"
        "15b. When the task is *only* to open a drawer/cabinet -- nothing is being placed into "
        "it -- rule 15 has no target object to check, so use this instead. Look for entities "
        "whose notes carry a \"[Precomputed: this object stands in the arm's approach path to "
        "<drawer>...]\" flag naming that fixture. Those sit where the gripper itself has to "
        "travel, so the handle cannot be reached until they are moved, and there is no object "
        "the task will later relocate anyway. For each flagged entity emit pick -> transport -> "
        "place with params.goal_region=\"table\" and params.place_surface=\"top\" *before* the "
        "open subtask; they stay on the table afterward. Move only flagged entities -- an "
        "object beside the approach is not in it.\n"
        "16. For closing a drawer/cabinet or a microwave, use a single subtask with skill=\"close\" "
        "and params.object naming the articulated_fixture entity (a drawer, e.g. bottom_drawer) or "
        "the microwave's own entity name (e.g. microwave_1). Formula: F[0,120](Closed(object)). "
        "This one subtask pushes it shut with a closed gripper -- it never grasps anything, so do "
        "not emit separate pick/transport/place subtasks for it. When a task says to place "
        "something into a drawer/microwave \"and close it\", emit the pick/transport/place "
        "subtasks for the object first, then this close subtask last, naming the same fixture.\n"
        "17. MANDATORY CHECK before emitting any pick/transport/place subtask. Look up that "
        "object's entity notes in the canonical scene entities list below. If its notes contain a "
        "\"[Precomputed: <object> already satisfies In(...)...]\" flag, its goal is already reached -- "
        "do NOT emit any pick/transport/place subtask for it. Plan only for the remaining objects "
        "the task still needs moved, and any other subtasks (e.g. close) the task still requires.\n"
        "18. MANDATORY CHECK before emitting any \"open\" or \"close\" subtask for a drawer/cabinet/"
        "microwave. Look up ONLY that exact fixture's own entity notes below -- never a sibling "
        "fixture's (e.g. top_drawer's notes say nothing about middle_drawer or bottom_drawer, and "
        "vice versa; check the one named in the subtask you are about to emit, nothing else). "
        "Opened(...) and Closed(...) are two DIFFERENT predicates about two DIFFERENT things (how "
        "far open vs. how far closed) -- a fixture's notes carry AT MOST ONE of the two flags below, "
        "and each flag says something only about its own named subtask, nothing about the other: "
        "a \"[Precomputed: this drawer is already ...% open, ... already satisfies Opened(...)...]\" "
        "flag means do NOT emit an \"open\" subtask for it -- it is already open, whatever left it "
        "that way (a task's own initial condition, not necessarily anything this plan did) -- but "
        "says nothing about Closed(...): if the task also asks to close this same fixture, still "
        "emit that close subtask per rule 16, exactly as if the Opened(...) flag were not there. "
        "Symmetrically, a \"...already satisfies Closed(...)...\" flag means do NOT emit a \"close\" "
        "subtask for it, but says nothing about Opened(...). Never treat one flag as evidence for "
        "the other -- if a fixture's notes carry neither flag, both open and close still need "
        "their own normal subtask exactly as the task requires. This is "
        "independent of rule 17: a fixture can already satisfy Opened/Closed while the object the "
        "task places into it still needs a normal pick/transport/place.\n\n"
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
