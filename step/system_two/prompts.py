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
        "5a. Safe(object, other, distance_m) keeps object -- or \"ee\" for the gripper -- at least distance_m from other, centre to centre in the plane. Upright(object, max_tilt_deg) keeps the carried object within max_tilt_deg of how it stood before the pick. When the instruction asks for either, write it as a top-level always constraint with the instruction's own numbers, never inside a pick/transport/place goal. Write the distance form only when the instruction states a distance; \"without touching\" or \"avoid\" with no number is Safe(other) alone -- never invent a distance.\n"
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
        "13b. Express a WAIT in the intervals, not with a new skill. If the instruction asks the "
        "robot to wait, pause, let something heat, or leave a gap before the next step, delay the "
        "NEXT subtask's interval by that many steps (20 per second). Worked example, for \"turn on "
        "the stove, wait 4 seconds, then put the pot on it\": F[0,150](TurnedOn('flat_stove_1')), "
        "then the pick opening at F[230,320] rather than F[150,240] -- the 80-step gap is the four "
        "seconds. System 1 holds position until each subtask's interval opens, so the gap is the "
        "wait. Do not invent a wait skill or predicate; the timing is the specification.\n"
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


def build_global_developer_prompt(scene_context: dict[str, Any] | None = None) -> str:
    """EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md).

    Asks the VLM for one global STL formula for the whole task instead of a
    pre-decomposed subtasks list. The ordered subtask sequence is derived
    afterward by step/system_two/stl_plan.py:decompose(), not by the model.
    Not used by the default build_developer_prompt pipeline.
    """
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
        "You are System 2 in STeP: a high-level robot planner that writes ONE STL "
        "formula for the entire task, not a pre-split sequence of subtasks.\n"
        "Convert the user's LIBERO tabletop manipulation instruction into one JSON "
        "STL specification. Return only JSON matching the supplied schema. Do not "
        "return Markdown.\n\n"
        "Important requirements:\n"
        "1. Formula JSON supports: true, predicate, and, always, eventually. Do not "
        "use not, implies, until, or a disjunction ('or' is not in this grammar) "
        "anywhere in the formula -- there is currently no way to decompose a "
        "negated, conditional, waited-on, or disjunctive goal into an executable "
        "subtask sequence (this matches the decomposition algorithm's own proven "
        "fragment: only conjunctions of eventually/always goals, nothing else), so "
        "restructure the task using only and/eventually/always instead.\n"
        "2. Use controller-step intervals, not seconds. LIBERO defaults to 20 Hz.\n"
        "2a. All intervals are ABSOLUTE positions on one shared global timeline for "
        "the whole task -- NOT a local clock that resets to 0 for each goal. "
        "Achievement goals execute strictly one at a time, in the order they "
        "appear, so their windows must form a chain of non-overlapping, "
        "non-decreasing absolute intervals: the first goal's window starts at 0, "
        "and every later goal's window must start at or after the exact step "
        "where the previous goal's own window ended -- never earlier, and never "
        "reusing the same absolute numbers for two different goals. Each goal "
        "still gets the same DURATION budget as before (see rule 5's templates for "
        "each skill's duration), you are just adding the running total of every "
        "goal that came before it as an offset. Worked example for three goals in "
        "sequence -- pick, transport, place -- where pick's own duration is 90 "
        "steps (its inner PreGrasp deadline lands in the first 40 of those 90), "
        "transport's is 180, and place's is 120: "
        "F[0,40](PreGrasp(...)) and F[0,90](Grasped(...)) for the pick (window "
        "[0,90]), then F[90,270](OverGoal(...)) for the transport (starts at 90, "
        "where pick ended, runs 180 more steps), then F[270,390]((Placed(...) & "
        "Released(...))) for the place (starts at 270, where transport ended, "
        "runs 120 more steps). Keep carrying this same running total forward "
        "through every remaining goal in the formula, however many there are.\n"
        "2b. When the user's instruction states a minimum delay between two events -- "
        "\"place it at least 5 seconds after picking it up\", \"wait 3 seconds before "
        "opening it\" -- encode it as an explicit GAP in the absolute timeline: "
        "start the later goal's window that many controller-steps (20/sec) after the "
        "earlier goal's own window ended, instead of immediately where it ended as in "
        "the plain worked example above. E.g. \"place at least 5 seconds (100 steps) "
        "after picking it up\", continuing the pick/transport/place example above (pick "
        "ends at 90, transport runs 90->270): place must not start before "
        "90+100=190, and since transport already carries it to 270 (past that), no gap "
        "is actually needed here -- only add one if the earlier goal's own end falls "
        "*before* the required delay would allow; e.g. if this had been a 250-step "
        "minimum delay after pick (pick ends at 90, so not before step 340), and "
        "transport still ends at 270, insert the gap on place's own window: place "
        "becomes F[340,460] instead of F[270,390] (place's own 120-step duration is "
        "unchanged, only its start shifts to satisfy the delay). Never shrink a "
        "duration to force a fit -- only shift a start later. System 1 will physically "
        "hold position (wait) if it reaches a goal before its window's own start_step, "
        "so a stated minimum delay is something it can actually satisfy, not just "
        "a label.\n"
        "3. Preserve the formula predicates expected by System 1: PreGrasp, Grasped, "
        "OverGoal, Placed, Released, Safe, Upright, CarryHigh, DescendOnlyNearGoal, "
        "InRegion, PouredInto, Opened, Closed, TurnedOn.\n"
        "3a. Safe(object, other, distance_m) keeps object -- or \"ee\" for the gripper -- "
        "at least distance_m from other, centre to centre in the plane. Upright(object, "
        "max_tilt_deg) keeps the carried object within max_tilt_deg of how it stood before "
        "the pick. When the instruction asks for either, write it as a GLOBAL CONSTRAINT "
        "(see 5a) with the instruction's own numbers, never inside an achievement goal. "
        "Write the distance form only when the instruction states a distance; \"without "
        "touching\" or \"avoid\" with no number is Safe(other) alone -- never invent a distance.\n"
        "4. Put all execution-relevant target data as predicate args. Semantic "
        "targets may be entity names. Explicit point targets use "
        "{\"type\":\"point\",\"xy\":[x,y],\"z\":z,\"radius\":r}.\n"
        "5. system2_plan.formula's top level MUST be {\"type\":\"and\",\"children\":[...]} "
        "(unless the whole task is a single atomic goal, in which case emit that one "
        "goal directly with no \"and\" wrapper). Each element of that top-level list "
        "is exactly one of:\n"
        "   (a) a GLOBAL CONSTRAINT: {\"type\":\"always\",\"interval\":null,\"child\":...} "
        "-- a safety/invariance condition that must hold for the entire task (e.g. "
        "'never enter the red zone'). Only add these when the user asks for them or "
        "the task clearly requires them -- do not invent safety clauses.\n"
        "   (b) one ATOMIC ACHIEVEMENT GOAL, in the order it should be executed. Do "
        "NOT reorder goals relative to the instruction's own sequencing, and do NOT "
        "split one achievement (e.g. a single pick) across two top-level list "
        "elements -- group it into one 'and' as shown below. These are the only "
        "achievement-goal shapes System 1 understands; use exactly these predicate "
        "groupings, one top-level element per achievement. The interval shown for "
        "each below (e.g. [0,40]) is that achievement's DURATION starting from "
        "wherever it falls on the global timeline -- per rule 2a, offset every "
        "number in it by the running total of every goal before it; only the very "
        "first goal in the whole formula actually keeps a literal [0,...]:\n"
        "     - Pick: {\"type\":\"and\",\"children\":["
        "{\"type\":\"eventually\",\"interval\":[0,40],\"child\":predicate PreGrasp(object, "
        "grasp_mode)}, {\"type\":\"eventually\",\"interval\":[0,90],\"child\":predicate "
        "Grasped(object[, height_m])}]}. For other graspable objects (cans, bottles, "
        "boxes, cartons, etc.), estimate the object's height in meters from the scene "
        "image and pass it as Grasped's second argument -- the object's known pose is "
        "at its base (table height), not its center, so without this the gripper "
        "descends into the object instead of closing around its middle. For bowls, "
        "mugs, ramekins, pans, cups, plates, saucers, or other rim-grasp objects, omit "
        "the height arg and instead add an object_hints entry (see below) with "
        "grasp_target=\"rim\".\n"
        "     - Transport: {\"type\":\"eventually\",\"interval\":[0,180],\"child\":predicate "
        "OverGoal(object, goal)}. Exactly these two args, nothing more. Name the "
        "target entity as the second arg (not a stale snapshot) so System 1 tracks "
        "its live position.\n"
        "     - Place: {\"type\":\"eventually\",\"interval\":[0,120],\"child\":{\"type\":\"and\","
        "\"children\":[predicate Placed(object, goal), predicate Released(object)]}}.\n"
        "     - Open (drawer/cabinet): {\"type\":\"eventually\",\"interval\":[0,120],"
        "\"child\":predicate Opened(object)} where object names the articulated_fixture "
        "entity (e.g. top_drawer). This one goal grasps the handle and pulls it open "
        "-- never emit separate pick/transport/place goals for a drawer or cabinet.\n"
        "     - Turn on (stove/knob/dial): {\"type\":\"eventually\",\"interval\":[0,150],"
        "\"child\":predicate TurnedOn(object)} where object names the knob entity "
        "(e.g. flat_stove_1). This one goal grasps the knob's rim and rotates it -- "
        "never emit separate pick/transport/place goals for a knob or dial.\n"
        "     - Close (drawer/cabinet/microwave): {\"type\":\"eventually\",\"interval\":"
        "[0,120],\"child\":predicate Closed(object)} where object names the "
        "articulated_fixture entity (a drawer, e.g. bottom_drawer) or the microwave's "
        "own entity name. Never grasps anything -- do not emit pick/transport/place "
        "goals for it. When a task says to place something 'and close it', emit the "
        "pick/transport/place goals for the object first, then this goal last, naming "
        "the same fixture.\n"
        "6. To place an object on top of a solid support's own frame (a cabinet, or "
        "the table itself -- not inside any drawer/container), name that support's "
        "own support_object entity exactly as it appears in the canonical scene "
        "entities list (for a cabinet, a separate entity from its drawer entities "
        "like top_drawer -- never reuse a drawer's own name) as the goal arg on both "
        "the transport and place goals for that object, AND add an object_hints entry "
        "for that object with place_surface=\"top\" (omit place_surface, or set it to "
        "\"inside\", for a normal place into a container/drawer -- that is still the "
        "default).\n"
        "7. object_hints is a top-level sibling of formula: "
        "{\"<object_entity_name>\": {\"grasp_target\"?: \"rim\", \"grasp_offset_xyz\"?: "
        "[x,y,z], \"place_surface\"?: \"top\"|\"inside\"}}. Use grasp_offset_xyz "
        "[0.0, 0.045, 0.05] as the default LIBERO bowl rim offset unless scene "
        "geometry clearly suggests another value. Only include an entry for an "
        "object that actually needs one of these hints.\n"
        "8. Use canonical scene entity names exactly when they match the request.\n"
        "9. MANDATORY CHECK before emitting any \"open\" goal for a drawer/cabinet -- "
        "and this rule only applies when an \"open\" goal is actually being emitted at "
        "all: if rule 12 already found the fixture satisfies Opened(...), there is no "
        "opening motion for anything to block, so skip straight to the plain pick -> "
        "transport -> place -> (rule 5's close, if the task needs it) with no detour, "
        "regardless of any clearance flag below. When an \"open\" goal IS being "
        "emitted: look up the entity notes (in the canonical scene entities list "
        "below) of the single object the task places into that fixture. If its notes "
        "contain a \"[Precomputed clearance to <cabinet>: ...m ... BLOCKS ...]\" flag "
        "naming that same fixture, that object is physically standing in the "
        "drawer's way and MUST be moved aside first. Emit exactly these 7 achievement "
        "goals in this order: pick(object) -> transport(object, aside) -> "
        "place(object, aside) -> open(fixture) -> pick(object) -> transport(object, "
        "fixture) -> place(object, fixture). For the two 'aside' goals just target "
        "goal=\"table\" and add place_surface=\"top\" to that object's object_hints -- "
        "do not compute coordinates for it. System 1 picks the actual spot on the "
        "table itself, using the real collision geometry of every obstacle in the "
        "scene (including fixtures like a stove or wine rack that are not listed as "
        "entities here, so you cannot reason about them). Never move any other "
        "object, flagged or not. If that one object has no such flag, emit the plain "
        "open -> pick -> transport -> place with no detour.\n"
        "10. When the task is *only* to open a drawer/cabinet -- nothing is being "
        "placed into it -- rule 9 has no target object to check, so use this "
        "instead. Look for entities whose notes carry a \"[Precomputed: this object "
        "stands in the arm's approach path to <drawer>...]\" flag naming that "
        "fixture. Those sit where the gripper itself has to travel, so the handle "
        "cannot be reached until they are moved, and there is no object the task "
        "will later relocate anyway. For each flagged entity emit pick -> transport "
        "-> place goals (goal=\"table\", place_surface=\"top\" in that object's "
        "object_hints) *before* the open goal; they stay on the table afterward. "
        "Move only flagged entities -- an object beside the approach is not in it.\n"
        "11. MANDATORY CHECK before emitting any pick/transport/place goal. Look up "
        "that object's entity notes in the canonical scene entities list below. If "
        "its notes contain a \"[Precomputed: <object> already satisfies In(...)...]\" "
        "flag, its goal is already reached -- do NOT emit any pick/transport/place "
        "goal for it. Plan only for the remaining objects the task still needs "
        "moved, and any other goals (e.g. close) the task still requires.\n"
        "12. MANDATORY CHECK before emitting any \"open\" or \"close\" goal for a "
        "drawer/cabinet/microwave. Look up ONLY that exact fixture's own entity "
        "notes below -- never a sibling fixture's (e.g. top_drawer's notes say "
        "nothing about middle_drawer or bottom_drawer, and vice versa; check the one "
        "named in the goal you are about to emit, nothing else). Opened(...) and "
        "Closed(...) are two DIFFERENT predicates about two DIFFERENT things (how "
        "far open vs. how far closed) -- a fixture's notes carry AT MOST ONE of the "
        "two flags below, and each flag says something only about its own named "
        "goal, nothing about the other: a \"[Precomputed: this drawer is already "
        "...% open, ... already satisfies Opened(...)...]\" flag means do NOT emit "
        "an \"open\" goal for it -- it is already open, whatever left it that way (a "
        "task's own initial condition, not necessarily anything this plan did) -- "
        "but says nothing about Closed(...): if the task also asks to close this "
        "same fixture, still emit that close goal per rule 5, exactly as if the "
        "Opened(...) flag were not there. Symmetrically, a \"...already satisfies "
        "Closed(...)...\" flag means do NOT emit a \"close\" goal for it, but says "
        "nothing about Opened(...). Never treat one flag as evidence for the other "
        "-- if a fixture's notes carry neither flag, both open and close still need "
        "their own normal goal exactly as the task requires. This is independent of "
        "rule 11: a fixture can already satisfy Opened/Closed while the object the "
        "task places into it still needs a normal pick/transport/place.\n"
        "13. MANDATORY CHECK before repeating an unchanged \"open\" or \"close\" goal "
        "for the same fixture on a recall. Look up that fixture's own entity notes. "
        "If they contain a \"[Precomputed: a previous attempt's <skill> subtask ... "
        "did not succeed ...]\" flag, do NOT just repeat the identical bare goal "
        "again -- that already failed once for a physical reason, not a planning "
        "mistake. From the scene image and the canonical entities list, identify "
        "whichever object is most plausibly responsible (for a drawer: an object "
        "shown resting inside its open cavity, or sitting right in front of its "
        "handle) and emit pick(object) -> transport(object, aside) -> place(object, "
        "aside) -> <same skill>(fixture) instead, exactly like rule 9's sequence "
        "(goal=\"table\", place_surface=\"top\" for the aside placement). Only skip "
        "this if no entity plausibly explains the failure, in which case retry the "
        "goal unchanged at most once more.\n\n"
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


def build_no_stl_developer_prompt(scene_context: dict[str, Any] | None = None) -> str:
    """Developer prompt for the no-STL baseline.

    Removes the specification layer and nothing else. The VLM still sees the same
    instruction, the same image and the same scene entities, and still chooses the
    same skills over the same robot -- it just has no formal language to state a
    requirement in. Any constraint in the instruction therefore survives only as
    English the planner read once, which is exactly the thing STeP claims to
    improve on.

    The output still uses the skill_graph schema so System 1 needs no special
    case, but every subtask carries a bare reachability formula with no safety
    term, no timing bound, and no ordering clause.
    """
    base = build_developer_prompt(scene_context)
    # Keep the skill/parameter rules; replace the specification guidance.
    return (
        base
        + "\n\nBASELINE MODE -- NO SPECIFICATION LAYER. Override anything above that "
        "asks for safety, timing or ordering clauses:\n"
        "B1. Emit ONLY the ordered skills needed to achieve the goal, with their "
        "params. Think of this as a plan, not a specification.\n"
        "B2. Every subtask's formula must be the bare achievement predicate with a "
        "single generous interval, e.g. F[0,400](Grasped(object)). Never emit "
        "Safe(...), never emit an always/G term, and never narrow an interval to "
        "encode a deadline or a delay.\n"
        "B3. If the instruction mentions a region to avoid, a distance to leave, an "
        "order to follow or a time to wait, do NOT translate it into the formula. "
        "Plan the skills you would anyway; the requirement is not part of the "
        "output.\n"
    )


def build_mpc_cost_developer_prompt(scene_context: dict[str, Any] | None = None) -> str:
    """Developer prompt for the VLM-MPC-Cost baseline.

    System 2 writes a direct MPC cost function over the predicate set instead of
    compiling the task into STL. The controller then minimizes that weighted sum
    rather than maximizing STL robustness, and the history log records costs
    instead of robustness traces.

    The point of the baseline is what a cost function cannot say. A weighted sum
    of predicate terms expresses "be near this" and "stay off that", but it has
    no interval, no ordering operator, and no notion of "after" -- so a timing or
    sequencing requirement in the instruction has nowhere to go except into the
    order the subtasks happen to be listed in, and into the weights.
    """
    base = build_developer_prompt(scene_context)
    return (
        base
        + "\n\nVLM-MPC-COST BASELINE -- NO TEMPORAL LOGIC. This overrides every "
        "instruction above about formulas, intervals, always/eventually/until, "
        "and robustness:\n"
        "C1. Do NOT emit a `formula` field. Emit `cost_terms` instead: a list of "
        "{predicate, args, weight} objects that are SUMMED into the controller's "
        "running cost for that subtask. The controller minimizes this sum.\n"
        "C2. Use the same predicate names and argument shapes as before, but as "
        "cost terms only. A larger weight means the controller cares more about "
        "that term. Terms naming a region to stay out of (Safe) push the arm "
        "away; terms naming a target (PreGrasp, Grasped, OverGoal, Placed) pull "
        "it toward that target.\n"
        "C3. There are NO intervals and NO temporal operators available. You "
        "cannot express a deadline, a wait, a dwell, or an ordering constraint as "
        "a cost term. Do not try to encode one by inventing a predicate. If the "
        "instruction asks for timing or ordering, express what you can through "
        "the subtask order and the weights, and accept that the rest is not "
        "representable in this output format.\n"
        "C4. Also emit `failure_threshold` per subtask: the cost value above "
        "which System 1 should give up on that subtask and recall to you. Pick a "
        "number on the same scale as the cost terms you wrote.\n"
    )
