"""EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md): unit tests for the
algorithmic formula-decomposition path (decompose/split_top_level_and/
infer_skill_from_predicates), independent of any VLM call.

Per SLICE (Kapoor/Kang/Meira-Goes, NFM 2024, Algorithm 3), goal intervals are
ABSOLUTE positions on one shared global timeline, not a per-goal-local clock
reset to 0 -- every helper below takes an explicit cumulative start offset
to build formulas the way the real prompt now asks the VLM to.
"""

from step.system_two.stl_plan import (
    DecompositionError,
    decompose,
    decompose_and_validate,
    goal_absolute_window,
    infer_skill_from_predicates,
    split_top_level_and,
    validate_global_stl_plan,
    validate_stl_plan,
)


def _predicate(name, args):
    return {"type": "predicate", "name": name, "args": args}


def _pick_goal(obj, start=0, mode="top", height_m=None):
    grasped_args = [obj] if height_m is None else [obj, height_m]
    return {
        "type": "and",
        "children": [
            {"type": "eventually", "interval": [start, start + 40], "child": _predicate("PreGrasp", [obj, mode])},
            {"type": "eventually", "interval": [start, start + 90], "child": _predicate("Grasped", grasped_args)},
        ],
    }  # absolute window: [start, start + 90]


def _transport_goal(obj, goal_region, start=0):
    return {"type": "eventually", "interval": [start, start + 180], "child": _predicate("OverGoal", [obj, goal_region])}


def _place_goal(obj, goal_region, start=0):
    return {
        "type": "eventually",
        "interval": [start, start + 120],
        "child": {
            "type": "and",
            "children": [_predicate("Placed", [obj, goal_region]), _predicate("Released", [obj])],
        },
    }


def _pick_transport_place(obj, goal_region, start=0, **pick_kwargs):
    pick = _pick_goal(obj, start=start, **pick_kwargs)
    transport_start = start + 90
    transport = _transport_goal(obj, goal_region, start=transport_start)
    place_start = transport_start + 180
    place = _place_goal(obj, goal_region, start=place_start)
    return {"type": "and", "children": [pick, transport, place]}, place_start + 120


def test_decompose_pick_transport_place_matches_legacy_shape():
    formula, _ = _pick_transport_place("cup_1", "goal_region")
    subtasks = decompose(formula)

    assert [s["skill"] for s in subtasks] == ["pick", "transport", "place"]
    assert [s["subtask_id"] for s in subtasks] == [0, 1, 2]
    assert subtasks[0]["params"] == {"object": "cup_1", "grasp_mode": "top"}
    assert subtasks[1]["params"] == {"object": "cup_1", "goal_region": "goal_region"}
    assert subtasks[2]["params"] == {"object": "cup_1", "goal_region": "goal_region"}

    plan = {"system2_plan": {"type": "skill_graph", "task_name": "t", "scene_summary": "s", "subtasks": subtasks}}
    assert validate_stl_plan(plan).ok


def test_decompose_carries_height_and_rim_hints():
    formula = {
        "type": "and",
        "children": [
            _pick_goal("can_1", height_m=0.12),
            _transport_goal("can_1", "table", start=90),
            _place_goal("can_1", "table", start=270),
        ],
    }
    subtasks = decompose(formula)
    assert subtasks[0]["params"]["object_height_m"] == 0.12

    rim_formula, _ = _pick_transport_place("bowl_1", "table")
    hints = {"bowl_1": {"grasp_target": "rim", "grasp_offset_xyz": [0.0, 0.045, 0.05]}}
    rim_subtasks = decompose(rim_formula, object_hints=hints)
    assert rim_subtasks[0]["params"]["grasp_target"] == "rim"
    assert rim_subtasks[0]["params"]["grasp_offset_xyz"] == [0.0, 0.045, 0.05]


def test_global_safety_constraint_is_folded_into_every_subtask():
    safety = {"type": "always", "interval": None, "child": _predicate("Safe", ["red_zone"])}
    formula = {"type": "and", "children": [safety, _transport_goal("cup_1", "goal_region")]}
    subtasks = decompose(formula)

    assert len(subtasks) == 1
    assert subtasks[0]["formula"]["type"] == "and"
    assert safety in subtasks[0]["formula"]["children"]


def test_single_atomic_goal_task_needs_no_and_wrapper():
    formula = {"type": "eventually", "interval": [0, 150], "child": _predicate("TurnedOn", ["stove_1"])}
    subtasks = decompose(formula)
    assert len(subtasks) == 1
    assert subtasks[0]["skill"] == "turn_on"
    assert subtasks[0]["params"] == {"object": "stove_1"}


def test_open_pick_transport_place_close_sequence_full_pipeline():
    formula = {
        "type": "and",
        "children": [
            {"type": "eventually", "interval": [0, 120], "child": _predicate("Opened", ["bottom_drawer"])},
            _pick_goal("bowl_1", start=120),
            _transport_goal("bowl_1", "bottom_drawer", start=210),
            _place_goal("bowl_1", "bottom_drawer", start=390),
            {"type": "eventually", "interval": [510, 630], "child": _predicate("Closed", ["bottom_drawer"])},
        ],
    }
    subtasks = decompose(formula)
    assert [s["skill"] for s in subtasks] == ["open", "pick", "transport", "place", "close"]

    plan = {"system2_plan": {"type": "skill_graph", "task_name": "t", "scene_summary": "s", "subtasks": subtasks}}
    assert validate_stl_plan(plan).ok


def test_split_top_level_and_separates_constraints_from_goals():
    safety = {"type": "always", "interval": None, "child": _predicate("Safe", ["red_zone"])}
    goal = _transport_goal("cup_1", "goal_region")
    constraints, goals = split_top_level_and({"type": "and", "children": [safety, goal]})
    assert constraints == [safety]
    assert goals == [goal]


def test_constraint_predicates_nested_in_a_goal_do_not_change_its_skill():
    # Shapes System 2 produced on the v2 benchmark: a constraint inside the goal.
    transport = {"type": "and", "children": [
        _transport_goal("bowl", "plate"),
        {"type": "always", "interval": None, "child": _predicate("Safe", ["bowl", "milk", 0.08])},
        {"type": "always", "interval": None, "child": _predicate("Upright", ["bowl", 5])},
    ]}
    assert infer_skill_from_predicates(transport) == "transport"
    place = {"type": "eventually", "interval": [0, 120], "child": {"type": "and", "children": [
        _predicate("Placed", ["bowl", "plate"]), _predicate("Released", ["bowl"]), _predicate("InRegion", ["bowl", "plate"]),
    ]}}
    assert infer_skill_from_predicates(place) == "place"
    try:
        infer_skill_from_predicates(_predicate("Safe", ["bowl", "milk", 0.08]))
        assert False, "a goal with only constraints names no skill"
    except DecompositionError:
        pass


def test_infer_skill_unknown_predicate_set_raises():
    weird = {"type": "predicate", "name": "Frobnicated", "args": ["x"]}
    try:
        infer_skill_from_predicates(weird)
        assert False, "expected DecompositionError"
    except DecompositionError:
        pass


def test_validate_global_stl_plan_accepts_stl_spec_shape():
    plan = {
        "system2_plan": {
            "type": "stl_spec",
            "task_name": "t",
            "scene_summary": "s",
            "formula": _transport_goal("cup_1", "goal_region"),
            "object_hints": {"cup_1": {"grasp_target": "rim"}},
        }
    }
    assert validate_global_stl_plan(plan).ok


def test_validate_global_stl_plan_rejects_wrong_type():
    plan = {
        "system2_plan": {
            "type": "skill_graph",
            "task_name": "t",
            "scene_summary": "s",
            "formula": _transport_goal("cup_1", "goal_region"),
        }
    }
    result = validate_global_stl_plan(plan)
    assert not result.ok


def test_decompose_and_validate_end_to_end_success():
    formula, _ = _pick_transport_place("cup_1", "goal_region")
    plan = {"system2_plan": {"type": "stl_spec", "task_name": "t", "scene_summary": "s", "formula": formula}}
    decomposed_plan, result = decompose_and_validate(plan)
    assert result.ok
    assert decomposed_plan is not None
    subtasks = decomposed_plan["system2_plan"]["subtasks"]
    assert [s["skill"] for s in subtasks] == ["pick", "transport", "place"]


def test_decompose_and_validate_surfaces_unknown_predicate_error():
    plan = {
        "system2_plan": {
            "type": "stl_spec",
            "task_name": "t",
            "scene_summary": "s",
            "formula": {"type": "predicate", "name": "Frobnicated", "args": ["x"]},
        }
    }
    decomposed_plan, result = decompose_and_validate(plan)
    assert decomposed_plan is None
    assert not result.ok
    assert any("Frobnicated" in e for e in result.errors)


# -- SLICE absolute-timeline semantics (the bug found on 2026-09-12) --------


def test_goal_absolute_window_reads_min_start_max_end():
    goal = _pick_goal("cup_1", start=90)  # PreGrasp [90,130], Grasped [90,180]
    assert goal_absolute_window(goal) == (90, 180)


def test_goal_absolute_window_none_for_untimed_goal():
    assert goal_absolute_window({"type": "predicate", "name": "Frobnicated", "args": ["x"]}) is None


def test_decompose_preserves_absolute_intervals_does_not_rebase_to_zero():
    """The whole point of SLICE: a later goal's interval stays absolute
    (e.g. [90,270] for a transport that starts where a 90-step pick ended),
    it must NOT come back out of decompose() rewritten to start at 0."""
    formula, _ = _pick_transport_place("cup_1", "goal_region")
    subtasks = decompose(formula)
    transport_formula = subtasks[1]["formula"]
    assert transport_formula["interval"] == [90, 270]
    place_formula = subtasks[2]["formula"]["child"] if subtasks[2]["formula"]["type"] == "eventually" else None
    assert subtasks[2]["formula"]["interval"] == [270, 390]


def test_decompose_rejects_overlapping_absolute_windows():
    """A transport starting at 50 while pick's own window still runs to 90
    is exactly the per-goal-local-reset bug this validation exists to catch."""
    formula = {
        "type": "and",
        "children": [_pick_goal("cup_1", start=0), _transport_goal("cup_1", "goal_region", start=50)],
    }
    try:
        decompose(formula)
        assert False, "expected DecompositionError for overlapping absolute windows"
    except DecompositionError as exc:
        assert "overlapping" in str(exc) or "starts before" in str(exc)


def test_decompose_rejects_two_goals_both_reset_to_zero():
    """The literal bug: two goals both written as if a local clock resets to
    0 for each one (F[0,180] then F[0,120]) instead of a shared timeline."""
    formula = {
        "type": "and",
        "children": [_transport_goal("cup_1", "goal_region", start=0), _place_goal("cup_1", "goal_region", start=0)],
    }
    try:
        decompose(formula)
        assert False, "expected DecompositionError"
    except DecompositionError:
        pass


def test_decompose_allows_touching_boundaries_not_just_gaps():
    """One goal's window ending exactly where the next begins (no gap, no
    overlap) is the normal, expected case -- must NOT raise."""
    formula = {
        "type": "and",
        "children": [_transport_goal("cup_1", "goal_region", start=0), _place_goal("cup_1", "goal_region", start=180)],
    }
    subtasks = decompose(formula)
    assert len(subtasks) == 2


def test_bounded_invariance_constraint_only_covers_overlapping_subtasks():
    """Algorithm 3: a constraint (l,h,p) is only 'active' during slices whose
    window falls inside [l,h] -- a bounded G[a,b] must NOT be folded into a
    subtask whose own window doesn't overlap it, unlike an unbounded
    (interval: null) constraint, which covers everything."""
    scoped_safety = {"type": "always", "interval": [0, 100], "child": _predicate("Safe", ["red_zone"])}
    formula = {
        "type": "and",
        "children": [
            scoped_safety,
            _pick_goal("cup_1", start=0),  # window [0,90] -- overlaps [0,100]
            _transport_goal("cup_1", "goal_region", start=90),  # window [90,270] -- overlaps [0,100]
            _place_goal("cup_1", "goal_region", start=270),  # window [270,390] -- does NOT overlap [0,100]
        ],
    }
    subtasks = decompose(formula)
    assert subtasks[0]["formula"]["type"] == "and" and scoped_safety in subtasks[0]["formula"]["children"]
    assert subtasks[1]["formula"]["type"] == "and" and scoped_safety in subtasks[1]["formula"]["children"]
    # place's window starts exactly where the constraint's [0,100] ends -- still
    # touches at 100? No: place window is [270,390], well past 100 -- excluded.
    assert subtasks[2]["formula"] == _place_goal("cup_1", "goal_region", start=270)


def test_unbounded_invariance_constraint_covers_every_subtask():
    unbounded_safety = {"type": "always", "interval": None, "child": _predicate("Safe", ["red_zone"])}
    formula, _ = _pick_transport_place("cup_1", "goal_region")
    formula["children"].insert(0, unbounded_safety)
    subtasks = decompose(formula)
    for sub in subtasks:
        assert sub["formula"]["type"] == "and" and unbounded_safety in sub["formula"]["children"]


def test_real_seven_goal_blocking_scenario_has_consistent_cumulative_timing():
    """Mirrors the real 'move bowl aside, open drawer, place bowl' scenario
    seen from the actual VLM (outputs/_stl_decomposition_experiment/) -- full
    cumulative chain across 7 goals must validate with no gaps/overlaps and
    every subtask's own formula must keep its own absolute window."""
    t = 0
    pick1 = _pick_goal("bowl_1", start=t); t += 90
    transport1 = _transport_goal("bowl_1", "table", start=t); t += 180
    place1 = _place_goal("bowl_1", "table", start=t); t += 120
    open_ = {"type": "eventually", "interval": [t, t + 120], "child": _predicate("Opened", ["top_drawer"])}; t += 120
    pick2 = _pick_goal("bowl_1", start=t); t += 90
    transport2 = _transport_goal("bowl_1", "top_drawer", start=t); t += 180
    place2 = _place_goal("bowl_1", "top_drawer", start=t); t += 120

    formula = {"type": "and", "children": [pick1, transport1, place1, open_, pick2, transport2, place2]}
    subtasks = decompose(formula)
    assert [s["skill"] for s in subtasks] == ["pick", "transport", "place", "open", "pick", "transport", "place"]
    assert goal_absolute_window(subtasks[-1]["formula"]) == (t - 120, t)
