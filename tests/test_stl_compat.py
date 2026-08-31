from step.system_two.stl_plan import normalize_legacy_pregrasp_grasp, validate_stl_plan


def _predicate(name, args):
    return {"type": "predicate", "name": name, "args": args}


def test_noop_on_already_unified_plan():
    """Today's schema/prompt shape (PreGrasp+Grasped inside one pick subtask's
    formula) must pass through completely unchanged."""
    subtasks = [
        {
            "subtask_id": 0,
            "skill": "pick",
            "description": "pick the cup",
            "params": {"object": "cup_1"},
            "formula": {
                "type": "and",
                "children": [
                    {"type": "eventually", "interval": [0, 40], "child": _predicate("PreGrasp", ["cup_1", "top"])},
                    {"type": "eventually", "interval": [0, 90], "child": _predicate("Grasped", ["cup_1"])},
                ],
            },
        },
        {
            "subtask_id": 1,
            "skill": "transport",
            "description": "move",
            "params": {"object": "cup_1", "goal_region": "goal_region"},
            "formula": {"type": "eventually", "interval": [0, 180], "child": _predicate("OverGoal", ["cup_1", "goal_region"])},
        },
    ]
    result = normalize_legacy_pregrasp_grasp(subtasks)
    assert result == subtasks


def test_merges_explicit_legacy_skill_names():
    subtasks = [
        {
            "subtask_id": 0,
            "skill": "pregrasp",
            "description": "approach",
            "params": {"object": "cup_1", "grasp_mode": "top"},
            "formula": {"type": "eventually", "interval": [0, 40], "child": _predicate("PreGrasp", ["cup_1", "top"])},
        },
        {
            "subtask_id": 1,
            "skill": "grasp",
            "description": "close",
            "params": {"object": "cup_1"},
            "formula": {"type": "eventually", "interval": [0, 90], "child": _predicate("Grasped", ["cup_1"])},
        },
        {
            "subtask_id": 2,
            "skill": "transport",
            "description": "move",
            "params": {"object": "cup_1", "goal_region": "goal_region"},
            "formula": {"type": "eventually", "interval": [0, 180], "child": _predicate("OverGoal", ["cup_1", "goal_region"])},
        },
    ]
    merged = normalize_legacy_pregrasp_grasp(subtasks)
    assert len(merged) == 2
    assert merged[0]["skill"] == "pick"
    assert merged[0]["subtask_id"] == 0
    assert merged[0]["params"]["object"] == "cup_1"
    assert merged[0]["params"]["grasp_mode"] == "top"
    assert merged[0]["formula"]["type"] == "and"
    assert merged[1]["skill"] == "transport"
    assert merged[1]["subtask_id"] == 1
    # Merged plan must still validate against the current schema.
    fake_plan = {"system2_plan": {"type": "skill_graph", "task_name": "t", "scene_summary": "s", "subtasks": merged}}
    assert validate_stl_plan(fake_plan).ok


def test_merges_pick_only_pregrasp_followed_by_pick_only_grasped():
    subtasks = [
        {
            "subtask_id": 0,
            "skill": "pick",
            "description": "approach",
            "params": {"object": "cup_1"},
            "formula": {"type": "eventually", "interval": [0, 40], "child": _predicate("PreGrasp", ["cup_1", "top"])},
        },
        {
            "subtask_id": 1,
            "skill": "pick",
            "description": "close",
            "params": {"object": "cup_1"},
            "formula": {"type": "eventually", "interval": [0, 90], "child": _predicate("Grasped", ["cup_1"])},
        },
    ]
    merged = normalize_legacy_pregrasp_grasp(subtasks)
    assert len(merged) == 1
    assert merged[0]["skill"] == "pick"


def test_does_not_merge_different_objects():
    subtasks = [
        {
            "subtask_id": 0,
            "skill": "pregrasp",
            "description": "approach cup",
            "params": {"object": "cup_1"},
            "formula": {"type": "eventually", "interval": [0, 40], "child": _predicate("PreGrasp", ["cup_1", "top"])},
        },
        {
            "subtask_id": 1,
            "skill": "grasp",
            "description": "close on bowl",
            "params": {"object": "bowl_1"},
            "formula": {"type": "eventually", "interval": [0, 90], "child": _predicate("Grasped", ["bowl_1"])},
        },
    ]
    merged = normalize_legacy_pregrasp_grasp(subtasks)
    assert len(merged) == 2
