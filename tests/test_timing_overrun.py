"""EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md): the soft, log-only timing-
overrun path -- decompose() attaching stl_deadline, executor.py flagging an
overrun on a SkillExecutionRecord, and libero_runner.py surfacing it into the
next System-2 recall's scene_context. No live env / rollout needed for any
of this -- it's pure data-plumbing around already-computed step counts."""

from step.libero_runner import _annotate_timing_overruns
from step.system_one.execution_types import SkillExecutionRecord
from step.system_one.executor import System1Executor
from step.system_two.stl_plan import decompose


def _predicate(name, args):
    return {"type": "predicate", "name": name, "args": args}


def _transport_goal(obj, goal_region, start=0):
    return {"type": "eventually", "interval": [start, start + 180], "child": _predicate("OverGoal", [obj, goal_region])}


def test_decompose_attaches_stl_deadline_matching_absolute_window():
    formula = _transport_goal("cup_1", "goal_region", start=90)
    subtasks = decompose(formula)
    assert subtasks[0]["stl_deadline"] == {"start_step": 90, "end_step": 270}


def test_annotate_timing_violation_noop_when_within_budget():
    sub = {"stl_deadline": {"start_step": 0, "end_step": 180}}
    record = SkillExecutionRecord(skill="transport", params={"object": "cup_1"}, steps=150, done=True, info={})
    System1Executor._annotate_timing_violation(None, record, sub, task_steps_so_far=0)
    assert "timing_violation" not in record.info


def test_annotate_timing_violation_noop_when_no_deadline_declared():
    """Legacy (--stl-mode skill_graph) subtasks never carry stl_deadline --
    must be silently skipped, not raise."""
    sub = {"params": {"object": "cup_1"}}
    record = SkillExecutionRecord(skill="transport", params={"object": "cup_1"}, steps=999, done=False, info={})
    System1Executor._annotate_timing_violation(None, record, sub, task_steps_so_far=0)
    assert "timing_violation" not in record.info


def test_annotate_timing_violation_flags_overrun_on_failure():
    sub = {"stl_deadline": {"start_step": 90, "end_step": 270}, "params": {"object": "cup_1"}}
    record = SkillExecutionRecord(skill="transport", params={"object": "cup_1"}, steps=250, done=False, info={"foo": "bar"})
    System1Executor._annotate_timing_violation(None, record, sub, task_steps_so_far=90)
    tv = record.info["timing_violation"]
    assert tv == {
        "skill": "transport",
        "object": "cup_1",
        "declared_deadline_step": 270,
        "actual_absolute_step": 340,
        "overrun_steps": 70,
        "subtask_succeeded": False,
    }
    assert record.info["foo"] == "bar"  # existing info preserved, not clobbered


def test_annotate_timing_violation_flags_overrun_even_on_late_success():
    sub = {"stl_deadline": {"start_step": 0, "end_step": 90}, "params": {"object": "cup_1"}}
    record = SkillExecutionRecord(skill="pick", params={"object": "cup_1"}, steps=120, done=True, info={})
    System1Executor._annotate_timing_violation(None, record, sub, task_steps_so_far=0)
    assert record.info["timing_violation"]["subtask_succeeded"] is True
    assert record.info["timing_violation"]["overrun_steps"] == 30


def test_annotate_timing_overruns_stamps_note_and_returns_object_names():
    fresh_context = {"entities": [{"name": "cup_1", "role": "object"}]}
    previous_summary = {
        "records": [
            {
                "skill": "transport",
                "info": {
                    "timing_violation": {
                        "skill": "transport",
                        "object": "cup_1",
                        "declared_deadline_step": 270,
                        "actual_absolute_step": 340,
                        "overrun_steps": 70,
                        "subtask_succeeded": False,
                    }
                },
            }
        ]
    }
    annotated = _annotate_timing_overruns(fresh_context, previous_summary)
    assert annotated == ["cup_1"]
    notes = fresh_context["entities"][0]["notes"]
    assert "340 controller-steps" in notes
    assert "exceeding its declared" in notes
    assert "270" in notes and "70" in notes
    assert "did not succeed" in notes


def test_annotate_timing_overruns_creates_entity_when_object_unknown():
    fresh_context = {"entities": []}
    previous_summary = {
        "records": [
            {
                "info": {
                    "timing_violation": {
                        "skill": "turn_on",
                        "object": None,
                        "declared_deadline_step": 150,
                        "actual_absolute_step": 200,
                        "overrun_steps": 50,
                        "subtask_succeeded": True,
                    }
                }
            }
        ]
    }
    annotated = _annotate_timing_overruns(fresh_context, previous_summary)
    assert annotated == ["turn_on"]
    assert len(fresh_context["entities"]) == 1
    assert "succeeded, but later than planned" in fresh_context["entities"][0]["notes"]


def test_annotate_timing_overruns_noop_on_clean_run():
    fresh_context = {"entities": [{"name": "cup_1", "role": "object"}]}
    previous_summary = {"records": [{"skill": "pick", "info": {}}]}
    assert _annotate_timing_overruns(fresh_context, previous_summary) == []
    assert fresh_context["entities"][0].get("notes") is None


def test_annotate_timing_overruns_noop_on_exception_fallback_summary():
    """A prior attempt that raised (executor exception, not a plan) leaves
    `summary` without a "records" key at all -- must not crash."""
    fresh_context = {"entities": []}
    previous_summary = {"success": False, "error": "RuntimeError: boom"}
    assert _annotate_timing_overruns(fresh_context, previous_summary) == []
