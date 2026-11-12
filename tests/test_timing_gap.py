"""The delay a formula declares must survive a run that overruns its own plan.

System 2 writes a nominal timeline -- pick 0-90, transport 90-270, place 430-550 --
and the 160 steps between 270 and 430 are the delay the instruction asked for. The
nominal timeline is not a stopwatch, though: a real LIBERO pick takes around 350
steps against that budgeted 90. The original check asked whether the run had
reached absolute step 430, which by then it always had, so every declared delay
was skipped. These pin the gap reading instead.
"""

from step.system_one.executor import System1Executor as E


def _f(lo, hi, kind="eventually"):
    return {"type": kind, "interval": [lo, hi], "child": {"type": "pred"}}


def test_declared_gap_is_held_even_when_the_run_has_overrun_the_plan():
    # transport's window ended at 270, place opens at 430, and the real run is
    # already at step 446 -- past 430, which is exactly when the old check gave up.
    gap, why = E._timing_gap_to_honour(430, 270, 446)
    assert gap == 160
    assert "gap" in why


def test_no_gap_declared_means_no_hold():
    assert E._timing_gap_to_honour(90, 90, 348) == (0, "")


def test_absolute_shortfall_still_applies_when_it_is_the_larger():
    # A plan running ahead of its own schedule: no declared gap, but the window
    # has not opened yet. This is the behaviour that existed before the fix.
    gap, why = E._timing_gap_to_honour(430, 430, 300)
    assert gap == 130
    assert "window" in why


def test_first_subtask_has_nothing_to_wait_after():
    assert E._timing_gap_to_honour(0, None, 0) == (0, "")
    assert E._timing_gap_to_honour(150, None, 0)[0] == 150


def test_window_bounds_skip_persistent_safety_terms():
    # A G(...) term holds from step 0 by definition. Counting it would make every
    # subtask look like it opens at 0 and ends at the horizon, so no gap would
    # ever be seen.
    formula = {"type": "and", "children": [
        {"type": "always", "interval": [0, 600], "child": {"type": "pred"}},
        _f(430, 550),
    ]}
    sub = {"formula": formula}
    assert E._subtask_interval_start(sub) == 430
    assert E._subtask_interval_end(sub) == 550


def test_window_end_is_the_latest_bound_across_a_conjunction():
    # A pick subtask carries two goals with different windows; the subtask ends
    # when the last of them must be met.
    sub = {"formula": {"type": "and", "children": [_f(0, 40), _f(0, 90)]}}
    assert E._subtask_interval_start(sub) == 0
    assert E._subtask_interval_end(sub) == 90


def test_the_libero_10_3_wait_gap_sits_before_transport():
    # Real plan from the benchmark: the delay was encoded by pushing transport's
    # start out to 170 while pick still ends at 90, not by moving place.
    assert E._timing_gap_to_honour(170, 90, 500)[0] == 80
