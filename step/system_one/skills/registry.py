"""Maps a subtask's skill name to the System1Executor method that
runs it under the OMPL/MPC motion pipeline (when
scene_context["platform"]["libero"]["motion_backend"] == "ompl_mpc").

Every handler here must accept (self, env, sub, *, task_steps_so_far=0,
outdir=None), though some ignore kwargs they don't need. Held as method
names (strings), not bound methods, since this module must not import
executor.py (which imports this registry)."""

from __future__ import annotations

OMPL_MPC_SKILL_HANDLERS: dict[str, str] = {
    "pick": "_run_pick_with_motion_pipeline",
    "place": "_run_place_with_motion_pipeline",
    "transport": "_run_transport_with_motion_pipeline",
    "open": "_run_open_drawer_with_motion_pipeline",
    "close": "_run_close_with_motion_pipeline",
    "turn_on": "_run_turn_knob_with_motion_pipeline",
}
