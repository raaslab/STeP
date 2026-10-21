"""Individual manipulation skills (pick, place, open-drawer, ...), each in
its own subdirectory, built on the shared motion primitives in the sibling
motion/ package (collision.py, kinematics.py, mpc.py, ompl_planner.py,
scene_query.py, config.py, types.py, panda.py, servo.py, time_parameterize.py,
articulation.py, attached_object.py).

Shared across more than one skill, so kept at this top level instead of
inside any one skill's subdirectory:

- standoff_search.py -- IK standoff-point search (pick, open_drawer, close).
- ik_utils.py -- IK-retry (solve_ik_robust) and collision-escape
  (escape_start_collision) helpers (pick, place, transport, open_drawer,
  close).
- registry.py -- maps a plan's skill name to the System1Executor
  method that runs it.

To add a new skill:

1. A subdirectory (e.g. pour/) holding its own (skill, adapter) pair
   -- pour_skill.py (planning, backend-agnostic) + libero_pour_adapter.py
   (execution against a live env, using servo.track_and_settle). Add a
   generator file too if it needs one, like pick's grasp_generator.py.
2. The skill's own config dataclass lives in its skill file (see
   DrawerConfig in open_drawer/open_drawer_skill.py), not in
   motion/config.py -- wire it into MotionConfig there.
3. Add a "pour": "_run_pour_with_motion_pipeline" entry to
   registry.OMPL_MPC_SKILL_HANDLERS, and the method itself on
   System1Executor.
4. Add "pour" to schemas/stl_plan_schema.json's skill enum.

tests/test_skill_registry_schema_consistency.py checks steps 3-4 stay in
sync.
"""
