"""What exactly does the rejected edge collide with?

The planner refuses the path on its own re-validation pass, so the waypoints are
never returned. Capture them by letting check_edge report but not veto, then walk
the path and name the colliding geom pair -- that distinguishes "the kitchen has
a genuinely narrow passage" from "our checker counts kitchen structure as an
obstacle it should not".
"""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import sys
sys.path.insert(0, _STEP_ROOT)
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
import robocasa
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.collision import CollisionChecker, CollisionPhase
from step.system_one.motion.ompl_planner import MotionPlanner
from step.system_one.motion.kinematics import IKSolver, IKConfig
from step.system_one.motion.config import MotionConfig
from step.system_one.motion import panda

cfg = load_composite_controller_config(robot="PandaOmron")
env = robosuite.make(env_name="PickPlaceCounterToStove", robots="PandaOmron",
                     controller_configs=cfg, has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False,
                     layout_ids=[1], style_ids=[1], seed=0)
env.reset()
scene = SceneQuery.from_robosuite_sim(env.sim)
mc = MotionConfig.from_scene_context({})
checker = CollisionChecker(scene, panda.JOINT_NAMES, panda.EE_SITE,
                           panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
q0 = scene.get_qpos(panda.JOINT_NAMES)
lo, hi = scene.joint_qpos_range(panda.JOINT_NAMES)
obj_xyz = np.asarray(env.sim.data.body_xpos[env.obj_body_id["obj"]], dtype=float)
sol = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig()).solve(
    scene.clone(), obj_xyz + np.array([0, 0, 0.12]), np.array([0.0, 1.0, 0.0, 0.0]), seed_q=q0)
q_goal = np.asarray(sol.q, dtype=float)

captured = {}
real_check_edge = checker.check_edge
def spy(a, b, **kw):
    r = real_check_edge(a, b, **kw)
    if r.in_collision and "first" not in captured:
        captured["first"] = (a.copy(), b.copy(), r)
    class Ok:
        in_collision = False
        contacts = []
        min_clearance = float("inf")
    return Ok()
checker.check_edge = spy

planner = MotionPlanner(panda.JOINT_NAMES, lo, hi, checker, config=mc.planner)
plan = planner.plan(q0, [q_goal], phase=CollisionPhase.FREE_SPACE)
print("with edge veto disabled -> success:", plan.success,
      "waypoints:", 0 if plan.geometric_path is None else len(plan.geometric_path))

if "first" in captured:
    a, b, r = captured["first"]
    print(f"\nfirst rejecting edge: {len(r.contacts)} contacts, min_clearance={r.min_clearance:.5f}")
    for c in r.contacts[:8]:
        print("   ", c)
    # Where is the EE along that edge?
    sc = scene.clone()
    for t in (0.0, 0.5, 1.0):
        sc.set_qpos(panda.JOINT_NAMES, a + t * (b - a))
        print(f"   t={t:.1f} EE={np.round(sc.site_pose(panda.EE_SITE)[0], 3)}")
else:
    print("no edge was rejected on this run")
