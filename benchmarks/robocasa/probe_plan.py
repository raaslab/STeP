"""First OMPL plan inside a RoboCasa kitchen.

Deliberately skips Contact-GraspNet: a straight top-down grasp above the object
isolates whether *planning* works in a 993-geom scene, rather than confounding it
with whether the grasp network likes a kitchen point cloud.
"""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import sys, time
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
target = obj_xyz + np.array([0.0, 0.0, 0.12])          # hover above the eggplant
down = np.array([0.0, 1.0, 0.0, 0.0])                   # top-down, wxyz

ik = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig())
t0 = time.time()
sol = ik.solve(scene.clone(), target, down, seed_q=q0)
print(f"IK: success={sol.success} pos_err={sol.position_error:.4f} ori_err={sol.orientation_error:.4f} iters={sol.iterations} reason={sol.reason}")
if not sol.success:
    raise SystemExit(f"IK failed ({sol.reason}) -- no valid goal to plan to")
q_goal = np.asarray(sol.q, dtype=float)
res = checker.check(q_goal, phase=CollisionPhase.FREE_SPACE)
print(f"goal config in_collision={res.in_collision} contacts={len(res.contacts)}")
for c in res.contacts[:4]:
    print("   ", c)

planner = MotionPlanner(panda.JOINT_NAMES, lo, hi, checker, config=mc.planner)
t0 = time.time()
plan = planner.plan(q0, [q_goal], phase=CollisionPhase.FREE_SPACE)
dt = time.time() - t0
wp = getattr(plan, "geometric_path", None)
n = 0 if wp is None else len(wp)
print(f"PLAN: success={plan.success} in {dt:.1f}s  waypoints={n}")
print("      reason  :", plan.failure_reason)
print("      metadata:", plan.metadata)
if plan.metadata.get("detail") == "post-hoc edge re-validation failed":
    i = plan.metadata["edge_index"]
    print(f"      -> the offending edge is {i}; re-checking its endpoints individually:")
    # Rebuild is not possible here, but the endpoints of a failing edge tell us
    # whether the kitchen geometry is the cause or the resolution is.
    print("         (edge re-validation resolution =",
          mc.planner.validity_resolution * float(np.max(hi - lo)), "rad)")
