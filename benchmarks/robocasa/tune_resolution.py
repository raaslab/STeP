"""How fine does edge checking need to be for a kitchen?

LIBERO's 0.01 was tuned on a tabletop where nothing is thin. A microwave door is,
and a 3.8 mm interpenetration falls between samples at that spacing. Tightening
validity_resolution helps twice over: RRTConnect's own search avoids the door
during planning, and the post-hoc re-check then has nothing to reject.
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
q0 = scene.get_qpos(panda.JOINT_NAMES)
lo, hi = scene.joint_qpos_range(panda.JOINT_NAMES)
obj_xyz = np.asarray(env.sim.data.body_xpos[env.obj_body_id["obj"]], dtype=float)
sol = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig()).solve(
    scene.clone(), obj_xyz + np.array([0, 0, 0.12]), np.array([0.0, 1.0, 0.0, 0.0]), seed_q=q0)
q_goal = np.asarray(sol.q, dtype=float)
span = float(np.max(hi - lo))

print(f"{'validity_res':>12} {'edge step':>10} {'success':>8} {'waypts':>7} {'time':>7}  reason")
for vr in (0.01, 0.005, 0.002, 0.001):
    mc = MotionConfig.from_scene_context({})
    mc.planner.validity_resolution = vr
    checker = CollisionChecker(scene, panda.JOINT_NAMES, panda.EE_SITE,
                               panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
    planner = MotionPlanner(panda.JOINT_NAMES, lo, hi, checker, config=mc.planner)
    t0 = time.time()
    plan = planner.plan(q0, [q_goal], phase=CollisionPhase.FREE_SPACE)
    dt = time.time() - t0
    n = 0 if plan.geometric_path is None else len(plan.geometric_path)
    reason = "" if plan.success else str(plan.failure_reason).split(".")[-1]
    print(f"{vr:>12.4f} {vr*span:>10.4f} {str(plan.success):>8} {n:>7} {dt:>6.1f}s  {reason}")
