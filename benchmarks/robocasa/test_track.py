"""Does the MPC actually follow the planned path?

Planning is direct (ratio ~1.1), so if the executed motion looks like a dance
the deviation is in tracking. Executes a planned path and compares where the
arm actually went against where the plan said to go.
"""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import sys
sys.path.insert(0, str(_HERE)); sys.path.insert(0, _STEP_ROOT)
import numpy as np
from constrained_task import make_env, build_constrained_context
from robocasa_env import RoboCasaManipulationEnv
from precheck_tasks import motion_config_for
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.collision import CollisionChecker, CollisionPhase
from step.system_one.motion.ompl_planner import MotionPlanner
from step.system_one.motion.kinematics import IKSolver, IKConfig
from step.system_one.motion.skills.contact_graspnet_generator import ContactGraspNetGenerator
from step.system_one.motion import panda

raw = make_env()
env = RoboCasaManipulationEnv(scene_context=build_constrained_context(raw, enforce=True), env=raw)
env.reset()
sc = SceneQuery.from_robosuite_sim(raw.sim)
mc = motion_config_for(raw); mc.contact_graspnet.enabled = True
mc.contact_graspnet.camera_names = ["robot0_agentview_center","robot0_eye_in_hand"]
ck = CollisionChecker(sc, panda.JOINT_NAMES, panda.EE_SITE,
                      panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
q0 = sc.get_qpos(panda.JOINT_NAMES)
lo, hi = sc.joint_qpos_range(panda.JOINT_NAMES)
body = raw.sim.model.body(raw.obj_body_id["obj"]).name
ik = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig())
planner = MotionPlanner(panda.JOINT_NAMES, lo, hi, ck, config=mc.planner)

plan = None
for c in ContactGraspNetGenerator(mc.contact_graspnet).generate(sc, [body]):
    ap = np.asarray(c.approach_direction_world, float); ap /= np.linalg.norm(ap)+1e-9
    tgt = np.asarray(c.grasp_pose_world[:3], float) - ap*mc.grasp.approach_standoff_m
    sol = ik.solve(sc.clone(), tgt, c.grasp_pose_world[3:], seed_q=q0)
    if not sol.success: continue
    p = planner.plan(q0, [np.asarray(sol.q,float)], phase=CollisionPhase.FREE_SPACE)
    if p.success: plan = np.asarray(p.geometric_path, float); break
if plan is None:
    raise SystemExit("no plan")

# Track it with the same servo the adapters use.
from step.system_one.motion.servo import _ServoMixin
print(f"planned path: {len(plan)} waypoints, {np.linalg.norm(np.diff(plan,axis=0),axis=1).sum():.2f} rad")
errs, actual = [], []
for i, qd in enumerate(plan[::4]):
    live = SceneQuery.from_robosuite_sim(raw.sim)
    qnow = live.get_qpos(panda.JOINT_NAMES)
    actual.append(qnow.copy())
    errs.append(float(np.linalg.norm(qnow - qd)))
    # crude joint servo: drive EE toward the waypoint's EE pose
    sc2 = sc.clone(); sc2.set_qpos(panda.JOINT_NAMES, qd)
    tgt_pos, tgt_quat = sc2.site_pose(panda.EE_SITE)
    cur_pos, _ = live.site_pose(panda.EE_SITE)
    a = np.zeros(7, dtype=np.float32)
    a[0:3] = np.clip((tgt_pos - cur_pos)*12.0, -1, 1)
    a[-1] = -1.0
    for _ in range(3):
        env._step_backend(a)
A = np.asarray(actual)
print(f"tracking error vs plan: mean {np.mean(errs):.3f} rad  max {np.max(errs):.3f} rad")
print(f"actual joint path length: {np.linalg.norm(np.diff(A,axis=0),axis=1).sum():.2f} rad")
print(f"   (planned was {np.linalg.norm(np.diff(plan,axis=0),axis=1).sum():.2f} rad)")
