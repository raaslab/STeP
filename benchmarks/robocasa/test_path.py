"""Is the PLANNED path a detour, or is execution wandering off a good plan?

Measures joint-space path length against the straight-line distance between
start and goal. A ratio near 1 means the plan is direct; a large ratio means
RRTConnect is routing the long way round, which is what a "360 degree turn"
looks like from outside.
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
import numpy as np, robosuite, robocasa
from robosuite.controllers import load_composite_controller_config
from precheck_tasks import motion_config_for
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.collision import CollisionChecker, CollisionPhase
from step.system_one.motion.ompl_planner import MotionPlanner
from step.system_one.motion.kinematics import IKSolver, IKConfig
from step.system_one.motion.skills.contact_graspnet_generator import ContactGraspNetGenerator
from step.system_one.motion import panda

cfg = load_composite_controller_config(robot="PandaOmron")
env = robosuite.make(env_name="PickPlaceCounterToStove", robots="PandaOmron", controller_configs=cfg,
                     has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
                     camera_names=["robot0_agentview_center","robot0_eye_in_hand"],
                     camera_heights=240, camera_widths=320, layout_ids=[1], style_ids=[1], seed=0)
env.reset()
sc = SceneQuery.from_robosuite_sim(env.sim)
mc = motion_config_for(env); mc.contact_graspnet.enabled = True
mc.contact_graspnet.camera_names = ["robot0_agentview_center","robot0_eye_in_hand"]
ck = CollisionChecker(sc, panda.JOINT_NAMES, panda.EE_SITE,
                      panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
q0 = sc.get_qpos(panda.JOINT_NAMES)
lo, hi = sc.joint_qpos_range(panda.JOINT_NAMES)
body = env.sim.model.body(env.obj_body_id["obj"]).name
ik = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig())
planner = MotionPlanner(panda.JOINT_NAMES, lo, hi, ck, config=mc.planner)

cands = ContactGraspNetGenerator(mc.contact_graspnet).generate(sc, [body])
print(f"{len(cands)} candidates; planning to each reachable one\n")
print(f"{'#':>2} {'direct(rad)':>11} {'path(rad)':>10} {'ratio':>6} {'wpts':>5}  max single-joint move")
for i, c in enumerate(cands[:6]):
    # Plan to the STANDOFF, which is what PickSkill plans to under FREE_SPACE;
    # the grasp pose itself touches the object and is only valid under
    # TERMINAL_APPROACH.
    ap = np.asarray(c.approach_direction_world, float)
    ap = ap / (np.linalg.norm(ap) + 1e-9)
    standoff = np.asarray(c.grasp_pose_world[:3], float) - ap * mc.grasp.approach_standoff_m
    sol = ik.solve(sc.clone(), standoff, c.grasp_pose_world[3:], seed_q=q0)
    if not sol.success:
        continue
    qg = np.asarray(sol.q, float)
    p = planner.plan(q0, [qg], phase=CollisionPhase.FREE_SPACE)
    if not p.success:
        print(f"{i:>2}  plan failed ({str(p.failure_reason).split('.')[-1]})"); continue
    W = np.asarray(p.geometric_path, float)
    direct = float(np.linalg.norm(qg - q0))
    path = float(np.linalg.norm(np.diff(W, axis=0), axis=1).sum())
    per_joint = np.abs(np.diff(W, axis=0)).sum(axis=0)
    print(f"{i:>2} {direct:11.3f} {path:10.3f} {path/max(direct,1e-6):6.2f} {len(W):>5}  "
          f"j{int(np.argmax(per_joint))+1}={per_joint.max():.2f} rad")
