"""Do Contact-GraspNet's 6-DoF grasps clear the elbow collisions that top-down ones cannot?

The privileged generator only proposes a straight-down approach, which in a
kitchen forces the elbow up into overhead cabinets and counter appliances --
0 of 8 candidates survived on CounterToCabinet and ToasterToCounter. CGN is
already installed and already proposes tilted approaches. Same check, both
generators, side by side.
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
from step.system_one.motion.kinematics import IKSolver, IKConfig
from step.system_one.motion.skills.grasp_generator import PrivilegedTopDownGraspGenerator
from step.system_one.motion.skills.contact_graspnet_generator import ContactGraspNetGenerator
from step.system_one.motion import panda

TASKS = ["PickPlaceCounterToCabinet", "PickPlaceToasterToCounter",
         "PickPlaceCounterToStove", "PickPlaceDrawerToCounter", "PickPlaceSinkToCounter"]


def tilt_deg(c):
    a = np.asarray(c.approach_direction_world, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(-a[2], -1.0, 1.0))))


def evaluate(sc, mc, ck, ik, q0, body, cands, maxw):
    fit = ok = 0
    tilts, blockers = [], {}
    for c in cands:
        if c.gripper_width is not None and c.gripper_width > maxw:
            continue
        fit += 1
        sol = ik.solve(sc.clone(), c.grasp_pose_world[:3], c.grasp_pose_world[3:], seed_q=q0)
        if not sol.success:
            continue
        r = ck.check(np.asarray(sol.q, float), phase=CollisionPhase.TERMINAL_APPROACH,
                     target_object_body=body)
        if not r.in_collision:
            ok += 1
            tilts.append(tilt_deg(c))
        elif r.contacts:
            k = r.contacts[0].geom1.split("_g")[0]
            blockers[k] = blockers.get(k, 0) + 1
    return fit, ok, tilts, blockers


print(f"{'task':<28} {'privileged (top-down)':>22} {'Contact-GraspNet (6-DoF)':>26}")
for task in TASKS:
    cfg = load_composite_controller_config(robot="PandaOmron")
    env = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=cfg,
                         has_renderer=False, has_offscreen_renderer=True,
                         use_camera_obs=False,
                         camera_names=["robot0_agentview_center", "robot0_eye_in_hand"],
                         camera_heights=240, camera_widths=320,
                         layout_ids=[1], style_ids=[1], seed=0)
    env.reset()
    sc = SceneQuery.from_robosuite_sim(env.sim)
    mc = motion_config_for(env)
    mc.contact_graspnet.enabled = True
    mc.contact_graspnet.camera_names = ["robot0_agentview_center", "robot0_eye_in_hand"]
    ck = CollisionChecker(sc, panda.JOINT_NAMES, panda.EE_SITE,
                          panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
    ik = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig())
    q0 = sc.get_qpos(panda.JOINT_NAMES)
    body = env.sim.model.body(env.obj_body_id["obj"]).name

    pg = PrivilegedTopDownGraspGenerator(mc.grasp)
    p_fit, p_ok, _, p_bl = evaluate(sc, mc, ck, ik, q0, body, pg.generate(sc, [body]),
                                    pg.gripper_max_opening_m)
    try:
        cg = ContactGraspNetGenerator(mc.contact_graspnet)
        cands = cg.generate(sc, [body])
        c_fit, c_ok, c_t, c_bl = evaluate(sc, mc, ck, ik, q0, body, cands, pg.gripper_max_opening_m)
        cgs = f"{len(cands):>3} cand {c_fit:>3} fit {c_ok:>3} free"
    except Exception as e:
        c_ok, c_t, c_bl = 0, [], {}
        cgs = f"error {type(e).__name__}"
    flag = "OK " if (p_ok or c_ok) else "!! "
    print(f"{flag}{task:<26} {f'{p_fit:>3} fit {p_ok:>3} free':>22} {cgs:>26}")
    if c_t:
        print(f"      CGN clear grasps tilt {min(c_t):.0f}-{max(c_t):.0f} deg from vertical")
    if not c_ok and c_bl:
        print(f"      CGN blocked by: {', '.join(list(c_bl)[:2])}")
    env.close()
