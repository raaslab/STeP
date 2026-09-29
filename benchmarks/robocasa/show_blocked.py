"""Render the arm at a blocked grasp pose so the collision is visible."""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import sys
sys.path.insert(0, str(_HERE)); sys.path.insert(0, _STEP_ROOT)
import numpy as np, cv2, robosuite, robocasa
from robosuite.controllers import load_composite_controller_config
from precheck_tasks import motion_config_for
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.collision import CollisionChecker, CollisionPhase
from step.system_one.motion.kinematics import IKSolver, IKConfig
from step.system_one.motion.skills.grasp_generator import PrivilegedTopDownGraspGenerator
from step.system_one.motion import panda

tiles = []
for task in ("PickPlaceCounterToCabinet", "PickPlaceToasterToCounter", "PickPlaceCounterToStove"):
    cfg = load_composite_controller_config(robot="PandaOmron")
    env = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=cfg,
                         has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
                         camera_names=["robot0_frontview"], camera_heights=480, camera_widths=640,
                         layout_ids=[1], style_ids=[1], seed=0)
    env.reset()
    sc = SceneQuery.from_robosuite_sim(env.sim)
    mc = motion_config_for(env)
    ck = CollisionChecker(sc, panda.JOINT_NAMES, panda.EE_SITE,
                          panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
    q0 = sc.get_qpos(panda.JOINT_NAMES)
    body = env.sim.model.body(env.obj_body_id["obj"]).name
    gen = PrivilegedTopDownGraspGenerator(mc.grasp)
    ik = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig())
    label, qshow = "no IK solution", None
    for c in gen.generate(sc, [body]):
        if c.gripper_width is not None and c.gripper_width > gen.gripper_max_opening_m:
            continue
        sol = ik.solve(sc.clone(), c.grasp_pose_world[:3], c.grasp_pose_world[3:], seed_q=q0)
        if not sol.success:
            continue
        q = np.asarray(sol.q, float)
        r = ck.check(q, phase=CollisionPhase.TERMINAL_APPROACH, target_object_body=body)
        qshow = q
        label = ("CLEAR" if not r.in_collision else
                 f"HITS {r.contacts[0].geom1.split('_g')[0]}")
        if not r.in_collision:
            break
    if qshow is not None:
        ids = [env.sim.model.joint(n).id for n in panda.JOINT_NAMES]
        env.sim.data.qpos[env.sim.model.jnt_qposadr[ids]] = qshow
        env.sim.forward()
    img = np.ascontiguousarray(env.sim.render(width=640, height=480, camera_name="robot0_frontview")[::-1])
    ok = label == "CLEAR"
    cv2.rectangle(img, (0, 0), (640, 52), (25, 25, 25), -1)
    cv2.putText(img, task, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, label, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 240, 120) if ok else (90, 90, 255), 1, cv2.LINE_AA)
    if not ok:
        cv2.rectangle(img, (2, 2), (637, 477), (70, 70, 255), 3)
    tiles.append(img); print(task, "->", label)
    env.close()
cv2.imwrite("blocked_postures.png", cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
print("wrote blocked_postures.png")
