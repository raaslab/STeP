"""Render the constrained vs unconstrained transport paths side by side.

Shows the same leg -- above the eggplant to above the pan -- planned with the
lit-burner keep-out enforced and with it only monitored, so the difference in
route is visible rather than only in the clearance numbers.
"""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import sys
sys.path.insert(0, str(_HERE))
sys.path.insert(0, _STEP_ROOT)
import numpy as np, cv2, imageio.v2 as imageio
import robosuite, robocasa
from robosuite.controllers import load_composite_controller_config
from constrained_task import build_constrained_context, burner_xy, ZONE_RADIUS_M
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.collision import CollisionChecker, CollisionPhase
from step.system_one.motion.ompl_planner import MotionPlanner
from step.system_one.motion.kinematics import IKSolver, IKConfig
from step.system_one.motion.config import MotionConfig
from step.system_one.motion import panda

CAM, W, H, RETRIES = "robot0_frontview", 640, 480, 8

cfg = load_composite_controller_config(robot="PandaOmron")
env = robosuite.make(env_name="PickPlaceCounterToStove", robots="PandaOmron",
                     controller_configs=cfg, has_renderer=False,
                     has_offscreen_renderer=True, use_camera_obs=False,
                     camera_names=[CAM], camera_heights=H, camera_widths=W,
                     layout_ids=[1], style_ids=[1], seed=0)
env.reset()
scene = SceneQuery.from_robosuite_sim(env.sim)
lo, hi = scene.joint_qpos_range(panda.JOINT_NAMES)
qh = scene.get_qpos(panda.JOINT_NAMES)
obj = np.asarray(env.sim.data.body_xpos[env.obj_body_id["obj"]], float)
pan = np.asarray(env.sim.data.body_xpos[env.obj_body_id["container"]], float)
bxy = burner_xy(env)[:2]
down = np.array([0.0, 1.0, 0.0, 0.0])
ik = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig())
q0 = np.asarray(ik.solve(scene.clone(), obj + [0, 0, 0.15], down, seed_q=qh).q, float)
qg = np.asarray(ik.solve(scene.clone(), pan + [0, 0, 0.15], down, seed_q=qh).q, float)

def plan_for(enforce):
    mc = MotionConfig.from_scene_context(build_constrained_context(env, enforce=enforce))
    mc.planner.validity_resolution = 0.001
    ck = CollisionChecker(scene, panda.JOINT_NAMES, panda.EE_SITE,
                          panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
    pl = MotionPlanner(panda.JOINT_NAMES, lo, hi, ck, config=mc.planner)
    for _ in range(RETRIES):
        p = pl.plan(q0, [qg], phase=CollisionPhase.FREE_SPACE)
        if p.success:
            return p.geometric_path
    return None

paths = {"STL OFF  (monitored only)": plan_for(False),
         "STL ON  (keep-out enforced)": plan_for(True)}
for k, v in paths.items():
    print(k, "->", "failed" if v is None else f"{len(v)} waypoints")

def clearance(q):
    scene.set_qpos(panda.JOINT_NAMES, q)
    return float(np.linalg.norm(scene.site_pose(panda.EE_SITE)[0][:2] - bxy))

def frames_for(path, label):
    out = []
    for q in path:
        env.sim.data.qpos[env.sim.model.jnt_qposadr[
            [env.sim.model.joint(n).id for n in panda.JOINT_NAMES]]] = q
        env.sim.forward()
        img = env.sim.render(width=W, height=H, camera_name=CAM)[::-1]
        img = np.ascontiguousarray(img)
        d = clearance(q)
        bad = d < ZONE_RADIUS_M
        cv2.rectangle(img, (0, 0), (W, 58), (20, 20, 20), -1)
        cv2.putText(img, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"burner clearance {d*100:5.1f} cm  (zone {ZONE_RADIUS_M*100:.0f} cm)",
                    (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (80, 80, 255) if bad else (120, 255, 120), 1, cv2.LINE_AA)
        if bad:
            cv2.rectangle(img, (2, 2), (W - 3, H - 3), (60, 60, 255), 4)
        out.append(img)
    return out

seqs = {k: frames_for(v, k) for k, v in paths.items() if v is not None}
n = max(len(v) for v in seqs.values())
pad = {k: v + [v[-1]] * (n - len(v)) for k, v in seqs.items()}
keys = list(pad)
writer = imageio.get_writer("keepout_ab.mp4", fps=25)
for i in range(n):
    writer.append_data(np.hstack([pad[k][i] for k in keys]))
for _ in range(40):
    writer.append_data(np.hstack([pad[k][-1] for k in keys]))
writer.close()
print("wrote keepout_ab.mp4")
