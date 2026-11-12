"""Are demo-derived grasps feasible in a fresh episode? IK + collision, no rollout."""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import sys, json, pathlib
sys.path.insert(0, str(_HERE)); sys.path.insert(0, _STEP_ROOT)
import numpy as np, robosuite, robocasa
from robosuite.controllers import load_composite_controller_config
from mimicgen_grasps import apply, CACHE
from precheck_tasks import motion_config_for
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.collision import CollisionChecker, CollisionPhase
from step.system_one.motion.kinematics import IKSolver, IKConfig
from step.system_one.motion import panda

task = sys.argv[1] if len(sys.argv) > 1 else "PickPlaceCounterToStove"
offs = json.loads((CACHE / f"{task}.json").read_text())
cfg = load_composite_controller_config(robot="PandaOmron")
env = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=cfg,
                     has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                     layout_ids=[1], style_ids=[1], seed=0)
env.reset()
sc = SceneQuery.from_robosuite_sim(env.sim)
mc = motion_config_for(env)
ck = CollisionChecker(sc, panda.JOINT_NAMES, panda.EE_SITE,
                      panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
ik = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig())
q0 = sc.get_qpos(panda.JOINT_NAMES)
bid = env.obj_body_id["obj"]
body = env.sim.model.body(bid).name
o_pos = np.asarray(env.sim.data.body_xpos[bid], float)
o_quat = np.asarray(env.sim.data.body_xquat[bid], float)
print(f"{task}: object at {np.round(o_pos,3)}, {len(offs)} demo offsets\n")
print(f"{'ep':>3} {'grasp pos':>22} {'IK':>5} {'collision':>10}  first contact")
ok = 0
for o in offs:
    gp, gq = apply(o, o_pos, o_quat)
    sol = ik.solve(sc.clone(), gp, gq, seed_q=q0)
    if not sol.success:
        print(f"{o['episode']:>3} {str(np.round(gp,3)):>22} {'FAIL':>5}"); continue
    r = ck.check(np.asarray(sol.q, float), phase=CollisionPhase.TERMINAL_APPROACH,
                 target_object_body=body)
    c = r.contacts[0].geom1.split("_g")[0] if r.contacts else ""
    if not r.in_collision: ok += 1
    print(f"{o['episode']:>3} {str(np.round(gp,3)):>22} {'ok':>5} {str(r.in_collision):>10}  {c}")
print(f"\nfeasible: {ok}/{len(offs)}")
