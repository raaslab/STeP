"""Per-task grasp feasibility pre-check -- no rollout, no System 2.

Answers, for each task, the question that every failed rollout so far turned out
to hinge on: is there any grasp candidate that fits the gripper, solves IK, and
is collision-free? Seconds per task instead of a 30-minute rollout, and it
separates "the pipeline is broken" from "this object cannot be grasped top-down".
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
import numpy as np
import robosuite
import robocasa  # noqa: F401  (registers kitchen envs)
from robosuite.controllers import load_composite_controller_config

from robocasa_fixtures import describe, surface_z, workspace_bounds
from step.system_one.motion.scene_query import SceneQuery
from step.system_one.motion.collision import CollisionChecker, CollisionPhase
from step.system_one.motion.kinematics import IKSolver, IKConfig
from step.system_one.motion.config import MotionConfig
from step.system_one.motion.skills.grasp_generator import PrivilegedTopDownGraspGenerator
from step.system_one.motion import panda

TASKS = ["PickPlaceCounterToCabinet", "PickPlaceCounterToStove",
         "PickPlaceDrawerToCounter", "PickPlaceSinkToCounter",
         "PickPlaceToasterToCounter"]


def motion_config_for(env):
    """Motion config with RoboCasa geometry, read from the fixtures."""
    ctx = {"platform": {"backend": "robocasa", "workspace_bounds": workspace_bounds(env),
                        "robocasa": {"motion": {
                            "planner": {"validity_resolution": 0.002, "timeout_sec": 40.0},
                            "grasp": {"standoff_distance_multipliers": [0.6, 1.0]},
                        }}}}
    return MotionConfig.from_scene_context(ctx)


def precheck(task, seed=0):
    cfg = load_composite_controller_config(robot="PandaOmron")
    env = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=cfg,
                         has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False, layout_ids=[1], style_ids=[1], seed=seed)
    env.reset()
    info = describe(env)
    sc = SceneQuery.from_robosuite_sim(env.sim)
    mc = motion_config_for(env)
    ck = CollisionChecker(sc, panda.JOINT_NAMES, panda.EE_SITE,
                          panda.ROBOT_BODIES, panda.GRIPPER_BODIES, config=mc.collision)
    q0 = sc.get_qpos(panda.JOINT_NAMES)
    body = env.sim.model.body(env.obj_body_id["obj"]).name
    gen = PrivilegedTopDownGraspGenerator(mc.grasp)
    cands = gen.generate(sc, [body])
    ik = IKSolver(panda.JOINT_NAMES, panda.EE_SITE, config=IKConfig())
    maxw = gen.gripper_max_opening_m

    n_fit = n_ik = n_free = 0
    blockers = {}
    for c in cands:
        w = c.gripper_width
        if w is not None and w > maxw:
            continue
        n_fit += 1
        sol = ik.solve(sc.clone(), c.grasp_pose_world[:3], c.grasp_pose_world[3:], seed_q=q0)
        if not sol.success:
            continue
        n_ik += 1
        r = ck.check(np.asarray(sol.q, float), phase=CollisionPhase.TERMINAL_APPROACH,
                     target_object_body=body)
        if not r.in_collision:
            n_free += 1
        elif r.contacts:
            key = f"{r.contacts[0].geom1}<->{r.contacts[0].geom2}"
            blockers[key] = blockers.get(key, 0) + 1
    env.close()
    return {"task": task, "obj_lang": info, "n": len(cands), "fit": n_fit,
            "ik": n_ik, "free": n_free, "blockers": blockers}


if __name__ == "__main__":
    print(f"{'task':<28} {'cands':>5} {'fits':>5} {'+IK':>5} {'+free':>6} {'reach':>6}  support")
    for t in TASKS:
        try:
            r = precheck(t)
            d = r["obj_lang"]
            flag = "OK " if r["free"] else "!! "
            print(f"{flag}{r['task']:<26} {r['n']:>5} {r['fit']:>5} {r['ik']:>5} {r['free']:>6} "
                  f"{d['reach_m']:>6} {d['support']}")
            if not r["free"] and r["blockers"]:
                for k, v in sorted(r["blockers"].items(), key=lambda x: -x[1])[:2]:
                    print(f"      blocked {v}x by {k}")
        except Exception as e:
            print(f"!! {t:<26} error {type(e).__name__}: {e}")
