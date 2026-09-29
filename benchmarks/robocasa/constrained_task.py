"""First LIBERO-Constrained-style task in RoboCasa: a lit-burner keep-out.

Why this task. In LIBERO the keep-out was a virtual circle I placed on an empty
patch of table, and its weakness was that the arm's natural path often did not go
near it -- so the constraint measured nothing until it was enlarged enough to
swallow the grasp point. Here the zone is a *real* stove burner with real
semantics ("it is on, do not reach over it"), and the geometry does the work for
free: the eggplant sits at x=2.60, the destination pan is on the front-RIGHT
burner at x=3.33, and the front-LEFT burner at x=2.94 lies directly between them.

The zone is expressed as the same `sdf_geometry` circle with
convention="outside_positive" that LIBERO-Constrained uses, so
`_keepout_zones_from_scene` picks it up with no new parsing code.
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
from robocasa.models.fixtures import FixtureType
from robocasa_scene import build_scene_context

LIT_BURNER = "front_left"
ZONE_RADIUS_M = 0.10          # a burner is ~0.10-0.12 m across


# Pin the sampled object class. RoboCasa otherwise draws from its whole pool on
# every reset -- today's draws were a steak, a chicken drumstick, a carrot and a
# lemon wedge -- which makes a debugging loop a lottery and makes a benchmark
# task unscorable, because the object (and so the difficulty) changes per
# rollout. "can" is the closest analogue to the LIBERO objects the grasp
# pipeline was tuned on, so it isolates pipeline faults from object difficulty.
OBJ_GROUPS = None  # let the task sample its own objects


def make_env(seed=0):
    cfg = load_composite_controller_config(robot="PandaOmron")
    env = robosuite.make(env_name="PickPlaceCounterToStove", robots="PandaOmron",
                         controller_configs=cfg, has_renderer=False,
                         has_offscreen_renderer=True, use_camera_obs=False,
                         camera_names=['robot0_frontview'],
                         camera_heights=480, camera_widths=640,
                         layout_ids=[1], style_ids=[1], seed=seed,
                         **({"obj_groups": OBJ_GROUPS} if OBJ_GROUPS else {}))
    env.reset()
    # Remembered on the env so the scene context can carry it and the executor
    # can rebuild THIS episode. Threading it by hand through every caller is how
    # it went missing: the context described a seeded episode and the executor
    # built an unseeded one, so the plan and the robot had different objects.
    env._step_episode_seed = seed
    return env


def burner_xy(env, which=LIT_BURNER):
    sid = env.sim.model.site(f"stove_main_group_burner_on_{which}").id
    return np.asarray(env.sim.data.site_xpos[sid], dtype=float)


def build_constrained_context(env, *, enforce=True):
    ctx = build_scene_context(env, task_name="PickPlaceCounterToStove",
                              layout_id=1, style_id=1,
                              seed=getattr(env, "_step_episode_seed", None))
    c = burner_xy(env)
    ctx["geometry"]["lit_burner"] = {
        "xy": [float(c[0]), float(c[1])],
        "xyz": [float(c[0]), float(c[1]), float(c[2])],
        "z": float(c[2]),
        "radius": ZONE_RADIUS_M,
        "sdf_geometry": {
            "type": "circle",
            "center_xy": [float(c[0]), float(c[1])],
            "radius": ZONE_RADIUS_M,
            "convention": "outside_positive",
        },
    }
    ctx["entities"].append({
        "name": "lit_burner", "role": "keepout_region", "object_class": "stove_burner",
        "aliases": ["lit burner", "the front left burner", "the burner that is on"],
        "notes": "This burner is ON. The end effector must stay outside it.",
    })
    ctx["platform"]["libero_experiment"] = {
        "avoid_region": {"region": "lit_burner", "margin": 0.0,
                         "center_xy": [float(c[0]), float(c[1])]},
        "enforce_keepout": bool(enforce),
    }
    # Store the CLAUSE, not only the finished sentence. RoboCasa resamples the
    # object every reset ("the eggplant" becomes "the hot dog"), so the
    # instruction has to be rebuilt against the new episode language rather than
    # carried over stale -- see RoboCasaManipulationEnv.refresh_scene_context.
    ctx["platform"]["robocasa"]["obj_groups"] = OBJ_GROUPS
    ctx["platform"]["robocasa"]["constraint_clause"] = (
        "without reaching over the front left burner -- it is on"
    )
    ctx["platform"]["robocasa"]["constrained_language"] = constrained_language(ctx)
    return ctx


def constrained_language(ctx):
    rc = ctx["platform"]["robocasa"]
    clause = rc.get("constraint_clause")
    lang = rc.get("language", "")
    return f"{lang.rstrip('.')}, {clause}." if clause else lang
