"""Object-relative grasp poses mined from RoboCasa's human demonstrations.

This is RoboCasa's own method, not a workaround. All 1,615 hours of their
synthetic data comes from MimicGen, which decomposes each human demo into
object-centric segments and re-applies them at the new pose of the relevant
object. There is no grasp network anywhere in RoboCasa -- which is why the repo
ships neither grasp annotations nor a generator.

What transfers between episodes is the gripper pose *relative to the object*:

    T_offset = inv(T_object) @ T_gripper        (recorded, at gripper close)
    T_grasp  = T_object_now @ T_offset          (applied, in a new episode)

A grasp obtained this way is reachable from the parked base, clear of the
cabinets, correctly tilted and inside the gripper's width -- because a human
achieved it in that kitchen. None of those four had to be searched for.

The transfer holds for a new *pose* of the same object, and not one step
further. RoboCasa samples a different object category every episode -- across
108 episodes of PickPlaceCounterToStove there are 27 of them -- so an
unfiltered pool mixes a carrot's offset with an egg's and a cream cheese
stick's. Applied to an eggplant those put the gripper 2-7 cm off the object,
which closes on air and shoves it. Every offset therefore records the category
it came from, and the generator uses only the matching ones.
"""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import glob
import json
import pathlib

import numpy as np

DATASET_ROOT = pathlib.Path(_ROBOCASA_DATASETS)
CACHE = pathlib.Path(str(_HERE / "grasp_offsets"))
A_GRIPPER = 11  # from the dataset's meta/modality.json, not the env action layout
# How far before the gripper closes to sample the pose (demos run at 20 Hz).
#
# A fixed number of timesteps is not a fixed distance -- it is however far that
# operator still had to travel -- so this does mix approach speed into the
# recorded offset. Measured over the four eggplant_0 demos, offset from the
# object origin at each backoff:
#
#     backoff      0        6        15
#     ep2       0.030    0.019    0.051
#     ep38      0.041    0.083    0.178
#     ep16      0.027    0.009    0.073
#     ep67      0.020    0.029    0.070
#
# At the close the four agree to within 2 cm and at six steps back they span 9x,
# so recording at the close looks like the cleaner measurement. It is not the
# more useful one: put through the same IK and collision gate, backoff 0 leaves
# 0 of 4 grasps feasible against 2 of 4 at backoff 6, because at the close the
# gripper is wrapped around the object and its geoms overlap the mesh. Kept at 6
# on that evidence. The honest reading is that this samples a pre-grasp on the
# approach, not the grasp -- which is what PickSkill wants to plan to anyway.
PRE_GRASP_BACKOFF_STEPS = 6
# Steps after the hand closes at which to ask whether it closed on the object.
# Contact needs a moment of simulation to register.
GRASP_SETTLE_STEPS = 10


def _mat(pos, quat_wxyz):
    w, x, y, z = [float(v) for v in quat_wxyz]
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def _quat(R):
    t = float(np.trace(R))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


FINGER_BODIES = ("gripper0_right_leftfinger", "gripper0_right_rightfinger",
                 "gripper0_right_finger_joint1_tip", "gripper0_right_finger_joint2_tip")


def _fingers_touching(env, body_id):
    """Were the gripper's fingers in contact with this object, at this state?

    The real test of whether a recorded moment is a grasp. Distance to the body
    origin cannot answer it: RoboCasa's origins are not centroids, and an
    elongated object's bounding sphere is far wider than the object at the point
    the hand closes -- an eggplant reports a 10 cm reach, so a grip site 9.6 cm
    away from its origin and nowhere near its surface passes a radius test and
    fails in the only way that matters, by closing on air.
    """
    sim = env.sim
    finger_geoms = set()
    for name in FINGER_BODIES:
        try:
            bid = sim.model.body(name).id
        except Exception:
            continue
        finger_geoms.update(
            g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid
        )
    obj_geoms = {g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == body_id}
    if not finger_geoms or not obj_geoms:
        return None  # cannot tell; do not drop on a check that did not run
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        pair = {c.geom1, c.geom2}
        if pair & finger_geoms and pair & obj_geoms:
            return True
    return False


def _object_category(meta):
    """Which object category this episode actually put on the plate."""
    for cfg in meta.get("object_cfgs") or []:
        if cfg.get("name") == "obj":
            return (cfg.get("info") or {}).get("cat")
    return None


def object_model(meta):
    """The specific mesh, e.g. "eggplant_0" -- the unit a grasp transfers within.

    Category is too coarse. RoboCasa ships several meshes per category and picks
    among them per episode: five "eggplant" demos here use eggplant_0, _1 and _2.
    Each mesh has its own body origin, so an offset measured against one places
    the gripper somewhere else entirely on another -- measured, an eggplant_1
    offset applied to eggplant_0 put the hand 9.6 cm away, where it closed on air.

    Taken as the model directory name rather than the path: the dataset records
    these as container paths under /root, which never match a local checkout.
    """
    for cfg in meta.get("object_cfgs") or []:
        if cfg.get("name") == "obj":
            path = (cfg.get("info") or {}).get("mjcf_path")
            if path:
                return pathlib.PurePosixPath(path).parent.name
    return None


def extract(task, n_episodes=8, split="pretrain", verbose=False):
    """Replay demos to the gripper-close step; record the object-relative grasp."""
    import json as _json
    import pandas as pd
    import robosuite
    import robocasa  # noqa: F401
    import robocasa.utils.lerobot_utils as LU
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    roots = sorted(glob.glob(str(DATASET_ROOT / "v1.0" / split / "*" / task / "*" / "lerobot")))
    if not roots:
        return []
    root = pathlib.Path(roots[0])

    env_meta = LU.get_env_metadata(root)
    kw = dict(env_meta["env_kwargs"])
    kw["env_name"] = env_meta["env_name"]
    kw["has_renderer"] = False
    kw["has_offscreen_renderer"] = False
    kw["use_camera_obs"] = False
    env = robosuite.make(**kw)

    n_eps = len(LU.get_episodes(root))
    out = []
    for ep in range(min(n_episodes, n_eps)):
        try:
            states = LU.get_episode_states(root, ep)
            pq = sorted(glob.glob(str(root / "data" / "chunk-*" / f"episode_{ep:06d}.parquet")))
            if not pq:
                continue
            a = np.stack(pd.read_parquet(pq[0])["action"].values)
            close = np.nonzero(a[:, A_GRIPPER] > 0.5)[0]
            if not len(close):
                continue
            # Step BACK from the close: at the closing step the fingers are
            # already wrapped around the object and overlapping its mesh, which
            # is a finished grasp, not a pose to plan to. A few steps earlier the
            # gripper is open and positioned around the object -- that is the
            # pre-grasp the planner needs.
            t = int(min(max(close[0] - PRE_GRASP_BACKOFF_STEPS, 0), len(states) - 1))
            ep_meta = LU.get_episode_meta(root, ep)
            category = _object_category(ep_meta)
            model = object_model(ep_meta)
            model_xml = LU.get_episode_model_xml(root, ep)
            meta_json = _json.dumps(ep_meta)

            def _goto(step):
                reset_to(env, {"states": states[int(step)], "model": model_xml,
                               "ep_meta": meta_json})

            # Is this close a grasp of THIS object? The gripper action column
            # closes for other reasons too -- a regrasp, a close after release, a
            # hand shutting mid-motion -- and each produces a well-formed offset
            # pointing at empty space that nothing downstream can tell from a real
            # one. Settling the question where it is actually decidable, in the
            # demo itself: at the moment the hand closed, were the fingers on the
            # object? A few steps later, to let the contact register.
            settle = int(min(close[0] + GRASP_SETTLE_STEPS, len(states) - 1))
            _goto(settle)
            touching = _fingers_touching(env, env.obj_body_id["obj"])
            if touching is False:
                if verbose:
                    print(f"   ep{ep} step{t}: {category or '?':<20} DROPPED -- fingers not on "
                          f"the object when the hand closed")
                continue

            _goto(t)
            sim = env.sim
            gid = sim.model.site("gripper0_right_grip_site").id
            g_pos = np.asarray(sim.data.site_xpos[gid], dtype=float)
            g_R = np.asarray(sim.data.site_xmat[gid], dtype=float).reshape(3, 3)
            bid = env.obj_body_id["obj"]
            T_obj = _mat(sim.data.body_xpos[bid], sim.data.body_xquat[bid])
            T_grip = np.eye(4); T_grip[:3, :3] = g_R; T_grip[:3, 3] = g_pos
            T_off = np.linalg.inv(T_obj) @ T_grip
            approach = (T_grip[:3, :3] @ np.array([0.0, 0.0, 1.0]))

            out.append({
                "episode": ep, "step": t, "task": task,
                "object": category, "model": model,
                "offset_pos": [float(v) for v in T_off[:3, 3]],
                "offset_quat": [float(v) for v in _quat(T_off[:3, :3])],
                "offset_dist": float(np.linalg.norm(T_off[:3, 3])),
                "tilt_deg": float(np.degrees(np.arccos(np.clip(-approach[2], -1, 1)))),
            })
            if verbose:
                print(f"   ep{ep} step{t}: {model or category or '?':<20} offset {np.round(T_off[:3,3],3)} "
                      f"|{np.linalg.norm(T_off[:3,3]):.3f}m| tilt {out[-1]['tilt_deg']:.0f}deg")
        except Exception as e:
            if verbose:
                print(f"   ep{ep}: {type(e).__name__}: {e}")
    env.close()
    return out


def apply(offset, obj_pos, obj_quat):
    """Turn a recorded offset into a world grasp pose for the object's pose now."""
    T = _mat(obj_pos, obj_quat) @ _mat(offset["offset_pos"], offset["offset_quat"])
    return T[:3, 3], _quat(T[:3, :3])


if __name__ == "__main__":
    import collections
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "PickPlaceCounterToStove"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    gs = extract(task, n_episodes=n, verbose=True)
    print(f"\n{task}: {len(gs)} object-relative grasps extracted")
    if gs:
        by_cat = collections.defaultdict(list)
        for g in gs:
            by_cat[g.get("model") or g.get("object") or "?"].append(g)
        print(f"  {len(by_cat)} object meshes (a grasp only transfers within one):")
        for cat, items in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            d = [g["offset_dist"] for g in items]
            t = [g["tilt_deg"] for g in items]
            print(f"    {cat:<22} {len(items):>2} grasps  offset {min(d):.3f}-{max(d):.3f} m  "
                  f"tilt {min(t):.0f}-{max(t):.0f} deg")
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / f"{task}.json").write_text(json.dumps(gs, indent=1))
        print(f"\n  saved {CACHE / (task + '.json')}")
