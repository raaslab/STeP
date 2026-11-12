"""Build a STeP scene context from a live RoboCasa kitchen episode.

RoboCasa names its manipulables generically -- `obj`, `container`, `obj_container`
-- so the MuJoCo model alone cannot tell System 2 what it is looking at. The
semantic identity lives in two other places: the sampled object's own category
(`env.objects[name].name` / its mjcf path) and the episode language in
`get_ep_meta()["lang"]`. This joins all three so the entity list carries real
names ("lemon", "pan") rather than positional handles, which is what every
predicate argument and every prompt reference depends on.
"""
import json
import pathlib

import numpy as np


def _xyz(env, body_name):
    bid = env.sim.model.body(body_name).id
    return [float(v) for v in env.sim.data.body_xpos[bid]]


def _quat(env, body_name):
    bid = env.sim.model.body(body_name).id
    return [float(v) for v in env.sim.data.body_xquat[bid]]


# Fixtures that are room structure rather than things a task can refer to.
# Without this the entity list is 44 entries, most of them walls and backings,
# which buries the stove and counter System 2 actually has to reason about.
_STRUCTURAL = ("wall", "floor", "ceiling", "backing", "room")


def object_categories(env):
    """Map each generic handle to its real category, from the episode metadata.

    RoboCasa's MJCFObject reports name "obj" and exposes no category attribute;
    the sampled category lives in ep_meta["object_cfgs"][i]["info"]["cat"] (and
    the asset path). Reading it here is what turns "obj" into "eggplant", which
    every predicate argument and prompt reference needs.
    """
    out = {}
    for cfg in env.get_ep_meta().get("object_cfgs", []) or []:
        name = cfg.get("name")
        info = cfg.get("info") or {}
        cat = info.get("cat")
        if not cat and info.get("mjcf_path"):
            cat = pathlib.Path(info["mjcf_path"]).parent.parent.name
        if name and cat:
            out[name] = str(cat)
    return out


def object_models(env):
    """Map each handle to its exact mesh ("eggplant_0"), not just its category.

    RoboCasa ships several meshes per category and samples among them per
    episode. They do not share a body origin, so anything measured against one
    mesh -- a demonstrated grasp offset above all -- is meaningless on another.
    """
    out = {}
    for cfg in env.get_ep_meta().get("object_cfgs", []) or []:
        name, path = cfg.get("name"), (cfg.get("info") or {}).get("mjcf_path")
        if name and path:
            out[name] = pathlib.PurePosixPath(path).parent.name
    return out


def build_scene_context(env, *, task_name, layout_id, style_id, seed=None):
    ep = env.get_ep_meta()
    lang = ep.get("lang", "")

    cats = object_categories(env)
    entities, geometry, live = [], {}, {}
    for handle in env.objects:
        cat = cats.get(handle, handle)
        body = env.obj_body_id.get(handle)
        if body is None:
            continue
        name = env.sim.model.body(body).name
        pos, quat = _xyz(env, name), _quat(env, name)
        role = "manipulated_object" if handle == "obj" else "support"
        entities.append({
            "name": handle,
            "role": role,
            "object_class": cat,
            "aliases": sorted({handle, cat, cat.replace("_", " ")}),
            "notes": f"RoboCasa handle '{handle}' -> body '{name}'. Episode language: {lang}",
        })
        geometry[handle] = {"xy": pos[:2], "xyz": pos, "z": pos[2], "radius": 0.05}
        live[handle] = {"xyz": pos, "xy": pos[:2], "z": pos[2],
                        "quat": quat, "source": "robocasa_sim"}

    # Fixtures are the kitchen's structure (stove, sink, cabinets). They are not
    # in env.objects, but constraints refer to them constantly ("the lit burner"),
    # so they have to reach System 2 as addressable entities too.
    for fname, fxtr in (getattr(env, "fixtures", {}) or {}).items():
        if any(tok in fname.lower() for tok in _STRUCTURAL):
            continue
        try:
            fpos = [float(v) for v in fxtr.pos]
        except Exception:
            continue
        entities.append({
            "name": fname, "role": "fixture", "object_class": type(fxtr).__name__,
            "aliases": sorted({fname, type(fxtr).__name__.lower()}),
            "notes": "RoboCasa kitchen fixture.",
        })
        geometry[fname] = {"xy": fpos[:2], "xyz": fpos, "z": fpos[2], "radius": 0.10}

    return {
        "platform": {
            "backend": "robocasa",
            "workspace_bounds": derive_workspace_bounds(env),
            "robocasa": {
                "task": task_name, "layout_id": layout_id,
                "style_id": style_id, "language": lang,
                # The episode this context describes. RoboCasa samples a fresh
                # object mesh and placement per episode, so an env built without
                # it is a DIFFERENT scene: measured, the context said eggplant_0
                # at x=2.601 while unseeded rebuilds gave garlic_1, LemonWedge004
                # and egg_0, one of them a metre away. Everything downstream --
                # the prompt, the keep-out geometry, the grasp routing's mesh --
                # describes this episode, so the executor has to rebuild it.
                "seed": seed,
                "motion_backend": "ompl_mpc",
                "camera_name": "robot0_frontview",
                "motion": {
                    "collision": {
                        # NOTE: this is a loosened tolerance, not a free win.
                        # At LIBERO's 6mm every one of 16 candidates is rejected
                        # for finger-vs-carrot penetration; at 15mm exactly one
                        # survives. 15mm of finger passing through the mesh is
                        # beyond real gripper compliance -- it will read as a
                        # successful grasp in sim that a real hand would not
                        # achieve. Recorded here so the number is visible rather
                        # than buried, and so a RoboCasa result is never quietly
                        # compared against a LIBERO one scored at 6mm.
                        "finger_overlap_tolerance_m": 0.015,
                        # Grasping produce puts the object against the palm, and
                        # the demonstrations show that is what works here: their
                        # own close steps penetrate the object by 1.1-6.9 mm at
                        # the palm and lift it anyway. 8 mm covers that and stays
                        # well under the 15 mm already allowed on the fingers.
                        # Applies to the grasp target only -- see
                        # CollisionConfig.palm_overlap_tolerance_m.
                        "palm_overlap_tolerance_m": 0.008,
                    },
                    "grasp": {
                        # The object sits ~0.78 m from the parked base against a
                        # ~0.85 m reach, so tall standoffs are past the joint
                        # limits. Measured directly: +5cm and +8cm solve IK
                        # cleanly, +12cm/+15cm/+20cm all fail on
                        # joint_limit_violation. LIBERO's (1.0, 2.5, 4.0) would
                        # try 8/20/32cm and waste most candidates on poses that
                        # can never be reached from here.
                        "standoff_distance_multipliers": [0.6, 1.0],
                        # RoboCasa names every manipulable "obj_main", so the
                        # object-class keyword table matches nothing and routes
                        # everything to the manual pose without ever trying
                        # GraspNet. See GraspSampleConfig.routing.
                        #
                        # The mesh is part of the route, not decoration: a demo
                        # grasp transfers to a new pose of the same object and no
                        # further, and this episode samples both its category and
                        # its mesh fresh, so the pool has to be narrowed to the
                        # exact mesh (eggplant_0, not "eggplant").
                        # graspnet_first, not demo:. Contact-GraspNet proposes
                        # 6-DoF grasps from the point cloud, so it finds the
                        # approach the geometry actually allows -- measured, every
                        # collision-free candidate it returns here is tilted 9 to
                        # 73 degrees off vertical, i.e. it comes in from the side
                        # rather than straight down. The demo route is kept for
                        # comparison but is too thin to rely on: 212 meshes across
                        # 504 demonstrations, 118 of them with a single grasp.
                        "routing": "graspnet_first",
                    },
                    "planner": {
                        # A kitchen needs 10x finer edge checking than LIBERO's
                        # tabletop default of 0.01. Measured on the transport
                        # leg of PickPlaceCounterToStove: 0.01 planned
                        # successfully 2/6, 0.002 10/12, and 0.001 12/12. The
                        # rejections were real -- a 3.8 mm interpenetration of
                        # the microwave door that fell between samples at the
                        # coarser spacing, caught by the planner's own post-hoc
                        # re-validation. Costs ~8 s per plan against ~1.5 s.
                        # 0.002 measured 10/12 at ~4 s; 0.001 was 12/12 but ~8 s,
                        # and a pick tries up to 16 candidates against the
                        # planner's own timeout, so the finer setting spent the
                        # whole budget and candidates died on planning_failed:
                        # timeout rather than on anything about the grasp.
                        "validity_resolution": 0.002,
                        "timeout_sec": 40.0,
                    },
                    "contact_graspnet": {
                        # Contact-GraspNet, not the privileged top-down
                        # generator. Measured across five pick-place tasks: the
                        # top-down generator produced ZERO collision-free
                        # candidates on four of them, because a straight-down
                        # approach in a kitchen drives the elbow into overhead
                        # cabinets and counter appliances. CGN's 6-DoF proposals
                        # clear them, and every clear grasp is tilted -- 9 to 73
                        # degrees off vertical. RoboCasa's own human demos never
                        # grasp straight down either.
                        "enabled": True,
                        "camera_names": ["robot0_agentview_center", "robot0_eye_in_hand"],
                        # Seed the retry viewpoints. They default to OS entropy,
                        # so each run saw different synthetic camera poses, hence
                        # a different point cloud and a different candidate set
                        # from the very same scene -- 3, 1, 2 and 32 candidates
                        # across four runs of one seeded episode, which makes two
                        # runs incomparable and any A/B meaningless.
                        "viewpoint_seed": 0,
                    },
                },
            },
        },
        "entities": entities,
        "geometry": geometry,
        # Mirrors LIBERO's own blocks; without them the executor falls back to
        # constructor defaults tuned for a 0.68 m tabletop.
        "execution_proxy": {
            "default_grasp_mode": "top",
            "max_subtask_steps": 900,
            "top_grasp_hover_z": 0.14,
            "top_grasp_close_xy_tol": 0.055,
            "top_grasp_close_z_tol": 0.025,
            "env_horizon": 1600,
            "post_success_steps": 80,
            "mpc_timing_enabled": True,
        },
        "placement_defaults": {"retreat_height_above_table": 0.28},
        "coordinate_convention": {
            "frame": "robosuite world xy", "left": "-x", "right": "+x",
            "front": "+y", "back": "-y", "offset_units": "meters",
        },
        "live_scene_state": {"objects": live},
    }


# Panda reach from its own base. Used to keep derived bounds inside something
# the arm can actually service, so a far-away counter on the other side of the
# kitchen does not widen the box into space the planner can never reach.
_PANDA_REACH_M = 0.9


def robot_base_xyz(env):
    """World position of the robot base, mobile or fixed."""
    for name in ("mobilebase0_base", "robot0_base", "base0_base"):
        try:
            return np.asarray(_xyz(env, name), dtype=float)
        except Exception:
            continue
    # Fall back to the end effector, which is at least inside the workspace.
    from step.system_one.motion import panda  # local import: optional dependency
    bid = env.sim.model.site(panda.EE_SITE).id
    return np.asarray(env.sim.data.site_xpos[bid], dtype=float)


def derive_workspace_bounds(env, *, margin_xy=0.30, z_below=0.10, z_above=0.45):
    """Axis-aligned workspace box for this episode, in world coordinates.

    LIBERO's bounds are a hardcoded tabletop box near the origin. RoboCasa places
    the kitchen in room coordinates -- the end effector sits at x=3.13 in the very
    first episode -- so those bounds reject every valid pose here. Derived instead
    from the things this episode actually involves (the manipulated object, its
    source and destination containers) and then clipped to the arm's reach about
    its own base, so the box is both relevant and reachable.
    """
    obj_pts = []
    for handle in ("obj", "obj_container", "container"):
        bid = env.obj_body_id.get(handle)
        if bid is not None:
            obj_pts.append(np.asarray(env.sim.data.body_xpos[bid], dtype=float))
    base = robot_base_xyz(env)
    # The base is included for xy (the arm works around where it stands) but
    # deliberately NOT for z: a mobile base reports z=0 at the floor, which
    # dragged the workspace floor down to -0.1 m when the task lives at counter
    # height ~0.95 m, and everything between was space the arm never works in.
    P = np.vstack(obj_pts + [base])
    Z = np.vstack(obj_pts) if obj_pts else P

    lo = P.min(axis=0) - np.array([margin_xy, margin_xy, z_below])
    hi = P.max(axis=0) + np.array([margin_xy, margin_xy, z_above])
    # Clip to what the arm can reach from where it is standing.
    lo = np.maximum(lo, base - _PANDA_REACH_M)
    hi = np.minimum(hi, base + _PANDA_REACH_M)
    # z is not reach-limited the same way: keep the vertical span the task needs.
    lo[2] = float(Z[:, 2].min() - z_below)
    hi[2] = float(Z[:, 2].max() + z_above)
    return {"x": [float(lo[0]), float(hi[0])],
            "y": [float(lo[1]), float(hi[1])],
            "z": [float(lo[2]), float(hi[2])]}
