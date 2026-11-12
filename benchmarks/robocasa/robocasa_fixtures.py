"""Read System 1's geometric parameters from RoboCasa's own fixtures.

LIBERO hands these over directly: `mujoco_arena.table_offset[2]` is the tabletop
height and the workspace is a fixed box. RoboCasa has neither, and earlier
versions of this port guessed both -- surface height from the bottom of whatever
object happened to be on it, workspace bounds from the spread of task objects.
Both were close enough to look right and wrong enough to matter (the guessed
surface was 0.9035 against a true 0.92, and carry height is derived from it).

RoboCasa fixtures already publish what we need: `height`, `size`, and
`get_bbox_points()` in world coordinates. This asks them.
"""
import numpy as np

# Panda reach from its own base, used to clip a counter-wide workspace down to
# the part the arm can actually service.
PANDA_REACH_M = 0.85


def _bbox_world(fxtr):
    """(min_xyz, max_xyz) for a fixture, in world coordinates."""
    pts = np.asarray(fxtr.get_bbox_points(), dtype=float)
    return pts.min(axis=0), pts.max(axis=0)


def supporting_fixture(env, handle="obj"):
    """The fixture whose top surface the object is resting on.

    Chosen by asking which fixture's top sits just below the object rather than
    by name, so it works for a counter, a sink basin or a cabinet shelf without
    the caller knowing which task it is looking at.
    """
    bid = getattr(env, "obj_body_id", {}).get(handle)
    if bid is None:
        return None
    obj_z = float(env.sim.data.body_xpos[bid][2])
    obj_xy = np.asarray(env.sim.data.body_xpos[bid][:2], dtype=float)
    best, best_gap = None, np.inf
    for name, fxtr in (getattr(env, "fixtures", {}) or {}).items():
        if any(tok in name.lower() for tok in ("wall", "floor", "ceiling", "backing", "room")):
            continue
        try:
            lo, hi = _bbox_world(fxtr)
        except Exception:
            continue
        # object must sit above this fixture's top, and within its footprint
        gap = obj_z - float(hi[2])
        inside = (lo[0] - 0.05 <= obj_xy[0] <= hi[0] + 0.05 and
                  lo[1] - 0.05 <= obj_xy[1] <= hi[1] + 0.05)
        if inside and -0.02 <= gap < best_gap:
            best, best_gap = (name, fxtr, float(hi[2])), gap
    return best


def surface_z(env, handle="obj", default=0.9):
    """Height of the surface the object rests on, from the fixture itself."""
    found = supporting_fixture(env, handle)
    return found[2] if found else default


def workspace_bounds(env, margin_xy=0.25, z_below=0.08, z_above=0.45):
    """Workspace box: the task's objects and their surface, clipped to arm reach."""
    pts = []
    for h in ("obj", "obj_container", "container"):
        bid = getattr(env, "obj_body_id", {}).get(h)
        if bid is not None:
            pts.append(np.asarray(env.sim.data.body_xpos[bid], dtype=float))
    if not pts:
        return None
    P = np.vstack(pts)
    try:
        bid = env.sim.model.body("mobilebase0_base").id
        base = np.asarray(env.sim.data.body_xpos[bid], dtype=float)
    except Exception:
        base = P.mean(axis=0)
    lo = np.minimum(P.min(axis=0) - margin_xy, base[:3] - PANDA_REACH_M)
    hi = np.maximum(P.max(axis=0) + margin_xy, base[:3] + PANDA_REACH_M)
    lo = np.maximum(lo, base - PANDA_REACH_M)
    hi = np.minimum(hi, base + PANDA_REACH_M)
    # z is not reach-clipped the same way; anchor it to the work surface, never
    # to the mobile base, which reports z=0 at the floor.
    surf = surface_z(env)
    lo[2] = surf - z_below
    hi[2] = float(P[:, 2].max()) + z_above
    return {"x": [float(lo[0]), float(hi[0])],
            "y": [float(lo[1]), float(hi[1])],
            "z": [float(lo[2]), float(hi[2])]}


def base_yaw(env):
    """Mobile-base yaw. The arm controller reads deltas in the base frame."""
    try:
        bid = env.sim.model.body("mobilebase0_base").id
    except Exception:
        return 0.0
    w, x, y, z = env.sim.data.body_xquat[bid]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def describe(env, handle="obj"):
    """One-line geometric summary, for checking a task before running it."""
    found = supporting_fixture(env, handle)
    bid = env.obj_body_id.get(handle)
    obj = np.asarray(env.sim.data.body_xpos[bid], dtype=float) if bid is not None else None
    base_bid = env.sim.model.body("mobilebase0_base").id
    base = np.asarray(env.sim.data.body_xpos[base_bid], dtype=float)
    reach = float(np.linalg.norm(obj[:2] - base[:2])) if obj is not None else float("nan")
    return {
        "support": found[0] if found else None,
        "surface_z": round(surface_z(env, handle), 4),
        "object_z": round(float(obj[2]), 4) if obj is not None else None,
        "reach_m": round(reach, 3),
        "base_yaw_deg": round(np.degrees(base_yaw(env)), 1),
        "bounds": workspace_bounds(env),
    }
