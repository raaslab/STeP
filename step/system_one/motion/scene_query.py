"""Shared MuJoCo scene-query layer: FK, Jacobians, geometry extents, and collision
primitives, all operating on a **scratch** ``mujoco.MjData`` cloned from a live
robosuite/LIBERO sim. Nothing in this module ever mutates the live episode's
``sim.data`` -- every public method either takes an explicit scratch ``SceneQuery``
or is called through ``SceneQuery.snapshot()``/``SceneQuery.from_robosuite_sim()``,
which copy ``qpos``/``qvel`` out of the live sim once and never write back to it.

Quaternion convention: **WXYZ** throughout this package. MuJoCo's own ``xquat``/
``xmat``-derived quaternions are natively WXYZ, which is why this convention was
chosen -- it needs no conversion at the MuJoCo boundary. Only convert to XYZW at
the point of calling ``scipy.spatial.transform.Rotation``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - mujoco is a hard dependency of this repo
    raise ImportError(
        "step.system_one.motion requires the 'mujoco' package (already a STeP "
        "dependency via robosuite/LIBERO)."
    ) from exc


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, both operands and result WXYZ."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(q, dtype=np.float64))
    return mat.reshape(3, 3)


def resolve_body_name(scene: "SceneQuery", canonical_name: str) -> str:
    """Map a ``SceneResolver`` canonical entity name (e.g. ``"alphabet_soup_1"``,
    as used throughout ``scene_context``/params) to the actual MuJoCo body name
    the LIBERO/robosuite model uses (e.g. ``"alphabet_soup_1_main"``). These
    differ by a ``"_main"`` suffix for LIBERO's free-body objects -- the
    resolver layer was never MuJoCo-body-name-aware (it only tracks XY/z
    geometry hints), so anything in ``step/system_one/motion/`` that needs a
    real body/geom lookup must go through this first.
    """
    for candidate in (f"{canonical_name}_main", canonical_name):
        try:
            scene.body_id(candidate)
            return candidate
        except KeyError:
            continue
    raise KeyError(
        f"Could not find a MuJoCo body for canonical object {canonical_name!r} "
        "(tried the '_main' suffix convention and the exact name)."
    )


def rotmat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(9))
    return quat


@dataclass
class BodyExtent:
    """Axis-aligned half-extents of a body's collision geoms, in the body's own
    local frame, plus the body-frame offset of the box center from the body
    origin (bodies' collision meshes aren't always centered on the body frame)."""

    half_extents: np.ndarray  # (3,)
    center_offset: np.ndarray  # (3,), local frame


class SceneQuery:
    """Read/query interface over a scratch MuJoCo model+data pair."""

    def __init__(self, model: "mujoco.MjModel", data: "mujoco.MjData") -> None:
        self.model = model
        self.data = data

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_robosuite_sim(cls, sim) -> "SceneQuery":
        """Clone a fresh MjData for ``sim``'s model and sync qpos/qvel from the
        live sim once. The returned SceneQuery's ``data`` is fully independent of
        ``sim.data`` from this point on -- stepping/mutating it never touches the
        live episode.
        """
        raw_model = sim.model._model
        raw_data = mujoco.MjData(raw_model)
        raw_data.qpos[:] = sim.data.qpos
        raw_data.qvel[:] = sim.data.qvel
        mujoco.mj_forward(raw_model, raw_data)
        return cls(raw_model, raw_data)

    def clone(self) -> "SceneQuery":
        """A fresh scratch copy of this SceneQuery's current state, for planners
        that need to try many candidate configurations without disturbing a
        shared baseline."""
        new_data = mujoco.MjData(self.model)
        new_data.qpos[:] = self.data.qpos
        new_data.qvel[:] = self.data.qvel
        mujoco.mj_forward(self.model, new_data)
        return SceneQuery(self.model, new_data)

    # ---------------------------------------------------------------- name lookups

    def body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"No body named {name!r} in model.")
        return bid

    def site_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise KeyError(f"No site named {name!r} in model.")
        return sid

    def joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"No joint named {name!r} in model.")
        return jid

    def geom_id(self, name: str) -> int:
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise KeyError(f"No geom named {name!r} in model.")
        return gid

    def geom_name(self, gid: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"

    # ---------------------------------------------------------------- FK / poses

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World (position, quat_wxyz) of a body's frame origin."""
        bid = self.body_id(name)
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()

    def site_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World (position, quat_wxyz) of a site."""
        sid = self.site_id(name)
        pos = self.data.site_xpos[sid].copy()
        quat = rotmat_to_quat_wxyz(self.data.site_xmat[sid])
        return pos, quat

    def hand_to_tcp_transform(
        self, hand_site: str = "gripper0_right_ft_frame", tcp_site: str = "gripper0_right_grip_site"
    ) -> np.ndarray:
        """Signed distance (m) along the hand frame's local +Z from the hand/
        flange site to the TCP/grip site, computed live from the current model
        instead of a hardcoded constant (unlike Cap-X's
        ``FRANKA_HAND_TO_FINGERTIP_Z_M``, which hardcodes 0.0584*2 m for the
        Franka hand geometry -- robosuite's Panda MJCF already exposes both
        frames as named sites, so we read the real transform instead of
        guessing/copying a constant that may not match this exact model).
        """
        hand_pos, hand_quat = self.site_pose(hand_site)
        tcp_pos, _ = self.site_pose(tcp_site)
        hand_R = quat_to_rotmat_wxyz(hand_quat)
        local_offset = hand_R.T @ (tcp_pos - hand_pos)
        return float(local_offset[2])

    # ---------------------------------------------------------------- geometry extents

    def body_geom_ids(self, body_name: str, include_children: bool = True) -> list[int]:
        bid = self.body_id(body_name)
        ids = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bid]
        if include_children:
            child_ids = {
                b for b in range(self.model.nbody) if self._is_descendant(b, bid)
            }
            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g] in child_ids:
                    ids.append(g)
        return sorted(set(ids))

    def body_collision_geom_ids(self, body_name: str, include_children: bool = True) -> list[int]:
        """Like ``body_geom_ids``, but excluding purely-visual geoms
        (``contype == 0 and conaffinity == 0``, robosuite/MuJoCo's convention
        for a render-only mesh that never participates in physics). Needed
        wherever a body's *physical* shape matters and its geom set includes
        such a geom -- e.g. a LIBERO container's high-poly visual mesh
        alongside the handful of simple boxes (floor + walls) that actually
        model its collision geometry. The visual mesh's own bounding box
        spans the whole hollow container as if it were solid, since an AABB
        can't represent concavity -- exactly the shape the simple box geoms
        are there to approximate correctly instead."""
        return [
            g
            for g in self.body_geom_ids(body_name, include_children=include_children)
            if self.model.geom_contype[g] != 0 or self.model.geom_conaffinity[g] != 0
        ]

    def _is_descendant(self, body_id: int, ancestor_id: int) -> bool:
        b = body_id
        seen = set()
        while b != 0 and b not in seen:
            seen.add(b)
            parent = int(self.model.body_parentid[b])
            if parent == ancestor_id:
                return True
            b = parent
        return False

    def _merged_extent_world(self, gids: list[int], body_pos: np.ndarray) -> BodyExtent | None:
        if not gids:
            return None
        mins = np.full(3, np.inf)
        maxs = np.full(3, -np.inf)
        for g in gids:
            gpos = self.data.geom_xpos[g]
            gmat = self.data.geom_xmat[g].reshape(3, 3)
            size = self.model.geom_size[g]
            gtype = self.model.geom_type[g]
            # Conservative world-aligned half-extent for this geom: rotate its
            # local half-extents by |R| (abs of rotation matrix), which never
            # underestimates the true rotated bounding box.
            local_half = self._geom_local_half_extent(gtype, size)
            world_half = np.abs(gmat) @ local_half
            mins = np.minimum(mins, gpos - world_half)
            maxs = np.maximum(maxs, gpos + world_half)
        center = (mins + maxs) / 2.0
        half = (maxs - mins) / 2.0
        return BodyExtent(half_extents=half, center_offset=center - body_pos)

    def body_extent_world(self, body_name: str) -> BodyExtent | None:
        """Axis-aligned bounding half-extents (world-frame-aligned, at the
        current pose) of all collision geoms belonging to ``body_name``, plus
        the world-frame offset of the box center from the body origin. Returns
        None if the body has no geoms (shouldn't happen for LIBERO objects)."""
        bid = self.body_id(body_name)
        return self._merged_extent_world(self.body_geom_ids(body_name), self.data.xpos[bid])

    def body_collision_extent_world(self, body_name: str) -> BodyExtent | None:
        """Like ``body_extent_world``, but over only the body's real
        collision geoms (see ``body_collision_geom_ids``) -- excludes a
        body's purely-visual mesh, whose own bounding box can extend past its
        actual physical collision shape (e.g. a container rendered with a
        smooth high-poly mesh but collision-modeled as a handful of simple
        boxes -- the mesh's AABB can disagree with where the real collidable
        surface is)."""
        bid = self.body_id(body_name)
        return self._merged_extent_world(self.body_collision_geom_ids(body_name), self.data.xpos[bid])

    def body_extent_local(self, body_name: str) -> BodyExtent | None:
        """Axis-aligned half-extents of a body's collision geoms **in the
        body's own local frame** -- i.e. invariant to the body's current
        world orientation, unlike ``body_extent_world``, whose world-axis-
        aligned box inflates under rotation (a box rotated 45 degrees needs a
        bigger world-aligned AABB to cover it, even though the object's own
        shape hasn't changed).

        Use this wherever the extent will be re-rotated by a *different*
        orientation later -- e.g. fitting collision spheres to an object at
        grasp time, to be checked at whatever orientation it's carried at
        during transport/place. Reusing ``body_extent_world`` there double-
        counts rotation (once baked into the stored extent at whatever angle
        the object happened to be at when grasped, again when the caller
        re-rotates it by the current orientation), inflating the fitted
        sphere far beyond the object's true size -- a real bug found via a
        soup can's fitted collision radius (~6cm) coming out barely smaller
        than the hover clearance above a basket rim, making every placement
        candidate spuriously read as colliding with the basket wall.
        """
        gids = self.body_geom_ids(body_name)
        if not gids:
            return None
        bid = self.body_id(body_name)
        body_pos = self.data.xpos[bid]
        body_mat = self.data.xmat[bid].reshape(3, 3)
        mins = np.full(3, np.inf)
        maxs = np.full(3, -np.inf)
        for g in gids:
            gpos = self.data.geom_xpos[g]
            gmat = self.data.geom_xmat[g].reshape(3, 3)
            size = self.model.geom_size[g]
            gtype = self.model.geom_type[g]
            local_half = self._geom_local_half_extent(gtype, size)
            # This geom's pose expressed in the body's own (world-orientation
            # -independent) local frame.
            rel_mat = body_mat.T @ gmat
            rel_pos = body_mat.T @ (gpos - body_pos)
            rel_half = np.abs(rel_mat) @ local_half
            mins = np.minimum(mins, rel_pos - rel_half)
            maxs = np.maximum(maxs, rel_pos + rel_half)
        center = (mins + maxs) / 2.0
        half = (maxs - mins) / 2.0
        return BodyExtent(half_extents=half, center_offset=center)

    def geom_world_aabb(self, gid: int) -> tuple[np.ndarray, np.ndarray]:
        """World-aligned (center, half_extents) bounding box for a single geom
        at its current pose -- the per-geom analogue of ``body_extent_world``.
        Needed wherever a body's *true* shape matters (e.g. a container
        assembled from separate wall/floor geoms): merging all of a body's
        geoms into one box would also cover the empty space between them
        (its hollow interior) as if it were solid."""
        gpos = self.data.geom_xpos[gid]
        gmat = self.data.geom_xmat[gid].reshape(3, 3)
        size = self.model.geom_size[gid]
        gtype = self.model.geom_type[gid]
        local_half = self._geom_local_half_extent(gtype, size)
        world_half = np.abs(gmat) @ local_half
        return gpos.copy(), world_half

    @staticmethod
    def _geom_local_half_extent(geom_type: int, size: np.ndarray) -> np.ndarray:
        # mjtGeom: 2=sphere, 3=capsule, 5=cylinder, 6=box, 7=mesh (size holds the
        # mesh's own AABB half-extent for mesh geoms in MuJoCo's compiled model).
        if geom_type in (mujoco.mjtGeom.mjGEOM_SPHERE,):
            r = float(size[0])
            return np.array([r, r, r])
        if geom_type in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            r, half_len = float(size[0]), float(size[1])
            return np.array([r, r, half_len + r])
        if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            # A plane's geom_size is a *rendering* hint (visual quad half-width
            # in xy, grid-line spacing in z) -- not physical geometry. Treating
            # it as a literal box half-extent (as the box/mesh fallback below
            # does) makes a ground plane's "half-extent" 12.5cm+ thick, so
            # anything resting *on* it reads as already deep inside the slab
            # (a real bug this exact fix was written against -- see
            # tests/test_collision.py::test_attached_sphere_ignores_ground_plane).
            # Model it as a thin, effectively-2D slab instead.
            half_xy = float(size[0]) if size[0] > 0 else 50.0  # 0 means "infinite" per MuJoCo convention
            half_xy2 = float(size[1]) if size[1] > 0 else 50.0
            return np.array([half_xy, half_xy2, 1e-4])
        # box, mesh, and anything else: MuJoCo stores a half-extent-shaped size
        # for box/mesh geoms; fall back to it directly (conservative for others).
        return np.asarray(size[:3], dtype=np.float64)

    # ---------------------------------------------------------------- joints

    def joint_qpos_range(self, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        lo = np.zeros(len(joint_names))
        hi = np.zeros(len(joint_names))
        for i, jn in enumerate(joint_names):
            jid = self.joint_id(jn)
            lo[i], hi[i] = self.model.jnt_range[jid]
        return lo, hi

    def joint_qpos_addrs(self, joint_names: list[str]) -> np.ndarray:
        return np.array([self.model.jnt_qposadr[self.joint_id(jn)] for jn in joint_names])

    def get_qpos(self, joint_names: list[str]) -> np.ndarray:
        addrs = self.joint_qpos_addrs(joint_names)
        return self.data.qpos[addrs].copy()

    def set_qpos(self, joint_names: list[str], q: np.ndarray) -> None:
        addrs = self.joint_qpos_addrs(joint_names)
        self.data.qpos[addrs] = q
        mujoco.mj_forward(self.model, self.data)

    def set_free_body_qpos(self, body_name: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        """Move a free-jointed body (e.g. a grasped object) to an explicit world
        pose in this scratch data -- used to test attached-object collision at
        hypothetical configurations. No-op on bodies without a free joint."""
        bid = self.body_id(body_name)
        jadr = self.model.body_jntadr[bid]
        if jadr < 0 or self.model.jnt_type[jadr] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Body {body_name!r} has no free joint; cannot reposition directly.")
        qadr = self.model.jnt_qposadr[jadr]
        self.data.qpos[qadr : qadr + 3] = pos
        self.data.qpos[qadr + 3 : qadr + 7] = quat_wxyz
        mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------------- Jacobian / kinematics

    def site_jacobian(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        """(jacp, jacr), each (3, nv) -- translational/rotational Jacobian of a
        site w.r.t. all DoFs in the model (caller slices to the arm's dofs)."""
        sid = self.site_id(site_name)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
        return jacp, jacr

    def dof_addrs(self, joint_names: list[str]) -> np.ndarray:
        return np.array([self.model.jnt_dofadr[self.joint_id(jn)] for jn in joint_names])

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------------- collision

    def compute_contacts(self) -> list[tuple[str, str, float]]:
        """Recompute contacts at the current qpos and return
        [(geom1_name, geom2_name, signed_distance)] for every active contact
        (``mj_collision`` already applies each geom-pair's ``margin``/``gap``, so
        distance here is already margin-adjusted -- negative means violating the
        configured margin, not just literal penetration)."""
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            out.append((self.geom_name(c.geom1), self.geom_name(c.geom2), float(c.dist)))
        return out
