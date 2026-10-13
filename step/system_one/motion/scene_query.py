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
from typing import Any

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
    resolver layer was never MuJoCo-body-name-aware, so anything in
    ``step/system_one/motion/`` that needs a real body/geom lookup must go
    through this first.

    A drawer/cabinet-level entity (e.g. ``"top_drawer"``) is neither of
    those -- it's a level_label alias ``infer_drawer_fixtures`` registers,
    never a real body name (the real body is something like
    ``"wooden_cabinet_1_cabinet_top"``, which doesn't derive from the alias
    by any string rule). Falls back to ``articulation.resolve_drawer``
    (lazily imported to avoid a circular import) before giving up.
    """
    # "_burner" before "_main": a stove's "_main"/"_base" frame body is not
    # its functional surface -- on the real flat_stove_1 asset, "_main" sits
    # well away (in a different local frame) from "_burner", which is what a
    # placed object actually needs to land on/near. Resolving to "_main"
    # would silently aim every hover/placement candidate at the wrong
    # position. "_main" still wins for every ordinary free-body object, so
    # this can't regress anything that isn't a stove.
    for candidate in (f"{canonical_name}_burner", f"{canonical_name}_main", canonical_name):
        try:
            scene.body_id(candidate)
            return candidate
        except KeyError:
            continue
    try:
        from .articulation import resolve_drawer

        return resolve_drawer(scene, canonical_name).drawer_body
    except KeyError:
        pass
    if canonical_name == "table":
        # "table" is the one canonical name every scene is assumed to have
        # (e.g. executor.py's _resolve_point_goal defaults an unset
        # params.goal_surface to it), but it's an exact MuJoCo body name only
        # in a kitchen scene's arena -- a living-room scene names it
        # "living_room_table", a study scene "study_table", and so on. Fall
        # back to any top-level body whose name contains "table", sorted by
        # (length, name) so the base table body wins over a longer,
        # more-specific sub-body/geom-holder when a scene exposes both.
        matches = sorted(
            (
                name
                for bid in range(scene.model.nbody)
                if (name := mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, bid))
                and "table" in name.lower()
            ),
            key=lambda n: (len(n), n),
        )
        if matches:
            return matches[0]
    raise KeyError(
        f"Could not find a MuJoCo body for canonical object {canonical_name!r} "
        "(tried the '_burner'/'_main' suffix conventions, the exact name, drawer-level "
        "resolution, and -- for 'table' specifically -- any body name containing 'table')."
    )


def object_within_container_footprint(
    scene: "SceneQuery", object_body: str, container_body: str, z_margin_m: float = 0.10
) -> bool | None:
    """Is ``object_body`` currently resting inside ``container_body``'s own
    footprint -- the same XY-AABB proxy ``libero_place_adapter.py`` uses for
    its own place-verification, plus a Z-height sanity check so an object
    merely transiting over the container isn't mistaken for one already
    resting inside it.

    Used by the pipeline-recall retry loop (``libero_runner.py``) to decide
    whether a place goal is already satisfied before recalling System 2, so
    a recall after an unrelated failure doesn't re-plan a redundant
    pick/place for an object that's already placed. Returns ``None`` (not a
    guess) if either body or the container's collision geometry can't be
    resolved, so the caller treats the goal state as unknown rather than
    unsatisfied.
    """
    try:
        object_pos, _ = scene.body_pose(object_body)
        container_pos, _ = scene.body_pose(container_body)
    except KeyError:
        return None
    container_extent = scene.body_collision_extent_world(container_body)
    if container_extent is None:
        return None
    half_xy = np.asarray(container_extent.half_extents[:2], dtype=np.float64)
    centre_xy = np.asarray(container_pos[:2], dtype=np.float64) + np.asarray(
        container_extent.center_offset[:2], dtype=np.float64
    )
    object_xy = np.asarray(object_pos[:2], dtype=np.float64)
    if not bool(np.all(np.abs(object_xy - centre_xy) <= half_xy)):
        return False
    container_bottom_z = float(
        container_pos[2] + container_extent.center_offset[2] - container_extent.half_extents[2]
    )
    container_top_z = float(
        container_pos[2] + container_extent.center_offset[2] + container_extent.half_extents[2]
    )
    return bool(container_bottom_z - z_margin_m <= float(object_pos[2]) <= container_top_z + z_margin_m)


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

    def __init__(self, model: "mujoco.MjModel", data: "mujoco.MjData", *, _render_sim: Any = None) -> None:
        self.model = model
        self.data = data
        # Optional: the live robosuite sim, kept only for its already-
        # established render context (see render_depth_and_body_ids) -- never
        # read/written for anything else, so this doesn't weaken the
        # scratch-data-only mutation guarantee the rest of this class relies
        # on. None for a SceneQuery built directly from (model, data), e.g.
        # in tests, where render_depth_and_body_ids falls back to
        # mujoco.Renderer.
        self._render_sim = _render_sim

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
        return cls(raw_model, raw_data, _render_sim=sim)

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

    def _gripper_name_alt(self, name: str) -> str | None:
        """The other robosuite gripper-naming convention for ``name``, if any.

        ``panda.py``'s constants (``gripper0_right_eef``, etc.) match
        robosuite's multi-arm-ready naming, where every gripper part gets a
        ``_right``/``_left`` side infix after ``gripper0_``. Older forks --
        e.g. the ``robosuite==1.4.0`` this repo's LIBERO-PRO checkout
        vendors -- only apply that infix to the top-level mount body and
        drop it everywhere else. Trying the other convention here, once,
        keeps every call site working against either install.
        """
        if "gripper0_right_" in name:
            return name.replace("gripper0_right_", "gripper0_", 1)
        if name.startswith("gripper0_") and not name.startswith("gripper0_right_"):
            return name.replace("gripper0_", "gripper0_right_", 1)
        return None

    def body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            alt = self._gripper_name_alt(name)
            if alt is not None:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, alt)
        if bid < 0:
            raise KeyError(f"No body named {name!r} in model.")
        return bid

    def site_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            alt = self._gripper_name_alt(name)
            if alt is not None:
                sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, alt)
        if sid < 0:
            raise KeyError(f"No site named {name!r} in model.")
        return sid

    def joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            alt = self._gripper_name_alt(name)
            if alt is not None:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, alt)
        if jid < 0:
            raise KeyError(f"No joint named {name!r} in model.")
        return jid

    def geom_id(self, name: str) -> int:
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            alt = self._gripper_name_alt(name)
            if alt is not None:
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, alt)
        if gid < 0 and name.startswith("geom_"):
            # Exact inverse of geom_name()'s fallback for a geom with no real
            # MJCF name: geom_name() synthesizes f"geom_{gid}" for those, and
            # any caller that round-trips a name from compute_contacts()/
            # geom_name() back through geom_id() (every CollisionChecker
            # check) needs that fallback to resolve, not raise.
            try:
                candidate = int(name[len("geom_"):])
            except ValueError:
                candidate = -1
            if 0 <= candidate < self.model.ngeom:
                gid = candidate
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

    def nearby_body_names(
        self, point_world: np.ndarray, radius_m: float, exclude: tuple[str, ...] = ()
    ) -> list[str]:
        """Names of free-jointed bodies (LIBERO's manipulable objects -- see
        ``set_free_body_qpos``'s same free-joint check) within ``radius_m`` of
        ``point_world``, excluding ``exclude``. Filtering to free-jointed
        bodies excludes the fixed robot/table/world bodies without needing
        any ``scene_context``/``SceneResolver`` plumbing, which this module
        doesn't have wired in. Used to assemble the wider ``pc_full`` scene
        context a learned grasp detector needs, beyond just the target
        object's own points."""
        point_world = np.asarray(point_world, dtype=np.float64).reshape(3)
        names = []
        for bid in range(self.model.nbody):
            jadr = self.model.body_jntadr[bid]
            if jadr < 0 or self.model.jnt_type[jadr] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not name or name in exclude:
                continue
            if float(np.linalg.norm(self.data.xpos[bid] - point_world)) <= radius_m:
                names.append(name)
        return names

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
        flange site to the TCP/grip site, computed live from the current
        model instead of a hardcoded constant (unlike Cap-X's
        ``FRANKA_HAND_TO_FINGERTIP_Z_M``) -- robosuite's Panda MJCF already
        exposes both frames as named sites, so we read the real transform
        rather than copy a constant that may not match this exact model.
        """
        hand_pos, hand_quat = self.site_pose(hand_site)
        tcp_pos, _ = self.site_pose(tcp_site)
        hand_R = quat_to_rotmat_wxyz(hand_quat)
        local_offset = hand_R.T @ (tcp_pos - hand_pos)
        return float(local_offset[2])

    def palm_clearance_above_tcp_m(
        self,
        hand_site: str = "gripper0_right_ft_frame",
        tcp_site: str = "gripper0_right_grip_site",
        palm_geom_name: str = "gripper0_right_hand_collision",
    ) -> float:
        """How far *before* the TCP, along the hand frame's approach axis,
        the palm/hand collision geometry's near face sits -- i.e. for a
        top-down grasp, how far above the TCP the palm's own lowest point
        already is, purely from the gripper's rigid geometry (pose-
        independent, like ``hand_to_tcp_transform``).

        Needed because the palm is not a point at the hand site: it's a
        sizeable collision mesh whose near face can sit well short of the
        naive "hand-to-tcp distance", so treating that distance alone as the
        gripper's reach silently assumes the palm can occupy space it
        physically can't -- e.g. for a squat object, it can put the palm
        into the object's top even though the TCP itself is still clear.
        """
        tcp_local_z = self.hand_to_tcp_transform(hand_site, tcp_site)
        palm_near_local_z = self._geom_far_local_z(palm_geom_name, hand_site)
        return tcp_local_z - palm_near_local_z

    def _geom_far_local_z(self, geom_name: str, hand_site: str) -> float:
        """A geom's own farthest extent along the hand frame's local +Z axis
        -- shared by ``palm_clearance_above_tcp_m`` (the palm sits before the
        TCP, so this is its near face) and ``finger_reach_below_tcp_m`` (the
        fingers extend past the TCP, so this is their far reach); which side
        of the TCP a geom ends up on falls out of the arithmetic.

        For a mesh geom, uses the real vertices transformed directly into
        the hand's local frame rather than rotating the geom's own AABB
        half-extent into that frame and re-bounding it there -- double-
        bounding a box through an arbitrary rotation always grows it, and
        the stock Panda hand's palm mesh sits at a slight off-axis angle to
        ``gripper0_right_ft_frame``, enough to meaningfully overstate the
        palm's near-face reach (real grasp depth given away to a bounding
        box the palm doesn't actually fill). Other geom types keep the
        cheaper AABB-rotation bound: a box is its own AABB in its local
        frame, and the primitives used elsewhere (finger geoms) are
        capsules/boxes, not meshes.
        """
        hand_pos, hand_quat = self.site_pose(hand_site)
        hand_R = quat_to_rotmat_wxyz(hand_quat)
        gid = self.geom_id(geom_name)
        gpos = self.data.geom_xpos[gid]
        gmat = self.data.geom_xmat[gid].reshape(3, 3)
        gtype = self.model.geom_type[gid]
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = self.model.geom_dataid[gid]
            vert_adr = self.model.mesh_vertadr[mesh_id]
            vert_num = self.model.mesh_vertnum[mesh_id]
            verts_local = self.model.mesh_vert[vert_adr : vert_adr + vert_num].reshape(-1, 3)
            verts_world = (gmat @ verts_local.T).T + gpos
            verts_hand_local_z = (hand_R.T @ (verts_world - hand_pos).T).T[:, 2]
            return float(np.max(verts_hand_local_z))
        local_half = self._geom_local_half_extent(gtype, self.model.geom_size[gid])
        rel_mat = hand_R.T @ gmat
        rel_pos = hand_R.T @ (gpos - hand_pos)
        return float(rel_pos[2] + (np.abs(rel_mat) @ local_half)[2])

    def finger_reach_below_tcp_m(
        self,
        hand_site: str = "gripper0_right_ft_frame",
        tcp_site: str = "gripper0_right_grip_site",
        finger_geom_names: tuple[str, ...] = (
            "gripper0_right_finger1_collision",
            "gripper0_right_finger2_collision",
            "gripper0_right_finger1_pad_collision",
            "gripper0_right_finger2_pad_collision",
        ),
    ) -> float:
        """How far *past* the TCP, along the hand frame's approach axis, the
        farthest-reaching finger geom extends -- i.e. for a top-down grasp,
        how far below the TCP the fingertips already reach, purely from the
        gripper's rigid geometry (pose-independent, mirroring
        ``palm_clearance_above_tcp_m`` for the opposite end of the approach).

        Needed so a grasp height doesn't ask the fingertips to reach below
        the surface a thin object rests on -- targeting a short object's
        exact geometric mid-height can otherwise put the rigid finger body
        into the floor even though the TCP itself is still comfortably
        above it.
        """
        tcp_local_z = self.hand_to_tcp_transform(hand_site, tcp_site)
        farthest = tcp_local_z
        for name in finger_geom_names:
            farthest = max(farthest, self._geom_far_local_z(name, hand_site))
        return farthest - tcp_local_z

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
        wherever a body's physical shape matters -- e.g. a LIBERO container's
        high-poly visual mesh alongside the simple boxes (floor + walls)
        that actually model its collision geometry. The visual mesh's own
        bounding box spans the whole hollow container as if it were solid,
        since an AABB can't represent concavity."""
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
        body's purely-visual mesh, whose own bounding box can extend past
        its actual physical collision shape."""
        bid = self.body_id(body_name)
        return self._merged_extent_world(self.body_collision_geom_ids(body_name), self.data.xpos[bid])

    def body_extent_local(self, body_name: str) -> BodyExtent | None:
        """Axis-aligned half-extents of a body's collision geoms **in the
        body's own local frame** -- i.e. invariant to the body's current
        world orientation, unlike ``body_extent_world``, whose world-axis-
        aligned box inflates under rotation.

        Use this wherever the extent will be re-rotated by a different
        orientation later -- e.g. fitting collision spheres to an object at
        grasp time, to be checked at whatever orientation it's carried at
        during transport/place. Reusing ``body_extent_world`` there would
        double-count rotation (once baked into the stored extent, again when
        the caller re-rotates it), inflating the fitted sphere far beyond
        the object's true size and making placement candidates spuriously
        read as colliding.
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

    def body_surface_points_local(self, body_name: str) -> np.ndarray:
        """Sample points on/near the real surface of a body's collision
        geometry (box corners, capsule/cylinder rings -- see
        ``_geom_sample_points_local``), all in the body's own local
        (pose-independent) frame. Shared by whatever needs the body's true
        shape rather than just its axis-aligned bounding box: fitting
        collision spheres to a carried object
        (``attached_object.fit_spheres_from_geoms``) and fitting an oriented
        bounding box for grasp candidates."""
        gids = self.body_collision_geom_ids(body_name)
        if not gids:
            return np.zeros((0, 3))
        bid = self.body_id(body_name)
        body_pos = self.data.xpos[bid]
        body_mat = self.data.xmat[bid].reshape(3, 3)
        point_sets = [self._geom_sample_points_local(g, body_mat, body_pos) for g in gids]
        non_empty = [p for p in point_sets if len(p)]
        return np.concatenate(non_empty, axis=0) if non_empty else np.zeros((0, 3))

    def _geom_sample_points_local(self, gid: int, body_mat: np.ndarray, body_pos: np.ndarray) -> np.ndarray:
        """A handful of extreme surface points of one geom, expressed in the
        body's local frame -- box corners for box/mesh geoms, a ring of
        points for round primitives."""
        gpos = self.data.geom_xpos[gid]
        gmat = self.data.geom_xmat[gid].reshape(3, 3)
        size = self.model.geom_size[gid]
        gtype = self.model.geom_type[gid]
        rel_mat = body_mat.T @ gmat
        rel_pos = body_mat.T @ (gpos - body_pos)

        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            return np.zeros((0, 3))
        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = float(size[0])
            local_pts = np.array(
                [[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r], [0, 0, 0]]
            )
        elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            r, half_len = float(size[0]), float(size[1])
            n_ring = 16
            angles = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
            # Slightly inflate the sampled ring radius so a finite number of
            # angular samples still safely bounds the true (continuous) round
            # surface at any angle in between.
            r_safe = r / np.cos(np.pi / n_ring)
            local_pts = np.array(
                [[r_safe * np.cos(a), r_safe * np.sin(a), z] for z in (-half_len, half_len) for a in angles]
                + [[0, 0, 0]]
            )
        else:  # box, mesh: 8 corners of the local half-extent box
            hx, hy, hz = self._geom_local_half_extent(gtype, size)
            local_pts = np.array(
                [[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            )
        return (rel_mat @ local_pts.T).T + rel_pos

    def body_surface_points_dense_local(
        self, body_name: str, points_per_edge: int = 4, n_ring: int = 24, n_bands: int = 3
    ) -> np.ndarray:
        """Like ``body_surface_points_local``, but dense enough for point-
        cloud-based grasp inference (a learned detector like Contact-GraspNet
        expects hundreds+ of points; the sparse version is only meant to
        bound a shape for OBB-fitting, not represent its surface). Still
        pure ground-truth mesh geometry -- grids box/mesh faces and adds
        more rings/latitude bands for round primitives, no camera rendering
        involved."""
        gids = self.body_collision_geom_ids(body_name)
        if not gids:
            return np.zeros((0, 3))
        bid = self.body_id(body_name)
        body_pos = self.data.xpos[bid]
        body_mat = self.data.xmat[bid].reshape(3, 3)
        point_sets = [
            self._geom_sample_points_dense_local(g, body_mat, body_pos, points_per_edge, n_ring, n_bands)
            for g in gids
        ]
        non_empty = [p for p in point_sets if len(p)]
        return np.concatenate(non_empty, axis=0) if non_empty else np.zeros((0, 3))

    def _geom_sample_points_dense_local(
        self, gid: int, body_mat: np.ndarray, body_pos: np.ndarray, points_per_edge: int, n_ring: int, n_bands: int
    ) -> np.ndarray:
        """Dense counterpart to ``_geom_sample_points_local`` -- same
        per-geom-type structure, denser sampling in each branch."""
        gpos = self.data.geom_xpos[gid]
        gmat = self.data.geom_xmat[gid].reshape(3, 3)
        size = self.model.geom_size[gid]
        gtype = self.model.geom_type[gid]
        rel_mat = body_mat.T @ gmat
        rel_pos = body_mat.T @ (gpos - body_pos)

        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            return np.zeros((0, 3))
        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = float(size[0])
            n_lat = max(3, n_ring // 3)
            pts = []
            for lat in np.linspace(-np.pi / 2.0, np.pi / 2.0, n_lat):
                n_lon = max(1, int(round(n_ring * np.cos(lat))))
                for lon in np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False):
                    pts.append(
                        [r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)]
                    )
            local_pts = np.array(pts) if pts else np.zeros((0, 3))
        elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            r, half_len = float(size[0]), float(size[1])
            angles = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
            r_safe = r / np.cos(np.pi / n_ring)
            bands = np.linspace(-half_len, half_len, max(2, n_bands))
            local_pts = np.array(
                [[r_safe * np.cos(a), r_safe * np.sin(a), z] for z in bands for a in angles]
            )
        else:  # box, mesh: grid every face, not just the 8 corners
            hx, hy, hz = self._geom_local_half_extent(gtype, size)
            n = max(2, int(points_per_edge))
            u = np.linspace(-1.0, 1.0, n)
            pts = []
            for sx in (-1.0, 1.0):
                for vy in u:
                    for vz in u:
                        pts.append([sx * hx, vy * hy, vz * hz])
            for sy in (-1.0, 1.0):
                for vx in u:
                    for vz in u:
                        pts.append([vx * hx, sy * hy, vz * hz])
            for sz in (-1.0, 1.0):
                for vx in u:
                    for vy in u:
                        pts.append([vx * hx, vy * hy, sz * hz])
            local_pts = np.array(pts)
        return (rel_mat @ local_pts.T).T + rel_pos

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
            # A plane's geom_size is a rendering hint (visual quad half-width
            # in xy, grid-line spacing in z), not physical geometry. Treating
            # it as a literal box half-extent (like the box/mesh fallback
            # below) makes a ground plane read as a thick slab, so anything
            # resting on it would read as already deep inside it. Model it as
            # a thin, effectively-2D slab instead.
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

    # ---------------------------------------------------------------- camera / rendering

    def camera_id(self, name: str) -> int:
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            raise KeyError(f"No camera named {name!r} in model.")
        return cid

    def camera_pose_world(self, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
        """(position, 3x3 rotation) of a camera in world frame. Camera-local
        +X is right, +Y is up, +Z points *away* from the view direction (the
        camera looks down its own local -Z) -- MuJoCo's own convention,
        matching ``scripts/visualize_grasp_poses.py``'s ``_camera_project``."""
        cid = self.camera_id(camera_name)
        return self.data.cam_xpos[cid].copy(), self.data.cam_xmat[cid].reshape(3, 3).copy()

    def camera_intrinsics(self, camera_name: str, height: int, width: int) -> np.ndarray:
        """Pinhole intrinsics matrix for a render of the given size, derived
        from the camera's vertical FOV (``model.cam_fovy``) -- the same
        formula ``visualize_grasp_poses.py`` already uses for its own forward
        projection, so a render captured here and one captured there agree."""
        cid = self.camera_id(camera_name)
        fovy_deg = float(self.model.cam_fovy[cid])
        f = 0.5 * height / np.tan(0.5 * np.deg2rad(fovy_deg))
        return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])

    def render_depth_and_body_ids(
        self, camera_name: str, height: int, width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Real single-camera render: (depth_m (H,W) float32, body_id (H,W)
        int64, -1 where no geom/an unknown object type was hit).

        Renders through ``self._render_sim`` (the live robosuite sim's own,
        already-established GL context -- set by ``from_robosuite_sim``) when
        available, falling back to a fresh ``mujoco.Renderer`` otherwise (a
        ``SceneQuery`` built directly from (model, data), e.g. in tests, has
        no live sim to reuse).

        Reusing the live sim's context is not just an optimization: creating
        (and especially closing) a second, independent GL context steals the
        process's thread-local "current" context away from robosuite's,
        which never re-binds its own on later ``render()`` calls -- so once
        stolen, every later robosuite render comes out as garbage for the
        rest of the process.
        """
        if self._render_sim is not None:
            return self._render_depth_and_body_ids_via_sim(camera_name, height, width)
        return self._render_depth_and_body_ids_via_mujoco_renderer(camera_name, height, width)

    def _render_depth_and_body_ids_via_sim(
        self, camera_name: str, height: int, width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        from robosuite.utils.camera_utils import get_real_depth_map

        sim = self._render_sim
        seg, depth_normalized = sim.render(
            width=width, height=height, camera_name=camera_name, depth=True, segmentation=True
        )
        # robosuite's raw sim.render() is OpenGL row order (row 0 = bottom of
        # image) -- flip to the standard top-to-bottom order
        # deproject_camera_depth assumes (mujoco.Renderer already returns
        # that order unflipped).
        seg = np.asarray(seg)[::-1]
        depth_normalized = np.asarray(depth_normalized, dtype=np.float32)[::-1]
        depth = np.asarray(get_real_depth_map(sim, depth_normalized), dtype=np.float32)

        obj_type, geom_id = seg[:, :, 0], seg[:, :, 1]
        body_id = np.full(geom_id.shape, -1, dtype=np.int64)
        is_geom = obj_type == mujoco.mjtObj.mjOBJ_GEOM
        body_id[is_geom] = self.model.geom_bodyid[geom_id[is_geom]]
        return depth, body_id

    def _render_depth_and_body_ids_via_mujoco_renderer(
        self, camera_name: str, height: int, width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback used only when there's no live sim to render through
        (``self._render_sim is None``). Two things to watch for:

        1. Offscreen multisampling (``model.vis.quality.offsamples``) blends
           object-id colors at silhouette edges, and ``mujoco.Renderer``'s
           segmentation decode has no bounds check against that, causing an
           ``IndexError`` at anti-aliased edges. Disabled here only for the
           duration of this one render (``self.model`` may be shared with a
           live robosuite env's own rendering, restored in ``finally``).
        2. ``mujoco.Renderer``'s segmentation mode must be enabled and then
           ``update_scene`` called again before rendering -- calling
           ``render()`` right after ``enable_segmentation_rendering()``
           without refreshing the scene reuses a stale id table and hits the
           same ``IndexError``.

        No row flip is needed on this path's output -- ``mujoco.Renderer``
        already returns standard top-to-bottom row order, unlike robosuite's
        own raw ``sim.render()`` (see ``_render_depth_and_body_ids_via_sim``).
        """
        old_samples = self.model.vis.quality.offsamples
        self.model.vis.quality.offsamples = 0
        renderer = None
        try:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            renderer.update_scene(self.data, camera=camera_name)
            renderer.enable_depth_rendering()
            depth = np.asarray(renderer.render(), dtype=np.float32).copy()
            renderer.disable_depth_rendering()

            renderer.enable_segmentation_rendering()
            renderer.update_scene(self.data, camera=camera_name)
            seg = np.asarray(renderer.render()).copy()  # (H, W, 2): (objid, objtype)
            renderer.disable_segmentation_rendering()
        finally:
            if renderer is not None:
                renderer.close()
            self.model.vis.quality.offsamples = old_samples

        geom_id, obj_type = seg[:, :, 0], seg[:, :, 1]
        body_id = np.full(geom_id.shape, -1, dtype=np.int64)
        is_geom = obj_type == mujoco.mjtObj.mjOBJ_GEOM
        body_id[is_geom] = self.model.geom_bodyid[geom_id[is_geom]]
        return depth, body_id

    def deproject_camera_depth(self, camera_name: str, depth: np.ndarray) -> np.ndarray:
        """(H, W) depth (meters, as returned by ``render_depth_and_body_ids``)
        -> (H, W, 3) world-frame points, inverting ``_camera_project``'s
        forward pinhole model (see ``camera_pose_world``/``camera_intrinsics``
        for the convention)."""
        height, width = depth.shape
        cam_pos, cam_mat = self.camera_pose_world(camera_name)
        K = self.camera_intrinsics(camera_name, height, width)
        f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        x_cam = (cols - cx) * depth / f
        y_cam = -(rows - cy) * depth / f
        z_cam = -depth
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        return pts_cam @ cam_mat.T + cam_pos

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
