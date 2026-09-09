"""Shared low-level servo/actuation helpers for executor-facing motion adapters.

Extracted from ``libero_pick_adapter.py`` so ``LiberoOpenDrawerAdapter`` doesn't
duplicate the same Cartesian-delta servo math: both adapters convert a
joint-space MPC/reference target into the clipped-delta OSC_POSE action the
legacy stage machine's ``BaseManipulationEnv._compose_full_action`` already
expects, via FK on a scratch ``SceneQuery`` -- never the live sim directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..orientation_utils import orientation_error_rotvec
from .scene_query import SceneQuery, quat_to_rotmat_wxyz


def is_settle_stalled(recent_errors: list[float], window: int, tol_m: float) -> bool:
    """True once a settle loop's rolling window of position errors shows no
    meaningful improvement over ``window`` consecutive steps.

    Both ``LiberoPickAdapter`` and ``LiberoOpenDrawerAdapter`` run a settle
    loop after their planned reference is exhausted: keep servoing toward
    the fixed final grasp/pull pose until the real (laggy) controller
    actually converges, up to ``max_settle_steps``. When the target sits
    right at/through a real contact boundary (a grasp pose that asks the
    fingers to close the last couple mm against the object itself, or a
    pull target the drawer physically resists at the end of its travel),
    the loop never converges *and* never gets closer -- it just repeatedly
    pushes into and eases back off the obstacle at full reactive gain for
    the entire remaining budget. That doesn't change where the arm ends up
    (it wasn't getting any closer anyway), it just visibly looks like the
    gripper hitting something and retrying. Detecting the plateau and
    breaking out early stops the pointless struggling without changing the
    final outcome.
    """
    if len(recent_errors) < window:
        return False
    tail = recent_errors[-window:]
    return (max(tail) - min(tail)) < tol_m


def real_nonpermitted_contacts(
    scene: SceneQuery,
    checker,
    phase,
    target_object_body: str | None,
    permitted_contact_geom_ids: set[int] | None = None,
) -> list[list]:
    """Real contacts from the *live* sim (not a hypothetical ``q``) involving
    the robot/gripper that ``CollisionChecker`` would not have permitted --
    i.e. a genuine collision, not the deliberate fingerpad<->target contact
    ``TERMINAL_APPROACH`` allows or the small finger-overlap tolerance.
    Reuses ``CollisionChecker``'s own ``_permitted``/
    ``_finger_overlap_within_tolerance`` logic so this reports exactly the
    same thing the planner itself would have flagged as a violation, not a
    cruder "any negative distance" check (which fires constantly on
    sub-millimeter numerical grazing that never mattered).

    ``permitted_contact_geom_ids``, when given, also excludes gripper<->target
    contact through those specific geoms -- mirrors ``CollisionChecker.
    check``'s own parameter of the same name (see its docstring), needed by
    a skill (e.g. ``close``) whose intended contact is made by the palm, not
    the fingerpads ``_permitted`` alone recognizes."""
    target_ids = set(scene.body_geom_ids(target_object_body)) if target_object_body else set()
    out: list[list] = []
    for g1n, g2n, dist in scene.compute_contacts():
        if dist >= 0:
            continue
        g1, g2 = scene.geom_id(g1n), scene.geom_id(g2n)
        if g1 not in checker.robot_geom_ids and g2 not in checker.robot_geom_ids:
            continue
        if checker._permitted(g1, g2, phase, target_ids):
            continue
        if permitted_contact_geom_ids and (
            (g1 in permitted_contact_geom_ids and g2 in checker.gripper_geom_ids)
            or (g2 in permitted_contact_geom_ids and g1 in checker.gripper_geom_ids)
        ):
            continue
        if checker._finger_overlap_within_tolerance(g1, g2, phase, target_ids, dist):
            continue
        out.append([g1n, g2n, round(float(dist), 5)])
    return out


@dataclass
class MotionTraceRecorder:
    """Accumulates a compact, textual/numeric stand-in for "what did this
    look like" for one MPC-tracked phase (descent/pull/retreat), for
    ``execution_summary.json``. Nobody debugging a run reviews the saved
    video frame-by-frame -- this is what lets a bad-looking run (arm
    wandering, visibly colliding, jerking) actually be diagnosed from the
    JSON alone."""

    ee_positions: list = field(default_factory=list)
    min_clearance_m: float = float("inf")
    replan_requested_steps: int = 0
    collision_events: list = field(default_factory=list)

    def record_step(
        self,
        step_index: int,
        ee_pos: np.ndarray,
        min_clearance: float,
        replan_requested: bool,
        bad_contacts: list,
    ) -> None:
        self.ee_positions.append([float(x) for x in ee_pos])
        if min_clearance < self.min_clearance_m:
            self.min_clearance_m = float(min_clearance)
        if replan_requested:
            self.replan_requested_steps += 1
        if bad_contacts:
            self.collision_events.append({"step": step_index, "contacts": bad_contacts})

    def summary(self) -> dict:
        base = {
            "n_steps": len(self.ee_positions),
            "min_clearance_m": None if self.min_clearance_m == float("inf") else self.min_clearance_m,
            "replan_requested_steps": self.replan_requested_steps,
            "n_collision_events": len(self.collision_events),
            # Capped so one badly-behaved run can't blow up the summary file;
            # the count above still reflects the true total.
            "collision_events": self.collision_events[:20],
        }
        if len(self.ee_positions) < 2:
            return base
        positions = np.asarray(self.ee_positions)
        deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        straight_line = float(np.linalg.norm(positions[-1] - positions[0]))
        actual = float(np.sum(deltas))
        base.update(
            {
                "start_ee_pos": positions[0].tolist(),
                "end_ee_pos": positions[-1].tolist(),
                "straight_line_distance_m": straight_line,
                "actual_path_length_m": actual,
                # 1.0 = the arm moved straight to its target; well below 1.0
                # means it wandered/oscillated rather than moving directly.
                "path_efficiency": (straight_line / actual) if actual > 1e-9 else 1.0,
                "max_single_step_jump_m": float(np.max(deltas)),
            }
        )
        return base


class _ServoMixin:
    """Requires ``self.env``, ``self.joint_names``, ``self.ee_site`` on the
    including class (set by each adapter's ``__init__``)."""

    def servo_to_home(self, n_steps: int = 60, step_callback=None) -> int:
        """Densely interpolate the arm back to its own reset configuration,
        gripper open, and return the steps consumed.

        A skill otherwise starts from wherever the previous one finished --
        a configuration chosen for *that* task, often mid-reach over the spot
        it just released something. Planning a fresh approach from there is a
        recurring failure (measured: an outright planner TIMEOUT from one
        such pose, a settle ending 0.53m from its target from another, and a
        NO_PATH_FOUND transporting an object away from a just-completed
        pick). Resetting first costs a fixed, cheap joint-space move and
        gives every skill the same known starting point. Shared by the pick
        and open-drawer adapters; a joint-space interpolation rather than one
        Cartesian move to home's end-effector pose, for the reason in
        DrawerConfig.return_home_steps.
        """
        home_q = np.asarray(self.env.env.robots[0].init_qpos, dtype=np.float64)
        current_q = SceneQuery.from_robosuite_sim(self.env.env.sim).get_qpos(self.joint_names)
        if float(np.linalg.norm(current_q - home_q)) < 1e-3:
            return 0
        steps = 0
        for target_q in np.linspace(current_q, home_q, max(2, int(n_steps))):
            live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, target_q)
            self._step_env(cart_pos, cart_ori, self.env.open_gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
        return steps

    def _servo_toward_joint_target(
        self, scene: SceneQuery, target_q: np.ndarray, pos_scale_m: float = 0.03, ori_gain: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray]:
        scratch = scene.clone()
        scratch.set_qpos(self.joint_names, target_q)
        target_pos, target_quat = scratch.site_pose(self.ee_site)
        target_R = quat_to_rotmat_wxyz(target_quat)

        ee_pos = self.env._eef_pos(self.env.last_raw_obs)
        ee_R = self.env._eef_rotmat(self.env.last_raw_obs)
        cart_pos = np.clip((target_pos - ee_pos) / pos_scale_m, -1.0, 1.0).astype(np.float32)
        ori_err = orientation_error_rotvec(ee_R, target_R)
        cart_ori = np.clip(ori_gain * ori_err, -1.0, 1.0).astype(np.float32)
        return cart_pos, cart_ori

    def _servo_toward_cartesian_pose(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        pos_scale_m: float = 0.03,
        ori_gain: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        target_R = quat_to_rotmat_wxyz(target_quat_wxyz)
        ee_pos = self.env._eef_pos(self.env.last_raw_obs)
        ee_R = self.env._eef_rotmat(self.env.last_raw_obs)
        pos_err = target_pos - ee_pos
        cart_pos = np.clip(pos_err / pos_scale_m, -1.0, 1.0).astype(np.float32)
        ori_err_vec = orientation_error_rotvec(ee_R, target_R)
        cart_ori = np.clip(ori_gain * ori_err_vec, -1.0, 1.0).astype(np.float32)
        return cart_pos, cart_ori, float(np.linalg.norm(pos_err)), float(np.linalg.norm(ori_err_vec))

    def _step_env(self, cart_pos: np.ndarray, cart_ori: np.ndarray, gripper_cmd: float) -> dict:
        full_action = self.env._compose_full_action(cart_pos, cart_ori, gripper_cmd)
        obs, _reward, _terminated, info = self.env._step_backend(full_action)
        self.env.last_raw_obs = obs
        return info
