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
from .collision import CollisionPhase
from .mpc import JointSpaceMPC, MPCContext
from .scene_query import SceneQuery, quat_to_rotmat_wxyz
from .types import MPCCostConfig


def wait_steps(adapter, n_steps: int, gripper_cmd: float, step_callback=None) -> int:
    """EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md, "wait skill" section).

    Hold the current end-effector pose still for ``n_steps`` ticks (zero
    Cartesian-delta each tick, ``gripper_cmd`` held constant by the caller).
    Used to satisfy a not-before STL timing constraint by literally waiting
    rather than releasing early. ``adapter`` only needs a
    ``_step_env(cart_pos, cart_ori, gripper_cmd)`` method, which every
    ``_ServoMixin`` subclass and ``LiberoPlaceAdapter`` has, so this one
    function serves pick, transport, and place.
    """
    zero = np.zeros(3, dtype=np.float32)
    steps = 0
    for _ in range(max(0, int(n_steps))):
        adapter._step_env(zero, zero, gripper_cmd)
        steps += 1
        if step_callback is not None:
            step_callback()
    return steps


def is_settle_stalled(recent_errors: list[float], window: int, tol_m: float) -> bool:
    """True once a settle loop's rolling window of position errors shows no
    meaningful improvement over ``window`` consecutive steps.

    Settle loops keep servoing toward a fixed final pose until the real
    (laggy) controller converges or ``max_settle_steps`` runs out. When the
    target sits right at a real contact boundary (e.g. fingers closing
    against the object, or a drawer resisting at the end of its travel), the
    loop never converges or gets any closer -- it just pushes into and eases
    off the obstacle for the rest of the budget. Detecting that plateau and
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
    ``TERMINAL_APPROACH`` allows. Reuses ``CollisionChecker``'s own
    ``_permitted``/``_finger_overlap_within_tolerance`` logic so this
    reports exactly what the planner itself would flag, not a cruder "any
    negative distance" check that fires on sub-millimeter numerical grazing.

    ``permitted_contact_geom_ids``, when given, also excludes gripper<->target
    contact through those specific geoms -- needed by a skill (e.g. ``close``)
    whose intended contact is made by the palm, not the fingerpads
    ``_permitted`` alone recognizes."""
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
        often mid-reach over the spot it just released something -- and
        planning a fresh approach from there was a recurring source of
        planner timeouts and failed transports. Resetting first costs a
        fixed, cheap joint-space move and gives every skill the same known
        starting point. Joint-space interpolation rather than one Cartesian
        move to home's end-effector pose, for the reason in
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


@dataclass
class TrackAndSettleResult:
    """Everything a ``Libero*Adapter`` needs to build its own
    ``plan_result.diagnostics[...]`` entry for one tracked leg. ``steps`` is
    just this call's own increment (add it to your running total);
    ``settle_steps`` is how many were spent past the end of ``reference``
    (the hold-and-converge portion)."""

    steps: int
    settle_steps: int
    converged: bool
    stalled: bool
    position_error_m: float
    orientation_error_rad: float
    wrong_obstacle_contacts: list | None = None

    def as_diagnostics(self) -> dict:
        # ``wrong_obstacle_contacts`` is always present (None when not
        # requested/not stalled) -- callers that check it
        # (``check_wrong_obstacle=True``) index it unconditionally, the same
        # way pick's original inline dict always carried the key.
        return {
            "steps": self.settle_steps,
            "converged": self.converged,
            "stalled": self.stalled,
            "position_error_m": self.position_error_m,
            "orientation_error_rad": self.orientation_error_rad,
            "wrong_obstacle_contacts": self.wrong_obstacle_contacts,
        }


def track_and_settle(
    adapter,
    reference: np.ndarray,
    *,
    joint_names: list[str],
    ee_site: str,
    approach_start_idx: int,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    collision_checker,
    mpc_horizon: int,
    mpc_cost_config: MPCCostConfig,
    replan_config,
    gripper_cmd: float,
    settle_position_tol_m: float,
    settle_orientation_tol_rad: float,
    stall_window_steps: int,
    stall_position_tol_m: float,
    max_settle_steps: int,
    mpc_step_kwargs: dict | None = None,
    steps_so_far: int = 0,
    max_steps: int | None = None,
    step_callback=None,
    terminal_approach_uses_mpc: bool = True,
    check_orientation_convergence: bool = True,
    abort_on_stall: bool = True,
    check_wrong_obstacle: bool = False,
    wrong_obstacle_graze_tolerance_m: float = 0.005,
    target_object_body: str | None = None,
    permitted_contact_geom_ids: set[int] | None = None,
    trace: "MotionTraceRecorder | None" = None,
) -> TrackAndSettleResult:
    """Track ``reference`` via a fresh collision-aware ``JointSpaceMPC``
    (FREE_SPACE below ``approach_start_idx``, TERMINAL_APPROACH at/after it --
    pass ``0`` for an always-TERMINAL_APPROACH leg, e.g. a pull/turn, or a
    number >= the eventual tracking budget for an always-FREE_SPACE leg, e.g.
    transport), then keep holding at ``(target_pos, target_quat_wxyz)`` until
    the real controller actually converges or stalls.

    This is the one control loop every ``Libero*Adapter.run()`` used to
    hand-roll independently (pick, open_drawer, close, transport, stove each
    had their own near-identical copy); a new skill adapter now calls this
    once per tracked leg instead of re-deriving it.

    ``mpc_step_kwargs`` are merged into every ``mpc.step()`` call -- different
    skills need different extra context there (e.g. transport has no single
    "target object" since the carried object is itself part of the robot's
    collision geometry for that leg). ``target_object_body`` here is used
    only for trace recording and the wrong-obstacle check below.

    ``terminal_approach_uses_mpc=False`` switches to plain joint-space
    reference tracking once TERMINAL_APPROACH starts -- needed by close's
    descent+push, where the MPC's smoothness/effort cost fights the
    deliberate stall-into-contact a push requires.

    ``check_orientation_convergence=False`` (stove's turn leg) ignores
    orientation error for the convergence gate, since the hand rotates with
    the knob and would never converge against a fixed pre-turn target.

    ``check_wrong_obstacle``/``abort_on_stall`` (pick only, so far): on a
    stall, tell a real stuck-against-a-different-object contact apart from a
    benign tracking-lag plateau. A benign plateau (``abort_on_stall=False``
    and no wrong-obstacle contact found) keeps the loop running instead of
    giving up, since bailing out there would hand the next phase a wrong
    starting XY with no way to correct it. A real wrong-obstacle contact, or
    ``abort_on_stall=True``, always breaks regardless.
    """
    mpc = JointSpaceMPC(joint_names, ee_site, horizon=mpc_horizon, cost_config=mpc_cost_config)
    ctx = MPCContext(reference=reference)
    mpc.reset()
    mpc_step_kwargs = mpc_step_kwargs or {}

    max_hold_steps = max_steps if max_steps is not None else len(reference) + max_settle_steps
    tracking_budget = min(max_hold_steps, len(reference) + max_settle_steps)

    local_steps = 0
    pos_err = ori_err = float("inf")
    converged = False
    stalled = False
    wrong_obstacle_contacts: list | None = None
    recent_pos_errs: list[float] = []
    t0 = 0
    while t0 < tracking_budget:
        phase = CollisionPhase.FREE_SPACE if t0 < approach_start_idx else CollisionPhase.TERMINAL_APPROACH
        live_scene = SceneQuery.from_robosuite_sim(adapter.env.env.sim)
        q_current = live_scene.get_qpos(joint_names)

        if terminal_approach_uses_mpc or phase == CollisionPhase.FREE_SPACE:
            mpc_result = mpc.step(
                live_scene,
                q_current,
                ctx,
                t0_step=t0,
                collision_checker=collision_checker,
                collision_phase=phase,
                replan_config=replan_config,
                **mpc_step_kwargs,
            )
            joint_target = mpc_result.joint_command
            step_min_clearance = mpc_result.min_clearance
            step_replan_requested = mpc_result.replan_requested
        else:
            # Deliberate contact phase (e.g. close's push): track the
            # planned reference directly, no MPC optimization. See this
            # function's own docstring for why.
            ref_idx = min(t0, len(reference) - 1)
            joint_target = reference[ref_idx]
            step_min_clearance = float("inf")
            step_replan_requested = False

        cart_pos, cart_ori = adapter._servo_toward_joint_target(live_scene, joint_target)
        adapter._step_env(cart_pos, cart_ori, gripper_cmd)
        local_steps += 1
        t0 += 1
        if step_callback is not None:
            step_callback()

        if trace is not None:
            post_scene = SceneQuery.from_robosuite_sim(adapter.env.env.sim)
            trace.record_step(
                t0,
                post_scene.site_pose(ee_site)[0],
                step_min_clearance,
                step_replan_requested,
                real_nonpermitted_contacts(
                    post_scene, collision_checker, phase, target_object_body, permitted_contact_geom_ids
                ),
            )

        if max_steps is not None and steps_so_far + local_steps >= max_steps:
            break

        if t0 >= len(reference):
            _, _, pos_err, ori_err = adapter._servo_toward_cartesian_pose(target_pos, target_quat_wxyz)
            orientation_ok = (not check_orientation_convergence) or (ori_err <= settle_orientation_tol_rad)
            if pos_err <= settle_position_tol_m and orientation_ok:
                converged = True
                break
            recent_pos_errs.append(pos_err)
            if is_settle_stalled(recent_pos_errs, stall_window_steps, stall_position_tol_m):
                stalled = True
                if check_wrong_obstacle:
                    stall_scene = SceneQuery.from_robosuite_sim(adapter.env.env.sim)
                    bad_contacts = real_nonpermitted_contacts(stall_scene, collision_checker, phase, target_object_body)
                    target_ids = set(stall_scene.body_geom_ids(target_object_body)) if target_object_body else set()
                    wrong_obstacle_contacts = [
                        c
                        for c in bad_contacts
                        if stall_scene.geom_id(c[0]) not in target_ids
                        and stall_scene.geom_id(c[1]) not in target_ids
                        and c[2] < -wrong_obstacle_graze_tolerance_m
                    ]
                    if wrong_obstacle_contacts or abort_on_stall:
                        break
                    # Benign plateau, not a real obstacle -- keep going. See
                    # this function's own docstring.
                    continue
                if abort_on_stall:
                    break

    settle_steps = max(0, t0 - len(reference))
    return TrackAndSettleResult(
        steps=local_steps,
        settle_steps=settle_steps,
        converged=converged,
        stalled=stalled,
        position_error_m=pos_err,
        orientation_error_rad=ori_err,
        wrong_obstacle_contacts=wrong_obstacle_contacts,
    )
