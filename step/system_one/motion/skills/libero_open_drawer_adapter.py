"""Executor-facing adapter for the ``open`` skill: wires ``OpenDrawerSkill``'s
grasp-handle + pull plan into a live ``LiberoManipulationEnv``.

Mirrors ``LiberoPickAdapter``'s structure (same time-parameterize -> MPC
tracking loop, same low-level servo helpers via ``_ServoMixin``), but differs
where the physics genuinely differs:

- No ``_is_grasped()`` check -- robosuite's ``_check_grasp`` needs a
  registered ``contact_geoms`` set that LIBERO never assigns to fixtures.
  Success is verified structurally instead: after tracking the planned pull,
  read the drawer joint's *live* qpos and check how far it actually travelled
  -- a strictly more direct signal than any contact heuristic, since if the
  fingers weren't really gripping the handle the joint simply won't have
  moved.
- No ``AttachedObjectManager`` registration -- the drawer stays connected to
  the cabinet via its own joint; nothing is carried afterward.
- Ends with a retreat (open gripper, servo back along the pull axis) instead
  of a hold, so a following subtask doesn't start with the gripper wedged
  against the handle.
"""

from __future__ import annotations

import numpy as np

from ..articulation import discover_drawers, measure_pull_axis_and_travel, resolve_drawer
from ..collision import CollisionChecker, CollisionPhase
from ..config import MotionConfig
from ..kinematics import IKSolver
from ..mpc import JointSpaceMPC, MPCContext
from ..ompl_planner import MotionPlanner
from .open_drawer_skill import OpenDrawerSkill
from ..panda import EE_SITE, FINGER_JOINT_NAMES, GRIPPER_BODIES, HAND_SITE, JOINT_NAMES, ROBOT_BODIES
from ..scene_query import SceneQuery
from ..servo import MotionTraceRecorder, _ServoMixin, is_settle_stalled, real_nonpermitted_contacts
from ..time_parameterize import TimeParameterizer
from ..types import MPCCostConfig, SkillFailureReason, SkillResult


class LiberoOpenDrawerAdapter(_ServoMixin):
    def __init__(self, env, motion_config: MotionConfig | None = None) -> None:
        self.env = env
        self.config = motion_config or MotionConfig()
        self.joint_names = JOINT_NAMES
        self.ee_site = EE_SITE
        self.hand_site = HAND_SITE

    def _build_scene(self) -> SceneQuery:
        return SceneQuery.from_robosuite_sim(self.env.env.sim)

    def _build_skill(self, scene: SceneQuery) -> OpenDrawerSkill:
        checker = CollisionChecker(
            scene, self.joint_names, self.ee_site, ROBOT_BODIES, GRIPPER_BODIES, self.config.collision
        )
        lo, hi = scene.joint_qpos_range(self.joint_names)
        ik = IKSolver(joint_names=self.joint_names, site_name=self.ee_site)
        planner = MotionPlanner(self.joint_names, lo, hi, checker, self.config.planner)
        return OpenDrawerSkill(ik, planner, checker, self.config)

    def _servo_to_home(self, step_callback=None) -> int:
        return self.servo_to_home(self.config.drawer.return_home_steps, step_callback)

    def run(
        self,
        drawer_ref: str,
        constraints: dict | None = None,
        max_steps: int = 600,
        step_callback=None,
    ) -> tuple[SkillResult, int, dict]:
        constraints = constraints or {}

        # Go to the robot's home configuration *before* planning the handle
        # approach. Whatever ran previously leaves the arm wherever it
        # finished -- typically hovering over the spot where it just
        # released an object, in a configuration chosen for that task, not
        # this one. Planning the long reach to a drawer handle from there
        # was a recurring failure (measured across rollouts: an outright
        # planner TIMEOUT from one such pose, and a settle that ended 0.53m
        # from the handle from another), and it is the same reasoning that
        # already sends the arm home *after* opening. Homing first costs a
        # fixed, cheap joint-space move and makes this skill start from the
        # one configuration its approach is reliably solvable from.
        pre_home_steps = self._servo_to_home(step_callback)

        scene = self._build_scene()
        skill = self._build_skill(scene)
        start_q = scene.get_qpos(self.joint_names)

        plan_result = skill.execute(scene, drawer_ref, start_q, constraints)
        if not plan_result.success:
            return plan_result, pre_home_steps, {}

        drawer = resolve_drawer(scene, drawer_ref)
        pull_axis_world, travel_m, qpos_closed, qpos_open = measure_pull_axis_and_travel(scene, drawer)

        traj = TimeParameterizer(self.config.time_param).parameterize(plan_result.planned_trajectory)
        if traj.positions is None:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = "time parameterization produced no reference"
            return plan_result, 0, {}

        grasp_position = np.asarray(plan_result.diagnostics["grasp_position"])
        grasp_quat_wxyz = np.asarray(plan_result.diagnostics["grasp_quat_wxyz"])

        approach_positions = np.asarray(plan_result.planned_trajectory.metadata["approach_positions"])
        full_reference = np.concatenate([traj.positions, approach_positions[1:]], axis=0)
        approach_start_idx = len(traj.positions)

        checker = skill.collision_checker
        drawer_body = drawer.drawer_body
        # NOTE: real obstacle spheres + a phased (free_space -> settle-to-
        # standoff -> descent) restructure were both tried here, to let the
        # MPC's existing clearance cost actively steer the FREE_SPACE leg
        # around real clutter per the intended RRT-gives-an-approximate-path
        # / MPC-locally-avoids-obstacles design. Iterated through two real,
        # fixed bugs in the process (sibling drawer bodies wrongly treated
        # as obstacles; the settle-to-standoff checkpoint converging on
        # Cartesian pose only, which a redundant 7-DoF arm can reach via a
        # different joint configuration than the one the descent reference
        # assumed) -- but even after both fixes, grip success stayed at
        # 0-1/10 against a measured 3/8-3/10 baseline for this single,
        # continuous-tracking design. The remaining suspect is the settle-
        # to-standoff checkpoint itself: it's a hard stop-and-converge with
        # a *different* control law (plain P-servo, not the MPC) inserted
        # into what was previously one continuous, receding-horizon-smoothed
        # MPC track from free_space straight into descent -- not re-verified
        # further given time spent. Reverted to this simpler, actually-
        # measured design rather than ship something worse; `build_static_
        # obstacle_spheres` (mpc.py) stays available for that follow-up.
        mpc = JointSpaceMPC(
            self.joint_names,
            self.ee_site,
            horizon=int(self.config.mpc.get("horizon", 16)),
            cost_config=MPCCostConfig(**{k: v for k, v in self.config.mpc.items() if k in MPCCostConfig.__dataclass_fields__}),
        )
        ctx = MPCContext(reference=full_reference)
        mpc.reset()

        cfg = self.config.execution
        max_hold_steps = max_steps if max_steps is not None else len(full_reference) + cfg.max_settle_steps
        tracking_budget = min(max_hold_steps, len(full_reference) + cfg.max_settle_steps)

        steps = pre_home_steps  # the homing move above is real env time too
        gripper_cmd = self.env.open_gripper_cmd
        pos_err = ori_err = float("inf")
        converged = False
        stalled = False
        recent_pos_errs: list[float] = []
        t0 = 0
        descent_trace = MotionTraceRecorder()
        while t0 < tracking_budget:
            phase = CollisionPhase.FREE_SPACE if t0 < approach_start_idx else CollisionPhase.TERMINAL_APPROACH
            live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            q_current = live_scene.get_qpos(self.joint_names)
            mpc_result = mpc.step(
                live_scene,
                q_current,
                ctx,
                t0_step=t0,
                collision_checker=checker,
                collision_phase=phase,
                target_object_body=drawer_body,
                replan_config=self.config.replan,
            )
            cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, mpc_result.joint_command)
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            t0 += 1
            if step_callback is not None:
                step_callback()
            post_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            descent_trace.record_step(
                t0,
                post_scene.site_pose(self.ee_site)[0],
                mpc_result.min_clearance,
                mpc_result.replan_requested,
                real_nonpermitted_contacts(post_scene, checker, phase, drawer_body),
            )
            if steps >= max_steps:
                break

            if t0 >= len(full_reference):
                _, _, pos_err, ori_err = self._servo_toward_cartesian_pose(grasp_position, grasp_quat_wxyz)
                if pos_err <= cfg.settle_position_tol_m and ori_err <= cfg.settle_orientation_tol_rad:
                    converged = True
                    break
                recent_pos_errs.append(pos_err)
                if is_settle_stalled(recent_pos_errs, cfg.stall_window_steps, cfg.stall_position_tol_m):
                    # Not getting any closer -- almost always because the
                    # grasp pose asks the fingers to close the last couple mm
                    # against the handle/drawer face itself. Continuing to
                    # command at full reactive gain toward an unreachable-any-
                    # closer target just repeatedly pushes into and eases off
                    # that surface for the rest of the budget (visibly the
                    # arm "hitting something and retrying" on video) without
                    # ever landing anywhere better -- stop here instead.
                    stalled = True
                    break

        settle_steps = max(0, t0 - len(full_reference))
        plan_result.diagnostics["settle"] = {
            "steps": settle_steps,
            "converged": converged,
            "stalled": stalled,
            "position_error_m": pos_err,
            "orientation_error_rad": ori_err,
        }

        # Close the gripper on the handle -- only *now* does tracking the
        # pull leg make physical sense (an earlier version tracked the pull
        # as part of the same pre-grasp reference, driving the arm all the
        # way to the "open" joint target while the gripper was still open,
        # so it never actually grasped the handle at all; see
        # OpenDrawerSkill.execute()'s comment on why the two legs are kept
        # separate).
        gripper_cmd = self.env.close_gripper_cmd
        close_dwell_steps = max(int(self.env.close_steps) * 2, 25)
        for _ in range(close_dwell_steps):
            hold_pos, hold_ori, _, _ = self._servo_toward_cartesian_pose(grasp_position, grasp_quat_wxyz)
            self._step_env(hold_pos, hold_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()

        # Diagnostic only (not a hard gate, unlike pick's `_is_grasped()`
        # check) -- robosuite's `_check_grasp` needs `contact_geoms` LIBERO
        # never registers for a fixture, so there's no equivalent built-in
        # signal; this is the same fingerpad<->target contact test
        # `CollisionChecker` itself uses to decide what contact is
        # *permitted* during the approach, just read directly off the real
        # sim instead of a hypothetical `q`. Recorded so a low `open_frac`
        # afterward can be told apart from "never actually gripped the
        # handle" vs. "gripped it but the pull didn't reach far enough".
        grip_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        drawer_geom_ids = set(grip_scene.body_geom_ids(drawer_body))
        gripped_handle = any(
            (g1 in checker.fingerpad_geom_ids and g2 in drawer_geom_ids)
            or (g2 in checker.fingerpad_geom_ids and g1 in drawer_geom_ids)
            for name1, name2, _dist in grip_scene.compute_contacts()
            for g1, g2 in [(grip_scene.geom_id(name1), grip_scene.geom_id(name2))]
        )
        # How far, and in which direction, the descent actually ended from
        # the handle it aimed at. A scalar settle error can't distinguish
        # "stopped short along the approach" from "ended up below the target"
        # -- and below is what matters here, since the drawers' handles are
        # only ~7cm apart vertically, so a downward miss lands on the *next
        # drawer's* handle.
        end_hand_pos = grip_scene.site_pose(self.ee_site)[0]
        handle_miss = np.asarray(end_hand_pos) - grasp_position
        plan_result.diagnostics["handle_approach_miss"] = {
            "delta_xyz": [float(v) for v in handle_miss],
            "distance_m": float(np.linalg.norm(handle_miss)),
            "dz_m": float(handle_miss[2]),
        }

        plan_result.diagnostics["handle_grip_verification"] = {"gripped": gripped_handle}

        # Hard gate, not just a diagnostic. Pulling without actually holding
        # the handle does not simply fail harmlessly: the hand is still
        # pressed against the cabinet, so dragging it along the pull axis
        # rakes whatever it *is* touching. Measured on a real rollout -- the
        # target (top) drawer ended 2.4% open while the middle drawer, never
        # asked for, was dragged 94% open, leaving the scene in a worse state
        # than before the skill ran and looking, in the user's words, "very
        # clunky". Failing here keeps the wrong-drawer damage from happening
        # at all; the diagnostics above say exactly why the grip missed.
        # Opening a drawer does not require a clean grasp: an open gripper
        # whose fingers straddle the handle hooks it, and pulling along the
        # joint axis opens the drawer regardless of whether the pads ever
        # made contact. So the gate is "is the handle actually captured
        # between the fingers", not "did we grip it" -- the strict pad test
        # alone would reject a perfectly good hook pull. What the gate is
        # really there to prevent is pulling while the hand is merely
        # *resting against* the cabinet somewhere, which is how a missed
        # approach dragged the middle drawer 94% open while its own target
        # barely moved.
        captured = gripped_handle or float(np.linalg.norm(handle_miss)) <= self.config.drawer.handle_capture_radius_m
        plan_result.diagnostics["handle_grip_verification"]["captured_between_fingers"] = bool(captured)
        if not captured:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = (
                "aborted before pulling: the handle is not between the fingers after the "
                f"descent (missed {drawer_body}'s handle by {np.linalg.norm(handle_miss):.3f}m, "
                f"dz={handle_miss[2]:+.3f}m). Pulling from here would drag whatever the hand "
                "is resting against."
            )
            steps += self._servo_to_home(step_callback)
            return plan_result, steps, {}

        # Track the pull leg with its own short MPC-tracked reference (fresh
        # context, not a continuation of `ctx`/`full_reference`, which only
        # ever covered the pre-grasp approach) -- still collision-checked
        # under TERMINAL_APPROACH exactly like the descent was. Runs until
        # actual Cartesian convergence (same settle pattern as the descent
        # phase above, same tolerances) rather than a fixed step count: a
        # fixed small budget measurably left the drawer only partially
        # pulled (e.g. 52-76% open against an 85%-of-travel target) because
        # the real controller's tracking lag ate most of a short budget,
        # exactly the same lag the descent phase's own settle loop exists to
        # absorb.
        pulled_q = np.asarray(plan_result.planned_trajectory.metadata["pulled_q"])
        pull_fraction_used = float(plan_result.diagnostics.get("pull_fraction_used") or 0.0)
        pulled_position = grasp_position + pull_fraction_used * travel_m * pull_axis_world
        current_q = SceneQuery.from_robosuite_sim(self.env.env.sim).get_qpos(self.joint_names)
        pull_reference = np.linspace(current_q, pulled_q, 15)
        pull_ctx = MPCContext(reference=pull_reference)
        mpc.reset()
        pull_budget = len(pull_reference) + cfg.max_settle_steps
        t0 = 0
        pull_converged = False
        pull_stalled = False
        recent_pull_errs: list[float] = []
        pull_trace = MotionTraceRecorder()
        while t0 < pull_budget and steps < max_steps:
            live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            q_current = live_scene.get_qpos(self.joint_names)
            mpc_result = mpc.step(
                live_scene,
                q_current,
                pull_ctx,
                t0_step=t0,
                collision_checker=checker,
                collision_phase=CollisionPhase.TERMINAL_APPROACH,
                target_object_body=drawer_body,
                replan_config=self.config.replan,
            )
            cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, mpc_result.joint_command)
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            t0 += 1
            if step_callback is not None:
                step_callback()
            post_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            pull_trace.record_step(
                t0,
                post_scene.site_pose(self.ee_site)[0],
                mpc_result.min_clearance,
                mpc_result.replan_requested,
                real_nonpermitted_contacts(post_scene, checker, CollisionPhase.TERMINAL_APPROACH, drawer_body),
            )
            if t0 >= len(pull_reference):
                _, _, pull_pos_err, pull_ori_err = self._servo_toward_cartesian_pose(pulled_position, grasp_quat_wxyz)
                if pull_pos_err <= cfg.settle_position_tol_m and pull_ori_err <= cfg.settle_orientation_tol_rad:
                    pull_converged = True
                    break
                recent_pull_errs.append(pull_pos_err)
                if is_settle_stalled(recent_pull_errs, cfg.stall_window_steps, cfg.stall_position_tol_m):
                    pull_stalled = True
                    break
        plan_result.diagnostics["pull_settle"] = {
            "steps": max(0, t0 - len(pull_reference)),
            "converged": pull_converged,
            "stalled": pull_stalled,
        }

        # Structural success check: how far did the joint actually travel,
        # not whether any contact heuristic fired.
        final_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        qpos_final = float(final_scene.get_qpos([drawer.joint_name])[0])
        denom = qpos_open - qpos_closed
        open_frac = (qpos_final - qpos_closed) / denom if abs(denom) > 1e-9 else 0.0
        plan_result.diagnostics["open_verification"] = {
            "joint_name": drawer.joint_name,
            "qpos_closed": qpos_closed,
            "qpos_open": qpos_open,
            "qpos_final": qpos_final,
            "open_frac": open_frac,
        }

        # Release and retreat clear of the handle regardless of outcome, so a
        # following subtask doesn't start with the gripper wedged in place.
        # Retreat *further* along the open direction from the arm's actual
        # current (pulled) position -- not backward from the stale pre-pull
        # `grasp_position` toward the closed end, which was a real bug: that
        # commanded the gripper to travel back through/past the handle in
        # the *closing* direction, shoving the drawer closed again right
        # after successfully opening it.
        gripper_cmd = self.env.open_gripper_cmd
        retreat_target = pulled_position + self.config.drawer.retreat_m * pull_axis_world
        retreat_trace = MotionTraceRecorder()
        for _ in range(self.config.drawer.retreat_steps):
            retreat_pos, retreat_ori, _, _ = self._servo_toward_cartesian_pose(retreat_target, grasp_quat_wxyz)
            self._step_env(retreat_pos, retreat_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
            post_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            # No MPC step here (retreat is a plain servo, not MPC-tracked),
            # so there's no min_clearance/replan_requested signal to report
            # -- still worth recording ee_pos/contacts, since this is
            # exactly the phase the retreat-direction bug lived in.
            retreat_trace.record_step(
                steps,
                post_scene.site_pose(self.ee_site)[0],
                float("inf"),
                False,
                real_nonpermitted_contacts(post_scene, checker, CollisionPhase.TERMINAL_APPROACH, drawer_body),
            )

        # Re-check *after* retreating, not before: the retreat itself is a
        # real, physically-contactful motion right next to the handle (the
        # gripper starts the retreat still close to it, gripper-open takes a
        # few sim steps to actually clear), so it's the true final state
        # that determines whether the drawer was actually left open -- an
        # earlier bug in the retreat direction silently dragged the drawer
        # back toward closed *after* this success check had already been
        # computed against the pre-retreat qpos, so the pre-retreat check
        # alone isn't trustworthy.
        post_retreat_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        qpos_post_retreat = float(post_retreat_scene.get_qpos([drawer.joint_name])[0])
        open_frac_post_retreat = (qpos_post_retreat - qpos_closed) / denom if abs(denom) > 1e-9 else 0.0
        plan_result.diagnostics["open_verification"]["open_frac_post_retreat"] = open_frac_post_retreat

        # Every drawer's travel, not just the target's. Verifying only the
        # joint we aimed at cannot tell "opened the right drawer" from
        # "shoved a neighbouring one open on the way past" -- and a hand that
        # stalls short of its handle and drags along the cabinet face does
        # exactly that. Cheap to record, and the only way to check the
        # complaint "it opened the middle drawer" against data rather than
        # against a low-resolution video frame.
        sibling_travel = {}
        for other in discover_drawers(post_retreat_scene):
            lo_o, hi_o = post_retreat_scene.joint_qpos_range([other.joint_name])
            span = float(hi_o[0] - lo_o[0])
            q_now = float(post_retreat_scene.get_qpos([other.joint_name])[0])
            q_closed = float(np.clip(0.0, lo_o[0], hi_o[0]))
            sibling_travel[other.level_label] = {
                "joint": other.joint_name,
                "qpos": q_now,
                "open_fraction_of_range": abs(q_now - q_closed) / span if span > 1e-9 else 0.0,
                "is_target": other.joint_name == drawer.joint_name,
            }
        plan_result.diagnostics["open_verification"]["all_drawers"] = sibling_travel

        # Return to the robot's own home/reset joint configuration --
        # regardless of outcome, same "leave the arm in a good state for
        # whatever runs next" reasoning as the retreat above. The handle-
        # approach IK solution that opened the drawer is often a genuinely
        # awkward configuration (elbow twisted, wrist near a joint limit);
        # see DrawerConfig.return_home_steps's own comment for why this is
        # a dense joint-space interpolation, not a single Cartesian move to
        # home's end-effector pose.
        home_q = np.asarray(self.env.env.robots[0].init_qpos, dtype=np.float64)
        current_q = post_retreat_scene.get_qpos(self.joint_names)
        home_reference = np.linspace(current_q, home_q, self.config.drawer.return_home_steps)
        return_home_trace = MotionTraceRecorder()
        for target_q in home_reference:
            live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, target_q)
            self._step_env(cart_pos, cart_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
            post_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            return_home_trace.record_step(
                steps,
                post_scene.site_pose(self.ee_site)[0],
                float("inf"),
                False,
                real_nonpermitted_contacts(post_scene, checker, CollisionPhase.FREE_SPACE, drawer_body),
            )

        # Stand-in for "what did this look like" without needing to watch
        # the saved video: per-phase path smoothness (path_efficiency near
        # 1.0 = moved directly to target; well below = wandered/oscillated),
        # worst clearance seen, and any real (non-permitted) collisions,
        # each phase-tagged so a bad-looking run can be localized to
        # descent/pull/retreat/return_home specifically.
        plan_result.diagnostics["motion_trace"] = {
            "descent": descent_trace.summary(),
            "pull": pull_trace.summary(),
            "retreat": retreat_trace.summary(),
            "return_home": return_home_trace.summary(),
        }

        if open_frac_post_retreat < self.config.drawer.open_threshold_frac:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = (
                f"drawer only reached open_frac={open_frac_post_retreat:.3f} after retreat "
                f"(was {open_frac:.3f} right after the pull), "
                f"below threshold {self.config.drawer.open_threshold_frac}"
            )
            return plan_result, steps, {}

        plan_result.selected_object = drawer.level_label + "_drawer"
        return plan_result, steps, plan_result.diagnostics["open_verification"]
