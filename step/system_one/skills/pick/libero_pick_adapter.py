"""Executor-facing adapter: wires the grasp-gen -> IK -> OMPL -> time-param ->
MPC pipeline into a live ``LiberoManipulationEnv`` for the ``pick`` skill.

Scope of this integration slice: **pick only**. ``transport``/``place`` still
run on the legacy stage machine even when ``motion_backend: "ompl_mpc"`` is
set -- see ``System1Executor``'s dispatch in ``executor.py``. This keeps the
new pipeline's grasp/IK/planning/collision logic independently verifiable
against the real env and gripper, without requiring new transport/place
motion-generation code for a pick-only smoke test to pass.

Action-space bridge: the MPC outputs joint-position targets, but LIBERO's
robosuite env (like the legacy stage machine) takes Cartesian-delta OSC_POSE
actions. Each joint target is converted to a Cartesian pose via forward
kinematics (on a scratch ``SceneQuery``, never the live sim) and tracked with
the same clipped-delta servo law the legacy stage machine already uses
(``BaseManipulationEnv._compose_full_action``), reusing a proven low-level
control primitive instead of inventing a second one.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np

from ...motion.attached_object import AttachedObjectManager
from ...motion.collision import CollisionChecker, CollisionPhase
from ...motion.config import MotionConfig
from .grasp_generator import (
    GraspGenerator,
    ManualDeepCenterGraspGenerator,
    PrivilegedTopDownGraspGenerator,
    _is_forced_manual_object_class,
    _is_rim_graspable_object_class,
    manual_pose_width_feasible,
)
from ...motion.kinematics import IKSolver
from ...motion.mpc import JointSpaceMPC, MPCContext
from ...motion.ompl_planner import MotionPlanner
from ...motion.panda import EE_SITE, GRIPPER_BODIES, HAND_SITE, JOINT_NAMES, ROBOT_BODIES
from .pick_skill import PickSkill
from ...motion.scene_query import SceneQuery, quat_to_rotmat_wxyz, resolve_body_name
from ...motion.servo import _ServoMixin, track_and_settle
from ...motion.time_parameterize import TimeParameterizer
from ...motion.types import AttachedObjectState, MPCCostConfig, SkillFailureReason, SkillResult

# Matches CollisionConfig.self_margin_m (0.005) -- the smallest margin
# already tolerated elsewhere in this pipeline's own collision checks. A
# contact against an unrelated neighbor closer than this is real; anything
# shallower is sub-mm meshing/numerical noise, not a genuine obstruction.
WRONG_OBSTACLE_GRAZE_TOLERANCE_M = 0.005

# Extra clearance above a cabinet's own measured top when extracting an
# object from inside one of its drawers (see _drawer_extraction_target_z),
# on top of just reaching that top height -- covers the attached object's
# own collision radius so it isn't still clipping the frame's top edge by
# the time transport's hover-candidate search begins.
DRAWER_EXTRACTION_MARGIN_M = 0.06


def _drawer_extraction_target_z(scene: SceneQuery, object_body_name: str) -> float | None:
    """If ``object_body_name`` currently sits inside a drawer -- its xy
    within some cabinet frame body's own footprint, and its z below that
    frame's measured top -- return the world-frame height the object needs
    to clear (that frame's top + DRAWER_EXTRACTION_MARGIN_M). Returns None
    if the object isn't inside any discovered cabinet's footprint (the
    common case: an object picked off an open surface).

    A cabinet's static frame occupies its own full footprint at every height
    below its top except where a drawer hollows out a slot, so an object
    whose xy falls inside that footprint while resting below the frame's top
    can only be inside a drawer. Needed because the flat
    post_grasp_carry_lift_m (tuned for objects starting on an open surface)
    isn't tall enough to clear a drawer opening on its own.
    """
    try:
        from ...motion.articulation import discover_cabinet_support_bodies
    except Exception:
        return None

    object_pos, _ = scene.body_pose(object_body_name)
    best_target_z: float | None = None
    for cabinet_body in discover_cabinet_support_bodies(scene):
        cab_pos, _ = scene.body_pose(cabinet_body)
        extent = scene.body_extent_world(cabinet_body)
        if extent is None:
            continue
        top_z = float(cab_pos[2] + extent.center_offset[2] + extent.half_extents[2])
        half_xy = extent.half_extents[:2]
        within_xy = bool(np.all(np.abs(object_pos[:2] - cab_pos[:2]) <= half_xy))
        if within_xy and float(object_pos[2]) < top_z:
            candidate_target = top_z + DRAWER_EXTRACTION_MARGIN_M
            if best_target_z is None or candidate_target > best_target_z:
                best_target_z = candidate_target
    return best_target_z


class LiberoPickAdapter(_ServoMixin):
    def __init__(self, env, motion_config: MotionConfig | None = None) -> None:
        self.env = env
        self.config = motion_config or MotionConfig()
        self.joint_names = JOINT_NAMES
        self.ee_site = EE_SITE
        self.hand_site = HAND_SITE
        self._pick_image_counter = 0

    # ------------------------------------------------------------ construction

    def _build_scene(self) -> SceneQuery:
        return SceneQuery.from_robosuite_sim(self.env.env.sim)

    def _build_pick_skill(self, scene: SceneQuery, finger_overlap_tolerance_m: float | None = None) -> PickSkill:
        collision_config = self.config.collision
        if finger_overlap_tolerance_m is not None:
            collision_config = dataclasses.replace(
                collision_config, finger_overlap_tolerance_m=finger_overlap_tolerance_m
            )
        checker = CollisionChecker(
            scene, self.joint_names, self.ee_site, ROBOT_BODIES, GRIPPER_BODIES, collision_config
        )
        lo, hi = scene.joint_qpos_range(self.joint_names)
        if self.config.contact_graspnet.enabled:
            # Lazily imported: this module (and the CUDA it may need) is only
            # ever touched when explicitly enabled -- see
            # contact_graspnet_generator.py's module docstring.
            #
            # GraspNet-only, not composited with the heuristic generator:
            # PickSkill's OR-set cost function tends to pick the heuristic's
            # cheap-but-possibly-wrong candidate over a real GraspNet one
            # (its zero-cost generator_score_penalty plus lower path cost
            # wins regardless of grasp quality). Once GraspNet is enabled, it
            # alone proposes candidates so that tradeoff can't reintroduce
            # the heuristic's known-bad grasps for rim/hollow objects.
            from .contact_graspnet_generator import ContactGraspNetGenerator

            gen: GraspGenerator = ContactGraspNetGenerator(self.config.contact_graspnet)
        else:
            gen = PrivilegedTopDownGraspGenerator(self.config.grasp)
        ik = IKSolver(joint_names=self.joint_names, site_name=self.ee_site)
        planner = MotionPlanner(self.joint_names, lo, hi, checker, self.config.planner)
        return PickSkill(gen, ik, planner, checker, self.config)

    # ------------------------------------------------------------ low-level stepping
    # (_servo_toward_joint_target / _servo_toward_cartesian_pose / _step_env are
    # provided by _ServoMixin -- shared verbatim with LiberoOpenDrawerAdapter.)

    def _current_q(self, scene: SceneQuery) -> np.ndarray:
        # Re-sync from the live sim every read -- the scene passed in at
        # construction time is a one-shot snapshot that goes stale as soon
        # as the arm moves.
        return SceneQuery.from_robosuite_sim(self.env.env.sim).get_qpos(self.joint_names)

    def _save_grasp_pose_image(
        self, scene: SceneQuery, candidates: list, plan_result: SkillResult, outdir: str | Path | None
    ) -> None:
        if outdir is None:
            return
        try:
            from .grasp_viz import save_grasp_candidates_image

            camera_name = getattr(self.env, "camera_name", "agentview")
            width = int(getattr(self.env, "camera_width", 512))
            height = int(getattr(self.env, "camera_height", 512))
            frame = self.env.render_rgb_frame(camera_name=camera_name, width=width, height=height)
            selected = plan_result.selected_grasp if plan_result.success else None
            out_path = Path(outdir) / f"grasp_poses_pick_{self._pick_image_counter}.png"
            save_grasp_candidates_image(frame, scene.model, scene.data, camera_name, candidates, out_path, selected)
            self._pick_image_counter += 1
        except Exception:
            # Purely observational -- a rendering hiccup here must never take
            # down the actual pick attempt.
            logging.getLogger(__name__).exception("Failed to save grasp-pose visualization")

    # ------------------------------------------------------------ closer-look fallback

    def _compute_hover_pose(
        self, scene: SceneQuery, object_body_name: str, extra_height_m: float = 0.15
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """A wrist-camera vantage point above the object, not a grasp pose.

        Reuses ``PrivilegedTopDownGraspGenerator``'s own object-center/height
        logic (already accounts for the object's real AABB and this robot's
        palm/finger clearance) purely to get a reasonable XY/Z above the
        object, then adds extra height so the wrist camera gets a clear
        top-down view without itself colliding. Orientation is a plain
        top-down look (no yaw), since this is just a vantage point.
        """
        heuristic = PrivilegedTopDownGraspGenerator(self.config.grasp)
        candidates = heuristic.generate(scene, [object_body_name])
        if not candidates:
            return None
        hover_pos = candidates[0].grasp_pose_world[:3].copy()
        hover_pos[2] += extra_height_m
        # Gripper approach axis (+Z) pointing straight down (-world Z) -- a
        # 180 degree rotation about world X, same top-down convention
        # grasp_generator.py's own _TOP_DOWN_QUAT_WXYZ uses; no yaw, since
        # this is a vantage point, not the eventual grasp.
        hover_quat = np.array([0.0, 1.0, 0.0, 0.0])
        return hover_pos, hover_quat

    def _move_to_hover(
        self,
        scene: SceneQuery,
        skill: PickSkill,
        start_q: np.ndarray,
        hover_pos: np.ndarray,
        hover_quat: np.ndarray,
        step_callback,
        max_steps: int,
    ) -> tuple[SceneQuery, np.ndarray, int] | None:
        """Physically move to ``hover_pos``/``hover_quat`` under the same
        collision-aware planning/tracking this file uses everywhere else --
        never a teleport. Returns ``(new_scene, new_start_q, steps_used)``, or
        ``None`` if the hover pose itself isn't IK-reachable or planning to it
        fails (the caller then just keeps the original failure)."""
        ik = skill.ik_solver.solve(scene.clone(), hover_pos, hover_quat, seed_q=start_q)
        if not ik.success:
            return None
        check = skill.collision_checker.check(ik.q, phase=CollisionPhase.FREE_SPACE)
        if check.in_collision:
            return None
        plan = skill.planner.plan(start_q, [ik.q], phase=CollisionPhase.FREE_SPACE)
        if not plan.success:
            return None
        traj = TimeParameterizer(self.config.time_param).parameterize(plan)
        if traj.positions is None:
            return None

        mpc = JointSpaceMPC(
            self.joint_names,
            self.ee_site,
            horizon=int(self.config.mpc.get("horizon", 16)),
            cost_config=MPCCostConfig(**{k: v for k, v in self.config.mpc.items() if k in MPCCostConfig.__dataclass_fields__}),
        )
        ctx = MPCContext(reference=traj.positions)
        mpc.reset()

        steps = 0
        for t0 in range(len(traj.positions)):
            live_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
            q_current = live_scene.get_qpos(self.joint_names)
            mpc_result = mpc.step(
                live_scene, q_current, ctx, t0_step=t0,
                collision_checker=skill.collision_checker, collision_phase=CollisionPhase.FREE_SPACE,
            )
            cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, mpc_result.joint_command)
            self._step_env(cart_pos, cart_ori, self.env.open_gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
            if steps >= max_steps:
                break

        return self._build_scene(), steps

    # ------------------------------------------------------------ public entry point

    def run(
        self,
        object_set: list[str],
        constraints: dict | None = None,
        max_steps: int = 600,
        step_callback=None,
        outdir: str | Path | None = None,
    ) -> tuple[SkillResult, int, AttachedObjectState | None]:
        """Returns (SkillResult, steps_executed, AttachedObjectState | None) so the executor can build a
        ``SkillExecutionRecord`` without re-deriving step counts.

        ``step_callback``, if given, is called with no arguments after every
        real env step -- the executor wires this to ``_capture_video_frame``
        so ``--save-video`` works for this backend too (this loop doesn't go
        through ``step_nominal()``, which is where that normally gets called).

        ``outdir``, if given, gets a ``grasp_poses_pick_N.png`` image of every
        candidate this call considered (green = PickSkill's actual selection,
        if any), saved automatically on every real run. Candidates are
        generated once here and handed straight to ``skill.execute()`` (its
        ``candidates=`` param) rather than regenerated, since a learned
        generator like ContactGraspNetGenerator re-runs real inference on
        every ``generate()`` call.
        """
        constraints = constraints or {}

        # Reset to home before planning the grasp, for the same reason the
        # open-drawer adapter does (see _ServoMixin.servo_to_home): a pick
        # usually follows a place, which leaves the arm in a configuration
        # chosen for that other task, and planning a fresh reach from there
        # is where this pipeline tends to fail.
        pre_home_steps = self.servo_to_home(self.config.drawer.return_home_steps, step_callback)

        scene = self._build_scene()
        skill = self._build_pick_skill(scene)
        start_q = scene.get_qpos(self.joint_names)

        # object_set comes from SceneResolver-canonical names (e.g.
        # "alphabet_soup_1"); the motion pipeline needs real MuJoCo body names
        # (e.g. "alphabet_soup_1_main") -- see resolve_body_name's docstring.
        body_names = [resolve_body_name(scene, name) for name in object_set]
        canonical_by_body = dict(zip(body_names, object_set))

        # Object classification, not a per-call heuristic: a rim-graspable
        # object (bowl/mug/cup) has no solid material at its center for the
        # manual pose to close on, so it always needs GraspNet's real
        # antipodal-contact search. Everything else is assigned to the manual
        # pose exclusively (skipping GraspNet/heuristic's slower fallback
        # searches, a deliberate speed/robustness tradeoff for the common
        # case) *unless* the manual pose's fixed axis (non-negotiable, see
        # ManualDeepCenterGraspGenerator's docstring) can't even physically
        # span the object's real width at that axis -- checked here up front
        # via manual_pose_width_feasible rather than discovered only after
        # every manual candidate is rejected. _is_forced_manual_object_class
        # (FORCE_MANUAL_KEYWORDS) routes a known-unsolved object class (e.g.
        # an upright book, whose narrow axis is vertical, so no top-down
        # close can ever span it) back to manual-only anyway, to fail fast
        # rather than burn a GraspNet pass on a pose nothing currently solves.
        manual_only = bool(body_names) and all(
            not _is_rim_graspable_object_class(b)
            and (
                manual_pose_width_feasible(scene, b, skill.grasp_generator.gripper_max_opening_m)
                or _is_forced_manual_object_class(b)
            )
            for b in body_names
        )

        manual_plan_result = None
        manual_candidates: list = []
        if self.config.contact_graspnet.enabled:
            manual_candidates = ManualDeepCenterGraspGenerator(
                skill.grasp_generator.gripper_max_opening_m
            ).generate(scene, body_names)
            if manual_candidates:
                manual_plan_result = skill.execute(
                    scene, body_names, start_q, constraints, candidates=manual_candidates
                )

        if manual_plan_result is not None and (manual_plan_result.success or manual_only):
            candidates, plan_result = manual_candidates, manual_plan_result
        else:
            candidates = skill.grasp_generator.generate(scene, body_names)
            plan_result = skill.execute(scene, body_names, start_q, constraints, candidates=candidates)

        if manual_only and not plan_result.success and plan_result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE:
            # The one fallback a manual-only object still gets: the same
            # widened-tolerance retry the heuristic path below uses as its own
            # last resort, applied directly to the manual candidate instead of
            # falling through to GraspNet/heuristic -- a small real mesh
            # protrusion can exceed the default compliance margin at an
            # otherwise-correct grasp. Cheap (one candidate, not a full
            # inference or multi-yaw search), so it doesn't undermine the
            # point of skipping those for this object class.
            relaxed_skill = self._build_pick_skill(scene, finger_overlap_tolerance_m=0.02)
            relaxed_plan_result = relaxed_skill.execute(
                scene, body_names, start_q, constraints, candidates=manual_candidates
            )
            candidates, plan_result = manual_candidates, relaxed_plan_result

        pre_steps = pre_home_steps  # the homing move above is real env time too
        needs_closer_look = (
            not plan_result.success
            and plan_result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE
            and len(body_names) == 1
            and not manual_only
        )
        if needs_closer_look:
            # The single-shot capture (from wherever the arm started) can be
            # too sparse for a small/thin object. Move the wrist camera close
            # to the object first, then recapture and regenerate from there,
            # rather than giving up on the first (distant) look.
            hover = self._compute_hover_pose(scene, body_names[0])
            if hover is not None:
                moved = self._move_to_hover(scene, skill, start_q, hover[0], hover[1], step_callback, max_steps)
                if moved is not None:
                    new_scene, hover_steps = moved
                    pre_steps = hover_steps
                    new_start_q = new_scene.get_qpos(self.joint_names)
                    new_candidates = skill.grasp_generator.generate(new_scene, body_names)
                    new_plan_result = skill.execute(
                        new_scene, body_names, new_start_q, constraints, candidates=new_candidates
                    )
                    # Only adopt the closer-look attempt if it actually did
                    # better, so diagnostics/video still reflect whichever
                    # attempt is being reported.
                    if new_plan_result.success or len(new_candidates) > len(candidates):
                        scene, start_q, candidates, plan_result = new_scene, new_start_q, new_candidates, new_plan_result

        if not plan_result.success and plan_result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE and not manual_only:
            # Last resort: GraspNet found nothing usable even after the closer
            # look above. Try the privileged heuristic generator's own
            # candidates as a genuinely separate, sequential attempt -- NOT
            # composited into the same OR-set GraspNet's candidates went
            # through (compositing was tried and reverted: the heuristic's
            # zero-cost generator_score_penalty let its cheap candidate beat a
            # real GraspNet one regardless of grasp quality). Running this
            # only after GraspNet has already failed outright avoids
            # reintroducing that regression.
            heuristic_candidates = PrivilegedTopDownGraspGenerator(self.config.grasp).generate(scene, body_names)
            if heuristic_candidates:
                heuristic_plan_result = skill.execute(
                    scene, body_names, start_q, constraints, candidates=heuristic_candidates
                )
                # Adopt this attempt's diagnostics either way, not just on
                # success -- GraspNet has already failed, so if the heuristic
                # also fails, its rejection reasons are what's worth seeing.
                candidates, plan_result = heuristic_candidates, heuristic_plan_result

            if not plan_result.success and plan_result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE:
                # Absolute last resort: even the heuristic's own candidates
                # were all rejected, which can happen for a genuinely real
                # reason (a small mesh protrusion not captured by the AABB
                # exceeding the default compliance margin). Retry the same
                # candidates through a skill built with a deliberately
                # widened tolerance for contact with the target object
                # specifically -- TERMINAL_APPROACH already only permits
                # gripper<->target contact, so this doesn't loosen any other
                # safety check. Better to attempt the grasp and let
                # _is_grasped() be the final arbiter than give up outright.
                relaxed_skill = self._build_pick_skill(scene, finger_overlap_tolerance_m=0.02)
                relaxed_plan_result = relaxed_skill.execute(
                    scene, body_names, start_q, constraints, candidates=heuristic_candidates
                )
                candidates, plan_result = heuristic_candidates, relaxed_plan_result

        self._save_grasp_pose_image(scene, candidates, plan_result, outdir)
        if not plan_result.success:
            return plan_result, pre_steps, None

        traj = TimeParameterizer(self.config.time_param).parameterize(plan_result.planned_trajectory)
        if traj.positions is None:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = "time parameterization produced no reference"
            return plan_result, pre_steps, None

        cand = plan_result.selected_grasp
        object_body_name = plan_result.selected_grasp.object_name  # real MuJoCo body name
        object_canonical_name = canonical_by_body.get(object_body_name, object_body_name)

        # PickSkill already validated (via check_edge, not blind execution --
        # see its module docstring) a short final descent from the standoff
        # point to the grasp pose; append it here, same as the OMPL-planned
        # standoff leg, at the same control rate.
        #
        # Tracked as two separate MPC phases, not one continuous reference:
        # the MPC's lookahead horizon blends across whatever boundary falls
        # within it, so a single combined reference lets the arm drift
        # diagonally through the airspace directly over the object while
        # still correcting XY. A fresh mpc.reset() plus a real
        # settle-to-convergence checkpoint between phases is what actually
        # enforces "get square above the object, then drop straight down."
        approach_positions = np.asarray(plan_result.planned_trajectory.metadata["approach_positions"])
        to_above_count = int(plan_result.planned_trajectory.metadata.get("to_above_count", 8))
        above_q = np.asarray(plan_result.planned_trajectory.metadata["above_q"])
        to_above_positions = approach_positions[:to_above_count]
        above_to_grasp_positions = approach_positions[to_above_count - 1 :]
        phase1_reference = np.concatenate([traj.positions, to_above_positions[1:]], axis=0)
        phase1_approach_start_idx = len(traj.positions)

        checker = skill.collision_checker
        # NOTE: real obstacle spheres (build_static_obstacle_spheres in mpc.py)
        # were tried here to let the MPC steer FREE_SPACE tracking around
        # clutter, but reverted after an A/B on the sibling `open` skill's
        # identical wiring showed a clear grip-success regression -- the
        # avoidance maneuver changes where the arm ends up at the
        # free_space/terminal_approach handoff, and a marginal final grasp
        # doesn't tolerate that perturbation. Stays off until that handoff is
        # made robust to it.
        cfg = self.config.execution
        steps = pre_steps
        gripper_cmd = self.env.open_gripper_cmd

        def _track_and_settle(reference, approach_start_idx, target_pos, target_quat, phase_label, abort_on_stall=True):
            # See ``track_and_settle``'s own docstring (servo.py) for the
            # shared loop this wraps. ``check_wrong_obstacle``/
            # ``abort_on_stall`` are this skill's own divergence from other
            # callers: a stalled settle can be resting against a neighboring
            # object instead of the intended grasp target, which closing the
            # gripper on would never produce a real grasp; phase 1 (settling
            # above the object, open air) tolerates a benign tracking-lag
            # plateau that phase 2 (the actual descent) must not.
            nonlocal steps
            result = track_and_settle(
                self,
                reference,
                joint_names=self.joint_names,
                ee_site=self.ee_site,
                approach_start_idx=approach_start_idx,
                target_pos=target_pos,
                target_quat_wxyz=target_quat,
                collision_checker=checker,
                mpc_horizon=int(self.config.mpc.get("horizon", 16)),
                mpc_cost_config=MPCCostConfig(
                    **{k: v for k, v in self.config.mpc.items() if k in MPCCostConfig.__dataclass_fields__}
                ),
                replan_config=self.config.replan,
                gripper_cmd=gripper_cmd,
                settle_position_tol_m=cfg.settle_position_tol_m,
                settle_orientation_tol_rad=cfg.settle_orientation_tol_rad,
                stall_window_steps=cfg.stall_window_steps,
                stall_position_tol_m=cfg.stall_position_tol_m,
                max_settle_steps=cfg.max_settle_steps,
                mpc_step_kwargs={"target_object_body": object_body_name},
                steps_so_far=steps,
                max_steps=max_steps,
                step_callback=step_callback,
                abort_on_stall=abort_on_stall,
                check_wrong_obstacle=True,
                wrong_obstacle_graze_tolerance_m=WRONG_OBSTACLE_GRAZE_TOLERANCE_M,
                target_object_body=object_body_name,
            )
            steps += result.steps
            diagnostics = result.as_diagnostics()
            diagnostics["phase"] = phase_label
            return diagnostics

        # Phase 1: get square above the object. Pure FREE_SPACE the whole way
        # -- no target contact expected or permitted yet -- ending with a real
        # settle-to-convergence check against the "above" waypoint itself, not
        # just running out of reference. This is the checkpoint that actually
        # enforces the vertical drop below: phase 2 only ever starts once the
        # arm is verified to be at the right XY, not merely handed a reference
        # that visits it.
        above_position = (
            cand.grasp_position - self.config.grasp.approach_standoff_m * cand.approach_direction_world
        )
        above_settle = _track_and_settle(
            phase1_reference,
            phase1_approach_start_idx,
            above_position,
            cand.grasp_quat_wxyz,
            "to_above",
            abort_on_stall=False,
        )
        if above_settle["wrong_obstacle_contacts"]:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = (
                "settle stalled against a different object, not the intended grasp target"
            )
            plan_result.diagnostics["wrong_obstacle_contacts"] = above_settle["wrong_obstacle_contacts"][:10]
            plan_result.diagnostics["settle_to_above"] = above_settle
            return plan_result, steps, None
        plan_result.diagnostics["settle_to_above"] = above_settle

        # Phase 2: the actual descent, always TERMINAL_APPROACH (contact with
        # the target is now expected) -- and, because phase 1 just verified
        # the arm is directly above the grasp XY, this is genuinely a
        # straight vertical drop, not a diagonal one.
        settle = _track_and_settle(
            above_to_grasp_positions, 0, cand.grasp_position, cand.grasp_quat_wxyz, "to_grasp"
        )
        if settle["wrong_obstacle_contacts"]:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.EXECUTION_FAILED
            plan_result.diagnostics["detail"] = (
                "settle stalled against a different object, not the intended grasp target"
            )
            plan_result.diagnostics["wrong_obstacle_contacts"] = settle["wrong_obstacle_contacts"][:10]
            plan_result.diagnostics["settle"] = settle
            return plan_result, steps, None
        plan_result.diagnostics["settle"] = settle

        # Close the gripper and dwell, continuing to servo toward the grasp
        # pose every step (not a single stale action) since the arm may still
        # be settling from tracking error when the approach segment ends.
        gripper_cmd = self.env.close_gripper_cmd
        close_dwell_steps = max(int(self.env.close_steps) * 3, 40)
        for _ in range(close_dwell_steps):
            hold_pos, hold_ori, _, _ = self._servo_toward_cartesian_pose(
                cand.grasp_position, cand.grasp_quat_wxyz
            )
            self._step_env(hold_pos, hold_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()

        grasped = bool(self.env._is_grasped())
        # Cheap and always worth recording: how far the object moved during the
        # approach (contact/pushing shows up here directly) and the final gap
        # between the gripper and the intended grasp point.
        final_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        ee_pos_final, _ = final_scene.site_pose(self.ee_site)
        obj_pos_after, _ = final_scene.body_pose(object_body_name)
        obj_pos_before, _ = scene.body_pose(object_body_name)
        plan_result.diagnostics["grasp_verification"] = {
            "grasped": grasped,
            "object_displacement_m": float(np.linalg.norm(obj_pos_after - obj_pos_before)),
            "gripper_to_target_error_m": float(np.linalg.norm(ee_pos_final - cand.grasp_position)),
        }
        if not grasped:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.GRASP_VERIFICATION_FAILED
            plan_result.diagnostics["detail"] = "gripper closed but _is_grasped() reported False"
            return plan_result, steps, None

        # Static contact at the exact grasp pose is not the same as a grasp
        # that survives being moved -- a grip can read as grasped immediately
        # after closing yet flicker and fail once the very next skill (which
        # starts with a lift) moves it. Catching that here, where PickSkill's
        # OR-set search could still pick a different candidate, is much
        # cheaper than discovering it downstream in transport, which has
        # nothing to fall back to. Reuses `lift_check_m` (already used at plan
        # time for the IK-reachability check) as the real, physically
        # simulated version of that same question.
        #
        # Interpolates the lift target in a dense ramp rather than commanding
        # a single fixed target far away: a fixed target saturates
        # _servo_toward_cartesian_pose's velocity clamp on the very first
        # step, producing a jerk that can knock a marginal grip loose before
        # it ever gets the chance to prove it holds under a gentler load.
        lift_target = cand.grasp_position + np.array([0.0, 0.0, self.config.grasp.lift_check_m])
        ee_pos_before_lift, _ = SceneQuery.from_robosuite_sim(self.env.env.sim).site_pose(self.ee_site)
        n_lift_steps = 20
        for lift_waypoint in np.linspace(ee_pos_before_lift, lift_target, n_lift_steps + 1)[1:]:
            lift_pos, lift_ori, _, _ = self._servo_toward_cartesian_pose(lift_waypoint, cand.grasp_quat_wxyz)
            self._step_env(lift_pos, lift_ori, gripper_cmd)
            steps += 1
            if step_callback is not None:
                step_callback()
        grasped_after_lift = bool(self.env._is_grasped())
        plan_result.diagnostics["grasp_verification"]["grasped_after_test_lift"] = grasped_after_lift
        if not grasped_after_lift:
            plan_result.success = False
            plan_result.failure_reason = SkillFailureReason.GRASP_VERIFICATION_FAILED
            plan_result.diagnostics["detail"] = "grasp did not survive test lift"
            return plan_result, steps, None

        # Carry lift: leave the arm high enough that the *next* skill (usually
        # transport) has room to plan, rather than wherever the deliberately
        # shallow test lift above happened to stop -- a low, table-adjacent
        # start pose routinely leaves RRTConnect almost no free corridor and
        # causes NO_PATH_FOUND.
        carry_lift_m = float(self.config.grasp.post_grasp_carry_lift_m)
        lift_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        extraction_target_z = _drawer_extraction_target_z(lift_scene, object_body_name)
        if extraction_target_z is not None:
            ee_now_z = float(lift_scene.site_pose(self.ee_site)[0][2])
            carry_lift_m = max(carry_lift_m, extraction_target_z - ee_now_z)
            plan_result.diagnostics["grasp_verification"]["drawer_extraction_target_z"] = extraction_target_z
        if carry_lift_m > 0.0:
            ee_now, _ = lift_scene.site_pose(self.ee_site)
            carry_target = ee_now + np.array([0.0, 0.0, carry_lift_m])
            for carry_waypoint in np.linspace(ee_now, carry_target, n_lift_steps + 1)[1:]:
                carry_pos, carry_ori, _, _ = self._servo_toward_cartesian_pose(
                    carry_waypoint, cand.grasp_quat_wxyz
                )
                self._step_env(carry_pos, carry_ori, gripper_cmd)
                steps += 1
                if step_callback is not None:
                    step_callback()
            plan_result.diagnostics["grasp_verification"]["carry_lift_m"] = carry_lift_m

        attached_mgr = AttachedObjectManager()
        attach_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
        attached_state = attached_mgr.attach(attach_scene, object_body_name, self.hand_site, step=steps)
        plan_result.diagnostics["attached_object"] = {
            "object_name": attached_state.object_name,
            "collision_radius_m": attached_state.collision_radius_m,
        }
        # Report back in SceneResolver-canonical form -- everything downstream
        # of this adapter (env.active_object_name, later subtask dispatch,
        # the execution-summary record) uses that convention, not raw MuJoCo
        # body names.
        plan_result.selected_object = object_canonical_name
        self.env.set_active_object(object_canonical_name)
        return plan_result, steps, attached_state
