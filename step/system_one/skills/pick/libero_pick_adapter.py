"""Executor-facing adapter: wires the grasp-gen -> IK -> OMPL -> time-param ->
MPC pipeline into a live LiberoManipulationEnv for the pick skill.

Scope: pick only -- transport/place still run on the legacy stage machine
even under motion_backend: "ompl_mpc", keeping this pipeline's grasp/IK/planning logic independently verifiable.

Action-space bridge: the MPC outputs joint-position targets, converted to a
Cartesian pose via forward kinematics and tracked with the legacy stage machine's own clipped-delta servo law.
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
    HandleSideGraspGenerator,
    _horizontal_alignment,
    ManualDeepCenterGraspGenerator,
    PrivilegedTopDownGraspGenerator,
    _is_forced_manual_object_class,
    _is_rim_graspable_object_class,
    manual_pose_width_feasible,
)
from ...motion.kinematics import IKSolver
from ...motion.mpc import JointSpaceMPC, MPCContext
from ...motion.ompl_planner import MotionPlanner
from ...motion.scene_query import quat_to_rotmat_wxyz
from ...motion.panda import EE_SITE, GRIPPER_BODIES, JOINT_NAMES, ROBOT_BODIES
from .pick_skill import PickSkill
from ...motion.scene_query import SceneQuery, quat_to_rotmat_wxyz, resolve_body_name
from ...motion.servo import _ServoMixin, track_and_settle
from ...motion.time_parameterize import TimeParameterizer
from ...motion.types import AttachedObjectState, MPCCostConfig, SkillFailureReason, SkillResult

# Matches CollisionConfig.self_margin_m -- a contact closer than this is real;
# anything shallower is sub-mm meshing/numerical noise, not a genuine obstruction.
WRONG_OBSTACLE_GRAZE_TOLERANCE_M = 0.005

# Extra clearance above a cabinet's measured top when extracting from a
# drawer -- covers the attached object's own collision radius before transport's hover search begins.
DRAWER_EXTRACTION_MARGIN_M = 0.06


def _drawer_extraction_target_z(scene: SceneQuery, object_body_name: str) -> float | None:
    """If object_body_name's xy falls within a cabinet frame's footprint and
    its z is below that frame's top, it's inside a drawer -- return the
    height needed to clear it (frame top + DRAWER_EXTRACTION_MARGIN_M).
    Returns None otherwise (the common case: picked off an open surface).
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


def _routing_names_a_generator(config) -> bool:
    """Has the scene context named the generator this pick must use?

    "object_class", the default, decides per object through a keyword table over
    MuJoCo body names. That table is a LIBERO artefact -- it matches names like
    akita_black_bowl_1, and a backend naming every task object obj_main matches
    nothing in it, so the manual pose wins every object by default.

    "graspnet_first" and "demo:<task>[:<mesh>]" name a generator outright, and
    then the table and the manual pre-attempt are both skipped. A pick told which
    method to use and silently running another reports diagnostics for a method
    that never ran -- which is exactly what happened when this covered only
    graspnet_first: the demo generator proposed grasps three times and the manual
    path's rejections were recorded as if they were the demo grasps' own.
    """
    routing = str(getattr(getattr(config, "grasp", None), "routing", "object_class"))
    return routing == "graspnet_first" or routing.startswith("demo:")


class LiberoPickAdapter(_ServoMixin):
    def __init__(self, env, motion_config: MotionConfig | None = None) -> None:
        self.env = env
        self.config = motion_config or MotionConfig()
        self.joint_names = JOINT_NAMES
        self.ee_site = EE_SITE
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
        routing = str(getattr(self.config.grasp, "routing", "object_class"))
        if routing.startswith("demo:"):
            # Grasps mined from the backend's own human demonstrations, applied
            # object-relative (MimicGen's transfer rule). "demo:<task>:<mesh>" --
            # the mesh matters, because the rule re-anchors a grasp to a new pose
            # of the SAME object and no further, and RoboCasa samples a new mesh
            # every episode. The backend ships under benchmarks/robocasa; set
            # STEP_ROBOCASA_DIR for a checkout elsewhere.
            import os as _os
            import sys as _sys

            _rc = _os.environ.get("STEP_ROBOCASA_DIR") or str(
                Path(__file__).resolve().parents[4] / "benchmarks" / "robocasa"
            )
            if _rc not in _sys.path:
                _sys.path.insert(0, _rc)
            from demo_grasp_generator import DemoGraspGenerator

            parts = routing.split(":")
            gen: GraspGenerator = DemoGraspGenerator(
                parts[1], model=parts[2] if len(parts) > 2 else None
            )
        elif self.config.contact_graspnet.enabled:
            # Lazy import (CUDA only touched when enabled); GraspNet-only, not
            # composited, since PickSkill's cost function would let the cheap heuristic candidate win regardless.
            from .contact_graspnet_generator import ContactGraspNetGenerator

            gen: GraspGenerator = ContactGraspNetGenerator(self.config.contact_graspnet)
        else:
            gen = PrivilegedTopDownGraspGenerator(self.config.grasp)
        ik = IKSolver(joint_names=self.joint_names, site_name=self.ee_site)
        planner = MotionPlanner(self.joint_names, lo, hi, checker, self.config.planner)
        return PickSkill(gen, ik, planner, checker, self.config)

    # ------------------------------------------------------------ low-level stepping
    # Stepping helpers are provided by _ServoMixin, shared with LiberoOpenDrawerAdapter.

    def _current_q(self, scene: SceneQuery) -> np.ndarray:
        # Re-sync from the live sim every read -- the scene passed at
        # construction time is a one-shot snapshot that goes stale as soon as the arm moves.
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
        Reuses PrivilegedTopDownGraspGenerator's object-center/height logic to
        get a reasonable XY/Z, then adds extra height for a clear top-down view. No yaw, since this is just a vantage point.
        """
        heuristic = PrivilegedTopDownGraspGenerator(self.config.grasp)
        candidates = heuristic.generate(scene, [object_body_name])
        if not candidates:
            return None
        hover_pos = candidates[0].grasp_pose_world[:3].copy()
        hover_pos[2] += extra_height_m
        # Gripper approach axis pointing straight down -- same top-down
        # convention as _TOP_DOWN_QUAT_WXYZ, with no yaw since this is just a vantage point.
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
        """Physically move to hover_pos/hover_quat under ordinary collision-aware
        planning/tracking, never a teleport. Returns (new_scene, new_start_q, steps_used), or None if the hover pose is unreachable or planning fails."""
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
        """Returns (SkillResult, steps_executed, AttachedObjectState | None).

        step_callback, if given, is called after every real env step -- wired
        to _capture_video_frame so --save-video works for this backend too.

        outdir, if given, saves a grasp_poses_pick_N.png of every candidate
        considered (green = selection). Candidates are generated once and reused, since a learned generator re-runs real inference every call.
        """
        constraints = constraints or {}

        # Reset to home before planning the grasp: a pick usually follows a
        # place, leaving the arm in a configuration that a fresh reach from tends to fail.
        pre_home_steps = self.servo_to_home(self.config.drawer.return_home_steps, step_callback)

        scene = self._build_scene()
        skill = self._build_pick_skill(scene)
        start_q = scene.get_qpos(self.joint_names)

        # object_set comes from SceneResolver-canonical names (e.g.
        # "alphabet_soup_1"); the motion pipeline needs real MuJoCo body names ("alphabet_soup_1_main").
        body_names = [resolve_body_name(scene, name) for name in object_set]
        canonical_by_body = dict(zip(body_names, object_set))

        # Every grasp attempt this pick makes; filled in by the generator chain below.
        attempts: list[dict] = []

        handle_dir = constraints.get("handle_approach_dir")
        if handle_dir is not None:
            # Re-grasp for a side-opening container: only a handle grasp whose
            # hand points along the insertion direction can go in under the roof.
            handle_dir = np.asarray(handle_dir, dtype=float)
            candidates = [
                c
                for c in HandleSideGraspGenerator(skill.grasp_generator.gripper_max_opening_m).generate(scene, body_names)
                if _horizontal_alignment(c.approach_direction_world, handle_dir) > 0.8
            ]
            if candidates:
                plan_result = skill.execute(scene, body_names, start_q, constraints, candidates=candidates)
            else:
                plan_result = SkillResult(
                    success=False,
                    failure_reason=SkillFailureReason.NO_FEASIBLE_CANDIDATE,
                    diagnostics={"detail": "no handle facing away from the insertion direction"},
                )
            pre_steps = pre_home_steps
        else:
            # A rim-graspable object always needs GraspNet; everything else uses
            # the fixed manual pose unless manual_pose_width_feasible says its axis can't span the object (checked up front, not after rejection).
            # A named generator skips the object-class table entirely.
            manual_only = (not _routing_names_a_generator(self.config)) and bool(body_names) and all(
                not _is_rim_graspable_object_class(b)
                and (
                    manual_pose_width_feasible(scene, b, skill.grasp_generator.gripper_max_opening_m)
                    or _is_forced_manual_object_class(b)
                )
                for b in body_names
            )

            # Every grasp attempt this pick makes, kept separately.
            #
            # The fallback chain below overwrites plan_result with whichever
            # attempt ran last, on the reasoning that once the primary generator
            # has definitively failed the fallback's rejections are the
            # interesting ones. That holds while the primary rarely fails. When
            # it is the method under test it inverts: the reported rejections
            # belong to the heuristic, the configured generator's own failure is
            # erased, and the run reads as evidence about a method that never got
            # scored. Recording each attempt keeps both.

            def _record(label, cands, result):
                diag = getattr(result, "diagnostics", None) or {}
                attempts.append({
                    "generator": label,
                    "n_candidates": len(cands),
                    "success": bool(getattr(result, "success", False)),
                    "failure_reason": str(getattr(result, "failure_reason", None)),
                    "rejection_reasons": list(diag.get("rejection_reasons") or []),
                    "detail": diag.get("detail"),
                })
                return result

            manual_plan_result = None
            manual_candidates: list = []
            # A named generator also means: do not spend a plan attempt on the
            # manual pose before that generator has been tried.
            if self.config.contact_graspnet.enabled and not _routing_names_a_generator(self.config):
                manual_candidates = ManualDeepCenterGraspGenerator(
                    skill.grasp_generator.gripper_max_opening_m
                ).generate(scene, body_names)
                if manual_candidates:
                    manual_plan_result = _record("manual", manual_candidates, skill.execute(
                        scene, body_names, start_q, constraints, candidates=manual_candidates
                    ))

            if manual_plan_result is not None and (manual_plan_result.success or manual_only):
                candidates, plan_result = manual_candidates, manual_plan_result
            else:
                candidates = skill.grasp_generator.generate(scene, body_names)
                plan_result = _record(
                    type(skill.grasp_generator).__name__, candidates,
                    skill.execute(scene, body_names, start_q, constraints, candidates=candidates),
                )

            if manual_only and not plan_result.success and plan_result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE:
                # Manual-only gets the same widened-tolerance retry the heuristic
                # path uses as a last resort -- cheap, since it's one candidate, not a full search.
                relaxed_skill = self._build_pick_skill(scene, finger_overlap_tolerance_m=0.02)
                relaxed_plan_result = _record("manual (relaxed)", manual_candidates, relaxed_skill.execute(
                    scene, body_names, start_q, constraints, candidates=manual_candidates
                ))
                candidates, plan_result = manual_candidates, relaxed_plan_result

            pre_steps = pre_home_steps  # the homing move above is real env time too
            needs_closer_look = (
                not plan_result.success
                and plan_result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE
                and len(body_names) == 1
                and not manual_only
            )
            if needs_closer_look:
                # The single-shot capture can be too sparse for a small/thin
                # object; move the wrist camera close, then recapture and regenerate from there.
                hover = self._compute_hover_pose(scene, body_names[0])
                if hover is not None:
                    moved = self._move_to_hover(scene, skill, start_q, hover[0], hover[1], step_callback, max_steps)
                    if moved is not None:
                        new_scene, hover_steps = moved
                        pre_steps = hover_steps
                        new_start_q = new_scene.get_qpos(self.joint_names)
                        new_candidates = skill.grasp_generator.generate(new_scene, body_names)
                        new_plan_result = _record(
                            type(skill.grasp_generator).__name__ + " (closer look)", new_candidates,
                            skill.execute(
                                new_scene, body_names, new_start_q, constraints, candidates=new_candidates
                            ),
                        )
                        # Only adopt the closer-look attempt if it actually did
                        # better, so diagnostics reflect whichever attempt is reported.
                        if new_plan_result.success or len(new_candidates) > len(candidates):
                            scene, start_q, candidates, plan_result = new_scene, new_start_q, new_candidates, new_plan_result

            if not plan_result.success and plan_result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE and not manual_only:
                # Last resort: try the heuristic generator as a separate sequential
                # attempt (not composited with GraspNet's OR-set) since its cheap candidate would otherwise always outscore a real one.
                heuristic_candidates = PrivilegedTopDownGraspGenerator(self.config.grasp).generate(scene, body_names)
                if heuristic_candidates:
                    heuristic_plan_result = _record(
                        "PrivilegedTopDownGraspGenerator (fallback)", heuristic_candidates,
                        skill.execute(
                            scene, body_names, start_q, constraints, candidates=heuristic_candidates
                        ),
                    )
                    # Adopt this attempt's diagnostics either way -- if the
                    # heuristic also fails, its rejection reasons are what's worth seeing.
                    candidates, plan_result = heuristic_candidates, heuristic_plan_result

                if not plan_result.success and plan_result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE:
                    # Absolute last resort: retry the same candidates with a
                    # widened contact tolerance for the target object -- TERMINAL_APPROACH already permits only gripper<->target contact.
                    relaxed_skill = self._build_pick_skill(scene, finger_overlap_tolerance_m=0.02)
                    relaxed_plan_result = _record(
                        "PrivilegedTopDownGraspGenerator (fallback, relaxed)", heuristic_candidates,
                        relaxed_skill.execute(
                            scene, body_names, start_q, constraints, candidates=heuristic_candidates
                        ),
                    )
                    candidates, plan_result = heuristic_candidates, relaxed_plan_result

        # Attached to whichever attempt ended up reported, so a failed pick always
        # says which generators were tried and how each one failed.
        plan_result.diagnostics["attempts"] = attempts
        if len(attempts) > 1:
            print("[pick] attempts: " + " | ".join(
                f"{a['generator']}={len(a['rejection_reasons']) or a['n_candidates']} "
                f"{'ok' if a['success'] else str(a['failure_reason']).split('.')[-1]}"
                for a in attempts))

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

        # Tracked as two separate MPC phases, not one continuous reference --
        # a single reference lets the MPC's lookahead blend across the boundary and drift diagonally.
        approach_positions = np.asarray(plan_result.planned_trajectory.metadata["approach_positions"])
        to_above_count = int(plan_result.planned_trajectory.metadata.get("to_above_count", 8))
        above_q = np.asarray(plan_result.planned_trajectory.metadata["above_q"])
        to_above_positions = approach_positions[:to_above_count]
        above_to_grasp_positions = approach_positions[to_above_count - 1 :]
        phase1_reference = np.concatenate([traj.positions, to_above_positions[1:]], axis=0)
        phase1_approach_start_idx = len(traj.positions)

        checker = skill.collision_checker
        # Real obstacle-avoidance steering is deliberately not used here: the
        # resulting free_space/terminal_approach handoff point shifts enough that a marginal final grasp doesn't tolerate it.
        cfg = self.config.execution
        steps = pre_steps
        gripper_cmd = self.env.open_gripper_cmd

        def _track_and_settle(reference, approach_start_idx, target_pos, target_quat, phase_label, abort_on_stall=True):
            # check_wrong_obstacle/abort_on_stall diverge per phase: phase 1
            # (settling in open air) tolerates a benign tracking-lag plateau that phase 2 (the actual descent) must not.
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

        # Phase 1: get square above the object, pure FREE_SPACE, ending with a
        # real settle-to-convergence check -- this is what enforces the vertical drop in phase 2.
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

        # Phase 2: the actual descent, always TERMINAL_APPROACH -- genuinely a
        # straight vertical drop, since phase 1 already verified the arm is directly above the grasp XY.
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
        # pose every step, since the arm may still be settling when the approach segment ends.
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
        # Cheap and worth recording: how far the object moved during approach
        # (contact/pushing shows here) and the final gripper-to-target gap.
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

        # Interpolated as a dense ramp, not a single fixed target: a fixed
        # target saturates the velocity clamp on the first step, jerking a marginal grip loose.
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

        # Carry lift: leave the arm high enough for the next skill to plan --
        # a low, table-adjacent start pose routinely leaves RRTConnect no free corridor, causing NO_PATH_FOUND.
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
        # ee_site, not the wrist ft_frame -- every consumer of this transform
        # (required_hand_pose_for_object_pose) solves IK/servos against ee_site, so using a different site here introduced a fixed ~9.7cm release-height offset.
        attached_state = attached_mgr.attach(attach_scene, object_body_name, self.ee_site, step=steps)
        # Which way was up before the grasp: the reference for G(Upright(...)).
        # Taken from the pre-grasp scene, since the lifts above may already
        # have tipped the object a little.
        _, rest_quat = scene.body_pose(object_body_name)
        attached_state.up_axis_object_frame = (
            quat_to_rotmat_wxyz(rest_quat).T @ np.array([0.0, 0.0, 1.0])
        )
        plan_result.diagnostics["attached_object"] = {
            "object_name": attached_state.object_name,
            "collision_radius_m": attached_state.collision_radius_m,
        }
        # Report back in SceneResolver-canonical form -- everything downstream
        # of this adapter uses that convention, not raw MuJoCo body names.
        plan_result.selected_object = object_canonical_name
        self.env.set_active_object(object_canonical_name)
        return plan_result, steps, attached_state
