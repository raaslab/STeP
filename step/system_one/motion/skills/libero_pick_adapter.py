"""Executor-facing adapter: wires the grasp-gen -> IK -> OMPL -> time-param ->
MPC pipeline into a live ``LiberoManipulationEnv`` for the ``pick`` skill.

Scope of this integration slice: **pick only**. ``transport``/``place`` still
run on the legacy stage machine even when ``motion_backend: "ompl_mpc"`` is
set -- see ``System1Executor``'s dispatch in ``executor.py``. This keeps the
integration's blast radius small and independently verifiable: the new
pipeline's grasp/IK/planning/collision logic is exercised against the real
env and gripper on every run, without also needing new transport/place
motion-generation code to be correct for a pick-only smoke test to pass.

Action-space bridge: the MPC outputs joint-position targets, but LIBERO's
robosuite env (like the legacy stage machine) takes Cartesian-delta OSC_POSE
actions -- there's no way to swap in a joint-position controller from here
without changing how ``LiberoManipulationEnv`` constructs its env (out of
scope for this pass). So each joint target is converted to a Cartesian pose
via forward kinematics (on a scratch ``SceneQuery``, never the live sim) and
tracked with the *same* clipped-delta servo law the legacy stage machine
already uses (``BaseManipulationEnv._compose_full_action``) -- this reuses a
proven, already-tuned low-level control primitive instead of inventing a
second one, and means the new pipeline is only responsible for *where* to
go, not the raw actuation.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np

from ..attached_object import AttachedObjectManager
from ..collision import CollisionChecker, CollisionPhase
from ..config import MotionConfig
from .grasp_generator import (
    GraspGenerator,
    ManualDeepCenterGraspGenerator,
    PrivilegedTopDownGraspGenerator,
    _is_forced_manual_object_class,
    _is_rim_graspable_object_class,
    manual_pose_width_feasible,
)
from ..kinematics import IKSolver
from ..mpc import JointSpaceMPC, MPCContext
from ..ompl_planner import MotionPlanner
from ..panda import EE_SITE, GRIPPER_BODIES, HAND_SITE, JOINT_NAMES, ROBOT_BODIES
from .pick_skill import PickSkill
from ..scene_query import SceneQuery, quat_to_rotmat_wxyz, resolve_body_name
from ..servo import _ServoMixin, is_settle_stalled, real_nonpermitted_contacts
from ..time_parameterize import TimeParameterizer
from ..types import AttachedObjectState, MPCCostConfig, SkillFailureReason, SkillResult

# Matches CollisionConfig.self_margin_m (0.005) -- the smallest margin
# already tolerated elsewhere in this pipeline's own collision checks. A
# contact against an unrelated neighbor closer than this is real; anything
# shallower is sub-mm meshing/numerical noise, not a genuine obstruction.
WRONG_OBSTACLE_GRAZE_TOLERANCE_M = 0.005


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
            # Lazily imported: this module (and the contact_graspnet_pytorch/
            # CUDA it needs) is only ever touched when explicitly enabled --
            # see contact_graspnet_generator.py's module docstring.
            #
            # GraspNet-only, not composited with the heuristic generator: a
            # real run showed PickSkill's OR-set cost function picking the
            # heuristic's cheap-but-wrong center-top candidate over every
            # real GraspNet candidate for a shallow rim object (ramekin) --
            # the heuristic's zero-cost `generator_score_penalty` (it doesn't
            # score itself) plus its lower path/reachability cost beat every
            # GraspNet candidate regardless of physical grasp quality. Once
            # GraspNet is enabled, it alone proposes candidates so this
            # tradeoff can't silently reintroduce the heuristic's known-bad
            # grasps for rim/hollow objects.
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
        # construction time is a one-shot snapshot, but the arm moves every
        # step, so joint readings must come from the live obs, not the
        # (now-stale) scratch clone used purely for FK.
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
        palm/finger clearance, e.g. it already knows cream_cheese_1 is ~2cm
        tall and won't put the target height inside the table) purely to get
        a reasonable XY/Z above the object, then adds extra height on top so
        the wrist camera gets a clear top-down view of it without itself
        being close enough to collide. Orientation is a plain top-down look
        (no yaw), since this is just a vantage point, not the eventual grasp.
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

        ``outdir``, if given, gets a ``grasp_poses_pick_N.png`` image of
        every candidate this call considered (green = PickSkill's actual
        selection, if any) -- a permanent, automatic record of what the
        generator proposed and what got picked, saved on every real run (not
        just when manually invoking ``scripts/visualize_grasp_poses.py``).
        This was previously the only way to see this, and only by re-running
        candidate generation from scratch against a scene snapshot -- doubly
        wasteful for a learned generator like ContactGraspNetGenerator,
        which re-runs real CUDA inference on every ``generate()`` call.
        Generating candidates *here*, once, and handing them straight to
        ``skill.execute()`` (its own ``candidates=`` param) avoids that.
        """
        constraints = constraints or {}

        # Reset to home before planning the grasp, for the same reason the
        # open-drawer adapter does (see _ServoMixin.servo_to_home): a pick
        # usually follows a place, which leaves the arm hovering over the
        # spot it just released something, in a configuration chosen for
        # that task. Planning a fresh reach from there is where this pipeline
        # keeps failing -- a NO_PATH_FOUND transporting away from a
        # just-finished pick, and a re-grasp that closed on nothing.
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
        # object (bowl/mug/cup, see _is_rim_graspable_object_class) has no
        # solid material at its center for the manual pose to close on, so it
        # always needs GraspNet's real antipodal-contact search -- everything
        # else (a can, a box like cream_cheese_1/butter_1) is assigned to the
        # manual pose exclusively, no GraspNet fallback at all -- *unless*
        # the manual pose's own fixed axis (non-negotiable, see
        # ManualDeepCenterGraspGenerator's docstring) can't even physically
        # span the object's real width at that axis (manual_pose_width_
        # feasible), the same "elongated object grasped along its long, not
        # short, axis" failure already documented for cream_cheese_1/
        # butter_1, just checked here before ever committing to manual-only
        # instead of discovered only after every manual candidate is
        # rejected. Confirmed on a real book (STUDY_SCENE1's black_book_1,
        # ~9.3cm at the fixed axis against an 8cm gripper): without this
        # check it fell to manual_only, exhausted 5 manual candidates all
        # rejected as gripper_width_incompatible, and never tried GraspNet
        # at all -- the pick failed having never actually attempted a real
        # grasp. Once fixed, GraspNet DID run for that book and STILL only
        # found wider candidates (11-17cm) than the manual pose's own
        # 9.3cm -- confirmed against a real screenshot that this book rests
        # standing upright, not flat, so its one genuinely narrow axis (page
        # thickness) is vertical, not horizontal, and no top-down close can
        # ever span it. _is_forced_manual_object_class (FORCE_MANUAL_
        # KEYWORDS) routes that object class back to manual-only despite
        # failing the width check -- fails fast and predictably rather than
        # burning a real GraspNet inference pass on a pose nothing currently
        # solves; still an open problem, not a fix, for the underlying
        # "grasp an upright book" case (a side approach, not top-down, is
        # what it actually needs).
        #
        # This is a deliberate speed/robustness tradeoff, not an oversight:
        # GraspNet's own "closer look" recapture + the heuristic's
        # exhaustive 75-candidate standoff search (see PickSkill's own
        # standoff_candidates call) are what make a failed pick on a small,
        # oddly-placed object slow -- confirmed against a real
        # cream_cheese_1 rollout, ~200s dominated by exactly those two
        # fallback stages. Skipping them entirely for the object classes the
        # manual pose already covers well trades a small amount of
        # robustness (a manual-only object with a genuinely infeasible fixed
        # pose just fails fast instead of exhausting every fallback) for a
        # large, predictable speed win on the common case.
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
            # widened-tolerance retry the heuristic path below uses as its
            # own absolute last resort, applied directly to the manual
            # candidate instead of falling through to GraspNet/heuristic.
            # Real, not hypothetical -- confirmed against the actual
            # butter_1/cream_cheese_1 meshes: a specific real protrusion
            # penetrates a finger by ~1.1-1.3mm at the correct (short-axis)
            # grasp, nearly double the default 6mm compliance margin. This
            # single retry is cheap (one candidate, not GraspNet's inference
            # or the heuristic's multi-yaw search) so it doesn't undermine
            # the whole point of skipping those for this object class.
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
            # The single-shot capture (from wherever the arm started, e.g.
            # its home pose) can be too sparse for a small/thin object --
            # confirmed against the real cream_cheese_1 asset (~2cm tall):
            # Contact-GraspNet found only 3 raw grasps total across every
            # viewpoint retry, vs. 32+ for a soup can from the same capture
            # setup. Move the wrist camera close to the object first, then
            # recapture and regenerate from there, rather than giving up on
            # the first (distant) look.
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
                    # better -- keep the original (scene, candidates,
                    # plan_result, start_q) if the second attempt is no
                    # improvement, so diagnostics/video still reflect
                    # whichever attempt is being reported.
                    if new_plan_result.success or len(new_candidates) > len(candidates):
                        scene, start_q, candidates, plan_result = new_scene, new_start_q, new_candidates, new_plan_result

        if not plan_result.success and plan_result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE and not manual_only:
            # Last resort: GraspNet found nothing usable even after the
            # closer look above. Rare for well-shaped objects, but real and
            # hard to fully eliminate for very small/thin ones (confirmed on
            # cream_cheese_1 -- see contact_graspnet_generator.py's module
            # docstring for the height-clamp history, and this file's own
            # closer-look comment for the sparse-candidate history).
            #
            # Try the privileged heuristic generator's own candidates as a
            # genuinely separate, sequential attempt -- NOT composited into
            # the same OR-set GraspNet's candidates went through.
            # Compositing was tried and reverted (see _build_pick_skill's own
            # comment): the heuristic's zero-cost generator_score_penalty let
            # its cheap center-top candidate beat every real GraspNet
            # candidate for a rim/hollow object (ramekin) regardless of
            # physical grasp quality. A sequential fallback that only ever
            # runs after GraspNet has already failed outright can't
            # reintroduce that regression -- GraspNet still gets first and
            # only say whenever it produces anything IK/collision-feasible.
            heuristic_candidates = PrivilegedTopDownGraspGenerator(self.config.grasp).generate(scene, body_names)
            if heuristic_candidates:
                heuristic_plan_result = skill.execute(
                    scene, body_names, start_q, constraints, candidates=heuristic_candidates
                )
                # Adopt this attempt's diagnostics either way, not just on
                # success -- GraspNet has already definitively failed by this
                # point, so if the heuristic also fails, ITS rejection
                # reasons are what's actually worth seeing, not GraspNet's
                # (silently keeping GraspNet's own diagnostics here made a
                # real heuristic-side failure invisible in a past run).
                candidates, plan_result = heuristic_candidates, heuristic_plan_result

            if not plan_result.success and plan_result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE:
                # Absolute last resort: even the heuristic's own center-top
                # candidates (sampled across every yaw) were all rejected.
                # Confirmed on cream_cheese_1's real collision data that this
                # can happen for a genuinely real reason -- a specific
                # sub-geom of the object's own mesh (a small, real protrusion
                # not captured by its overall AABB) penetrates a finger by
                # ~1.1cm, nearly double the default 6mm
                # finger_overlap_tolerance_m compliance margin -- not a
                # marginal/borderline case the normal tolerance is meant to
                # paper over. Rather than refuse the pick entirely, retry the
                # same heuristic candidates once more through a skill built
                # with a deliberately widened tolerance for contact with the
                # *target object specifically* (TERMINAL_APPROACH already
                # only ever permits gripper<->target contact, never contact
                # with the table/a neighbor/the robot itself -- widening this
                # one number doesn't loosen any of those other real safety
                # checks). Better to attempt a plain center grasp and let
                # _is_grasped() be the real, final arbiter of whether it
                # actually worked than to give up outright.
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
        # Two separate references, not one, and tracked as two separate MPC
        # phases below (see _track_reference's own call sites): PickSkill's
        # own plan is already correctly vertical for the final descent (its
        # standoff_q/above_q/grasp_q metadata share identical XY, verified
        # directly), but a single continuous reference handed to one
        # continuously-running MPC does not execute that faithfully. The
        # MPC's lookahead horizon (mpc.step's own `horizon` steps) blends
        # across whatever boundary happens to fall within it -- confirmed via
        # a real rollout's Cartesian trace: with one combined reference, the
        # arm was still correcting XY toward the grasp target while Z had
        # already dropped down to the "above" checkpoint's height, i.e. it
        # was moving diagonally through the exact airspace directly over the
        # object this waypoint exists to protect. A fresh mpc.reset() plus a
        # real settle-to-convergence checkpoint between phases is what
        # actually enforces "get square above the object, then drop
        # straight down" -- not just handing the controller a reference that
        # happens to visit those points in order.
        approach_positions = np.asarray(plan_result.planned_trajectory.metadata["approach_positions"])
        to_above_count = int(plan_result.planned_trajectory.metadata.get("to_above_count", 8))
        above_q = np.asarray(plan_result.planned_trajectory.metadata["above_q"])
        to_above_positions = approach_positions[:to_above_count]
        above_to_grasp_positions = approach_positions[to_above_count - 1 :]
        phase1_reference = np.concatenate([traj.positions, to_above_positions[1:]], axis=0)
        phase1_approach_start_idx = len(traj.positions)

        checker = skill.collision_checker
        # NOTE: real obstacle spheres (build_static_obstacle_spheres in
        # mpc.py) were tried here, feeding the MPC's existing (previously
        # always-inert) clearance cost so it could actively steer FREE_SPACE
        # tracking around real clutter. Reverted after an A/B on the
        # sibling `open` skill's identical wiring measured a clear
        # regression (0/10 grip success with obstacles active vs. 3/8 at
        # baseline without) -- even scoped to FREE_SPACE only, the avoidance
        # maneuver changes where the arm ends up at the free_space/
        # terminal_approach handoff, and a marginal final grasp doesn't
        # tolerate that perturbation. Not re-validated here for pick
        # specifically, but the mechanism and risk are identical, so this
        # stays off until the handoff itself is made robust to it (e.g. an
        # explicit settle-to-standoff checkpoint before the final descent).
        cfg = self.config.execution
        steps = pre_steps
        gripper_cmd = self.env.open_gripper_cmd

        def _track_and_settle(reference, approach_start_idx, target_pos, target_quat, phase_label, abort_on_stall=True):
            # Track `reference` via a *fresh* MPC (its own mpc.reset(), its
            # own lookahead window that starts at index 0 of this segment
            # only), then -- still under the same collision-aware MPC, not a
            # separate blind servo -- keep holding at the final waypoint
            # (mpc.py clamps t0_step past the reference's end, see
            # JointSpaceMPC._cost_and_grad's `ridx`) until the arm actually
            # converges on `target_pos`/`target_quat`.
            #
            # An earlier version stopped calling the MPC the moment the
            # nominal reference was exhausted and handed the rest of the job
            # to a separate `_settle_toward_grasp_pose` helper: a plain
            # Cartesian P-servo with *no collision awareness at all*. That was
            # fine as long as the reference's time-parameterized duration
            # closely matched how fast the real Cartesian-delta controller
            # could actually track it -- but measured on a real run, the arm
            # was still 4.7cm / 16 degrees from the grasp pose the moment the
            # reference ran out (the time-parameterizer's velocity-limit-based
            # duration estimate doesn't account for the real controller's
            # tracking dynamics). Handing a gap that large to a
            # collision-blind servo is exactly what let the gripper collide
            # with the object while still mid-rotation instead of cleanly
            # finishing its approach. Continuing the same MPC loop keeps every
            # remaining step collision-checked.
            nonlocal steps
            mpc = JointSpaceMPC(
                self.joint_names,
                self.ee_site,
                horizon=int(self.config.mpc.get("horizon", 16)),
                cost_config=MPCCostConfig(
                    **{k: v for k, v in self.config.mpc.items() if k in MPCCostConfig.__dataclass_fields__}
                ),
            )
            ctx = MPCContext(reference=reference)
            mpc.reset()
            max_hold_steps = max_steps if max_steps is not None else len(reference) + cfg.max_settle_steps
            tracking_budget = min(max_hold_steps, len(reference) + cfg.max_settle_steps)

            pos_err = ori_err = float("inf")
            converged = False
            stalled = False
            wrong_obstacle_contacts = None
            recent_pos_errs: list[float] = []
            t0 = 0
            phase = CollisionPhase.FREE_SPACE
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
                    target_object_body=object_body_name,
                    replan_config=self.config.replan,
                )
                cart_pos, cart_ori = self._servo_toward_joint_target(live_scene, mpc_result.joint_command)
                self._step_env(cart_pos, cart_ori, gripper_cmd)
                steps += 1
                t0 += 1
                if step_callback is not None:
                    step_callback()
                if steps >= max_steps:
                    break

                if t0 >= len(reference):
                    _, _, pos_err, ori_err = self._servo_toward_cartesian_pose(target_pos, target_quat)
                    if pos_err <= cfg.settle_position_tol_m and ori_err <= cfg.settle_orientation_tol_rad:
                        converged = True
                        break
                    recent_pos_errs.append(pos_err)
                    if is_settle_stalled(recent_pos_errs, cfg.stall_window_steps, cfg.stall_position_tol_m):
                        # Not getting any closer -- see LiberoOpenDrawerAdapter's
                        # identical check for why continuing to hunt at full
                        # reactive gain against a target it can't reach any
                        # closer to (almost always a real contact boundary) is
                        # pointless and visibly looks like the gripper
                        # repeatedly bumping and retrying. Still checked (and
                        # still aborts) even when abort_on_stall=False: a real
                        # obstacle is dangerous regardless of which phase hit
                        # it. What abort_on_stall=False actually skips is only
                        # the "no wrong-obstacle contact, so just give up
                        # anyway" branch further below -- see there for why.
                        stalled = True
                        # But *which* boundary it stalled against matters: this
                        # check was written assuming that boundary is always the
                        # intended grasp target, and unconditionally proceeds to
                        # close the gripper either way. Confirmed via a real
                        # rollout it is not always true -- a stalled settle can
                        # be resting against a *neighboring* object instead (the
                        # milk carton next to this exact can), sustained contact
                        # (100+ steps, several mm of real penetration) the whole
                        # time, not the can at all. Closing the gripper there
                        # doesn't produce a grasp; it just holds that position
                        # until _is_grasped() reports the obvious False --
                        # cheaper and clearer to catch here, by name, than to
                        # let the close-dwell run first and report a generic
                        # "gripper closed but _is_grasped() reported False".
                        stall_scene = SceneQuery.from_robosuite_sim(self.env.env.sim)
                        target_geom_ids = set(stall_scene.body_geom_ids(object_body_name))
                        bad_contacts = real_nonpermitted_contacts(stall_scene, checker, phase, object_body_name)
                        # Ignore grazes smaller than WRONG_OBSTACLE_GRAZE_TOLERANCE_M:
                        # this check is about a *sustained, real* contact against
                        # a neighbor (confirmed against the milk-carton case --
                        # several mm of penetration, 100+ steps), not sub-mm
                        # meshing noise. Confirmed against a real butter_1
                        # rollout: a -0.00015m (0.15mm) palm graze against
                        # tomato_sauce_1 -- 30x smaller than this pipeline's own
                        # self_margin_m (0.005) -- was tripping this exact check
                        # and failing an otherwise-clean pick outright.
                        wrong_obstacle_contacts = [
                            c for c in bad_contacts
                            if stall_scene.geom_id(c[0]) not in target_geom_ids
                            and stall_scene.geom_id(c[1]) not in target_geom_ids
                            and c[2] < -WRONG_OBSTACLE_GRAZE_TOLERANCE_M
                        ]
                        if wrong_obstacle_contacts or abort_on_stall:
                            break
                        # abort_on_stall=False and nothing's actually in the
                        # way: this is phase 1 (settling to the "above"
                        # checkpoint, open air, nothing to contact) plateauing
                        # on ordinary tracking lag, not a real obstacle --
                        # confirmed on a real butter_1 rollout, stalling
                        # ~3.2cm short with zero wrong-obstacle contacts and
                        # never closing that gap even given the full
                        # max_settle_steps budget. Bailing out there just
                        # handed phase 2 a wrong starting XY with no way to
                        # correct it (its own reference is a fixed joint-space
                        # interpolation, not a re-plan) -- confirmed by the
                        # same rollout's descent ending measurably off-center.
                        # Letting the loop keep running instead costs nothing
                        # dangerous (every step remains collision-checked by
                        # the same MPC) and phase 2's own settle-to-convergence
                        # step, later, is what actually recovers full
                        # precision -- confirmed: 3.16cm short here still
                        # finished at 0.17mm from the true grasp target.

            return {
                "phase": phase_label,
                "steps": max(0, t0 - len(reference)),
                "stalled": stalled,
                "converged": converged,
                "position_error_m": pos_err,
                "orientation_error_rad": ori_err,
                "wrong_obstacle_contacts": wrong_obstacle_contacts,
            }

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
        # Cheap and always worth recording: how far the object moved during
        # the approach (contact/pushing shows up here directly) and the
        # final gap between the gripper and the intended grasp point --
        # this is what actually exposed the collision-model bugs this file's
        # history refers to above, not print statements gated behind a debug
        # flag that nobody remembers to set.
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
        # that survives being moved -- confirmed by a real rollout where
        # _is_grasped() read True immediately after closing, only for the
        # very next subtask (transport, which starts with a few-cm lift) to
        # report the object gone 2 steps in, with _is_grasped() flickering
        # True/False/True as the fingers "closed" without ever settling on a
        # stable bilateral grip. That failure is far more useful caught here
        # -- where PickSkill's own OR-set search could in principle pick a
        # different candidate next time -- than discovered downstream in
        # transport, which has no candidate to fall back to and simply
        # drops the object. Reuses `lift_check_m` (already used at plan time
        # to confirm a lift target is IK-reachable, see PickSkill's own
        # `post_grasp_lift_infeasible` check) as the real, physically
        # simulated version of that same question.
        # A dense ramp, not a single fixed target repeated every step: with
        # a fixed target 5cm away, _servo_toward_cartesian_pose's own
        # pos_err/pos_scale_m (0.03) saturates the very first commanded
        # step at maximum Cartesian velocity -- a sudden jerk, not a gentle
        # lift. Measured directly against the deterministic heuristic-
        # generator candidate for this exact can: a marginal grip that held
        # a stable, unmoving 37-step static contact (real contacts, gap
        # plateaued under load) lost all contact within a single sim step
        # of that first saturated command, and by the time it grasped-
        # checked at the end had fallen back to its original resting
        # height. Interpolating the target itself, not just clipping the
        # commanded velocity (already 0.03 m/step max via pos_scale_m),
        # keeps each step's position error small throughout, not just
        # capped-but-still-large on step one.
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

        # Carry lift: leave the arm high enough that the *next* skill has room
        # to plan, rather than wherever the (deliberately shallow) test lift
        # above happened to stop. Not cosmetic -- measured on the real
        # libero_object task: transporting from the post-test-lift pose
        # (~6cm off the table, object attached) had RRTConnect thrash and
        # return NO_PATH_FOUND for all 8 candidate goals, 85.8s to fail;
        # lifting 10cm first, the identical transport found a direct path and
        # succeeded in 1.4s. The attached object's own collision spheres are
        # tight (2 x 2.3cm radius, measured) -- it's the low, table-adjacent
        # start pose that leaves the planner almost no free corridor to
        # expand into, not an inflated object model.
        carry_lift_m = float(self.config.grasp.post_grasp_carry_lift_m)
        if carry_lift_m > 0.0:
            ee_now, _ = SceneQuery.from_robosuite_sim(self.env.env.sim).site_pose(self.ee_site)
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
