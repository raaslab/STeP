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

import logging
from pathlib import Path

import numpy as np

from .attached_object import AttachedObjectManager
from .collision import CollisionChecker, CollisionPhase
from .config import MotionConfig
from .grasp_generator import GraspGenerator, PrivilegedTopDownGraspGenerator
from .kinematics import IKSolver
from .mpc import JointSpaceMPC, MPCContext
from .ompl_planner import MotionPlanner
from .panda import EE_SITE, GRIPPER_BODIES, HAND_SITE, JOINT_NAMES, ROBOT_BODIES
from .pick_skill import PickSkill
from .scene_query import SceneQuery, quat_to_rotmat_wxyz, resolve_body_name
from .servo import _ServoMixin, is_settle_stalled
from .time_parameterize import TimeParameterizer
from .types import AttachedObjectState, MPCCostConfig, SkillFailureReason, SkillResult


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

    def _build_pick_skill(self, scene: SceneQuery) -> PickSkill:
        checker = CollisionChecker(
            scene, self.joint_names, self.ee_site, ROBOT_BODIES, GRIPPER_BODIES, self.config.collision
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
        scene = self._build_scene()
        skill = self._build_pick_skill(scene)
        start_q = scene.get_qpos(self.joint_names)

        # object_set comes from SceneResolver-canonical names (e.g.
        # "alphabet_soup_1"); the motion pipeline needs real MuJoCo body names
        # (e.g. "alphabet_soup_1_main") -- see resolve_body_name's docstring.
        body_names = [resolve_body_name(scene, name) for name in object_set]
        canonical_by_body = dict(zip(body_names, object_set))

        candidates = skill.grasp_generator.generate(scene, body_names)
        plan_result = skill.execute(scene, body_names, start_q, constraints, candidates=candidates)

        pre_steps = 0
        needs_closer_look = (
            not plan_result.success
            and plan_result.failure_reason == SkillFailureReason.NO_FEASIBLE_CANDIDATE
            and len(body_names) == 1
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
        approach_positions = np.asarray(plan_result.planned_trajectory.metadata["approach_positions"])
        full_reference = np.concatenate([traj.positions, approach_positions[1:]], axis=0)
        approach_start_idx = len(traj.positions)

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
        mpc = JointSpaceMPC(
            self.joint_names,
            self.ee_site,
            horizon=int(self.config.mpc.get("horizon", 16)),
            cost_config=MPCCostConfig(**{k: v for k, v in self.config.mpc.items() if k in MPCCostConfig.__dataclass_fields__}),
        )
        ctx = MPCContext(reference=full_reference)
        mpc.reset()

        # Track the planned reference, then -- still under the *same*
        # collision-aware MPC, not a separate blind servo -- keep holding at
        # its final waypoint (mpc.py clamps t0_step past the reference's end,
        # see JointSpaceMPC._cost_and_grad's `ridx`) until the arm actually
        # converges on the true grasp pose.
        #
        # An earlier version stopped calling the MPC the moment the nominal
        # reference was exhausted and handed the rest of the job to a
        # separate `_settle_toward_grasp_pose` helper: a plain Cartesian
        # P-servo with *no collision awareness at all*. That was fine as long
        # as the reference's time-parameterized duration closely matched how
        # fast the real Cartesian-delta controller could actually track it --
        # but measured on a real run, the arm was still 4.7cm / 16 degrees
        # from the grasp pose the moment the reference ran out (the
        # time-parameterizer's velocity-limit-based duration estimate doesn't
        # account for the real controller's tracking dynamics). Handing a gap
        # that large to a collision-blind servo is exactly what let the
        # gripper collide with the object while still mid-rotation instead of
        # cleanly finishing its approach. Continuing the same MPC loop keeps
        # every remaining step collision-checked.
        cfg = self.config.execution
        max_hold_steps = max_steps if max_steps is not None else len(full_reference) + cfg.max_settle_steps
        tracking_budget = min(max_hold_steps, len(full_reference) + cfg.max_settle_steps)

        steps = pre_steps
        gripper_cmd = self.env.open_gripper_cmd
        pos_err = ori_err = float("inf")
        converged = False
        stalled = False
        recent_pos_errs: list[float] = []
        t0 = 0
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

            if t0 >= len(full_reference):
                _, _, pos_err, ori_err = self._servo_toward_cartesian_pose(cand.grasp_position, cand.grasp_quat_wxyz)
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
                    # repeatedly bumping and retrying.
                    stalled = True
                    break

        settle_steps = max(0, t0 - len(full_reference))
        plan_result.diagnostics["settle"] = {
            "steps": settle_steps,
            "stalled": stalled,
            "converged": converged,
            "position_error_m": pos_err,
            "orientation_error_rad": ori_err,
        }

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
