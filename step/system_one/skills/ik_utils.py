"""Shared IK-retry and collision-escape utilities used by pick, place,
transport, open_drawer, and close.

solve_ik_robust tries the full restart budget and keeps the result
closest to the seed, rather than returning on the first success: accepting
the first success can land in a distant, unrelated arm configuration and
produce a needlessly long path.
"""

from __future__ import annotations

import numpy as np

from ..motion.collision import CollisionChecker, CollisionPhase
from ..motion.kinematics import IKSolver
from ..motion.scene_query import SceneQuery
from ..motion.types import AttachedObjectState, IKResult


def solve_ik_robust(
    ik_solver: IKSolver,
    scene: SceneQuery,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    seed_q: np.ndarray,
    rng: np.random.Generator,
    n_restarts: int = 8,
    perturb_scale: float = 0.6,
) -> IKResult:
    """Retries a failed DLS solve from perturbed seeds, keeping whichever
    success lands closest to seed_q rather than the first one found."""
    best_result = ik_solver.solve(scene.clone(), target_pos, target_quat_wxyz, seed_q=seed_q)
    best_distance = float(np.linalg.norm(best_result.q - seed_q)) if best_result.success else float("inf")
    lo, hi = scene.joint_qpos_range(ik_solver.joint_names)
    for attempt in range(n_restarts):
        # Perturbing near seed_q only searches one basin, which is useless
        # when the real solution isn't anywhere near it; full-range seeds cover that case.
        if attempt < n_restarts // 2:
            seed = np.clip(seed_q + rng.uniform(-perturb_scale, perturb_scale, size=seed_q.shape), lo, hi)
        else:
            seed = rng.uniform(lo, hi)
        result = ik_solver.solve(scene.clone(), target_pos, target_quat_wxyz, seed_q=seed)
        if result.success:
            distance = float(np.linalg.norm(result.q - seed_q))
            if distance < best_distance:
                best_result, best_distance = result, distance
    return best_result


def yaw_quat_wxyz(yaw_rad: float) -> np.ndarray:
    """Quaternion for a rotation of yaw_rad about the world +Z axis."""
    half = yaw_rad / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def escape_start_collision(
    ik_solver: IKSolver,
    collision_checker: CollisionChecker,
    scene: SceneQuery,
    attached: AttachedObjectState | None,
    goal_container_body: str | None,
    start_q: np.ndarray,
    max_lift_m: float = 0.20,
    step_m: float = 0.01,
) -> np.ndarray:
    """Retreats start_q in small collision-checked steps until clear;
    tries multiple directions since a laterally embedded object needs a sideways escape, not just a lift."""
    check = collision_checker.check(
        start_q, phase=CollisionPhase.FREE_SPACE, attached=attached, goal_container_body=goal_container_body
    )
    if not check.in_collision:
        return start_q

    joint_names = collision_checker.joint_names
    ee_site = collision_checker.ee_site
    scratch = scene.clone()
    scratch.set_qpos(joint_names, start_q)
    hand_pos, hand_quat = scratch.site_pose(ee_site)

    directions = {"up": np.array([0.0, 0.0, 1.0])}
    # Needs a single known goal body; skipped (not guessed) when none is
    # given, as when PickSkill calls this with no goal.
    if goal_container_body is not None:
        container_pos, _ = scene.body_pose(goal_container_body)
        toward_center_xy = container_pos[:2] - hand_pos[:2]
        norm = float(np.linalg.norm(toward_center_xy))
        toward_center_xy = toward_center_xy / norm if norm > 1e-6 else np.zeros(2)
        directions["up_toward_center"] = np.array([toward_center_xy[0], toward_center_xy[1], 1.0])
    # Tried first: the directions above ignore where the obstacle actually
    # is, and can walk the object further into it if it sits between start and goal.
    obstacle_xy_positions = []
    for contact in check.contacts:
        for geom_name in (contact.geom1, contact.geom2):
            if geom_name.startswith("attached:"):
                continue
            try:
                gid = scene.geom_id(geom_name)
            except KeyError:
                continue
            obstacle_xy_positions.append(scene.data.geom_xpos[gid][:2])
    if obstacle_xy_positions:
        obstacle_xy = np.mean(obstacle_xy_positions, axis=0)
        away_xy = hand_pos[:2] - obstacle_xy
        norm = float(np.linalg.norm(away_xy))
        away_xy = away_xy / norm if norm > 1e-6 else np.zeros(2)
        directions = {"away_from_obstacle": np.array([away_xy[0], away_xy[1], 0.3]), **directions}
    n_steps = max(1, int(round(max_lift_m / step_m)))
    for direction in directions.values():
        unit = direction / float(np.linalg.norm(direction))
        q = start_q
        for i in range(1, n_steps + 1):
            target_pos = hand_pos + unit * step_m * i
            ik = ik_solver.solve(scratch, target_pos, hand_quat, seed_q=q)
            if not ik.success:
                continue
            q = ik.q
            c = collision_checker.check(
                q, phase=CollisionPhase.FREE_SPACE, attached=attached, goal_container_body=goal_container_body
            )
            if not c.in_collision:
                return q
    return start_q  # couldn't escape within budget; planner will report start_in_collision honestly
