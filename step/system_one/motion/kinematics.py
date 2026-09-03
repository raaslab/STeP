"""Damped-least-squares (Levenberg-Marquardt style) inverse kinematics on a
scratch MuJoCo model/data, via ``SceneQuery``. No pinocchio/ikpy dependency --
built directly on the ``mj_jacSite`` Jacobian STeP already has through MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene_query import SceneQuery, quat_mul_wxyz


def quat_error_wxyz(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """Small-angle axis-angle error rotating q_current towards q_target,
    expressed in world frame (3,). Sign-robust to the quaternion double-cover."""
    qc = q_current if q_current[0] >= 0 else -q_current
    qt = q_target if q_target[0] >= 0 else -q_target
    q_conj_c = np.array([qc[0], -qc[1], -qc[2], -qc[3]])
    q_err = quat_mul_wxyz(qt, q_conj_c)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


@dataclass
class IKConfig:
    max_iterations: int = 150
    position_tol_m: float = 0.002
    orientation_tol_rad: float = 0.02
    damping: float = 1e-3
    step_scale: float = 1.0
    orientation_weight: float = 1.0


from .types import IKFailureReason, IKResult  # noqa: E402  (avoid circular import at module load)


class IKSolver:
    """Iterative DLS IK for a named end-effector site over a fixed joint chain."""

    def __init__(
        self,
        joint_names: list[str],
        site_name: str = "gripper0_grip_site",
        config: IKConfig | None = None,
    ) -> None:
        self.joint_names = list(joint_names)
        self.site_name = site_name
        self.config = config or IKConfig()

    def solve(
        self,
        scene: SceneQuery,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        seed_q: np.ndarray | None = None,
    ) -> IKResult:
        """Solve IK against a **scratch** SceneQuery (mutated in place across
        iterations -- pass ``scene.clone()`` if the caller needs the original
        preserved). Joint limits are enforced by clamping every iterate."""
        cfg = self.config
        lo, hi = scene.joint_qpos_range(self.joint_names)
        q = np.clip(
            seed_q.copy() if seed_q is not None else scene.get_qpos(self.joint_names), lo, hi
        )
        scene.set_qpos(self.joint_names, q)
        dof_addrs = scene.dof_addrs(self.joint_names)

        pos_err = ori_err = np.inf
        for it in range(cfg.max_iterations):
            site_pos, site_quat = scene.site_pose(self.site_name)
            e_pos = target_pos - site_pos
            e_ori = quat_error_wxyz(site_quat, target_quat_wxyz)
            pos_err = float(np.linalg.norm(e_pos))
            ori_err = float(np.linalg.norm(e_ori))
            if pos_err <= cfg.position_tol_m and ori_err <= cfg.orientation_tol_rad:
                return IKResult(
                    success=True,
                    q=q.copy(),
                    position_error=pos_err,
                    orientation_error=ori_err,
                    iterations=it,
                    reason=IKFailureReason.NONE,
                )

            jacp, jacr = scene.site_jacobian(self.site_name)
            J = np.vstack([jacp[:, dof_addrs], cfg.orientation_weight * jacr[:, dof_addrs]])
            e = np.concatenate([e_pos, cfg.orientation_weight * e_ori])

            JJt = J @ J.T
            lam2 = cfg.damping**2 * np.eye(JJt.shape[0])
            dq = J.T @ np.linalg.solve(JJt + lam2, e)
            q = np.clip(q + cfg.step_scale * dq, lo, hi)
            scene.set_qpos(self.joint_names, q)

        # Ran out of iterations: report whichever failure mode is most informative.
        at_limit = np.any(np.isclose(q, lo, atol=1e-6) | np.isclose(q, hi, atol=1e-6))
        reason = (
            IKFailureReason.JOINT_LIMIT_VIOLATION if at_limit else IKFailureReason.NO_CONVERGENCE
        )
        return IKResult(
            success=False,
            q=None,
            position_error=pos_err,
            orientation_error=ori_err,
            iterations=cfg.max_iterations,
            reason=reason,
        )
