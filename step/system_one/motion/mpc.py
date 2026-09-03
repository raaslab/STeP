"""Joint-space receding-horizon MPC tracker.

Restructured from ``step/system_one/nominal_mpc.py``'s ``STLCartesianMPC``:
same overall shape (single-integrator rollout, Adam-warm-started differentiable
cost, softplus barriers, STL timing-window costs), but the state is now the
7-DoF joint configuration instead of a 3D Cartesian EE position, tracking a
pre-planned, already-collision-checked time-parameterized OMPL path instead of
chasing a hand-tuned stage-machine's analytic waypoints.

Two cost tiers, both differentiable but for different reasons:

* Tracking/terminal/control/smoothness/joint-limit costs are pure functions of
  the joint-space rollout -- exactly as cheap and analytic as the legacy MPC's
  Cartesian costs were.
* Clearance and STL-timing costs need the end-effector's *Cartesian* position,
  which requires forward kinematics. Rather than finite-differencing through
  MuJoCo's (non-differentiable) collision engine every iteration -- which
  would be far too slow for a 20 Hz control loop -- both costs are expressed
  as smooth penalties around a small set of Cartesian anchor points (obstacle
  centers / STL event targets), and their gradient w.r.t. the joint rollout is
  obtained *exactly* (not numerically) by chaining the cost's Cartesian
  gradient through the site Jacobian MuJoCo already gives us
  (``SceneQuery.site_jacobian``, i.e. ``mj_jacSite``) -- no autodiff needed.
  This mirrors the legacy MPC's own design (its SDF costs are also a fast
  analytic *approximation* of true geometry, not full collision, with the
  real 3D collision check instead gating planning and triggering replans).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .collision import CollisionChecker, CollisionPhase
from .scene_query import SceneQuery, quat_to_rotmat_wxyz
from .types import MPCCostConfig, MPCStepResult


def _softplus(x: np.ndarray, beta: float = 8.0) -> np.ndarray:
    return np.logaddexp(0.0, beta * x) / beta


def _softplus_grad(x: np.ndarray, beta: float = 8.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-beta * x))


@dataclass
class TimingWindow:
    """One STL "not-before start_step / deadline by end_step" constraint on an
    end-effector-space event target, exactly as flattened by
    ``executor._timing_windows_for_subtask`` today for the legacy MPC -- the
    same flattening step feeds this one, only the destination changed."""

    predicate: str
    event_target_xyz: np.ndarray
    start_step: int | None
    end_step: int | None
    radius: float = 0.03


@dataclass
class ObstacleSphere:
    center: np.ndarray
    radius: float


# Bodies whose collision extent exceeds this along any axis are treated as
# structure (table, floor, cabinet carcass), never as an "obstacle sphere" --
# sphere-fitting something table-sized would repel the arm from the entire
# tabletop, not just an object sitting on it. This is well above any real
# LIBERO tabletop object (groceries/bowls/bottles all measure a few cm to
# ~10cm) and well below furniture-scale structure.
_OBSTACLE_MAX_HALF_EXTENT_M = 0.15


def build_static_obstacle_spheres(
    scene: SceneQuery,
    candidate_body_names: list[str],
    exclude_bodies: set[str],
    n_spheres: int = 1,
) -> list["ObstacleSphere"]:
    """Sphere-approximate every body in ``candidate_body_names`` except
    ``exclude_bodies`` (the current pick/place/open target is always
    excluded by the caller -- it's the intended destination, not something
    to steer away from) so the MPC's own clearance cost (see
    ``JointSpaceMPC._clearance_cost_and_grad``) can actively push the
    tracked trajectory away from real, un-planned-for clutter while it
    tracks a reference, rather than blindly re-executing a pre-planned path
    on the assumption the real controller follows it closely enough that
    whatever the plan avoided stays avoided. Bodies larger than
    ``_OBSTACLE_MAX_HALF_EXTENT_M`` (table, floor, cabinet carcass) are
    skipped -- see that constant's docstring.

    Static by design: called once per skill invocation, not per step --
    every candidate body is assumed not to move for the duration of the
    tracked motion (true for every LIBERO object during a single pick/
    place/open subtask; a grasped/carried object is handled separately via
    ``AttachedObjectState``, not this obstacle list).
    """
    from .attached_object import fit_spheres_from_geoms

    spheres: list[ObstacleSphere] = []
    for name in candidate_body_names:
        if name in exclude_bodies:
            continue
        try:
            body_pos, body_quat = scene.body_pose(name)
        except KeyError:
            continue
        if not scene.body_collision_geom_ids(name):
            continue
        extent = scene.body_collision_extent_world(name)
        if extent is None or float(np.max(extent.half_extents)) > _OBSTACLE_MAX_HALF_EXTENT_M:
            continue
        R = quat_to_rotmat_wxyz(body_quat)
        for local_offset, radius in fit_spheres_from_geoms(scene, name, n_spheres):
            spheres.append(ObstacleSphere(center=body_pos + R @ np.asarray(local_offset), radius=float(radius)))
    return spheres


@dataclass
class MPCContext:
    reference: np.ndarray  # (T, n_joints) time-parameterized target positions
    obstacles: list[ObstacleSphere] = field(default_factory=list)
    timing_windows: list[TimingWindow] = field(default_factory=list)
    joint_limits_lo: np.ndarray | None = None
    joint_limits_hi: np.ndarray | None = None


class JointSpaceMPC:
    def __init__(
        self,
        joint_names: list[str],
        ee_site: str,
        horizon: int = 16,
        num_iters: int = 24,
        step_size: float = 0.15,
        action_scale: float = 0.15,  # rad, per-iteration max joint delta
        cost_config: MPCCostConfig | None = None,
    ) -> None:
        self.joint_names = list(joint_names)
        self.n_joints = len(joint_names)
        self.ee_site = ee_site
        self.horizon = horizon
        self.num_iters = num_iters
        self.step_size = step_size
        self.action_scale = action_scale
        self.cfg = cost_config or MPCCostConfig()
        self._prev_actions = np.zeros((horizon, self.n_joints))
        self._prev_action_applied = np.zeros(self.n_joints)

    def reset(self) -> None:
        self._prev_actions[:] = 0.0
        self._prev_action_applied[:] = 0.0

    # ------------------------------------------------------------ rollout

    def _rollout(self, q0: np.ndarray, u: np.ndarray) -> np.ndarray:
        q = np.zeros((self.horizon + 1, self.n_joints))
        q[0] = q0
        for t in range(self.horizon):
            q[t + 1] = q[t] + self.action_scale * u[t]
        return q

    # ------------------------------------------------------------ FK-chained costs

    def _ee_and_jacobian(self, scene: SceneQuery, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scene.set_qpos(self.joint_names, q)
        ee_pos, _ = scene.site_pose(self.ee_site)
        jacp, _ = scene.site_jacobian(self.ee_site)
        dof = scene.dof_addrs(self.joint_names)
        return ee_pos, jacp[:, dof]

    def _clearance_cost_and_grad(
        self, scene: SceneQuery, q_traj: np.ndarray, obstacles: list[ObstacleSphere]
    ) -> tuple[float, np.ndarray]:
        if not obstacles:
            return 0.0, np.zeros_like(q_traj[1:])
        cost = 0.0
        grad = np.zeros_like(q_traj)
        for t in range(1, self.horizon + 1):
            ee_pos, J = self._ee_and_jacobian(scene, q_traj[t])
            for obs in obstacles:
                diff = ee_pos - obs.center
                dist = float(np.linalg.norm(diff) + 1e-9)
                violation = obs.radius - dist  # >0 => inside the obstacle sphere
                penalty = _softplus(np.array([violation]))[0]
                cost += self.cfg.clearance_weight * penalty
                dpenalty_ddist = -_softplus_grad(np.array([violation]))[0]
                ddist_dee = diff / dist
                dcost_dee = self.cfg.clearance_weight * dpenalty_ddist * ddist_dee
                grad[t] += dcost_dee @ J
        return cost, grad[1:]

    def _timing_cost_and_grad(
        self, scene: SceneQuery, q_traj: np.ndarray, windows: list[TimingWindow], t0_step: int
    ) -> tuple[float, np.ndarray]:
        if not windows:
            return 0.0, np.zeros_like(q_traj[1:])
        cost = 0.0
        grad = np.zeros_like(q_traj)
        for t in range(1, self.horizon + 1):
            global_step = t0_step + t
            ee_pos, J = self._ee_and_jacobian(scene, q_traj[t])
            for w in windows:
                diff = ee_pos - w.event_target_xyz
                dist = float(np.linalg.norm(diff) + 1e-9)
                if w.start_step is not None and global_step < w.start_step:
                    # Not-before: penalize being inside the event radius too early.
                    violation = w.radius - dist
                    penalty = _softplus(np.array([violation]))[0]
                    cost += self.cfg.timing_weight * penalty
                    dpenalty_ddist = -_softplus_grad(np.array([violation]))[0]
                    dcost_dee = self.cfg.timing_weight * dpenalty_ddist * (diff / dist)
                    grad[t] += dcost_dee @ J
                if w.end_step is not None:
                    remaining = w.end_step - global_step
                    horizon_frac = max(0.0, 1.0 - remaining / max(self.horizon, 1))
                    urgency = horizon_frac**2
                    cost += self.cfg.stl_weight * urgency * float(dist**2)
                    grad[t] += self.cfg.stl_weight * urgency * 2.0 * diff @ J
        return cost, grad[1:]

    # ------------------------------------------------------------ analytic joint-space costs

    def _cost_and_grad(
        self, scene: SceneQuery, q0: np.ndarray, u: np.ndarray, ctx: MPCContext, t0_step: int
    ) -> tuple[float, np.ndarray]:
        q_traj = self._rollout(q0, u)
        ref = ctx.reference
        T = self.horizon

        cost = 0.0
        grad_q = np.zeros_like(q_traj)

        for t in range(1, T + 1):
            ridx = min(t0_step + t - 1, len(ref) - 1)
            ramp = 0.25 + 0.75 * (t / T)
            err = q_traj[t] - ref[ridx]
            w = self.cfg.terminal_weight if t == T else self.cfg.tracking_weight * ramp
            cost += w * float(np.dot(err, err))
            grad_q[t] += 2.0 * w * err

        cost += self.cfg.control_weight * float(np.sum(u**2))
        grad_u_control = 2.0 * self.cfg.control_weight * u

        smooth = np.diff(u, axis=0, prepend=self._prev_action_applied[None, :])
        cost += self.cfg.smoothness_weight * float(np.sum(smooth**2))

        if ctx.joint_limits_lo is not None and ctx.joint_limits_hi is not None:
            for t in range(1, T + 1):
                lo_violation = ctx.joint_limits_lo - q_traj[t]
                hi_violation = q_traj[t] - ctx.joint_limits_hi
                cost += self.cfg.joint_limit_weight * float(
                    np.sum(_softplus(lo_violation) ** 2 + _softplus(hi_violation) ** 2)
                )
                grad_q[t] += self.cfg.joint_limit_weight * (
                    -2.0 * _softplus(lo_violation) * _softplus_grad(lo_violation)
                    + 2.0 * _softplus(hi_violation) * _softplus_grad(hi_violation)
                )

        clearance_cost, clearance_grad = self._clearance_cost_and_grad(scene, q_traj, ctx.obstacles)
        cost += clearance_cost
        grad_q[1:] += clearance_grad

        timing_cost, timing_grad = self._timing_cost_and_grad(scene, q_traj, ctx.timing_windows, t0_step)
        cost += timing_cost
        grad_q[1:] += timing_grad

        # Backprop through the single-integrator rollout: dq[t]/du[k] = action_scale for k<t.
        grad_u = np.zeros_like(u)
        running = np.zeros(self.n_joints)
        for t in range(T, 0, -1):
            running += grad_q[t]
            grad_u[t - 1] = self.action_scale * running

        grad_u += grad_u_control
        # Smoothness gradient (finite-difference chain).
        grad_u[0] += 2.0 * self.cfg.smoothness_weight * (u[0] - self._prev_action_applied)
        for t in range(1, T):
            d = u[t] - u[t - 1]
            grad_u[t] += 2.0 * self.cfg.smoothness_weight * d
            grad_u[t - 1] -= 2.0 * self.cfg.smoothness_weight * d

        return cost, grad_u

    # ------------------------------------------------------------ public API

    def step(
        self,
        scene: SceneQuery,
        q_current: np.ndarray,
        ctx: MPCContext,
        t0_step: int,
        collision_checker: CollisionChecker | None = None,
        replan_config=None,
        collision_phase: CollisionPhase = CollisionPhase.FREE_SPACE,
        target_object_body: str | None = None,
        goal_container_body: str | None = None,
        attached=None,
    ) -> MPCStepResult:
        """One receding-horizon tick: optimize a horizon of actions from
        ``q_current``, return the *first* joint command, then shift the warm
        start for next tick (the actual "recede" happens because the caller
        re-invokes ``step`` with the new ``q_current`` on the next control
        tick -- this class holds no episode-long state besides the warm
        start)."""
        u = np.concatenate([self._prev_actions[1:], self._prev_actions[-1:]], axis=0)

        m = np.zeros_like(u)
        v = np.zeros_like(u)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        cost = 0.0
        for it in range(1, self.num_iters + 1):
            cost, grad = self._cost_and_grad(scene, q_current, u, ctx, t0_step)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**it)
            v_hat = v / (1 - beta2**it)
            u -= self.step_size * m_hat / (np.sqrt(v_hat) + eps)
            u = np.clip(u, -1.0, 1.0)

        self._prev_actions = u
        joint_command = q_current + self.action_scale * u[0]
        self._prev_action_applied = u[0].copy()

        tracking_error = float(np.linalg.norm(q_current - ctx.reference[min(t0_step, len(ctx.reference) - 1)]))
        min_clearance = float("inf")
        replan_requested = False
        replan_reason = None
        if collision_checker is not None:
            result = collision_checker.check(
                joint_command,
                phase=collision_phase,
                target_object_body=target_object_body,
                goal_container_body=goal_container_body,
                attached=attached,
            )
            min_clearance = result.min_clearance
            if replan_config is not None:
                if tracking_error > replan_config.tracking_error_m:
                    replan_requested = True
                    replan_reason = "tracking_error_exceeded"
                elif result.in_collision or min_clearance < replan_config.collision_margin_m:
                    replan_requested = True
                    replan_reason = "collision_margin_violated"

        return MPCStepResult(
            joint_command=joint_command,
            tracking_error=tracking_error,
            min_clearance=min_clearance,
            cost_breakdown={"total": cost},
            stl_robustness={},
            replan_requested=replan_requested,
            replan_reason=replan_reason,
        )
