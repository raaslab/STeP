from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any
import numpy as np


@dataclass
class STLStageContext:
    target_xyz: np.ndarray
    knife_xy: np.ndarray
    knife_radius: float
    table_z: float
    # Deprecated compatibility fields for the old analytic circular keepout.
    # Geometric safety costs are now supplied exclusively through sdf_constraints.
    always_keepout: bool = False
    carry_active: bool = False
    carry_min_z: float = 0.0
    goal_xy: np.ndarray | None = None
    goal_radius: float = 0.06
    mug_xy_radius: float = 0.0
    mug_pos: np.ndarray | None = None
    # Optional axis-aligned workspace box for the end-effector rollout.
    # If provided, the MPC adds smooth barrier penalties for every predicted
    # state outside [workspace_min_xyz, workspace_max_xyz].
    workspace_min_xyz: np.ndarray | None = None
    workspace_max_xyz: np.ndarray | None = None
    workspace_weight: float = 80.0
    sdf_constraints: list[Any] = field(default_factory=list)

    # Optional finite-horizon temporal STL costs.  Each entry is a generic
    # predicate window dictionary prepared by BaseManipulationEnv, e.g.
    # {"predicate": "grasped", "start_sec": 3.0, "end_sec": 5.0,
    #  "elapsed_sec": 2.2, "event_target_xyz": np.array(...)}.
    # The MPC treats these as soft robustness costs over the predicted EE
    # rollout rather than as task-specific if/else gates.
    timing_windows: list[Any] = field(default_factory=list)
    dt_sec: float = 0.05


class STLCartesianMPC:
    """
    A tiny optimization-based MPC in end-effector delta-position space with
    smooth STL-style soft robustness penalties.

    Internal model:
        p_{t+1} = p_t + action_scale * u_t,  ||u_t||_inf <= 1

    The controller does *not* model full robot dynamics. It is a practical,
    end-effector-space nominal planner that is easy to modify.
    """

    def __init__(
        self,
        action_scale_m: float = 0.035,
        horizon: int = 14,
        num_iters: int = 22,
        step_size: float = 0.18,
        stage_weight: float = 0.22,
        terminal_weight: float = 22.0,
        control_weight: float = 0.03,
        smooth_weight: float = 0.90,
        table_clearance_z: float = 0.020,
        knife_weight: float = 22.0,
        carry_weight: float = 18.0,
        goal_over_weight: float = 8.0,
        barrier_beta: float = 36.0,
    ):
        self.action_scale_m = float(action_scale_m)
        self.horizon = int(horizon)
        self.num_iters = int(num_iters)
        self.step_size = float(step_size)
        self.stage_weight = float(stage_weight)
        self.terminal_weight = float(terminal_weight)
        self.control_weight = float(control_weight)
        self.smooth_weight = float(smooth_weight)
        self.table_clearance_z = float(table_clearance_z)
        self.knife_weight = float(knife_weight)
        self.carry_weight = float(carry_weight)
        self.goal_over_weight = float(goal_over_weight)
        self.barrier_beta = float(barrier_beta)
        self.prev_solution = np.zeros((self.horizon, 3), dtype=np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        self.prev_solution.fill(0.0)
        self.prev_action.fill(0.0)

    def _project(self, u: np.ndarray) -> np.ndarray:
        return np.clip(u, -1.0, 1.0)

    def _goal_init(self, pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = target - pos
        norm = float(np.linalg.norm(delta)) + 1e-8
        direction = delta / norm
        guess = np.repeat(direction[None, :], self.horizon, axis=0)
        return self._project(0.70 * guess)

    def _warm_start(self, pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        if float(np.linalg.norm(self.prev_solution[0])) < 1e-6:
            return self._goal_init(pos, target)
        shifted = np.zeros_like(self.prev_solution)
        shifted[:-1] = self.prev_solution[1:]
        shifted[-1] = self.prev_solution[-1]
        return self._project(0.82 * shifted + 0.18 * self._goal_init(pos, target))

    def _rollout(self, pos: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.zeros((self.horizon + 1, 3), dtype=np.float32)
        x[0] = pos
        for t in range(self.horizon):
            x[t + 1] = x[t] + self.action_scale_m * u[t]
        return x

    def _softplus_sq(self, violation: float) -> tuple[float, float]:
        beta = self.barrier_beta
        soft = math.log1p(math.exp(beta * violation)) / beta
        sig = 1.0 / (1.0 + math.exp(-beta * violation))
        return soft * soft, 2.0 * soft * sig


    def _timing_window_cost_and_grad(
        self,
        x: np.ndarray,
        ctx: STLStageContext,
        grad_x: np.ndarray,
    ) -> float:
        """Add smooth robustness costs for local timing-window predicates.

        For a bounded window start<=predicate<=end, the MPC receives two
        differentiable STL-style costs:

        * before ``start``: penalize predicted states that would make the
          predicate true too early, using a continuous event-target proxy;
        * inside/near the window: increase attraction to the event target as
          the deadline approaches.

        The predicate itself is still checked by the online STL monitor.  This
        cost only shapes the receding-horizon policy so it naturally waits,
        approaches, or accelerates toward the event at the right time.
        """
        if not ctx.timing_windows:
            return 0.0

        total = 0.0
        dt = max(float(getattr(ctx, "dt_sec", 0.05)), 1e-6)
        horizon_time = dt * float(self.horizon)

        for window in ctx.timing_windows:
            try:
                target = np.asarray(window.get("event_target_xyz"), dtype=np.float32)
            except Exception:
                continue
            if target.shape != (3,) or not np.all(np.isfinite(target)):
                continue

            early_hold_target = None
            try:
                hold_candidate = window.get("early_hold_xyz")
                if hold_candidate is not None:
                    hold_arr = np.asarray(hold_candidate, dtype=np.float32)
                    if hold_arr.shape == (3,) and np.all(np.isfinite(hold_arr)):
                        early_hold_target = hold_arr
            except Exception:
                early_hold_target = None

            start_sec = window.get("start_sec")
            end_sec = window.get("end_sec")
            elapsed_sec = float(window.get("elapsed_sec", 0.0) or 0.0)
            event_radius = float(window.get("event_radius", 0.035) or 0.035)
            early_weight = float(window.get("early_weight", 180.0) or 180.0)
            deadline_weight = float(window.get("deadline_weight", 40.0) or 40.0)
            window_weight = float(window.get("window_weight", 8.0) or 8.0)
            early_hold_weight = float(window.get("early_hold_weight", 260.0) or 260.0)

            for t in range(1, self.horizon + 1):
                tau = elapsed_sec + dt * float(t)
                err = x[t] - target
                dist = float(np.linalg.norm(err) + 1e-8)

                # Not-before semantics: before the lower bound, the robustness
                # of !predicate is approximated by dist(event_proxy)-radius.
                # Violating that margin means the predicted EE state is close
                # enough to trigger the event too early.
                if start_sec is not None and tau < float(start_sec):
                    pen = event_radius - dist
                    term, coeff = self._softplus_sq(float(pen))
                    total += early_weight * term
                    grad_x[t] += early_weight * coeff * (-(err / dist))

                    # A pure "not event" barrier can be satisfied by moving
                    # past the event target. Add a generic precursor-hold cost
                    # when available (e.g., pregrasp for Grasped) so the MPC
                    # waits in a safe predecessor state before the window opens.
                    if early_hold_target is not None:
                        hold_err = x[t] - early_hold_target
                        total += early_hold_weight * float(np.dot(hold_err, hold_err))
                        grad_x[t] += 2.0 * early_hold_weight * hold_err

                # Not-after semantics: once the window is open, pull the
                # predicted rollout toward the event proxy.  The weight ramps
                # up as the deadline comes within the MPC horizon.
                if end_sec is not None:
                    lower_ok = start_sec is None or tau >= float(start_sec)
                    if lower_ok:
                        time_left = float(end_sec) - tau
                        urgency = max(0.0, 1.0 - max(time_left, 0.0) / max(horizon_time, dt))
                        w = window_weight + deadline_weight * urgency * urgency
                        total += w * float(np.dot(err, err))
                        grad_x[t] += 2.0 * w * err
                elif start_sec is not None and tau >= float(start_sec):
                    # Lower-bounded-only window: after it opens, the event is
                    # useful but not urgent.
                    w = window_weight
                    total += w * float(np.dot(err, err))
                    grad_x[t] += 2.0 * w * err

        return float(total)

    def _cost_and_grad(self, pos: np.ndarray, ctx: STLStageContext, u: np.ndarray) -> tuple[float, np.ndarray]:
        x = self._rollout(pos, u)
        grad_x = np.zeros_like(x)
        grad_u = np.zeros_like(u)
        cost = 0.0

        for t in range(1, self.horizon + 1):
            err = x[t] - ctx.target_xyz
            w = self.stage_weight * (0.25 + 0.75 * t / self.horizon)
            cost += w * float(np.dot(err, err))
            grad_x[t] += 2.0 * w * err

        errT = x[-1] - ctx.target_xyz
        cost += self.terminal_weight * float(np.dot(errT, errT))
        grad_x[-1] += 2.0 * self.terminal_weight * errT

        cost += self.control_weight * float(np.sum(u * u))
        grad_u += 2.0 * self.control_weight * u

        diff0 = u[0] - self.prev_action
        cost += self.smooth_weight * float(np.dot(diff0, diff0))
        grad_u[0] += 2.0 * self.smooth_weight * diff0
        for t in range(1, self.horizon):
            d = u[t] - u[t - 1]
            cost += self.smooth_weight * float(np.dot(d, d))
            grad_u[t] += 2.0 * self.smooth_weight * d
            grad_u[t - 1] -= 2.0 * self.smooth_weight * d

        floor_z = ctx.table_z + self.table_clearance_z
        for t in range(1, self.horizon + 1):
            pen = floor_z - float(x[t, 2])
            term, coeff = self._softplus_sq(pen)
            cost += 12.0 * term
            grad_x[t, 2] += -12.0 * coeff

        # Safety/keepout is intentionally SDF-only.  The older analytic
        # circular keepout around ctx.knife_xy has been removed from the MPC
        # objective to avoid double-counting geometry when an SDF is present.
        # Avoidance constraints should activate entries in ctx.sdf_constraints,
        # which are handled by the robustness penalty block below.

        if ctx.carry_active:
            for t in range(1, self.horizon + 1):
                pen = float(ctx.carry_min_z) - float(x[t, 2])
                term, coeff = self._softplus_sq(pen)
                cost += self.carry_weight * term
                grad_x[t, 2] += -self.carry_weight * coeff
                if ctx.goal_xy is not None:
                    goal_delta = x[t, :2] - ctx.goal_xy
                    dist = float(np.linalg.norm(goal_delta) + 1e-8)
                    pen_goal = dist - float(ctx.goal_radius)
                    term_goal, coeff_goal = self._softplus_sq(pen_goal)
                    cost += 0.25 * self.goal_over_weight * term_goal
                    grad_x[t, 0] += 0.25 * self.goal_over_weight * coeff_goal * (goal_delta[0] / dist)
                    grad_x[t, 1] += 0.25 * self.goal_over_weight * coeff_goal * (goal_delta[1] / dist)

        cost += self._timing_window_cost_and_grad(x, ctx, grad_x)

        if ctx.workspace_min_xyz is not None and ctx.workspace_max_xyz is not None:
            lo = np.asarray(ctx.workspace_min_xyz, dtype=np.float32)
            hi = np.asarray(ctx.workspace_max_xyz, dtype=np.float32)
            weight = float(ctx.workspace_weight)
            for t in range(1, self.horizon + 1):
                for j in range(3):
                    # Lower-bound violation: lo[j] - x[t, j] > 0
                    pen_low = float(lo[j] - x[t, j])
                    term_low, coeff_low = self._softplus_sq(pen_low)
                    cost += weight * term_low
                    grad_x[t, j] += -weight * coeff_low

                    # Upper-bound violation: x[t, j] - hi[j] > 0
                    pen_high = float(x[t, j] - hi[j])
                    term_high, coeff_high = self._softplus_sq(pen_high)
                    cost += weight * term_high
                    grad_x[t, j] += weight * coeff_high

        if ctx.sdf_constraints:
            object_xy_offset = None
            if ctx.mug_pos is not None:
                object_xy_offset = np.asarray(ctx.mug_pos[:2] - pos[:2], dtype=np.float32)

            for sdf_c in ctx.sdf_constraints:
                weight = float(getattr(sdf_c, "weight", 10.0))
                for t in range(1, self.horizon + 1):
                    if getattr(sdf_c, "applies_to_kind", lambda _k: False)("ee"):
                        xy = x[t, :2]
                        rho = float(sdf_c.robustness_xy(xy))
                        grad_rho = np.asarray(sdf_c.grad_xy(xy), dtype=np.float32)
                        term, coeff = self._softplus_sq(-rho)
                        cost += weight * term
                        grad_x[t, 0] += -weight * coeff * float(grad_rho[0])
                        grad_x[t, 1] += -weight * coeff * float(grad_rho[1])

                    if object_xy_offset is not None and getattr(sdf_c, "applies_to_kind", lambda _k: False)("object"):
                        xy = x[t, :2] + object_xy_offset
                        rho = float(sdf_c.robustness_xy(xy))
                        grad_rho = np.asarray(sdf_c.grad_xy(xy), dtype=np.float32)
                        term, coeff = self._softplus_sq(-rho)
                        cost += weight * term
                        grad_x[t, 0] += -weight * coeff * float(grad_rho[0])
                        grad_x[t, 1] += -weight * coeff * float(grad_rho[1])

        lam = np.zeros_like(x)
        lam[-1] = grad_x[-1]
        for t in reversed(range(self.horizon)):
            lam[t] += grad_x[t] + lam[t + 1]
            grad_u[t] += self.action_scale_m * lam[t + 1]
        return float(cost), grad_u

    def act(self, pos: np.ndarray, ctx: STLStageContext) -> np.ndarray:
        u = self._warm_start(pos, ctx.target_xyz)
        m = np.zeros_like(u)
        v = np.zeros_like(u)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for k in range(1, self.num_iters + 1):
            _, grad = self._cost_and_grad(pos, ctx, u)
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            m_hat = m / (1.0 - beta1**k)
            v_hat = v / (1.0 - beta2**k)
            u = self._project(u - self.step_size * m_hat / (np.sqrt(v_hat) + eps))
        self.prev_solution = u.copy()
        self.prev_action = u[0].copy()
        return u[0].astype(np.float32)
