"""Joint-space OMPL RRTConnect motion planner.

Targets the OMPL 2.0.2 (conda-forge, nanobind) Python bindings actually
installed here, whose API differs from the older Boost.Python wrapper (states
come from SpaceInformation.allocState(), validity checker is a plain callable).

Never falls back to unchecked straight-line interpolation -- if OMPL isn't
available, every call returns a structured
MotionPlan(success=False, failure_reason=PLANNER_UNAVAILABLE) via
MotionPlanner.unavailable_plan().
"""

from __future__ import annotations

import time

import numpy as np

from .collision import CollisionChecker, CollisionPhase
from .config import PlannerConfig
from .types import AttachedObjectState, MotionPlan, PlannerFailureReason

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou

    OMPL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by test_ompl_planner's skip path
    OMPL_AVAILABLE = False


class PlannerUnavailableError(RuntimeError):
    pass


class MotionPlanner:
    """OMPL RRTConnect over the arm's joint angles."""

    def __init__(
        self,
        joint_names: list[str],
        joint_limits_lo: np.ndarray,
        joint_limits_hi: np.ndarray,
        collision_checker: CollisionChecker,
        config: PlannerConfig | None = None,
    ) -> None:
        self.joint_names = list(joint_names)
        self.dim = len(joint_names)
        self.joint_limits_lo = np.asarray(joint_limits_lo, dtype=np.float64)
        self.joint_limits_hi = np.asarray(joint_limits_hi, dtype=np.float64)
        self.collision_checker = collision_checker
        self.config = config or PlannerConfig()

    @staticmethod
    def unavailable_plan(reason: str = "ompl is not installed") -> MotionPlan:
        return MotionPlan(
            success=False,
            failure_reason=PlannerFailureReason.PLANNER_UNAVAILABLE,
            metadata={"detail": reason},
        )

    def _make_space(self):
        space = ob.RealVectorStateSpace(self.dim)
        bounds = ob.RealVectorBounds(self.dim)
        for i in range(self.dim):
            bounds.setLow(i, float(self.joint_limits_lo[i]))
            bounds.setHigh(i, float(self.joint_limits_hi[i]))
        space.setBounds(bounds)
        return space

    def _q_from_state(self, state) -> np.ndarray:
        return np.array([state[i] for i in range(self.dim)], dtype=np.float64)

    def _state_from_q(self, si, q: np.ndarray):
        state = si.allocState()
        for i in range(self.dim):
            state[i] = float(q[i])
        return state

    def plan(
        self,
        start_q: np.ndarray,
        goal_qs: list[np.ndarray],
        phase: CollisionPhase = CollisionPhase.FREE_SPACE,
        target_object_body: str | None = None,
        attached: AttachedObjectState | None = None,
        goal_container_body: str | None = None,
        permitted_contact_geom_ids: set[int] | None = None,
    ) -> MotionPlan:
        """Plan from start_q to the best-reachable of goal_qs (already
        IK-filtered by the caller).

        Retrying this exact (start_q, goal_qs) pair does *not* resample a
        different solution: OMPL seeds its global RNG once per process and
        refuses later reseeds, so identical calls return identical paths.
        Retries only help when the problem itself differs.
        """
        if not OMPL_AVAILABLE:
            return self.unavailable_plan()

        t0 = time.time()
        start_q = np.asarray(start_q, dtype=np.float64)

        def _check(q):
            return self.collision_checker.check(
                q,
                phase=phase,
                target_object_body=target_object_body,
                attached=attached,
                goal_container_body=goal_container_body,
                permitted_contact_geom_ids=permitted_contact_geom_ids,
            )

        start_result = _check(start_q)
        if start_result.in_collision:
            return MotionPlan(
                success=False,
                failure_reason=PlannerFailureReason.START_IN_COLLISION,
                planning_time_sec=time.time() - t0,
                metadata={"contacts": [(c.geom1, c.geom2) for c in start_result.contacts]},
            )

        valid_goals = []
        for gq in goal_qs:
            gq = np.asarray(gq, dtype=np.float64)
            res = _check(gq)
            if not res.in_collision:
                valid_goals.append(gq)
        if not valid_goals:
            return MotionPlan(
                success=False,
                failure_reason=PlannerFailureReason.GOAL_IN_COLLISION,
                planning_time_sec=time.time() - t0,
            )

        # Try the straight-line path first -- RRTConnect returns whatever its
        # trees connect, not necessarily direct, and simplification can't recover a path it never found.
        edge_resolution = self.config.validity_resolution * float(
            np.max(self.joint_limits_hi - self.joint_limits_lo)
        )
        for gq in valid_goals:
            direct = self.collision_checker.check_edge(
                start_q,
                gq,
                resolution=edge_resolution,
                phase=phase,
                target_object_body=target_object_body,
                attached=attached,
                goal_container_body=goal_container_body,
                permitted_contact_geom_ids=permitted_contact_geom_ids,
            )
            if not direct.in_collision:
                return MotionPlan(
                    success=True,
                    joint_names=self.joint_names,
                    geometric_path=np.linspace(start_q, gq, int(self.config.interpolation_density)),
                    planning_time_sec=time.time() - t0,
                    failure_reason=PlannerFailureReason.NONE,
                    metadata={"solve_status": "Direct path", "n_goals_tried": len(goal_qs)},
                )

        ou.RNG.setSeed(int(self.config.seed))
        space = self._make_space()
        ss = og.SimpleSetup(space)

        def is_valid(state) -> bool:
            q = self._q_from_state(state)
            return not _check(q).in_collision

        ss.setStateValidityChecker(is_valid)
        si = ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(float(self.config.validity_resolution))

        start_state = self._state_from_q(si, start_q)
        goal_states = ob.GoalStates(si)
        for gq in valid_goals:
            goal_states.addState(self._state_from_q(si, gq))
        ss.setStartState(start_state)
        ss.setGoal(goal_states)

        planner = og.RRTConnect(si)
        planner.setRange(float(self.config.range))
        ss.setPlanner(planner)

        solved = ss.solve(float(self.config.timeout_sec))
        planning_time = time.time() - t0
        if not solved:
            return MotionPlan(
                success=False,
                failure_reason=PlannerFailureReason.TIMEOUT,
                planning_time_sec=planning_time,
            )
        if str(solved) != "Exact solution":
            # "Approximate solution" means the trees never actually connected
            # within budget; treating it as success would execute a path that never reaches the target.
            reason = (
                PlannerFailureReason.APPROXIMATE_SOLUTION_ONLY
                if str(solved) == "Approximate solution"
                else PlannerFailureReason.NO_PATH_FOUND
            )
            return MotionPlan(
                success=False,
                failure_reason=reason,
                planning_time_sec=planning_time,
                metadata={"status": str(solved)},
            )

        ss.simplifySolution(float(self.config.simplify_time_sec))
        path = ss.getSolutionPath()
        path.interpolate(int(self.config.interpolation_density))
        waypoints = np.array(
            [[path.getState(i)[j] for j in range(self.dim)] for i in range(path.getStateCount())]
        )

        # Defense-in-depth: re-validate the whole interpolated path at our own
        # resolution, independent of OMPL's internal edge checking (never trust vertices alone).
        for i in range(len(waypoints) - 1):
            edge = self.collision_checker.check_edge(
                waypoints[i],
                waypoints[i + 1],
                resolution=self.config.validity_resolution
                * float(np.max(self.joint_limits_hi - self.joint_limits_lo)),
                phase=phase,
                target_object_body=target_object_body,
                attached=attached,
                goal_container_body=goal_container_body,
                permitted_contact_geom_ids=permitted_contact_geom_ids,
            )
            if edge.in_collision:
                return MotionPlan(
                    success=False,
                    failure_reason=PlannerFailureReason.NO_PATH_FOUND,
                    planning_time_sec=planning_time,
                    metadata={"detail": "post-hoc edge re-validation failed", "edge_index": i},
                )

        return MotionPlan(
            success=True,
            joint_names=self.joint_names,
            geometric_path=waypoints,
            planning_time_sec=planning_time,
            failure_reason=PlannerFailureReason.NONE,
            metadata={"solve_status": str(solved), "n_goals_tried": len(goal_qs)},
        )
