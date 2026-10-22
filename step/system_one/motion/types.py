"""Typed result/contract dataclasses shared across the motion pipeline.

Kept dependency-free (stdlib + numpy only) so every other module in this package,
including test doubles, can import from here without pulling in MuJoCo/OMPL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- grasping


@dataclass
class GraspCandidate:
    """One candidate 6-DoF grasp for a named object.

    Quaternion convention: WXYZ everywhere in this package (matches Cap-X's
    libero_privileged.py and robosuite's own LIBERO conventions). Convert
    to XYZW only at a library boundary that requires it.

    grasp_pose_world is the panda_hand / TCP frame target (what
    robot0_eef_pos/gripper0_right_grip_site reports), already
    converted from any fingertip-frame grasp-net-style output.

    No separate pregrasp pose is stored here -- PickSkill derives an
    internal standoff (this pose offset back along
    approach_direction_world), plans the approach under
    CollisionPhase.FREE_SPACE, then validates a short final descent
    under TERMINAL_APPROACH. Permitting gripper<->target contact for the
    whole approach instead would let RRTConnect shove the object anywhere
    along the way.
    """

    object_name: str
    grasp_pose_world: np.ndarray  # (7,) [x, y, z, qw, qx, qy, qz]
    approach_direction_world: np.ndarray  # (3,) unit vector, gripper -> object at contact
    gripper_width: float | None
    generator_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.grasp_pose_world = np.asarray(self.grasp_pose_world, dtype=np.float64).reshape(7)
        self.approach_direction_world = np.asarray(
            self.approach_direction_world, dtype=np.float64
        ).reshape(3)

    @property
    def grasp_position(self) -> np.ndarray:
        return self.grasp_pose_world[:3]

    @property
    def grasp_quat_wxyz(self) -> np.ndarray:
        return self.grasp_pose_world[3:7]


# --------------------------------------------------------------------------- IK


class IKFailureReason(str, Enum):
    NONE = "none"
    NO_CONVERGENCE = "no_convergence"
    JOINT_LIMIT_VIOLATION = "joint_limit_violation"
    SITE_NOT_FOUND = "site_not_found"


@dataclass
class IKResult:
    success: bool
    q: np.ndarray | None  # (n_joints,) solution, only set if success
    position_error: float
    orientation_error: float
    iterations: int
    reason: IKFailureReason = IKFailureReason.NONE


# --------------------------------------------------------------------------- collision


@dataclass
class Contact:
    geom1: str
    geom2: str
    distance: float  # negative = penetrating, positive = separated (margin-adjusted)


@dataclass
class CollisionResult:
    in_collision: bool
    contacts: list[Contact] = field(default_factory=list)
    min_clearance: float = float("inf")

    @property
    def is_valid(self) -> bool:
        return not self.in_collision


# --------------------------------------------------------------------------- planning


class PlannerFailureReason(str, Enum):
    NONE = "none"
    IK_FAILED = "ik_failed"
    START_IN_COLLISION = "start_in_collision"
    GOAL_IN_COLLISION = "goal_in_collision"
    NO_PATH_FOUND = "no_path_found"
    TIMEOUT = "timeout"
    APPROXIMATE_SOLUTION_ONLY = "approximate_solution_only"
    PLANNER_UNAVAILABLE = "planner_unavailable"


@dataclass
class MotionPlan:
    """A time-parameterized joint-space trajectory, or a structured failure."""

    success: bool
    joint_names: list[str] = field(default_factory=list)
    # Geometric path straight out of OMPL (post simplify+interpolate), joint positions only.
    geometric_path: np.ndarray | None = None  # (T, n_joints)
    # Time-parameterized reference: filled in by time_parameterize.py.
    times: np.ndarray | None = None  # (T,)
    positions: np.ndarray | None = None  # (T, n_joints)
    velocities: np.ndarray | None = None  # (T, n_joints)
    accelerations: np.ndarray | None = None  # (T, n_joints)
    planning_time_sec: float = 0.0
    failure_reason: PlannerFailureReason = PlannerFailureReason.NONE
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- MPC


@dataclass
class MPCCostConfig:
    """Modular MPC cost weights, loaded from configs/libero_template.json's motion.mpc block."""

    tracking_weight: float = 1.0
    terminal_weight: float = 4.0
    control_weight: float = 0.05
    smoothness_weight: float = 0.1
    joint_limit_weight: float = 2.0
    clearance_weight: float = 2.0
    stl_weight: float = 1.0
    timing_weight: float = 1.0


@dataclass
class MPCStepResult:
    """One receding-horizon MPC tick."""

    joint_command: np.ndarray  # (n_joints,) next joint position target
    tracking_error: float
    min_clearance: float
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    stl_robustness: dict[str, float] = field(default_factory=dict)
    replan_requested: bool = False
    replan_reason: str | None = None


# --------------------------------------------------------------------------- attached object


@dataclass
class AttachedObjectState:
    object_name: str
    grasp_transform_hand_from_object: np.ndarray  # (4,4), object pose expressed in hand frame
    collision_radius_m: float
    attached_at_step: int
    # Sphere approximation of the object's collision geometry, in the
    # OBJECT's own frame: [(local_offset_xyz, radius), ...].
    collision_spheres_object_frame: list[tuple[np.ndarray, float]] = field(default_factory=list)


# --------------------------------------------------------------------------- execution / skill


@dataclass
class ExecutionResult:
    success: bool
    steps_executed: int
    final_tracking_error: float
    replans: int
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillFailureReason(str, Enum):
    NONE = "none"
    NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"
    GRASP_VERIFICATION_FAILED = "grasp_verification_failed"
    EXECUTION_FAILED = "execution_failed"
    PLANNER_UNAVAILABLE = "planner_unavailable"


@dataclass
class SkillResult:
    """Typed result for the unified Pick (and future transport/place) skill."""

    success: bool
    selected_object: str | None = None
    selected_grasp: GraspCandidate | None = None
    planned_trajectory: MotionPlan | None = None
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    stl_robustness: dict[str, float] = field(default_factory=dict)
    failure_reason: SkillFailureReason = SkillFailureReason.NONE
    diagnostics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- candidate scoring


@dataclass
class CandidateScore:
    """Per-(object, grasp) score used to rank OR-set candidates. Lower total_cost wins;
    a candidate with feasible=False is never selected regardless of cost."""

    object_name: str
    grasp: GraspCandidate
    plan: MotionPlan | None
    feasible: bool
    total_cost: float
    breakdown: dict[str, float] = field(default_factory=dict)
    rejection_reason: str | None = None
