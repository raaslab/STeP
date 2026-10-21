"""Shared name constants for the Panda + parallel-jaw gripper model used by
every LIBERO task in this repo (robosuite's default Panda). Centralized here
so tests/conftest.py and the executor-facing adapter never drift apart.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES = [f"robot0_joint{i}" for i in range(1, 8)]
# The gripper's own open/close DOF -- deliberately not part of JOINT_NAMES or
# the planned state space (IK/OMPL/MPC only ever reason about the 7 arm joints).
FINGER_JOINT_NAMES = ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]
# Both joints' MJCF ranges are [0, 0.04] and [-0.04, 0] respectively -- 0.0 is
# "closed" for both.
FINGER_CLOSED_QPOS = np.array([0.0, 0.0])
ROBOT_BODIES = ["robot0_base"] + [f"robot0_link{i}" for i in range(8)] + ["robot0_right_hand"]
GRIPPER_BODIES = [
    "gripper0_right_right_gripper",
    "gripper0_right_eef",
    "gripper0_right_leftfinger",
    "gripper0_right_finger_joint1_tip",
    "gripper0_right_rightfinger",
    "gripper0_right_finger_joint2_tip",
]
EE_SITE = "gripper0_right_grip_site"
HAND_SITE = "gripper0_right_ft_frame"
