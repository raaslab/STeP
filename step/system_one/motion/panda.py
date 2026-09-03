"""Shared name constants for the Panda + parallel-jaw gripper model used by
every LIBERO task in this repo (robosuite's default Panda). Centralized here
so ``tests/conftest.py`` and the executor-facing adapter never drift apart.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES = [f"robot0_joint{i}" for i in range(1, 8)]
# The gripper's own open/close DOF -- deliberately *not* part of
# `JOINT_NAMES`/the planned state space (IK, OMPL, and the MPC all only ever
# reason about the 7 arm joints). That means neither planner ever simulates
# what the *closed* finger geometry looks like at a candidate grasp
# configuration -- see CollisionChecker.check_gripper_closure, which exists
# specifically to validate this separately, since it's a real, previously
# unchecked gap (a configuration whose *open*-finger geometry clears the
# target can still have its *closed* fingers -- particularly the rigid
# finger segment, which reaches further than the pad, see
# scene_query.finger_reach_below_tcp_m -- collide with something the open
# fingers swept past cleanly).
FINGER_JOINT_NAMES = ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]
# Both joints' MJCF ranges are [0, 0.04] and [-0.04, 0] respectively (measured
# directly against the loaded model) -- 0.0 is "closed" for both.
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
