"""Hardware drivers for the real-robot backend.

These wrap third-party SDKs (``ur_rtde``, Robotiq socket protocol) into small
classes that the real-robot environment can use without polluting the rest of
the codebase. All imports of the real SDKs are deferred to construction time
so that the sim backend keeps working without ur_rtde installed.
"""

from .robotiq_gripper import RobotiqGripper
from .ur3e_controller import UR3eController, UR3eMockController

__all__ = ["RobotiqGripper", "UR3eController", "UR3eMockController"]
