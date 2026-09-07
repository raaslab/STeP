"""Minimal STeP package.

This first pass keeps System 1 execution in the legacy tree and moves System 2
to direct STL-plan generation.
"""

import os

# mujoco reads MUJOCO_GL once, at its own import time, to pick a GL backend --
# must be set before the first `import mujoco` anywhere in the process, hence
# here (this module's own import always happens first). Default to EGL: the
# default (GLX, via an existing X connection) intermittently fails with a
# GLXMakeCurrent BadAccess when a *second* independent GL context is created
# in the same process (confirmed on this dev setup -- robosuite/LIBERO's own
# rendering already holds one such context). EGL creates its own headless
# context and doesn't hit this. setdefault: never overrides a value the user
# already exported.
os.environ.setdefault("MUJOCO_GL", "egl")

__all__ = ["__version__"]

__version__ = "0.1.0"
