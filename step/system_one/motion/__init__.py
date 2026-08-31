"""Grasp-generation -> IK -> OMPL RRTConnect -> time-parameterization -> MPC pipeline.

This subpackage is the LIBERO-only replacement motion-generation core described in
``STeP/step/system_one/motion/README.md``. It is additive: the legacy stage-machine
in ``base_env.py``/``nominal_mpc.py`` is untouched and remains the default backend.
Opt in via ``scene_context["platform"]["libero"]["motion_backend"] = "ompl_mpc"``.

Design references: Cap-X (https://github.com/capgym/cap-x, MIT License, Copyright (c)
2026 Max Fu). Where a specific function or numeric approach is adapted from Cap-X, the
docstring says so explicitly and reproduces the required attribution.
"""
