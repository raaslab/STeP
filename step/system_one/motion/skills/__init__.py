"""Individual manipulation skills (pick, place, open-drawer, ...) built on the
shared motion primitives one level up (collision.py, kinematics.py, mpc.py,
ompl_planner.py, scene_query.py, config.py, types.py, panda.py, servo.py,
time_parameterize.py, articulation.py, attached_object.py).

Each skill lives here as a (skill, generator, LiberoXxxAdapter) file group --
e.g. pick_skill.py + grasp_generator.py/contact_graspnet_generator.py +
libero_pick_adapter.py -- so a new skill (see turn_knob_skill.py) only ever
needs to add files here, never touch the shared primitives above.
"""
