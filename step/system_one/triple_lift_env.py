"""Three-cube Lift variant: same task geometry as Lift, three pickable blocks (registered as ``TriplePlaceLift``)."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class TriplePlaceLift(Lift):
    """Table-top scene with three small cubes (same size family as Lift)."""

    def _load_model(self):
        super(Lift, self)._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.cube0 = BoxObject(
            name="cube0",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[1, 0, 0, 1],
            material=redwood,
            rng=self.rng,
        )
        self.cube1 = BoxObject(
            name="cube1",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[0, 1, 0, 1],
            material=greenwood,
            rng=self.rng,
        )
        self.cube2 = BoxObject(
            name="cube2",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[0, 0.35, 1, 1],
            material=bluewood,
            rng=self.rng,
        )
        self.cubes = [self.cube0, self.cube1, self.cube2]

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cubes,
                # Wider spawn area avoids intermittent placement failures for 3 cubes.
                x_range=[-0.15, 0.0],
                y_range=[-0.15, 0.15],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cubes,
        )

    def _setup_references(self):
        super(Lift, self)._setup_references()
        self.cube_body_ids = [self.sim.model.body_name2id(c.root_body) for c in self.cubes]
        self.cube = self.cubes[0]
        self.cube_body_id = self.cube_body_ids[0]

    def _setup_observables(self):
        observables: OrderedDict = super(Lift, self)._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cube_pos(obs_cache):
                idx = int(getattr(self, "_stl_active_block_idx", 0))
                idx = max(0, min(idx, len(self.cube_body_ids) - 1))
                return np.array(self.sim.data.body_xpos[self.cube_body_ids[idx]])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                idx = int(getattr(self, "_stl_active_block_idx", 0))
                idx = max(0, min(idx, len(self.cube_body_ids) - 1))
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_ids[idx]]), to="xyzw")

            sensors = [cube_pos, cube_quat]
            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors += [
                self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False

    def visualize(self, vis_settings):
        super(Lift, self).visualize(vis_settings=vis_settings)
        if vis_settings.get("grippers"):
            idx = int(getattr(self, "_stl_active_block_idx", 0))
            idx = max(0, min(idx, len(self.cubes) - 1))
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubes[idx])
