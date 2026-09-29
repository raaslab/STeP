"""RoboCasa backend for STeP's System 1.

Mirrors LiberoManipulationEnv against RoboCasa's kitchen environments. The
differences that actually matter:

* RoboCasa names manipulables generically (`obj`, `container`, `obj_container`),
  so every name lookup goes through the episode metadata's category map rather
  than through MuJoCo body names -- see robocasa_scene.object_categories.
* The robot is a PandaOmron, whose mobile base adds four joints beyond the seven
  the planner controls. The base is left parked: planning happens in the arm's
  7 DoF exactly as on LIBERO, so nothing in the motion stack changes.
* Success is RoboCasa's own `_check_success()` rather than a BDDL goal state.
"""
import os as _os, pathlib as _pathlib
# Repo-relative so this runs from any checkout. Override the dataset location
# with ROBOCASA_DATASETS; override the STeP root with STEP_ROOT.
_HERE = _pathlib.Path(__file__).resolve().parent
_STEP_ROOT = _os.environ.get("STEP_ROOT") or str(_HERE.parents[1])
_ROBOCASA_DATASETS = _os.environ.get("ROBOCASA_DATASETS") or str(
    _pathlib.Path(_STEP_ROOT).parent / "robocasa_ws" / "robocasa" / "datasets")
import sys
from typing import Any, Dict

sys.path.insert(0, _STEP_ROOT)
import numpy as np

from step.system_one.base_env import BaseManipulationEnv


class RoboCasaManipulationEnv(BaseManipulationEnv):
    def __init__(self, *, scene_context: dict[str, Any], env, horizon: int | None = None,
                 camera_name: str = "robot0_frontview",
                 camera_height: int = 480, camera_width: int = 640, **kwargs):
        self._episode_terminated = False
        self._last_obs: Dict[str, Any] = {}
        self.scene_context = scene_context
        self.env = env
        self.camera_name = camera_name
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self._active_handle = "obj"
        platform = scene_context.get("platform", {}) or {}
        # LIBERO reads this from mujoco_arena.table_offset; RoboCasa's KitchenArena
        # has no such attribute, so it silently stayed at BaseManipulationEnv's
        # 0.8 default while a kitchen counter sits at ~0.95. Everything derived
        # from it is then wrong: carry_z = table_z + carry_height put the carry
        # height BELOW the counter surface, so the arm dragged along it instead
        # of lifting over.
        super().__init__(
            workspace_bounds=platform.get("workspace_bounds"),
            table_z=self._surface_z(),
            **kwargs,
        )

    def _surface_z(self) -> float:
        """Height of the surface the task's objects rest on."""
        zs = []
        for handle in ("obj_container", "container", "obj"):
            bid = getattr(self.env, "obj_body_id", {}).get(handle)
            if bid is None:
                continue
            pos = self.env.sim.data.body_xpos[bid]
            try:
                ext = self.env.sim.model.body(bid).name
                from step.system_one.motion.scene_query import SceneQuery
                sc = SceneQuery.from_robosuite_sim(self.env.sim)
                half = sc.body_collision_extent_world(ext).half_extents[2]
            except Exception:
                half = 0.0
            zs.append(float(pos[2]) - float(half))
        return float(min(zs)) if zs else 0.8

    # ------------------------------------------------------------- backend
    def _reset_backend(self) -> Dict[str, Any]:
        obs = self.env.reset()
        self._episode_terminated = False
        self._last_obs = obs
        self._refresh_scene_objects(obs)
        # RoboCasa resamples the manipulated object on every reset -- the same
        # task class gives an eggplant one episode and a hot dog the next, and
        # the episode language changes with it. Any scene context built before
        # this call is therefore stale: its entity names, categories and
        # positions all describe the previous episode. Rebuild it here so
        # System 2 is prompted about the scene that actually exists.
        self.refresh_scene_context()
        return obs

    def refresh_scene_context(self) -> dict:
        """Rebuild the scene context from the live episode (see _reset_backend)."""
        from robocasa_scene import build_scene_context
        rc = (self.scene_context.get("platform", {}) or {}).get("robocasa", {}) or {}
        fresh = build_scene_context(self.env,
                                    task_name=rc.get("task", ""),
                                    layout_id=rc.get("layout_id"),
                                    style_id=rc.get("style_id"))
        # Preserve anything the task layered on top (the keep-out zone and the
        # enforcement switch), which the generic builder knows nothing about.
        for key in ("geometry", "entities"):
            if key == "geometry":
                for name, g in (self.scene_context.get("geometry") or {}).items():
                    if isinstance(g, dict) and "sdf_geometry" in g:
                        fresh["geometry"][name] = g
            else:
                keep = [e for e in (self.scene_context.get("entities") or [])
                        if e.get("role") == "keepout_region"]
                fresh["entities"].extend(keep)
        # The constraint clause is episode-independent; the sentence it forms is
        # not, because the object changed. Carry the clause across and rebuild.
        old_rc = (self.scene_context.get("platform") or {}).get("robocasa", {}) or {}
        if old_rc.get("obj_groups"):
            fresh["platform"]["robocasa"]["obj_groups"] = old_rc["obj_groups"]
        if old_rc.get("constraint_clause"):
            fresh["platform"]["robocasa"]["constraint_clause"] = old_rc["constraint_clause"]
            from constrained_task import constrained_language
            fresh["platform"]["robocasa"]["constrained_language"] = constrained_language(fresh)
        if "libero_experiment" in (self.scene_context.get("platform") or {}):
            fresh["platform"]["libero_experiment"] = self.scene_context["platform"]["libero_experiment"]
        self.scene_context = fresh
        return fresh

    def _base_yaw_rotation(self):
        """World->base yaw rotation for this robot's mobile base.

        RoboCasa's arm controller declares ``input_ref_frame: "base"``, so the
        Cartesian deltas it receives are interpreted in the BASE frame. STeP's
        servo code computes them in the WORLD frame, which is identical for
        LIBERO (fixed base, zero yaw) but not here: the PandaOmron parks facing
        whatever fixture the task references, measured at 90 degrees for the
        stove tasks. Sending world deltas unrotated makes every commanded +x
        come out as +y, which is exactly the wandering trajectory it produced.
        """
        try:
            bid = self.env.sim.model.body("mobilebase0_base").id
        except Exception:
            return None
        w, x, y, z = self.env.sim.data.body_xquat[bid]
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        if abs(yaw) < 1e-6:
            return None
        c, s_ = np.cos(-yaw), np.sin(-yaw)
        return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])

    def _step_backend(self, full_action: np.ndarray):
        action = np.asarray(full_action, dtype=np.float32).reshape(-1)
        if action.size > 0:
            self._last_gripper_cmd = float(action[-1])
        # Rotate the world-frame position/orientation deltas into the base frame
        # the controller actually expects (see _base_yaw_rotation).
        R = self._base_yaw_rotation()
        if R is not None and action.size >= 6:
            action = action.copy()
            action[0:3] = (R @ action[0:3]).astype(np.float32)
            action[3:6] = (R @ action[3:6]).astype(np.float32)
        # PandaOmron's composite controller expects the mobile base's DOFs too.
        # The planner only ever commands the arm, so pad with zeros rather than
        # letting robosuite reject a short action -- a parked base is exactly
        # the intended behaviour, not a silent truncation.
        need = int(self.env.action_dim)
        if action.size < need:
            action = np.concatenate([action, np.zeros(need - action.size, dtype=np.float32)])
        elif action.size > need:
            action = action[:need]
        if self._episode_terminated or bool(getattr(self.env, "done", False)):
            self._episode_terminated = True
            return self._last_obs, 0.0, True, {"episode_already_terminated": True}
        obs, reward, done, info = self.env.step(action)
        self._episode_terminated = bool(done)
        self._last_obs = obs
        self._refresh_scene_objects(obs)
        return obs, reward, done, info

    # --------------------------------------------------------- observation
    def _refresh_scene_objects(self, obs: Dict[str, Any]) -> None:
        for handle in getattr(self.env, "obj_body_id", {}):
            p = self._body_xpos(handle)
            if p is not None:
                self.scene_objects[handle] = p

    def _body_xpos(self, handle: str):
        bid = getattr(self.env, "obj_body_id", {}).get(handle)
        if bid is None:
            return None
        return np.asarray(self.env.sim.data.body_xpos[bid], dtype=np.float32).copy()

    def _eef_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        sid = self.env.sim.model.site("gripper0_right_grip_site").id
        return np.asarray(self.env.sim.data.site_xpos[sid], dtype=np.float32).copy()

    def _eef_rotmat(self, obs: Dict[str, Any]) -> np.ndarray:
        sid = self.env.sim.model.site("gripper0_right_grip_site").id
        return np.asarray(self.env.sim.data.site_xmat[sid], dtype=np.float32).reshape(3, 3).copy()

    def _cube_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        p = self._body_xpos(self._active_handle)
        return p if p is not None else np.zeros(3, dtype=np.float32)

    def _object_pos_by_name(self, object_name: str | None):
        if object_name is None:
            return None
        if object_name in getattr(self.env, "obj_body_id", {}):
            return self._body_xpos(object_name)
        # Categories ("eggplant") and fixtures ("stove_main_group") both reach
        # here, because that is how System 2 refers to things.
        from robocasa_scene import object_categories
        for handle, cat in object_categories(self.env).items():
            if object_name in (cat, cat.replace("_", " ")):
                return self._body_xpos(handle)
        # Fixtures (stove, sink, cabinets) are not in obj_body_id at all; they
        # carry their own pose. Constraints refer to them constantly, so they
        # have to resolve here too.
        fx = (getattr(self.env, "fixtures", {}) or {}).get(object_name)
        if fx is not None and getattr(fx, "pos", None) is not None:
            return np.asarray(fx.pos, dtype=np.float32).copy()
        try:
            bid = self.env.sim.model.body(object_name).id
            return np.asarray(self.env.sim.data.body_xpos[bid], dtype=np.float32).copy()
        except Exception:
            return super()._object_pos_by_name(object_name)

    def _active_pick_object(self):
        return self.env.objects.get(self._active_handle)

    def set_active_object(self, name: str | None) -> None:
        if not name:
            return
        if name in getattr(self.env, "obj_body_id", {}):
            self._active_handle = name
            return
        from robocasa_scene import object_categories
        for handle, cat in object_categories(self.env).items():
            if name in (cat, cat.replace("_", " ")):
                self._active_handle = handle
                return

    def _is_grasped(self) -> bool:
        try:
            obj = self._active_pick_object()
            if obj is not None:
                gripper = self.env.robots[0].gripper
                if isinstance(gripper, dict):
                    gripper = gripper.get("right", next(iter(gripper.values())))
                important = getattr(gripper, "important_geoms", {})
                if "left_finger" in important and "right_finger" in important:
                    arg = [important["left_finger"], important["right_finger"]]
                else:
                    arg = gripper
                return bool(self.env._check_grasp(gripper=arg, object_geoms=obj))
        except Exception:
            pass
        return False

    def _is_success(self) -> bool:
        try:
            return bool(self.env._check_success())
        except Exception:
            return False

    # ------------------------------------------------------------- render
    def render_rgb_frame(self, camera_name: str = None, width: int = 640, height: int = 480):
        # Callers pass LIBERO's camera names ("agentview"), which do not exist in
        # a RoboCasa kitchen -- robosuite raises rather than falling back, which
        # killed a whole rollout the moment video recording was switched on.
        # Resolve to a camera this model actually has.
        name = camera_name or self.camera_name
        if name not in self._camera_names():
            name = self.camera_name
        return self.env.sim.render(width=width, height=height, camera_name=name)[::-1]

    def _camera_names(self) -> set:
        m = self.env.sim.model
        return {m.camera(i).name for i in range(m.ncam)}

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def make_robocasa_env(*, scene_context, runtime, render=False, env_horizon=None,
                      debug_dir=None, **_ignored):
    """Factory registered with step.system_one.env_factory.register_backend.

    Builds the RoboCasa kitchen env declared in the scene context and wraps it
    in the adapter. Registered rather than imported by env_factory because
    RoboCasa needs robosuite 1.5 while LIBERO pins 1.4 -- the two cannot share a
    process, so the caller that has the right environment registers its own.
    """
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    import robocasa  # noqa: F401  (registers the kitchen environments)

    rc = (scene_context.get("platform", {}) or {}).get("robocasa", {}) or {}
    task = rc.get("task") or "PickPlaceCounterToStove"
    robot = rc.get("robot", "PandaOmron")
    cam = rc.get("camera_name", "robot0_frontview")
    cfg = load_composite_controller_config(robot=robot)
    raw = robosuite.make(
        env_name=task, robots=robot, controller_configs=cfg,
        has_renderer=bool(render), has_offscreen_renderer=True,
        use_camera_obs=False, camera_names=[cam],
        camera_heights=int(rc.get("camera_height", 480)),
        camera_widths=int(rc.get("camera_width", 640)),
        layout_ids=[rc["layout_id"]] if rc.get("layout_id") else None,
        style_ids=[rc["style_id"]] if rc.get("style_id") else None,
        # Rebuild the SAME episode the scene context describes. Without this the
        # executor sampled a fresh one: the plan named the eggplant the context
        # saw, and the arm was handed a garlic, or a lemon wedge a metre further
        # along the counter. Every rollout before this was planning for one scene
        # and executing in another.
        **({"seed": rc["seed"]} if rc.get("seed") is not None else {}),
        horizon=int(env_horizon) if env_horizon else 1000,
        **({"obj_groups": rc["obj_groups"]} if rc.get("obj_groups") else {}),
    )
    return RoboCasaManipulationEnv(scene_context=scene_context, env=raw,
                                   camera_name=cam)
