"""Record a rollout in the shape `score.py` reads.

Any policy can be scored on LIBERO-Constrained -- the measures only need where the
gripper went, where the objects went, and whether the gripper was closed. This
writes those into the two files the scorer expects, so a VLA evaluation does not
have to reverse-engineer a format.

    from benchmarks.libero_constrained.record import RolloutRecorder

    rec = RolloutRecorder(out_dir, task)
    obs = env.set_init_state(init_states[task["init_state_id"]])
    for _ in range(max_steps):
        action = policy.select_action(obs, task["prompt"])
        obs, reward, done, info = env.step(action)
        rec.step(env, gripper_closed=bool(action[-1] > 0))
        if done:
            break
    rec.save(success=bool(done))
"""

import json
import pathlib

import numpy as np

# Bodies that are scenery rather than manipulable objects.
_SCENERY = ("world", "table", "robot", "panda", "gripper", "mount", "link", "floor", "wall")


class RolloutRecorder:
    """Collects per-step state and writes trajectory.npz + execution_summary.json.

    Object names and the end-effector site are discovered from the environment on
    the first step, so nothing has to be declared up front.
    """

    def __init__(self, out_dir, task: dict, ee_site_candidates=(
        "gripper0_grip_site", "gripper0_right_grip_site", "grip_site",
    )):
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.ee_site_candidates = ee_site_candidates
        self._names: list[str] | None = None
        self._bodies: dict[str, int] = {}
        self._ee: list[list[float]] = []
        self._obj: list[list[list[float]]] = []
        self._quat: list[list[list[float]]] = []
        self._grasped: list[float] = []
        self._over_goal: list[float] = []

    @staticmethod
    def _sim(env):
        return getattr(getattr(env, "env", None), "sim", None) or getattr(env, "sim", None)

    def _discover(self, sim):
        """Object names as the manifest spells them, mapped to their body ids.

        MuJoCo suffixes an object's root body -- akita_black_bowl_1 is the body
        akita_black_bowl_1_main -- while the manifest names objects without it.
        Left unstripped, every measure that looks an object up by name silently
        finds nothing and reports unscored.
        """
        by_name: dict[str, int] = {}
        for bid in range(1, int(sim.model.nbody)):
            raw = sim.model.body_id2name(bid)
            if not raw or any(s in raw for s in _SCENERY):
                continue
            name = raw
            for suffix in ("_main", "_base", "_root"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            # Keep the first body claiming a name: that is the object's root,
            # and later ones are its sub-geoms.
            by_name.setdefault(name, bid)
        return by_name

    def _ee_pos(self, sim):
        for site in self.ee_site_candidates:
            try:
                return np.asarray(sim.data.site_xpos[sim.model.site_name2id(site)], dtype=float)
            except Exception:
                continue
        return np.full(3, np.nan)

    def step(self, env, *, gripper_closed: bool, over_goal: bool = False):
        """Record one control step. Call once per env.step()."""
        sim = self._sim(env)
        if sim is None:
            raise RuntimeError("could not reach the MuJoCo sim from this env")
        if self._names is None:
            self._bodies = self._discover(sim)
            self._names = sorted(self._bodies)

        self._ee.append([float(v) for v in self._ee_pos(sim)])
        row, qrow = [], []
        for name in self._names:
            bid = self._bodies.get(name)
            if bid is None:
                row.append([float("nan")] * 3)
                qrow.append([float("nan")] * 4)
                continue
            row.append([float(v) for v in sim.data.body_xpos[bid][:3]])
            qrow.append([float(v) for v in sim.data.body_xquat[bid][:4]])
        self._obj.append(row)
        self._quat.append(qrow)
        self._grasped.append(1.0 if gripper_closed else 0.0)
        # over_goal is only needed by the dwell measure. A policy that cannot
        # report it leaves it false and dwell scores as not held, which is the
        # honest answer rather than a guess.
        self._over_goal.append(1.0 if over_goal else 0.0)

    def save(self, *, success: bool, model: str = "", extra: dict | None = None) -> None:
        if not self._ee:
            raise RuntimeError("no steps recorded")
        np.savez(
            self.out_dir / "trajectory.npz",
            eef_pos=np.asarray(self._ee, dtype=np.float32),
            object_pos=np.asarray(self._obj, dtype=np.float32),
            object_quat=np.asarray(self._quat, dtype=np.float32),
            object_names=np.asarray(self._names, dtype=object),
            grasped=np.asarray(self._grasped, dtype=np.float32),
            over_goal=np.asarray(self._over_goal, dtype=np.float32),
            # Present so the array set matches what the STeP runner writes;
            # nothing in the scorer reads it.
            cube_pos=np.asarray(self._ee, dtype=np.float32),
        )
        summary = {
            "success": bool(success),
            # A VLA has no subtasks. The measures that need them -- stage
            # ordering, and the wait between stages -- will report as unscored
            # rather than being guessed at from elapsed time.
            "records": [],
            "provenance": {"model": model, "benchmark_task": self.task["id"]},
        }
        if extra:
            summary.update(extra)
        (self.out_dir / "execution_summary.json").write_text(json.dumps(summary, indent=2))
