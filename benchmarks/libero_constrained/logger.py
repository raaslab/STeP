"""The benchmark's rollout log: what any policy records, so any policy can be scored.

Call `step()` once per control step and `save()` at the end. Everything is read
straight from the simulator, so the log does not depend on how the policy
works -- a VLA, a planner, or teleoperation writes the same file, and spec.py
scores them all the same way.

Per step it records:

  eef           gripper position (the grip site)
  gripper       finger opening, metres
  held          which object both fingers are touching (-1: none), from contacts
  pos, quat     every object and fixture: position, and orientation (wxyz)
  joints        every drawer, knob and door joint
  predicates    each of the task's BDDL goal conditions, judged by LIBERO itself

The last line is what makes event rules exact: "after the stove is on" is
LIBERO's own Turnon predicate turning true, not a threshold of ours.

    from benchmarks.libero_constrained.logger import BenchmarkLogger
    log = BenchmarkLogger(env)          # LIBERO's OffScreenRenderEnv
    for each step:
        env.step(action); log.step()
    log.save("benchmark_log.npz")
"""

import numpy as np

FORMAT = 2


def _domain(env):
    """LIBERO's problem instance -- the object that owns the BDDL state."""
    seen = env
    for _ in range(4):
        if hasattr(seen, "_eval_predicate") and hasattr(seen, "objects_dict"):
            return seen
        seen = getattr(seen, "env", None)
        if seen is None:
            break
    raise TypeError("not a LIBERO environment: no BDDL problem found")


class BenchmarkLogger:
    def __init__(self, env, hz=20, extra_predicates=()):
        self.d = _domain(env)
        self.sim = self.d.sim
        m = self.sim.model
        self.hz = hz

        self.names, self._bodies = [], []
        for name, obj in list(self.d.objects_dict.items()) + list(self.d.fixtures_dict.items()):
            try:
                self._bodies.append(m.body_name2id(obj.root_body))
                self.names.append(name)
            except Exception:
                continue
        self._root_of = self._map_bodies_to_roots()

        self.joint_names, self._joint_addr = [], []
        for j in range(m.njnt):
            name = m.joint_id2name(j) or ""
            if int(m.jnt_type[j]) in (2, 3) and not name.startswith(("robot", "gripper", "mount")):
                self.joint_names.append(name)
                self._joint_addr.append(int(m.jnt_qposadr[j]))

        self._finger_q = [int(m.jnt_qposadr[m.joint_name2id(n)])
                          for n in ("gripper0_finger_joint1", "gripper0_finger_joint2")]
        self._grip_site = m.site_name2id("gripper0_grip_site")
        self._left = self._subtree(m.body_name2id("gripper0_leftfinger"))
        self._right = self._subtree(m.body_name2id("gripper0_rightfinger"))

        goal = list(self.d.parsed_problem.get("goal_state", []))
        self.predicates = [list(p) for p in goal] + [list(p) for p in extra_predicates]
        self.predicate_names = [" ".join(str(x) for x in p) for p in self.predicates]

        self.rows = {k: [] for k in ("eef", "gripper", "held", "pos", "quat", "joints", "predicates")}

    def _subtree(self, root):
        m = self.sim.model
        out = {root}
        for b in range(m.nbody):
            p = b
            while p not in (0, root):
                p = int(m.body_parentid[p])
            if p == root:
                out.add(b)
        return out

    def _map_bodies_to_roots(self):
        """body id -> index into self.names, for every body under an object."""
        m = self.sim.model
        index = {b: i for i, b in enumerate(self._bodies)}
        out = {}
        for b in range(m.nbody):
            p = b
            while p not in index and p != 0:
                p = int(m.body_parentid[p])
            if p in index:
                out[b] = index[p]
        return out

    def _held(self):
        m, d = self.sim.model, self.sim.data
        left, right = set(), set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for finger, other in ((b1, b2), (b2, b1)):
                obj = self._root_of.get(other)
                if obj is None:
                    continue
                if finger in self._left:
                    left.add(obj)
                elif finger in self._right:
                    right.add(obj)
        both = left & right
        return min(both) if both else -1

    def step(self):
        d = self.sim.data
        r = self.rows
        r["eef"].append(np.array(d.site_xpos[self._grip_site], dtype=np.float32))
        r["gripper"].append(float(d.qpos[self._finger_q[0]] - d.qpos[self._finger_q[1]]))
        r["held"].append(self._held())
        r["pos"].append(np.array([d.body_xpos[b] for b in self._bodies], dtype=np.float32))
        r["quat"].append(np.array([d.body_xquat[b] for b in self._bodies], dtype=np.float32))
        r["joints"].append(np.array([d.qpos[a] for a in self._joint_addr], dtype=np.float32))
        vals = []
        for p in self.predicates:
            try:
                vals.append(bool(self.d._eval_predicate(p)))
            except Exception:
                vals.append(False)
        r["predicates"].append(np.array(vals, dtype=bool))

    def save(self, path):
        r = self.rows
        T = len(r["eef"])
        np.savez_compressed(
            path,
            format=FORMAT,
            hz=self.hz,
            object_names=np.array(self.names, dtype=object),
            object_pos=np.stack(r["pos"]) if T else np.zeros((0, len(self.names), 3), np.float32),
            object_quat_wxyz=np.stack(r["quat"]) if T else np.zeros((0, len(self.names), 4), np.float32),
            eef_pos=np.stack(r["eef"]) if T else np.zeros((0, 3), np.float32),
            gripper_width=np.array(r["gripper"], dtype=np.float32),
            held_index=np.array(r["held"], dtype=np.int16),
            joint_names=np.array(self.joint_names, dtype=object),
            joint_qpos=np.stack(r["joints"]) if T and self.joint_names else np.zeros((T, 0), np.float32),
            predicate_names=np.array(self.predicate_names, dtype=object),
            predicate_values=np.stack(r["predicates"]) if T and self.predicates else np.zeros((T, 0), bool),
        )
