"""One scorer for every constraint: a task's STL rule, run on the logged motion.

Each task in the manifest carries `spec`, a formal rule over signals that any
policy can log -- where the gripper and the objects were, how they were turned,
whether something was held -- sampled at the control rate. Scoring a rollout is
evaluating that rule on its trajectory. There is no per-task scoring code, and
nothing here knows how the trajectory was produced: a VLA, a planner, or a
person on a spacemouse is scored the same way.

The result is STL robustness, not just pass/fail. Positive means the rule held,
negative means it broke, and the size says by how much. Each predicate is
divided by the unit of its signal (UNITS), so the number reads in centimetres,
degrees, cm/s or seconds rather than mixing metres with degrees.

Formula, as JSON:

  {"pred": [SIGNAL, ">=" | "<=", c]}
  {"not": F}   {"and": [F, ...]}   {"or": [F, ...]}   {"implies": [F, G]}
  {"G": F, "in": [a, b]}           always, over seconds a..b from now (default: all)
  {"F": F, "in": [a, b]}           eventually
  {"U": [F, G], "in": [a, b]}      F holds until G does
  {"end": F}                       F at the last step -- for final-state rules

Signals:

  {"dist": [A, B], "plane": "xy"|"xyz", "b_at": "now"|"start"}
  {"disp": A}                 how far A has moved from where it started (3D)
  {"height": A}               A's height above where it started
  {"tilt": A}                 degrees A has tipped from how it started
  {"speed": A}                A's speed, m/s
  {"rel": [A, B], "axis": "x"|"y"|"z"}   A minus B along one axis
  {"held": true | A}          1 while the gripper holds something (or A), else 0
  {"bddl": "turnon flat_stove_1"}  1 while that LIBERO goal predicate is true
  {"joint": NAME}             an articulated joint's position (knob, drawer, door)
  {"count_near": C, "radius": r, "exclude": [A, ...]}   objects within r of C, in xy

  A or B is an object name as logged, or "eef" for the gripper.
"""

import math

import numpy as np

# Robustness units, so a margin reads in units a person thinks in.
UNITS = {
    "dist": 0.01,      # cm
    "disp": 0.01,      # cm
    "height": 0.01,    # cm
    "rel": 0.01,       # cm
    "tilt": 1.0,       # degrees
    "speed": 0.01,     # cm/s
    "joint": 0.01,
    # Yes/no and counting signals only switch a rule on or off ("while held,
    # ..."). A tiny unit makes their margin huge, so in max/min they decide
    # which branch counts and never stand in for the margin of the measured
    # quantity -- otherwise "held" (margin 0.5) would mask a speed overrun of
    # 10 cm/s inside an implication.
    "held": 1e-4,
    "bddl": 1e-4,
    "count_near": 1e-4,
}

# Speed is a finite difference of positions logged at the control rate, which
# is noisy step to step; average over a short window so one jittery sample is
# not a "speeding" violation.
SPEED_WINDOW_S = 0.25


class Unscorable(Exception):
    """The trajectory lacks a signal the rule needs."""


# ---------------------------------------------------------------- trajectory

class Trajectory:
    """The logged rollout, with the signals the rules are written over.

    Reads the benchmark's own log (logger.py, benchmark_log.npz) when there is
    one, and otherwise STeP's older trajectory.npz, which lacks fixtures, the
    contact-based grasp and LIBERO's predicates.
    """

    def __init__(self, npz, hz):
        files = set(npz.files)
        self.hz = float(npz["hz"]) if "hz" in files else float(hz)
        self.names = [str(n) for n in npz["object_names"]]
        self.pos = np.asarray(npz["object_pos"], dtype=float)
        if "object_quat_wxyz" in files:
            self.quat = np.asarray(npz["object_quat_wxyz"], dtype=float)
        elif "object_quat" in files:
            # robosuite observations carry quaternions as xyzw
            self.quat = np.asarray(npz["object_quat"], dtype=float)[..., [3, 0, 1, 2]]
        else:
            self.quat = None
        self.eef = np.asarray(npz["eef_pos"], dtype=float)
        if "held_index" in files:
            self.held_idx = np.asarray(npz["held_index"], dtype=int)
            self.held_raw = (self.held_idx >= 0).astype(float)
        else:
            self.held_idx = None
            self.held_raw = np.asarray(npz["grasped"], dtype=float) if "grasped" in files else None
        self.joint_names = [str(n) for n in npz["joint_names"]] if "joint_names" in files else []
        self.joints = np.asarray(npz["joint_qpos"], dtype=float) if "joint_qpos" in files else None
        if self.joints is not None and self.joints.size == 0:
            self.joints = None
        self.pred_names = [str(n) for n in npz["predicate_names"]] if "predicate_names" in files else []
        self.preds = (np.asarray(npz["predicate_values"], dtype=float)
                      if "predicate_values" in files and len(self.pred_names) else None)
        # The logs are appended side by side each step but by separate code, so
        # trim to the shortest rather than let one dropped row shift the rest.
        series = [self.eef, self.pos] + [a for a in (self.quat, self.held_raw, self.held_idx,
                                                     self.joints, self.preds) if a is not None]
        self.T = min(len(a) for a in series)
        for attr in ("eef", "pos", "quat", "held_raw", "held_idx", "joints", "preds"):
            a = getattr(self, attr)
            if a is not None:
                setattr(self, attr, a[:self.T])

    @classmethod
    def load(cls, path, hz=20):
        """A log file, or a run directory (benchmark log preferred)."""
        import pathlib
        path = pathlib.Path(path)
        if path.is_dir():
            for name in ("benchmark_log.npz", "trajectory.npz"):
                if (path / name).exists():
                    path = path / name
                    break
            else:
                raise Unscorable(f"no rollout log in {path}")
        return cls(np.load(path, allow_pickle=True), hz)

    def position(self, name):
        if name == "eef":
            return _ffill(self.eef)
        if name not in self.names:
            raise Unscorable(f"{name} was not logged")
        return _ffill(self.pos[:, self.names.index(name), :])

    def orientation(self, name):
        if self.quat is None or name not in self.names:
            raise Unscorable(f"no orientation logged for {name}")
        return _ffill(self.quat[:, self.names.index(name), :])


def _ffill(a):
    """Carry the last good sample over gaps; a signal that is never good is unscorable."""
    a = np.array(a, dtype=float)
    ok = np.isfinite(a).all(axis=-1) if a.ndim > 1 else np.isfinite(a)
    if not ok.any():
        raise Unscorable("signal has no finite samples")
    idx = np.where(ok, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(ok))
    idx[:first] = first
    return a[idx]


def _quat_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------- signals

def signal(traj, s):
    """(T,) values of one signal."""
    kind = next(iter(k for k in s if k in UNITS))
    if kind == "dist":
        a, b = s["dist"]
        pa, pb = traj.position(a), traj.position(b)
        if s.get("b_at") == "start":
            pb = np.broadcast_to(pb[0], pb.shape)
        k = 2 if s.get("plane", "xy") == "xy" else 3
        return np.linalg.norm(pa[:, :k] - pb[:, :k], axis=1)
    if kind == "disp":
        p = traj.position(s["disp"])
        return np.linalg.norm(p - p[0], axis=1)
    if kind == "height":
        p = traj.position(s["height"])
        return p[:, 2] - p[0, 2]
    if kind == "tilt":
        q = traj.orientation(s["tilt"])
        up = _quat_mat(q[0]).T @ np.array([0.0, 0.0, 1.0])
        z = np.array([(_quat_mat(qi) @ up)[2] for qi in q])
        return np.degrees(np.arccos(np.clip(z, -1.0, 1.0)))
    if kind == "speed":
        p = traj.position(s["speed"])
        w = max(1, int(round(SPEED_WINDOW_S * traj.hz)))
        v = np.zeros(len(p))
        if len(p) > w:
            v[w:] = np.linalg.norm(p[w:] - p[:-w], axis=1) * traj.hz / w
            v[:w] = v[w]
        return v
    if kind == "rel":
        a, b = s["rel"]
        ax = "xyz".index(s["axis"])
        return traj.position(a)[:, ax] - traj.position(b)[:, ax]
    if kind == "held":
        who = s["held"]
        if who is True:
            if traj.held_raw is None:
                raise Unscorable("grasp was not logged")
            return np.nan_to_num(traj.held_raw)
        if traj.held_idx is None:
            raise Unscorable("which object is held was not logged (needs the benchmark log)")
        if who not in traj.names:
            raise Unscorable(f"{who} was not logged")
        return (traj.held_idx == traj.names.index(who)).astype(float)
    if kind == "bddl":
        if traj.preds is None or s["bddl"] not in traj.pred_names:
            raise Unscorable(f"LIBERO predicate '{s['bddl']}' was not logged")
        return traj.preds[:, traj.pred_names.index(s["bddl"])]
    if kind == "joint":
        if traj.joints is None or s["joint"] not in traj.joint_names:
            raise Unscorable(f"joint {s['joint']} was not logged")
        return _ffill(traj.joints[:, traj.joint_names.index(s["joint"])])
    if kind == "count_near":
        c = traj.position(s["count_near"])
        skip = set(s.get("exclude", [])) | {s["count_near"]}
        n = np.zeros(traj.T)
        for name in traj.names:
            if name in skip:
                continue
            try:
                p = traj.position(name)
            except Unscorable:
                continue
            n += np.linalg.norm(p[:, :2] - c[:, :2], axis=1) <= float(s["radius"])
        return n
    raise ValueError(f"unknown signal {s}")


# ---------------------------------------------------------------- robustness

def _window(traj, f):
    """[a, b] seconds -> step offsets; no interval means the rest of the run."""
    if "in" not in f:
        return 0, traj.T
    a, b = f["in"]
    return int(round(a * traj.hz)), int(round(b * traj.hz)) + 1


def _sliding(x, lo, hi, reduce):
    """out[t] = reduce(x[t+lo : t+hi]); windows past the end are cut short,
    and a window that starts past the end sees the last value."""
    T = len(x)
    out = np.empty(T)
    for t in range(T):
        a, b = min(t + lo, T - 1), min(t + hi, T)
        out[t] = reduce(x[a:max(b, a + 1)])
    return out


def robustness(traj, f):
    """(T,) robustness of formula f at every step."""
    if "pred" in f:
        sig, op, c = f["pred"]
        kind = next(iter(k for k in sig if k in UNITS))
        x = signal(traj, sig)
        r = (x - float(c)) if op == ">=" else (float(c) - x)
        return r / UNITS[kind]
    if "not" in f:
        return -robustness(traj, f["not"])
    if "and" in f:
        return np.min([robustness(traj, g) for g in f["and"]], axis=0)
    if "or" in f:
        return np.max([robustness(traj, g) for g in f["or"]], axis=0)
    if "implies" in f:
        a, b = f["implies"]
        return np.maximum(-robustness(traj, a), robustness(traj, b))
    if "G" in f:
        lo, hi = _window(traj, f)
        return _sliding(robustness(traj, f["G"]), lo, hi, np.min)
    if "F" in f:
        lo, hi = _window(traj, f)
        return _sliding(robustness(traj, f["F"]), lo, hi, np.max)
    if "U" in f:
        left, right = (robustness(traj, g) for g in f["U"])
        lo, hi = _window(traj, f)
        T = traj.T
        out = np.empty(T)
        for t in range(T):
            # min of `left` over [t, t'), for every t' from t on
            held = np.minimum.accumulate(left[t:])
            before = np.concatenate([[math.inf], held[:-1]])
            cand = np.minimum(right[t:], before)
            a, b = min(lo, len(cand) - 1), min(hi, len(cand))
            out[t] = cand[a:max(b, a + 1)].max()
        return out
    if "end" in f:
        r = robustness(traj, f["end"])
        return np.full(traj.T, r[-1])
    raise ValueError(f"unknown formula {f}")


def evaluate(traj, spec):
    """Robustness of the whole rule, read at the start of the run."""
    return float(robustness(traj, spec)[0])


def score(traj, spec):
    """{"held": bool|None, "robustness": float|None, "detail": str}."""
    try:
        r = evaluate(traj, spec)
    except Unscorable as exc:
        return {"held": None, "robustness": None, "detail": str(exc)}
    return {"held": r >= 0.0, "robustness": round(r, 2), "detail": f"robustness {r:+.2f}"}
