"""The benchmark's constraint types, each as an STL template.

Every task's rule is built from one of these, so the set of constraint types is
a short list a reader can check, and each one is a recognised requirement in
robot manipulation:

  keep-out     stay a distance from something (a person, a hot surface)
  no-contact   leave an object exactly where it stands
  upright      keep a carried object level (anything with liquid in it)
  deadline     finish a step within a time budget
  dwell        hold still over the destination before letting go
  wait         after one event, keep something still for a set time
  order        one thing must happen before another
  condition    a step may begin only once something is true
  placement    end with the object at a given offset or on a given side

Rules read only logged motion -- positions, orientations, joint positions and
the grasp flag -- so they score any policy, not just one that keeps STeP's
subtask records. "Lifted" is defined from the object's height, not from the
grasp flag, for the same reason.
"""

# An object counts as lifted once its centre is this far above where it began;
# below that, it is still being picked up or set down.
LIFTED_M = 0.02
# "At" a destination, in the plane. Containers in LIBERO measure 15-20 cm across.
ARRIVAL_M = 0.10
# Still enough to count as holding steady.
STEADY_MPS = 0.03
# Moved, for an object that is supposed to stay put.
MOVED_M = 0.01


def P(sig, op, c):
    return {"pred": [sig, op, c]}


def AND(*fs):
    return {"and": list(fs)}


def lifted(obj):
    return P({"height": obj}, ">=", LIFTED_M)


def at(obj, dest, radius=ARRIVAL_M):
    return P({"dist": [obj, dest]}, "<=", radius)


def moved(obj, tol=MOVED_M):
    return P({"disp": obj}, ">=", tol)


# ---------------------------------------------------------------- templates

def keepout(objs, region, distance):
    """G( dist(o, region) >= d ) for every o -- typically the gripper and the
    carried object. Measured against where the region started."""
    return {"G": AND(*[P({"dist": [o, region], "b_at": "start"}, ">=", distance) for o in objs])}


def no_contact(obj, move_tol=MOVED_M, tilt_deg=10.0):
    return {"G": AND(P({"disp": obj}, "<=", move_tol), P({"tilt": obj}, "<=", tilt_deg))}


def margin(carried, obstacle, distance):
    return AND(keepout([carried], obstacle, distance), no_contact(obstacle))


def upright(obj, max_tilt_deg):
    return {"G": {"implies": [lifted(obj), P({"tilt": obj}, "<=", max_tilt_deg)]}}


def deadline(obj, dest, seconds):
    return {"F": AND(at(obj, dest), {"not": lifted(obj)}), "in": [0, seconds]}


def dwell(obj, dest, seconds):
    """Eventually: for `seconds`, the object is lifted, over the destination,
    and steady -- held still before it is let go."""
    steady = AND(lifted(obj), at(obj, dest), P({"speed": obj}, "<=", STEADY_MPS))
    return {"F": {"G": steady, "in": [0, seconds]}}


def order(first, second, dest):
    """`second` does not reach dest until `first` has."""
    return {"U": [{"not": at(second, dest)}, at(first, dest)]}


def precondition_empty(container, target, radius=ARRIVAL_M):
    """When `target` first arrives, nothing else is in `container`."""
    empty = P({"count_near": container, "radius": radius, "exclude": [target]}, "<=", 0.5)
    return {"U": [{"not": at(target, container, radius)}, AND(at(target, container, radius), empty)]}


def untouched(obj, tol):
    return {"G": P({"disp": obj}, "<=", tol)}


def wait_after(event, still_obj, seconds):
    """`still_obj` stays put until `event`, and for `seconds` after it."""
    hold = {"G": {"not": moved(still_obj)}, "in": [0, seconds]}
    return {"U": [{"not": moved(still_obj)}, AND(event, hold)]}


def gate(event, forbidden):
    """`forbidden` does not happen until `event` has."""
    return {"U": [{"not": forbidden}, event]}


def joint_at_least(name, q):
    return P({"joint": name}, ">=", q)


def placement_offset(target, ref, axis, want, tol):
    rel = {"rel": [target, ref], "axis": axis}
    return {"end": AND(P(rel, ">=", want - tol), P(rel, "<=", want + tol))}


def placement_side(target, ref, axis, sign, tol):
    rel = {"rel": [target, ref], "axis": axis}
    return {"end": P(rel, ">=", tol) if sign > 0 else P(rel, "<=", -tol)}


# ---------------------------------------------------------------- v1 scoring -> spec

def from_v1_scoring(task, manifest_tasks):
    """The formal rule for a task scored by one of v1's hand-written scorers,
    or None where the old scorer needs a signal the trajectory does not log
    (articulated joints, for wait and stage) -- those wait on joint logging."""
    s = task["scoring"]
    t = s["type"]
    if t == "obstacle":
        return no_contact(s["object"], s.get("move_tol_m", MOVED_M), s.get("tilt_tol_deg", 10.0))
    if t == "margin":
        return AND(keepout([s["object"]], s["obstacle"], s["margin_m"]),
                   no_contact(s["obstacle"], s.get("move_tol_m", MOVED_M), s.get("tilt_tol_deg", 10.0)))
    if t == "untouched":
        return untouched(s["object"], s.get("tolerance_m", 0.02))
    if t == "order":
        return order(s["expected_first"], s["expected_second"], s["destination"])
    if t == "precondition" and s.get("container"):
        return precondition_empty(s["container"], s["target"])
    if t == "placement_offset":
        return placement_offset(s["target"], s["reference"], s["axis"], s["sign"] * s["distance_m"], s["tolerance_m"])
    if t == "placement_relative":
        return placement_side(s["target"], s["reference"], s["axis"], s["sign"], s.get("tolerance_m", 0.02))
    if t == "dwell" and s.get("object") and s.get("destination"):
        return dwell(s["object"], s["destination"], s["seconds"])
    if t == "combo":
        parts = [from_v1_scoring(manifest_tasks[p], manifest_tasks) for p in s["parts"]]
        return None if any(p is None for p in parts) else AND(*parts)
    return None
