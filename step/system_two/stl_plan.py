from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Formula = dict[str, Any]


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]


def validate_stl_plan(plan: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    system2_plan = plan.get("system2_plan")
    if not isinstance(system2_plan, dict):
        return ValidationResult(False, ["missing object: system2_plan"])

    if system2_plan.get("type") != "skill_graph":
        errors.append("system2_plan.type must be 'skill_graph'")

    subtasks = system2_plan.get("subtasks")
    if not isinstance(subtasks, list) or not subtasks:
        errors.append("system2_plan.subtasks must be a non-empty list")
        return ValidationResult(False, errors)

    for expected_id, subtask in enumerate(subtasks):
        if not isinstance(subtask, dict):
            errors.append(f"subtask {expected_id} must be an object")
            continue
        sid = subtask.get("subtask_id")
        if sid != expected_id:
            errors.append(f"subtask {expected_id} has subtask_id={sid}; expected {expected_id}")
        skill = subtask.get("skill")
        if skill not in {"pick", "transport", "place", "pour", "open", "close", "turn_on"}:
            errors.append(f"subtask {expected_id} has unsupported skill={skill!r}")
        if not isinstance(subtask.get("params"), dict):
            errors.append(f"subtask {expected_id} params must be an object")
        if not isinstance(subtask.get("formula"), dict):
            errors.append(f"subtask {expected_id} formula must be an object")
        else:
            errors.extend(_validate_formula(subtask["formula"], f"subtask {expected_id}.formula"))

    return ValidationResult(not errors, errors)


def _validate_formula(formula: Formula, path: str) -> list[str]:
    errors: list[str] = []
    ftype = formula.get("type")
    if ftype == "true":
        return errors
    if ftype == "predicate":
        if not formula.get("name"):
            errors.append(f"{path}.name is required for predicates")
        if not isinstance(formula.get("args"), list):
            errors.append(f"{path}.args must be a list")
        return errors
    if ftype == "not":
        return _validate_child(formula, "child", path)
    if ftype == "and":
        children = formula.get("children")
        if not isinstance(children, list) or not children:
            return [f"{path}.children must be a non-empty list"]
        for i, child in enumerate(children):
            errors.extend(_validate_formula(child, f"{path}.children[{i}]"))
        return errors
    if ftype == "implies":
        errors.extend(_validate_child(formula, "left", path))
        errors.extend(_validate_child(formula, "right", path))
        return errors
    if ftype in {"always", "eventually"}:
        errors.extend(_validate_interval(formula.get("interval"), f"{path}.interval"))
        errors.extend(_validate_child(formula, "child", path))
        return errors
    if ftype == "until":
        errors.extend(_validate_interval(formula.get("interval"), f"{path}.interval"))
        errors.extend(_validate_child(formula, "left", path))
        errors.extend(_validate_child(formula, "right", path))
        return errors
    return [f"{path}.type has unsupported formula type {ftype!r}"]


def _validate_child(formula: Formula, key: str, path: str) -> list[str]:
    child = formula.get(key)
    if not isinstance(child, dict):
        return [f"{path}.{key} must be an object"]
    return _validate_formula(child, f"{path}.{key}")


def _validate_interval(interval: Any, path: str) -> list[str]:
    if interval is None:
        return []
    if not isinstance(interval, list) or len(interval) != 2:
        return [f"{path} must be null or [start, end]"]
    if not all(isinstance(x, int) and x >= 0 for x in interval):
        return [f"{path} values must be non-negative integers"]
    if interval[1] < interval[0]:
        return [f"{path} end must be >= start"]
    return []


def validate_global_stl_plan(plan: dict[str, Any]) -> ValidationResult:
    """EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md).

    Validates the single-global-formula plan shape (system2_plan.type ==
    "stl_spec", a formula instead of subtasks) produced by
    build_global_developer_prompt / schemas/stl_global_plan_schema.json.
    """
    errors: list[str] = []
    system2_plan = plan.get("system2_plan")
    if not isinstance(system2_plan, dict):
        return ValidationResult(False, ["missing object: system2_plan"])

    if system2_plan.get("type") != "stl_spec":
        errors.append("system2_plan.type must be 'stl_spec'")

    formula = system2_plan.get("formula")
    if not isinstance(formula, dict):
        errors.append("system2_plan.formula must be an object")
        return ValidationResult(False, errors)
    errors.extend(_validate_formula(formula, "system2_plan.formula"))

    object_hints = system2_plan.get("object_hints", {})
    if object_hints is not None and not isinstance(object_hints, dict):
        errors.append("system2_plan.object_hints must be an object")

    return ValidationResult(not errors, errors)


# EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md): predicate-set -> skill lookup
# used by decompose()/infer_skill_from_predicates() to replace the
# natural-language skill-grounding rules (7-13/16) in the *legacy*
# build_developer_prompt. Each key is the exact set of predicate names
# expected inside one top-level achievement-goal conjunct.
PREDICATE_TO_SKILL: dict[frozenset[str], str] = {
    frozenset({"PreGrasp", "Grasped"}): "pick",
    frozenset({"Grasped"}): "pick",
    frozenset({"OverGoal"}): "transport",
    frozenset({"Placed", "Released"}): "place",
    frozenset({"Placed"}): "place",
    frozenset({"Opened"}): "open",
    frozenset({"Closed"}): "close",
    frozenset({"TurnedOn"}): "turn_on",
    frozenset({"PouredInto"}): "pour",
}


class DecompositionError(ValueError):
    """Raised by decompose()/infer_skill_from_predicates() when a formula's
    top-level structure doesn't match a known, decomposable shape (e.g. a
    top-level implies/or, or a predicate set with no registered skill)."""


# Predicates that constrain how a skill runs rather than which skill it is.
# System 2 sometimes nests one inside a goal (e.g. Safe/Upright next to
# OverGoal); they stay in the subtask formula for System 1, but do not count
# toward the skill signature.
CONSTRAINT_PREDICATES = frozenset({"Safe", "Upright", "CarryHigh", "DescendOnlyNearGoal", "InRegion"})


def infer_skill_from_predicates(formula: Formula) -> str:
    """EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md). Maps one achievement-goal
    formula to the skill that satisfies it, by its predicate-name set."""
    names = frozenset(_collect_predicate_names(formula)) - CONSTRAINT_PREDICATES
    skill = PREDICATE_TO_SKILL.get(names)
    if skill is None:
        raise DecompositionError(
            f"cannot infer skill: predicate set {sorted(names)!r} does not match any "
            f"registered skill signature {[sorted(k) for k in PREDICATE_TO_SKILL]!r}"
        )
    return skill


def _is_global_constraint(formula: Formula) -> bool:
    return isinstance(formula, dict) and formula.get("type") == "always"


def split_top_level_and(formula: Formula) -> tuple[list[Formula], list[Formula]]:
    """EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md).

    Splits a global formula's top-level conjunction into (a) persistent
    global constraints (top-level always conjuncts) and (b) the ordered
    sequential achievement goals (everything else), in source order. A
    formula whose top level is not and is treated as one single sequential
    goal with no global constraints (the single-atomic-goal task case).
    """
    if not isinstance(formula, dict):
        raise DecompositionError("formula must be an object")
    if formula.get("type") != "and":
        return [], [formula]

    global_constraints: list[Formula] = []
    sequential_goals: list[Formula] = []
    for child in formula.get("children", []):
        if _is_global_constraint(child):
            global_constraints.append(child)
        else:
            sequential_goals.append(child)
    return global_constraints, sequential_goals


def _collect_intervals(formula: Formula) -> list[tuple[int, int]]:
    if not isinstance(formula, dict):
        return []
    intervals: list[tuple[int, int]] = []
    interval = formula.get("interval")
    if isinstance(interval, list) and len(interval) == 2:
        intervals.append((interval[0], interval[1]))
    for key in ("child", "left", "right"):
        if key in formula:
            intervals.extend(_collect_intervals(formula[key]))
    for child in formula.get("children", []) or []:
        intervals.extend(_collect_intervals(child))
    return intervals


def goal_absolute_window(goal: Formula) -> tuple[int, int] | None:
    """EXPERIMENTAL (SLICE, see STL_DECOMPOSITION_PLAN.md).

    The [min-start, max-end] absolute-time window spanned by one achievement
    goal's own time-bounded sub-formulas (e.g. a pick's PreGrasp/Grasped
    intervals), per Kapoor/Kang/Meira-Goes NFM 2024 Algorithm 3 (SLICE):
    atomic task formulas carry ABSOLUTE bounds on one shared global timeline,
    not a per-task-local clock reset to 0. Returns None if the goal has no
    time-bounded nodes at all (e.g. a bare untimed predicate).
    """
    intervals = _collect_intervals(goal)
    if not intervals:
        return None
    return min(s for s, _ in intervals), max(e for _, e in intervals)


def _validate_sequential_timing(sequential_goals: list[Formula]) -> None:
    """EXPERIMENTAL (SLICE, see STL_DECOMPOSITION_PLAN.md).

    Sequential achievement goals execute strictly one at a time, so on the
    global timeline their absolute windows must be non-overlapping and
    non-decreasing in source order (SLICE's own timeline-partitioning
    invariant, Algorithm 3 lines 12-28: the active-constraint set changes at
    each slice boundary, which for a purely sequential chain like ours means
    one goal's window must end at or before the next one's begins). Goals
    with no time-bounded nodes are skipped -- nothing to check.
    """
    previous_end: int | None = None
    for idx, goal in enumerate(sequential_goals):
        window = goal_absolute_window(goal)
        if window is None:
            continue
        start, end = window
        if previous_end is not None and start < previous_end:
            raise DecompositionError(
                f"sequential goal {idx} has absolute window [{start},{end}], which starts "
                f"before the previous goal's window ended at step {previous_end} -- "
                "achievement goals must occupy non-overlapping, non-decreasing absolute "
                "time windows on one shared global timeline (SLICE-style decomposition), "
                "not per-goal-local windows that each reset to 0"
            )
        previous_end = end


def _derive_params(skill: str, goal: Formula, object_hints: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if skill == "pick":
        pregrasp_args = find_predicate_args(goal, "PreGrasp") or []
        grasped_args = find_predicate_args(goal, "Grasped") or []
        obj = (pregrasp_args[0] if pregrasp_args else None) or (grasped_args[0] if grasped_args else None)
        params["object"] = obj
        if len(pregrasp_args) > 1:
            params["grasp_mode"] = pregrasp_args[1]
        if len(grasped_args) > 1:
            params["object_height_m"] = grasped_args[1]
    elif skill == "transport":
        args = find_predicate_args(goal, "OverGoal") or []
        params["object"] = args[0] if args else None
        if len(args) > 1:
            params["goal_region"] = args[1]
    elif skill == "place":
        args = find_predicate_args(goal, "Placed") or []
        params["object"] = args[0] if args else None
        if len(args) > 1:
            params["goal_region"] = args[1]
    elif skill in ("open", "close"):
        pred_name = "Opened" if skill == "open" else "Closed"
        args = find_predicate_args(goal, pred_name) or []
        params["object"] = args[0] if args else None
    elif skill == "turn_on":
        args = find_predicate_args(goal, "TurnedOn") or []
        params["object"] = args[0] if args else None
    elif skill == "pour":
        args = find_predicate_args(goal, "PouredInto") or []
        params["object"] = args[0] if args else None
        if len(args) > 1:
            params["goal_region"] = args[1]

    obj = params.get("object")
    if obj is not None and obj in object_hints:
        params = {**object_hints[obj], **params}
    return params


def _describe_subtask(skill: str, params: dict[str, Any]) -> str:
    obj = params.get("object", "?")
    goal_region = params.get("goal_region")
    if goal_region is not None:
        return f"{skill}({obj}, {goal_region})"
    return f"{skill}({obj})"


def _constraint_covers_window(constraint: Formula, window: tuple[int, int] | None) -> bool:
    """EXPERIMENTAL (SLICE, see STL_DECOMPOSITION_PLAN.md).

    Per Algorithm 3, a constraint (l, h, p) is "active" during a slice iff
    the slice's time range falls inside [l, h] -- an invariance constraint is
    NOT automatically global just because it's a top-level always; a bounded
    always[a,b] only covers subtasks whose own absolute window overlaps
    [a,b]. interval: null (this repo's "holds for the entire task"
    convention, per rule 5a in build_global_developer_prompt) or a goal with
    no timed nodes of its own (window is None) always applies -- both cases
    correspond to a constraint whose bound already equals (or exceeds) the
    whole horizon.
    """
    interval = constraint.get("interval")
    if interval is None or window is None:
        return True
    lo, hi = interval
    start, end = window
    return start <= hi and end >= lo


def decompose(
    formula: Formula,
    object_hints: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md).

    Deterministically derives the ordered subtasks list (today's
    stl_plan.py/System 1/stl_monitor.py shape) from one global STL
    formula, instead of the VLM emitting subtasks directly. Every top-level
    always conjunct is a persistent constraint and is folded into every
    subtask whose own absolute window it covers (see _constraint_covers_window
    -- an unbounded/interval: null one covers all of them, matching today's
    prompt, which only ever asks for that shape); every other top-level
    conjunct becomes exactly one subtask, in source order (see
    split_top_level_and).

    Per SLICE (Kapoor/Kang/Meira-Goes, NFM 2024, Algorithm 3), each goal's
    time-bounded sub-formulas carry ABSOLUTE bounds on one shared global
    timeline -- e.g. a pick spanning [0,90] followed by a transport spanning
    [90,270] -- and are validated (_validate_sequential_timing) rather than
    rewritten: this function does NOT rebase intervals to a per-subtask-local
    0, it only groups/orders/labels what the formula already specifies.
    """
    object_hints = object_hints or {}
    global_constraints, sequential_goals = split_top_level_and(formula)
    _validate_sequential_timing(sequential_goals)

    subtasks: list[dict[str, Any]] = []
    for idx, goal in enumerate(sequential_goals):
        skill = infer_skill_from_predicates(goal)
        params = _derive_params(skill, goal, object_hints)
        window = goal_absolute_window(goal)
        applicable_constraints = [c for c in global_constraints if _constraint_covers_window(c, window)]
        combined_formula = {"type": "and", "children": [goal, *applicable_constraints]} if applicable_constraints else goal
        subtask: dict[str, Any] = {
            "subtask_id": idx,
            "skill": skill,
            "description": _describe_subtask(skill, params),
            "params": params,
            "formula": combined_formula,
        }
        if window is not None:
            # EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md): the goal's own
            # absolute deadline, exposed for System 1's soft, log-only
            # overrun check (executor.py's _annotate_timing_violation) --
            # explicitly NOT consumed as a hard timeout (max_subtask_steps is
            # a separate, unrelated per-skill safety cap; see the "don't wire
            # this into max_step" discussion in the plan doc).
            subtask["stl_deadline"] = {"start_step": window[0], "end_step": window[1]}
        subtasks.append(subtask)
    return subtasks


def decompose_and_validate(plan: dict[str, Any]) -> tuple[dict[str, Any] | None, ValidationResult]:
    """EXPERIMENTAL (see STL_DECOMPOSITION_PLAN.md).

    One-call wrapper around validate_global_stl_plan + decompose +
    validate_stl_plan, for callers (e.g. libero_runner.py) that just want a
    subtasks-shape plan or a single, unified list of errors -- whichever
    stage failed. Returns (decomposed_plan, ok_result) on success, or
    (None, failing_result) otherwise.
    """
    global_result = validate_global_stl_plan(plan)
    if not global_result.ok:
        return None, global_result

    system2_plan = plan["system2_plan"]
    try:
        subtasks = decompose(system2_plan["formula"], system2_plan.get("object_hints"))
    except DecompositionError as exc:
        return None, ValidationResult(False, [str(exc)])

    decomposed_plan = {
        "system2_plan": {
            "type": "skill_graph",
            "task_name": system2_plan["task_name"],
            "scene_summary": system2_plan["scene_summary"],
            "subtasks": subtasks,
        }
    }
    subtask_result = validate_stl_plan(decomposed_plan)
    if not subtask_result.ok:
        return None, subtask_result
    return decomposed_plan, subtask_result


_LEGACY_PREGRASP_SKILLS = {"pregrasp", "pre_grasp", "pre-grasp"}
_LEGACY_GRASP_SKILLS = {"grasp"}


def normalize_legacy_pregrasp_grasp(subtasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge any adjacent (pregrasp, grasp)-shaped subtask pair into one
    unified pick subtask.

    Compatibility boundary for the unified Pick(object_or_object_set,
    constraints) skill (step/system_one/motion/pick_skill.py): externally,
    System 1 now expects exactly one pick subtask per grasp, with
    PreGrasp/Grasped expressed as two predicate nodes *inside* that one
    subtask's formula -- which is already the only shape today's schema/
    prompt (schemas/stl_plan_schema.json, step/system_two/prompts.py rule 7)
    ever produces. This function exists for any *external or legacy* plan data
    that might still emit them as two separate subtasks (e.g. an older
    task-spec compiler, a hand-authored eval plan, or a different System 2
    version) -- it is a no-op on any plan already shaped like today's.

    Detects a subtask pair shaped like separate pregrasp/grasp steps by either
    (a) an explicit legacy skill value ("pregrasp"/"grasp", not in
    the current schema's enum) or (b) a pick-skill subtask whose formula
    contains only a PreGrasp predicate immediately followed by another
    subtask for the same object whose formula contains only a Grasped
    predicate. Merges the pair into one pick subtask combining both
    formulas under and and re-numbers subtask_id sequentially.
    """
    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(subtasks):
        cur = subtasks[i]
        nxt = subtasks[i + 1] if i + 1 < len(subtasks) else None
        if nxt is not None and _is_legacy_pregrasp_grasp_pair(cur, nxt):
            merged.append(_merge_pregrasp_grasp(cur, nxt))
            i += 2
        else:
            merged.append(dict(cur))
            i += 1
    for idx, sub in enumerate(merged):
        sub["subtask_id"] = idx
    return merged


def _is_legacy_pregrasp_grasp_pair(cur: dict[str, Any], nxt: dict[str, Any]) -> bool:
    cur_skill = str(cur.get("skill", "")).lower()
    nxt_skill = str(nxt.get("skill", "")).lower()
    cur_obj = (cur.get("params") or {}).get("object")
    nxt_obj = (nxt.get("params") or {}).get("object")
    same_object = cur_obj is None or nxt_obj is None or cur_obj == nxt_obj
    if not same_object:
        return False
    if cur_skill in _LEGACY_PREGRASP_SKILLS and nxt_skill in _LEGACY_GRASP_SKILLS:
        return True
    if cur_skill == "pick" and nxt_skill in _LEGACY_GRASP_SKILLS | {"pick"}:
        cur_only_pregrasp = _formula_only_contains(cur.get("formula"), {"PreGrasp"})
        nxt_only_grasped = _formula_only_contains(nxt.get("formula"), {"Grasped"})
        return cur_only_pregrasp and nxt_only_grasped
    return False


def _formula_only_contains(formula: Formula | None, allowed_names: set[str]) -> bool:
    if not isinstance(formula, dict):
        return False
    names = _collect_predicate_names(formula)
    return bool(names) and names.issubset(allowed_names)


def _collect_predicate_names(formula: Formula) -> set[str]:
    if not isinstance(formula, dict):
        return set()
    ftype = formula.get("type")
    if ftype == "predicate":
        name = formula.get("name")
        return {name} if name else set()
    names: set[str] = set()
    for key in ("child", "left", "right"):
        if key in formula:
            names |= _collect_predicate_names(formula[key])
    for child in formula.get("children", []) or []:
        names |= _collect_predicate_names(child)
    return names


def _merge_pregrasp_grasp(cur: dict[str, Any], nxt: dict[str, Any]) -> dict[str, Any]:
    params = dict(nxt.get("params") or {})
    for k, v in (cur.get("params") or {}).items():
        params.setdefault(k, v)
    formula = {"type": "and", "children": [cur.get("formula", {"type": "true"}), nxt.get("formula", {"type": "true"})]}
    description = cur.get("description") or nxt.get("description") or "Pick (normalized from legacy pregrasp+grasp)"
    return {
        "subtask_id": cur.get("subtask_id", 0),
        "skill": "pick",
        "description": description,
        "params": params,
        "formula": formula,
    }


def find_predicate_args(formula: Formula, name: str) -> list[Any] | None:
    """Recursively find the first predicate node named name and return its args.

    Used to pull execution hints (e.g. an object-height arg on Grasped) out of
    a subtask's formula tree without needing a full STL interpreter.
    """
    if not isinstance(formula, dict):
        return None
    ftype = formula.get("type")
    if ftype == "predicate":
        return formula.get("args") if formula.get("name") == name else None
    if ftype in {"not", "always", "eventually"}:
        return find_predicate_args(formula.get("child", {}), name)
    if ftype == "and":
        for child in formula.get("children", []) or []:
            found = find_predicate_args(child, name)
            if found is not None:
                return found
        return None
    if ftype in {"implies", "until"}:
        return find_predicate_args(formula.get("left", {}), name) or find_predicate_args(
            formula.get("right", {}), name
        )
    return None


def find_all_predicate_args(formula: Formula, name: str) -> list[list[Any]]:
    """Args of every predicate node named name, in tree order.

    find_predicate_args stops at the first match, which is right for an
    execution hint but not for constraints: a subtask can carry several
    G(Safe(...)) terms, one per thing to keep clear of.
    """
    if not isinstance(formula, dict):
        return []
    ftype = formula.get("type")
    if ftype == "predicate":
        return [list(formula.get("args") or [])] if formula.get("name") == name else []
    if ftype in {"not", "always", "eventually"}:
        return find_all_predicate_args(formula.get("child", {}), name)
    if ftype == "and":
        out: list[list[Any]] = []
        for child in formula.get("children", []) or []:
            out.extend(find_all_predicate_args(child, name))
        return out
    if ftype in {"implies", "until"}:
        return (find_all_predicate_args(formula.get("left", {}), name)
                + find_all_predicate_args(formula.get("right", {}), name))
    return []


def formula_to_string(formula: Formula) -> str:
    ftype = formula.get("type")
    if ftype == "true":
        return "True"
    if ftype == "predicate":
        args = ", ".join(_arg_to_string(arg) for arg in formula.get("args", []))
        return f"{formula.get('name')}({args})"
    if ftype == "not":
        return f"!({formula_to_string(formula['child'])})"
    if ftype == "and":
        return "(" + " & ".join(formula_to_string(child) for child in formula.get("children", [])) + ")"
    if ftype == "implies":
        return f"({formula_to_string(formula['left'])} -> {formula_to_string(formula['right'])})"
    if ftype in {"always", "eventually"}:
        op = "G" if ftype == "always" else "F"
        return f"{op}{_interval_to_string(formula.get('interval'))}({formula_to_string(formula['child'])})"
    if ftype == "until":
        return (
            f"({formula_to_string(formula['left'])} "
            f"U{_interval_to_string(formula.get('interval'))} "
            f"{formula_to_string(formula['right'])})"
        )
    return str(formula)


def print_formulas(plan: dict[str, Any]) -> None:
    print("=== STL formulas by subtask ===")
    for subtask in plan["system2_plan"]["subtasks"]:
        print(f"[subtask {subtask['subtask_id']} | skill={subtask['skill']}]")
        print(formula_to_string(subtask["formula"]))
    print("=== End STL formulas ===")


def _interval_to_string(interval: Any) -> str:
    if not interval:
        return ""
    return f"[{interval[0]},{interval[1]}]"


def _arg_to_string(arg: Any) -> str:
    if isinstance(arg, str):
        return repr(arg)
    if isinstance(arg, dict) and arg.get("type") == "point":
        xy = arg.get("xy", [None, None])
        radius = arg.get("radius")
        z = arg.get("z")
        if z is None:
            return f"PointTarget(x={xy[0]}, y={xy[1]}, r={radius})"
        return f"PointTarget(x={xy[0]}, y={xy[1]}, z={z}, r={radius})"
    if isinstance(arg, dict) and arg.get("type") == "pass_through_region":
        xy = arg.get("xy", [None, None])
        return f"PassThrough(x={xy[0]}, y={xy[1]}, r={arg.get('radius')})"
    return str(arg)
