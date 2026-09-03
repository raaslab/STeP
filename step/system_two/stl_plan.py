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
        if skill not in {"pick", "transport", "place", "pour", "open"}:
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


_LEGACY_PREGRASP_SKILLS = {"pregrasp", "pre_grasp", "pre-grasp"}
_LEGACY_GRASP_SKILLS = {"grasp"}


def normalize_legacy_pregrasp_grasp(subtasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge any adjacent (pregrasp, grasp)-shaped subtask pair into one
    unified ``pick`` subtask.

    Compatibility boundary for the unified ``Pick(object_or_object_set,
    constraints)`` skill (``step/system_one/motion/pick_skill.py``): externally,
    System 1 now expects exactly one ``pick`` subtask per grasp, with
    PreGrasp/Grasped expressed as two predicate nodes *inside* that one
    subtask's formula -- which is already the only shape today's schema/
    prompt (`schemas/stl_plan_schema.json`, `step/system_two/prompts.py` rule 7)
    ever produces. This function exists for any *external or legacy* plan data
    that might still emit them as two separate subtasks (e.g. an older
    task-spec compiler, a hand-authored eval plan, or a different System 2
    version) -- it is a no-op on any plan already shaped like today's.

    Detects a subtask pair shaped like separate pregrasp/grasp steps by either
    (a) an explicit legacy ``skill`` value (``"pregrasp"``/``"grasp"``, not in
    the current schema's enum) or (b) a ``pick``-skill subtask whose formula
    contains only a ``PreGrasp`` predicate immediately followed by another
    subtask for the same object whose formula contains only a ``Grasped``
    predicate. Merges the pair into one ``pick`` subtask combining both
    formulas under ``and`` and re-numbers ``subtask_id`` sequentially.
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
    """Recursively find the first predicate node named `name` and return its args.

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
