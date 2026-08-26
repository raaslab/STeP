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
        if skill not in {"pick", "transport", "place", "pour"}:
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
