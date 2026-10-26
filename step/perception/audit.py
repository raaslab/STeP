"""Records every read of simulator ground truth made while vision mode is on.

Each privileged entry point calls note(); in vision mode the hit is counted
(and raises under STEP_PERCEPTION_STRICT=1 unless it is an allowed use:
the benchmark's own success check, or experiment scene construction that
happens before the robot is given control). The runner writes the counts
next to the rollout so a vision-mode result can show it never read
object state.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path


class PrivilegedAccessError(RuntimeError):
    pass


_vision_active = False
_hits: Counter = Counter()


def set_vision_active(active: bool) -> None:
    global _vision_active
    _vision_active = bool(active)


def vision_active() -> bool:
    return _vision_active


def note(entry_point: str, *, allowed: str | None = None) -> None:
    if not _vision_active:
        return
    _hits[(entry_point, allowed or "")] += 1
    if allowed is None and os.environ.get("STEP_PERCEPTION_STRICT") == "1":
        raise PrivilegedAccessError(f"vision mode read simulator ground truth via {entry_point}")


def report() -> dict:
    disallowed = {k: v for (k, a), v in _hits.items() if not a}
    allowed = {f"{k} ({a})": v for (k, a), v in _hits.items() if a}
    return {"vision_mode": _vision_active, "disallowed": disallowed, "allowed": allowed}


def write_report(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report(), indent=1))
