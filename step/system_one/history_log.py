"""Rolling history log for System-2 recall.

The log is deliberately small and JSON-friendly. It stores the most recent
`window_sec` seconds of records, sampled at `interval_sec`, so System 2 can
replan from recent execution evidence instead of a one-off failure summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class STLHistoryLog:
    interval_sec: float = 1.0
    window_sec: float = 20.0
    records: list[dict[str, Any]] = field(default_factory=list)
    _last_wall_time: float | None = None
    _start_wall_time: float | None = None

    def reset(self) -> None:
        self.records.clear()
        self._last_wall_time = None
        self._start_wall_time = None

    def maybe_add(self, record: dict[str, Any], *, force: bool = False) -> bool:
        now = time.monotonic()
        if self._start_wall_time is None:
            self._start_wall_time = now
        if (
            not force
            and self._last_wall_time is not None
            and now - self._last_wall_time < float(self.interval_sec)
        ):
            return False

        rec = dict(record)
        rec.setdefault("t_wall", now)
        rec.setdefault("t_rel", now - float(self._start_wall_time))
        self.records.append(rec)
        self._last_wall_time = now
        self._trim(now)
        return True

    def add(self, record: dict[str, Any]) -> None:
        self.maybe_add(record, force=True)

    def _trim(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        cutoff = now - float(self.window_sec)
        self.records = [
            r for r in self.records if float(r.get("t_wall", now)) >= cutoff
        ]

    def latest(self) -> dict[str, Any] | None:
        return self.records[-1] if self.records else None

    def to_payload(self) -> dict[str, Any]:
        self._trim()
        return {
            "interval_sec": float(self.interval_sec),
            "window_sec": float(self.window_sec),
            "num_records": len(self.records),
            "records": list(self.records),
        }
