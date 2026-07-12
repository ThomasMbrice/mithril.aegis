"""
Shared JSONL formats for the real-cluster harness's self-reporting streams.

Every process in the real run (trainer, sensor via the trainer, collector,
``chaos_inject.real_injector``) reads or writes one of these two streams, and
all timestamps come from ``time.time()`` (wall-clock, not ``time.monotonic()``)
so that ``realbench/collector/align.py`` can align them against the chaos-inject
log per test_suite.md §4.5.4's "same clock" requirement.

Step-log format (§4.5.4b): ``{wall_clock, global_step, loss}`` per line.
Heartbeat format (§1.2 of the real-cluster plan): ``{wall_clock, step}`` per
rank, refreshed every training step, polled by ``RankHeartbeatSensor`` to
detect peer death.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StepRecord:
    wall_clock: float
    global_step: int
    loss: float


class StepLogWriter:
    """Appends one JSON line per training step. Flushed every write."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a", buffering=1)

    def write(self, global_step: int, loss: float) -> None:
        record = {"wall_clock": time.time(), "global_step": global_step, "loss": loss}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()


class StepLogReader:
    """Tails a step-log file from the last-seen byte offset."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._offset = 0

    def tail_new(self) -> list[StepRecord]:
        """Return any step records written since the last call."""
        if not self._path.exists():
            return []
        records: list[StepRecord] = []
        with open(self._path) as fh:
            fh.seek(self._offset)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                records.append(
                    StepRecord(
                        wall_clock=data["wall_clock"],
                        global_step=data["global_step"],
                        loss=data["loss"],
                    )
                )
            self._offset = fh.tell()
        return records

    def read_all(self) -> list[StepRecord]:
        """Read the whole file from the start (for post-hoc report generation)."""
        self._offset = 0
        return self.tail_new()


def append_jsonl(path: str | Path, record: dict) -> None:
    """
    Append one JSON record to a shared log file, flushed to disk.

    Generic helper for the harness's other small append-only logs
    (``RankHeartbeatSensor``'s detection log, ``chaos_inject.real_injector``'s
    fault-fire log) so every wall-clock-stamped stream in the harness uses
    the same durability guarantee as ``StepLogWriter``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", buffering=1) as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: str | Path) -> list[dict]:
    """Read every record from a JSONL log, empty list if the file doesn't exist yet."""
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@dataclass(frozen=True)
class HeartbeatRecord:
    wall_clock: float
    step: int


class HeartbeatWriter:
    """
    Refreshes a single rank's liveness file every step.

    Overwrites in place (not append-only like the step log) — only the most
    recent heartbeat matters, and this file is read by every *other* rank's
    ``RankHeartbeatSensor`` to detect this rank's death.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def touch(self, step: int) -> None:
        record = {"wall_clock": time.time(), "step": step}
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(record, fh)
        os.replace(tmp, self._path)  # atomic — readers never see a partial write


class HeartbeatReader:
    """Reads a single rank's most recent heartbeat, if any."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def read(self) -> HeartbeatRecord | None:
        if not self._path.exists():
            return None
        try:
            with open(self._path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            # Torn read racing the writer's os.replace — treat as "no update yet".
            return None
        return HeartbeatRecord(wall_clock=data["wall_clock"], step=data["step"])
