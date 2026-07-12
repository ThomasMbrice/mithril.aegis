"""
Fixed, step-keyed B1 trace (test_suite.md §4.5.3) for Phase 1.

Only B1 is real-injectable on this allocation (single-node 8xA100 — no
B3/B4 targets, and B0 is sim-only per test_suite.md §4.5.5, see
chaos_inject/real_injector.py). There is deliberately no trace_b3.py/
trace_b4.py in this package.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_trace(target_step: int, target_rank: int) -> list[dict]:
    return [{"step": target_step, "fault": "B1", "target_rank": target_rank}]


def write_trace(path: str | Path, target_step: int, target_rank: int) -> None:
    Path(path).write_text(json.dumps(build_trace(target_step, target_rank), indent=2))


def suggest_target_step(
    steps_per_sec: float, steps_budget: int, warmup_secs: float = 60.0
) -> int:
    """
    Pick a target step using a real measured steps/sec (from Phase 0's
    report) rather than a fixed fraction of the step budget, so the fault
    fires a fixed amount of *wall-clock* time in — comfortably past
    allocator/JIT warm-up regardless of how fast or slow this workload's
    steps/sec turns out to be — and well before the run's step budget ends.
    """
    target_step = max(1, int(warmup_secs * steps_per_sec))
    ceiling = int(steps_budget * 0.8)
    if target_step >= ceiling:
        raise ValueError(
            f"suggest_target_step: computed target_step={target_step} leaves no "
            f"headroom before steps_budget={steps_budget} (80% ceiling={ceiling}). "
            "Increase --steps or lower --warmup-secs."
        )
    return target_step
