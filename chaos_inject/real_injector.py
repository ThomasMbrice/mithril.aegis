"""
RealFaultInjector — standalone OS-level fault process for the real-cluster
harness (test_suite.md §4.5.2/§4.5.3).

**This is a separate mechanism from ``chaos_inject.harness.ChaosHarness``,
not an extension of it.** ``ChaosHarness`` injects synthetic
``TelemetryEvent``s directly onto an in-process asyncio UTP — it never
touches the OS or another process, and is what every existing unit/
integration test uses. ``RealFaultInjector`` (this module) is a standalone
process that watches a real training job's step-log file and sends a real
OS signal to a real process. It does not import or use ``ChaosHarness``/UTP
at all. The two share only the ``FaultSignal``/trace-format vocabulary, not
code: ``ChaosHarness`` exercises the EPE/layers in-process for software
tests; ``RealFaultInjector`` creates a real fault for the real sensor
(``realbench.sensors.rank_heartbeat.RankHeartbeatSensor``) to detect.

Trace format (test_suite.md §4.5.3), one JSON array in a file:
    [{"step": 5000, "fault": "B1", "target_rank": 3}, ...]

Determinism: this process watches the *same* step-log every system-under-
test uses, and fires at the *same* trace-defined step regardless of how
fast/slow that particular run's steps/sec is — so the same fault lands at
the same point in training for every condition (test_suite.md §6.1).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from realbench.training.step_log import StepLogReader, append_jsonl

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECS = 0.5

# Per test_suite.md §4.5.5: B0 requires >=2 physical RNICs per node with
# spare bandwidth for R2CCL to migrate onto. Confirmed for this allocation:
# single NIC/plane per node -> B0 cannot be hardware-validated here. This
# flag makes the real ip-link path structurally unreachable from the trace
# dispatcher below. Flip only after re-confirming the topology has changed
# (see test_suite.md §4.5.5 Appendix — "check this on day one").
B0_HARDWARE_VALIDATED = False


@dataclass(frozen=True)
class TraceEvent:
    step: int
    fault: str
    target_rank: int | None = None
    rnic: str | None = None
    duration_secs: float | None = None


def load_trace(path: str | Path) -> list[TraceEvent]:
    with open(path) as fh:
        raw = json.load(fh)
    return [
        TraceEvent(
            step=e["step"],
            fault=e["fault"],
            target_rank=e.get("target_rank"),
            rnic=e.get("rnic"),
            duration_secs=e.get("duration_secs"),
        )
        for e in raw
    ]


def read_rank_pid(pid_dir: str | Path, rank: int) -> int:
    path = Path(pid_dir) / f"pid_rank{rank}.txt"
    return int(path.read_text().strip())


class B0RealInjector:
    """
    Real NIC-down/up injection for B0 — present for future dual-NIC
    hardware, not wired into the trace dispatcher while
    ``B0_HARDWARE_VALIDATED`` is False. Unit-testable in isolation by
    mocking ``subprocess.run``.
    """

    def __init__(self, run=subprocess.run) -> None:
        self._run = run

    def fire(self, rnic: str, duration_secs: float) -> None:
        self._run(["ip", "link", "set", rnic, "down"], check=True)
        time.sleep(duration_secs)
        self._run(["ip", "link", "set", rnic, "up"], check=True)


def _fire_b1(event: TraceEvent, pid_dir: str | Path, out_path: str | Path, current_step: int) -> None:
    if event.target_rank is None:
        raise ValueError(f"B1 trace event at step {event.step} missing target_rank")
    pid = read_rank_pid(pid_dir, event.target_rank)
    logger.warning(
        "[real_injector] firing B1: SIGKILL rank %d (pid %d) at observed step %d (trace step %d)",
        event.target_rank, pid, current_step, event.step,
    )
    try:
        os.kill(pid, signal.SIGKILL)
        fired = True
    except ProcessLookupError:
        # Target already exited (e.g. the run finished before this step was
        # reached, or the trace's target step was set too close to --steps).
        # Log clearly and record a non-fatal miss rather than crashing the
        # injector — a real Phase 1 run should size --steps well past the
        # trace's target step (see realbench/phase1_per_tier/trace_b1.py).
        logger.error(
            "[real_injector] rank %d (pid %d) already exited — fault not delivered", event.target_rank, pid
        )
        fired = False
    append_jsonl(out_path, {
        "wall_clock": time.time(),
        "step": current_step,
        "fault_fired": fired,
        "fault_signal": "B1",
        "target_rank": event.target_rank,
        **({} if fired else {"reason": f"target rank {event.target_rank} (pid {pid}) already exited"}),
    })


def _skip_b0(event: TraceEvent, out_path: str | Path, current_step: int) -> None:
    logger.warning(
        "[real_injector] skipping B0 at observed step %d (trace step %d): sim-only "
        "on this allocation (single NIC/plane topology, see test_suite.md §4.5.5)",
        current_step, event.step,
    )
    append_jsonl(out_path, {
        "wall_clock": time.time(),
        "step": current_step,
        "fault_fired": False,
        "fault_signal": "B0",
        "target_rank": event.target_rank,
        "reason": "sim-only — single NIC/plane topology, see test_suite.md §4.5.5",
    })


def run(trace_path: str | Path, step_log_path: str | Path, pid_dir: str | Path, out_path: str | Path) -> None:
    trace = sorted(load_trace(trace_path), key=lambda e: e.step)
    reader = StepLogReader(step_log_path)
    pending = list(trace)
    current_step = -1

    while pending:
        for rec in reader.tail_new():
            current_step = max(current_step, rec.global_step)

        fired_this_pass = [e for e in pending if e.step <= current_step]
        for event in fired_this_pass:
            if event.fault == "B1":
                _fire_b1(event, pid_dir, out_path, current_step)
            elif event.fault == "B0":
                if B0_HARDWARE_VALIDATED:
                    # Unreachable until the constant above is deliberately
                    # flipped by a human who has re-confirmed the NIC
                    # topology — see B0RealInjector.
                    raise NotImplementedError(
                        "B0_HARDWARE_VALIDATED=True but the real dispatch path "
                        "for B0 has not been wired up — implement before flipping the flag."
                    )
                _skip_b0(event, out_path, current_step)
            else:
                raise ValueError(
                    f"Unsupported fault class {event.fault!r} for real injection — "
                    "only B1 is real-injectable on a single-node allocation "
                    "(no B2/B3/B4 targets available, see realbench/phase1_per_tier)."
                )
        pending = [e for e in pending if e not in fired_this_pass]

        if pending:
            time.sleep(POLL_INTERVAL_SECS)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", required=True)
    p.add_argument("--step-log", required=True)
    p.add_argument("--pid-dir", required=True,
                    help="Directory containing pid_rank{r}.txt files written by w1_llama7b.py.")
    p.add_argument("--out", required=True, help="Path to write chaos_log.jsonl")
    args = p.parse_args()

    run(args.trace, args.step_log, args.pid_dir, args.out)


if __name__ == "__main__":
    main()
