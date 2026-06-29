"""Fault specifications — what to inject, where, and when."""

from __future__ import annotations

from dataclasses import dataclass, field

from aegis.telemetry.events import FaultSignal


@dataclass
class FaultSpec:
    """Specification for a single injected fault event."""

    fault_signal: FaultSignal
    rank: int
    node: str
    nic_id: str | None = None
    rack_id: str | None = None
    delay_secs: float = 0.0           # wait before first injection
    repeat: int = 1                    # how many times to inject this fault
    repeat_interval_secs: float = 0.1  # delay between repeated injections


@dataclass
class BurstSpec:
    """
    A correlated burst of faults.

    Used to test §3.3 (correlation window: B1 burst → B4 re-classification)
    and IT-3 / IT-5.
    """

    faults: list[FaultSpec]
    inter_fault_delay_secs: float = 0.05  # delay between faults in the burst
