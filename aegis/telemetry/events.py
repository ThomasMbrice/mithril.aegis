"""
Unified event schema for the AEGIS Telemetry Plane.

Every fault in the system is represented as a TelemetryEvent.  The schema
is deliberately flat and frozen so events can be safely passed across
coroutines and stored in audit logs without mutation bugs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FaultSignal(str, Enum):
    """Raw signals emitted by primitive sensors.  One signal per fault event."""

    # B0 — transient link / single NIC
    NIC_PORT_FLAP = "nic_port_flap"
    LINK_FLUCTUATION = "link_fluctuation"
    RDMA_TIMEOUT = "rdma_timeout"

    # B1 — single GPU / node death
    GPU_FELL_OFF_BUS = "gpu_fell_off_bus"
    NODE_CRASH = "node_crash"
    OOM_KILLED_RANK = "oom_killed_rank"

    # B2 — software crash, recoverable in place
    CUDA_KERNEL_CRASH = "cuda_kernel_crash"
    RUNTIME_BUG = "runtime_bug"
    HUNG_RANK = "hung_rank"

    # B3 — node-level state loss (unrecoverable hardware)
    NODE_UNRECOVERABLE = "node_unrecoverable"

    # B4 — rack-level / cluster outage
    RACK_POWER_LOSS = "rack_power_loss"
    FABRIC_PARTITION = "fabric_partition"
    CLUSTER_WIDE_OUTAGE = "cluster_wide_outage"

    # Catch-all
    UNKNOWN = "unknown"


class BlastRadius(int, Enum):
    """
    Blast-radius tier — how much of the job a fault invalidates.

    Inherits from int so tiers are directly comparable with > < >= <=.
    Design invariant: faults only ever escalate upward (B0 → B4).
    """

    B0 = 0  # transient link / single NIC
    B1 = 1  # single GPU / node death
    B2 = 2  # software crash, recoverable in place
    B3 = 3  # node-level state loss
    B4 = 4  # rack-level / cluster outage


@dataclass(frozen=True)
class TelemetryEvent:
    """
    Single fault event on the Unified Telemetry Plane.

    Schema: {timestamp, rank, node, nic_id, fault_signal, raw_payload, epoch}
    as specified in §4 Layer A1.
    """

    rank: int
    node: str
    fault_signal: FaultSignal
    raw_payload: dict[str, Any]
    epoch: int
    timestamp: float = field(default_factory=time.monotonic)
    nic_id: str | None = None
    rack_id: str | None = None
    source: str = "unknown"
