"""
Per-tier isolation fault traces.

One trace per blast-radius tier — each injects only that fault class, so
we measure each tier's recovery cost in isolation and cleanly attribute
the result to a single system under test.

All traces are deterministic — same events at same steps, every run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aegis.telemetry.events import BlastRadius, FaultSignal


@dataclass(frozen=True)
class TraceFaultEvent:
    """
    A single fault event in a benchmark trace.

    Keyed to training step (not wall-clock) for deterministic replay
    regardless of per-system throughput differences.
    """

    step: int
    fault_signal: FaultSignal
    target_rank: int
    target_node: str
    target_nic: str | None = None
    target_rack: str | None = None
    duration_secs: float = 0.0
    expected_tier: BlastRadius = BlastRadius.B0


@dataclass
class FaultTrace:
    """
    A named, ordered sequence of fault events.

    The cardinal rule: only the FT system varies between runs; the trace
    is identical for all systems under test.
    """

    name: str
    events: list[TraceFaultEvent] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)


# ---------------------------------------------------------------------------
# Per-tier isolation traces (FT-B0 through FT-B4)

# FT-B0: NIC port flap — exercises B0 transport tier only
FT_B0 = FaultTrace(
    name="FT-B0",
    events=[
        TraceFaultEvent(
            step=5000,
            fault_signal=FaultSignal.NIC_PORT_FLAP,
            target_rank=0,
            target_node="node0",
            target_nic="nic0",
            target_rack="rack0",
            duration_secs=30.0,
            expected_tier=BlastRadius.B0,
        ),
    ],
)

# FT-B1: Node crash — exercises B1 compute tier only
FT_B1 = FaultTrace(
    name="FT-B1",
    events=[
        TraceFaultEvent(
            step=5000,
            fault_signal=FaultSignal.NODE_CRASH,
            target_rank=2,
            target_node="node1",
            target_nic=None,
            target_rack="rack0",
            duration_secs=0.0,
            expected_tier=BlastRadius.B1,
        ),
    ],
)

# FT-B2: CUDA kernel crash — exercises B2 storage T1 tier
FT_B2 = FaultTrace(
    name="FT-B2",
    events=[
        TraceFaultEvent(
            step=8000,
            fault_signal=FaultSignal.CUDA_KERNEL_CRASH,
            target_rank=3,
            target_node="node1",
            target_nic=None,
            target_rack="rack0",
            duration_secs=0.0,
            expected_tier=BlastRadius.B2,
        ),
    ],
)

# FT-B3: Node unrecoverable — exercises B3 storage T2 tier
FT_B3 = FaultTrace(
    name="FT-B3",
    events=[
        TraceFaultEvent(
            step=10000,
            fault_signal=FaultSignal.NODE_UNRECOVERABLE,
            target_rank=4,
            target_node="node2",
            target_nic=None,
            target_rack="rack1",
            duration_secs=0.0,
            expected_tier=BlastRadius.B3,
        ),
    ],
)

# FT-B4: Rack power loss — exercises B4 storage T3 tier
# Two nodes on the same rack fail simultaneously → correlated B4 event
# Step 12500 is mid-interval (last checkpoint at 12000) so vanilla rollback
# is significant: 500 steps × 0.4s/step + 120s restore ≈ 320s > AEGIS 270s.
FT_B4 = FaultTrace(
    name="FT-B4",
    events=[
        TraceFaultEvent(
            step=12500,
            fault_signal=FaultSignal.RACK_POWER_LOSS,
            target_rank=4,
            target_node="node2",
            target_nic=None,
            target_rack="rack1",
            duration_secs=0.0,
            expected_tier=BlastRadius.B4,
        ),
        TraceFaultEvent(
            step=12500,
            fault_signal=FaultSignal.RACK_POWER_LOSS,
            target_rank=6,
            target_node="node3",
            target_nic=None,
            target_rack="rack1",
            duration_secs=0.0,
            expected_tier=BlastRadius.B4,
        ),
    ],
)
