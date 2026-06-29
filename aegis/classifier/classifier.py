"""
Failure Classifier (FC) — maps raw telemetry signals to blast-radius tiers.

Currently rule-based (§4 Layer A2 spec: "Start rule-based; leave hook for
learned classifier later").  The interface is intentionally thin so that
the implementation can be swapped for a model-backed version in Phase 3
(§7.1 learned failure classifier) without touching the EPE.
"""

from __future__ import annotations

from ..telemetry.events import BlastRadius, FaultSignal, TelemetryEvent

# -----------------------------------------------------------------------
# Signal → tier table
#
# The "dangerous direction" the test plan calls out explicitly (UT-A2):
# never misclassify B4 as B1 or below.  This table is the single source
# of truth for that constraint.
# -----------------------------------------------------------------------

_SIGNAL_TO_TIER: dict[FaultSignal, BlastRadius] = {
    # B0 — transient link / single NIC
    FaultSignal.NIC_PORT_FLAP: BlastRadius.B0,
    FaultSignal.LINK_FLUCTUATION: BlastRadius.B0,
    FaultSignal.RDMA_TIMEOUT: BlastRadius.B0,
    # B1 — single GPU / node death
    FaultSignal.GPU_FELL_OFF_BUS: BlastRadius.B1,
    FaultSignal.NODE_CRASH: BlastRadius.B1,
    FaultSignal.OOM_KILLED_RANK: BlastRadius.B1,
    # B2 — software crash, recoverable in place
    FaultSignal.CUDA_KERNEL_CRASH: BlastRadius.B2,
    FaultSignal.RUNTIME_BUG: BlastRadius.B2,
    FaultSignal.HUNG_RANK: BlastRadius.B2,
    # B3 — node-level state loss
    FaultSignal.NODE_UNRECOVERABLE: BlastRadius.B3,
    # B4 — rack-level / cluster outage
    FaultSignal.RACK_POWER_LOSS: BlastRadius.B4,
    FaultSignal.FABRIC_PARTITION: BlastRadius.B4,
    FaultSignal.CLUSTER_WIDE_OUTAGE: BlastRadius.B4,
    # Unknown — conservative safe default: recoverable in place, not dangerous
    FaultSignal.UNKNOWN: BlastRadius.B2,
}


class FailureClassifier:
    """
    Classify a telemetry event into a blast-radius tier.

    Interface contract: classify(event) → BlastRadius is synchronous and
    side-effect-free.  The EPE calls this on every event from the UTP
    dispatch loop (§4 Layer A2).
    """

    def classify(self, event: TelemetryEvent) -> BlastRadius:
        """Return the blast-radius tier for this event."""
        return _SIGNAL_TO_TIER.get(event.fault_signal, BlastRadius.B2)

    def classify_signal(self, signal: FaultSignal) -> BlastRadius:
        """Convenience: classify a raw signal without constructing an event."""
        return _SIGNAL_TO_TIER.get(signal, BlastRadius.B2)
