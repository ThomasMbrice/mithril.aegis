"""
UT-A2 — Failure Classifier.

Pass criteria (§5.1):
  - Classification accuracy: every signal maps to its correct tier
  - Critical safety: ZERO misclassification of B4 signals as B1 or lower
  - Unknown signals receive a conservative (safe) default
"""

from __future__ import annotations

import pytest

from aegis.classifier.classifier import FailureClassifier
from aegis.telemetry.events import BlastRadius, FaultSignal, TelemetryEvent


@pytest.fixture
def fc() -> FailureClassifier:
    return FailureClassifier()


def _event(signal: FaultSignal) -> TelemetryEvent:
    return TelemetryEvent(rank=0, node="node0", fault_signal=signal, raw_payload={}, epoch=0)


# -----------------------------------------------------------------------
# Full signal → tier coverage
# -----------------------------------------------------------------------

@pytest.mark.parametrize("signal,expected", [
    # B0
    (FaultSignal.NIC_PORT_FLAP,       BlastRadius.B0),
    (FaultSignal.LINK_FLUCTUATION,    BlastRadius.B0),
    (FaultSignal.RDMA_TIMEOUT,        BlastRadius.B0),
    # B1
    (FaultSignal.GPU_FELL_OFF_BUS,    BlastRadius.B1),
    (FaultSignal.NODE_CRASH,          BlastRadius.B1),
    (FaultSignal.OOM_KILLED_RANK,     BlastRadius.B1),
    # B2
    (FaultSignal.CUDA_KERNEL_CRASH,   BlastRadius.B2),
    (FaultSignal.RUNTIME_BUG,         BlastRadius.B2),
    (FaultSignal.HUNG_RANK,           BlastRadius.B2),
    # B3
    (FaultSignal.NODE_UNRECOVERABLE,  BlastRadius.B3),
    # B4
    (FaultSignal.RACK_POWER_LOSS,     BlastRadius.B4),
    (FaultSignal.FABRIC_PARTITION,    BlastRadius.B4),
    (FaultSignal.CLUSTER_WIDE_OUTAGE, BlastRadius.B4),
])
def test_signal_to_tier(fc: FailureClassifier, signal: FaultSignal, expected: BlastRadius) -> None:
    assert fc.classify(_event(signal)) == expected


# -----------------------------------------------------------------------
# Critical safety property (§5.1 UT-A2): B4 never misclassified as B1 or below
# -----------------------------------------------------------------------

@pytest.mark.parametrize("signal", [
    FaultSignal.RACK_POWER_LOSS,
    FaultSignal.FABRIC_PARTITION,
    FaultSignal.CLUSTER_WIDE_OUTAGE,
])
def test_b4_never_classified_as_b1_or_lower(
    fc: FailureClassifier, signal: FaultSignal
) -> None:
    """
    THE DANGEROUS DIRECTION: a B4 misclassified as B1 would cause the EPE to
    attempt neighbor-absorb on what is actually a rack-level outage, wasting
    time and likely failing anyway.  This must never happen.
    """
    tier = fc.classify(_event(signal))
    assert tier >= BlastRadius.B1, (
        f"DANGEROUS: {signal.value} was classified as {tier.name} (below B1)"
    )
    assert tier == BlastRadius.B4, (
        f"INCORRECT: {signal.value} classified as {tier.name}, expected B4"
    )


def test_unknown_signal_conservative_default(fc: FailureClassifier) -> None:
    """UNKNOWN maps to B2 — recoverable in place, not catastrophically wrong."""
    tier = fc.classify(_event(FaultSignal.UNKNOWN))
    assert tier == BlastRadius.B2


def test_classify_signal_matches_classify_event(fc: FailureClassifier) -> None:
    """classify_signal() and classify() must agree for all known signals."""
    for signal in FaultSignal:
        event_tier = fc.classify(_event(signal))
        signal_tier = fc.classify_signal(signal)
        assert event_tier == signal_tier, (
            f"Mismatch for {signal}: classify={event_tier}, classify_signal={signal_tier}"
        )
