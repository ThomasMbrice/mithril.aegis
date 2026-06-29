"""
IT-1 — NIC flap during AllReduce.

Scenario: single NIC port flaps while an AllReduce is in progress.
Pass criterion (§5.2):
  - B0 fast-path handles it in transport layer
  - NO neighbor-absorb (B1) triggered
  - NO checkpoint rollback triggered
  - Fault is NOT escalated beyond B0
"""

from __future__ import annotations

import asyncio

from aegis.runtime import AegisRuntime
from aegis.telemetry.events import FaultSignal, TelemetryEvent
from chaos_inject.faults import FaultSpec
from chaos_inject.harness import ChaosHarness


async def _settle(rt: AegisRuntime, expected: int = 1, timeout: float = 0.5) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while len(rt.epe.history) < expected:
        if asyncio.get_event_loop().time() >= deadline:
            break
        await asyncio.sleep(0.02)


async def test_nic_flap_stays_at_b0(runtime: AegisRuntime, chaos: ChaosHarness) -> None:
    await chaos.inject(FaultSpec(
        fault_signal=FaultSignal.NIC_PORT_FLAP,
        rank=0,
        node="node0",
        nic_id="nic0",
    ))
    await _settle(runtime)

    audit = runtime.epe.escalation_audit()
    assert len(audit) == 1

    r = audit[0]
    assert r["signal"] == FaultSignal.NIC_PORT_FLAP.value
    assert r["original_tier"] == "B0", "NIC flap should classify as B0"
    assert r["final_tier"] == "B0", f"NIC flap escalated to {r['final_tier']} — unexpected"
    assert not r["escalated"], "B0 should not escalate for a simple NIC flap"
    assert r["success"], "B0 transport recovery should succeed"


async def test_nic_flap_does_not_trigger_neighbor_absorb(runtime: AegisRuntime) -> None:
    """B1 (ComputeLayer) must never be invoked for a B0 fault."""
    await runtime.utp.publish(TelemetryEvent(
        rank=0, node="node0", fault_signal=FaultSignal.NIC_PORT_FLAP,
        raw_payload={}, epoch=0, nic_id="nic0",
    ))
    await _settle(runtime)

    audit = runtime.epe.escalation_audit()
    assert all(r["final_tier"] == "B0" for r in audit), (
        "Neighbor-absorb (B1+) triggered by a NIC flap — should never happen"
    )


async def test_multiple_nic_flaps_are_independent(runtime: AegisRuntime, chaos: ChaosHarness) -> None:
    """Three NIC flaps on different nodes should each be handled at B0."""
    for i in range(3):
        await chaos.inject(FaultSpec(
            fault_signal=FaultSignal.NIC_PORT_FLAP,
            rank=i,
            node=f"node{i}",
            nic_id="nic0",
        ))
    await _settle(runtime, expected=3)

    audit = runtime.epe.escalation_audit()
    assert len(audit) == 3
    assert all(r["final_tier"] == "B0" for r in audit)
    assert all(r["success"] for r in audit)
