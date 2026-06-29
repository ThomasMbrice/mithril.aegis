"""
IT-2 — Node death during active R²CCL migration.

Scenario: a node crashes while a NIC migration is in progress on another node.
Pass criterion (§5.2):
  - Epoch boundary is clean: each fault gets its own epoch
  - Neighbor absorbs from a consistent state (no torn gradient)
  - No double-escalation or interference between the two concurrent faults
  - URC reports surviving ranks from a consistent epoch boundary
"""

from __future__ import annotations

import asyncio

from aegis.runtime import AegisRuntime
from aegis.telemetry.events import FaultSignal, TelemetryEvent
from chaos_inject.faults import FaultSpec
from chaos_inject.harness import ChaosHarness


async def _settle(rt: AegisRuntime, expected: int, timeout: float = 0.5) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while len(rt.epe.history) < expected:
        if asyncio.get_event_loop().time() >= deadline:
            break
        await asyncio.sleep(0.02)


async def test_node_death_gets_clean_epoch(runtime: AegisRuntime, chaos: ChaosHarness) -> None:
    """Each fault must increment the epoch — no epoch is shared between faults."""
    await chaos.inject(FaultSpec(
        fault_signal=FaultSignal.NIC_PORT_FLAP, rank=0, node="node0", nic_id="nic0",
    ))
    await chaos.inject(FaultSpec(
        fault_signal=FaultSignal.NODE_CRASH, rank=1, node="node1",
    ))
    await _settle(runtime, expected=2)

    history = runtime.epe.history
    assert len(history) == 2
    epochs = [r.epoch for r in history]
    assert len(set(epochs)) == 2, f"Epoch collision: {epochs}"
    assert epochs[1] > epochs[0], "Epochs must be strictly increasing"


async def test_concurrent_b0_and_b1_both_succeed(runtime: AegisRuntime) -> None:
    """
    Inject B0 and B1 near-simultaneously.
    Both should succeed at their respective tiers with no interference.
    """
    await asyncio.gather(
        runtime.utp.publish(TelemetryEvent(
            rank=0, node="node0", fault_signal=FaultSignal.NIC_PORT_FLAP,
            raw_payload={}, epoch=0, nic_id="nic0",
        )),
        runtime.utp.publish(TelemetryEvent(
            rank=1, node="node1", fault_signal=FaultSignal.NODE_CRASH,
            raw_payload={}, epoch=0,
        )),
    )
    await _settle(runtime, expected=2)

    audit = runtime.epe.escalation_audit()
    assert len(audit) == 2

    tiers = {r["signal"]: r["final_tier"] for r in audit}
    assert tiers[FaultSignal.NIC_PORT_FLAP.value] == "B0"
    assert tiers[FaultSignal.NODE_CRASH.value] == "B1"
    assert all(r["success"] for r in audit)


async def test_urc_excludes_dead_rank(runtime: AegisRuntime) -> None:
    """
    After a node crash, URC consensus must use surviving ranks only.
    The dead rank's epoch must not count in the global minimum.
    """
    # Seed: rank 0 at epoch 5, rank 1 at epoch 5
    runtime.consensus.report_epoch(rank=0, epoch=5)
    runtime.consensus.report_epoch(rank=1, epoch=5)

    # rank 1 dies
    runtime.consensus.clear_rank(rank=1)

    decision = runtime.consensus.agree(active_ranks=[0], current_epoch=6)
    assert decision.agreed
    assert decision.min_valid_epoch == 5
    assert 1 not in decision.surviving_ranks
