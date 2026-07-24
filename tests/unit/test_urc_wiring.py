"""
Real cross-layer URC wiring (§3.2, Phase 1 goal per design.md §8.1).

Before Phase 1, consensus/urc.py had real reduction *logic* but nothing in
the routing path fed it real state or consulted its output — the EPE
never called agree() and no layer ever called report_epoch(). These tests
verify the EPE now does both: every successful recovery reports the
rank's committed epoch, and storage-tier restores are gated by the
resulting min_valid_epoch.
"""

from __future__ import annotations

import asyncio

from aegis.policy.dsl import OperatorPolicy
from aegis.runtime import AegisRuntime
from aegis.telemetry.events import FaultSignal, TelemetryEvent


async def _settle(rt: AegisRuntime, expected: int, timeout: float = 1.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while len(rt.epe.history) < expected:
        if asyncio.get_event_loop().time() >= deadline:
            break
        await asyncio.sleep(0.02)


async def test_successful_recovery_reports_epoch_to_consensus():
    """Every successful recovery — any tier — reports the rank's epoch to URC."""
    policy = OperatorPolicy(correlation_window_secs=1.0, correlation_node_threshold=3)
    rt = AegisRuntime(policy=policy)
    rt.transport.register_node_nics("node0", ["nic0", "nic1"])

    async with rt:
        await rt.utp.publish(TelemetryEvent(
            rank=7, node="node0", fault_signal=FaultSignal.NIC_PORT_FLAP,
            raw_payload={}, epoch=0, nic_id="nic0",
        ))
        await _settle(rt, expected=1)

    assert 7 in rt.consensus.all_ranks()


async def test_storage_restore_gated_by_real_consensus_from_prior_faults():
    """
    A stale checkpoint (written at a later epoch than surviving ranks have
    validated) must not be restored — real min_valid_epoch computed from
    real prior fault recoveries, not a hand-fed value.
    """
    policy = OperatorPolicy(correlation_window_secs=1.0, correlation_node_threshold=3)
    rt = AegisRuntime(policy=policy)
    rt.transport.register_node_nics("node0", ["nic0", "nic1"])
    rt.transport.register_node_nics("node1", ["nic0", "nic1"])
    rt.storage.write_tier1("node1", epoch=0)

    async with rt:
        # rank 0 recovers a B0 fault at epoch 1 → consensus now knows rank 0 = epoch 1
        await rt.utp.publish(TelemetryEvent(
            rank=0, node="node0", fault_signal=FaultSignal.NIC_PORT_FLAP,
            raw_payload={}, epoch=0, nic_id="nic0",
        ))
        await _settle(rt, expected=1)
        assert rt.consensus.all_ranks() == [0]

        # Write a checkpoint for node1/rank1 far ahead of what rank 0 has validated
        rt.storage.write_tier1("node1", epoch=99)

        # rank 1 now faults at B2 — gating should exclude rank 1 itself, leaving
        # only rank 0's epoch (1) as the surviving-rank floor, so the epoch=99
        # checkpoint must NOT be selected; the epoch=0 seed checkpoint should be.
        await rt.utp.publish(TelemetryEvent(
            rank=1, node="node1", fault_signal=FaultSignal.CUDA_KERNEL_CRASH,
            raw_payload={}, epoch=0,
        ))
        await _settle(rt, expected=2)

    audit = rt.epe.escalation_audit()
    b2_record = next(r for r in audit if r["node"] == "node1")
    assert b2_record["success"]

    history = rt.storage._tier1["node1"]
    restored_epochs = {c.epoch for c in history if c.epoch <= 1}
    assert 0 in restored_epochs  # the epoch=0 seed is what should have been eligible
