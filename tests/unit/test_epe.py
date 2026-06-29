"""
EPE unit tests — routing, escalation invariant, correlation window.

These tests use the full AegisRuntime (from conftest) with stub layers so
the routing logic is exercised end-to-end without GPU hardware.
"""

from __future__ import annotations

import asyncio

import pytest

from aegis.policy.dsl import OperatorPolicy
from aegis.runtime import AegisRuntime
from aegis.telemetry.events import FaultSignal, TelemetryEvent
from chaos_inject.faults import BurstSpec, FaultSpec
from chaos_inject.harness import ChaosHarness


def _event(
    signal: FaultSignal,
    node: str = "node0",
    rank: int = 0,
    rack_id: str | None = None,
) -> TelemetryEvent:
    return TelemetryEvent(
        rank=rank,
        node=node,
        fault_signal=signal,
        raw_payload={},
        epoch=0,
        rack_id=rack_id,
    )


async def _wait_for_processing(rt: AegisRuntime, expected: int, timeout: float = 0.5) -> None:
    """Poll until the EPE has processed at least *expected* faults."""
    deadline = asyncio.get_event_loop().time() + timeout
    while len(rt.epe.history) < expected:
        if asyncio.get_event_loop().time() >= deadline:
            break
        await asyncio.sleep(0.02)


# -----------------------------------------------------------------------
# Basic routing per tier
# -----------------------------------------------------------------------

async def test_b0_routes_to_transport_layer(runtime: AegisRuntime) -> None:
    await runtime.utp.publish(_event(FaultSignal.NIC_PORT_FLAP, node="node0"))
    await _wait_for_processing(runtime, 1)

    audit = runtime.epe.escalation_audit()
    assert len(audit) == 1
    assert audit[0]["original_tier"] == "B0"
    assert audit[0]["final_tier"] == "B0"
    assert audit[0]["success"]
    assert not audit[0]["escalated"]


async def test_b1_routes_to_compute_layer(runtime: AegisRuntime) -> None:
    await runtime.utp.publish(_event(FaultSignal.NODE_CRASH, node="node0"))
    await _wait_for_processing(runtime, 1)

    audit = runtime.epe.escalation_audit()
    assert audit[0]["original_tier"] == "B1"
    assert audit[0]["final_tier"] == "B1"
    assert audit[0]["success"]
    assert audit[0]["degraded"]  # MeCeFO fallback = degraded=True


async def test_b2_routes_to_storage_tier1(runtime: AegisRuntime) -> None:
    await runtime.utp.publish(_event(FaultSignal.CUDA_KERNEL_CRASH, node="node0"))
    await _wait_for_processing(runtime, 1)

    audit = runtime.epe.escalation_audit()
    assert audit[0]["original_tier"] == "B2"
    assert audit[0]["final_tier"] == "B2"
    assert audit[0]["success"]


async def test_b4_routes_to_storage_tier3(runtime: AegisRuntime) -> None:
    await runtime.utp.publish(_event(FaultSignal.RACK_POWER_LOSS, node="node0"))
    await _wait_for_processing(runtime, 1)

    audit = runtime.epe.escalation_audit()
    assert audit[0]["original_tier"] == "B4"
    assert audit[0]["final_tier"] == "B4"
    assert audit[0]["success"]


# -----------------------------------------------------------------------
# One-directional escalation invariant
# -----------------------------------------------------------------------

async def test_escalation_invariant_holds_for_all_signals(runtime: AegisRuntime) -> None:
    """
    Property: final_tier >= original_tier for every processed fault.
    De-escalation must be structurally impossible.
    """
    signals = [
        FaultSignal.NIC_PORT_FLAP,
        FaultSignal.NODE_CRASH,
        FaultSignal.CUDA_KERNEL_CRASH,
        FaultSignal.NODE_UNRECOVERABLE,
        FaultSignal.RACK_POWER_LOSS,
    ]
    for sig in signals:
        await runtime.utp.publish(_event(sig, node="node0"))

    await _wait_for_processing(runtime, len(signals))

    for record in runtime.epe.escalation_audit():
        assert record["escalation_valid"], (
            f"INVARIANT VIOLATED: {record['signal']} "
            f"went from {record['original_tier']} → {record['final_tier']}"
        )


# -----------------------------------------------------------------------
# Escalation on missing prerequisite (no multi-NIC → fallthrough from B0)
# -----------------------------------------------------------------------

async def test_b0_escalates_when_no_backup_nic() -> None:
    """Node registered with only one NIC cannot absorb B0 — escalates to B1."""
    policy = OperatorPolicy(correlation_window_secs=1.0, correlation_node_threshold=3)
    rt = AegisRuntime(policy=policy)
    # Single NIC only — B0 can_handle() returns False
    rt.transport.register_node_nics("loner", ["nic0"])
    rt.compute.register_neighbor("loner", "node1")
    rt.storage.write_tier1("loner", epoch=0)
    rt.storage.write_tier2("loner", epoch=0)
    rt.storage.write_tier3(epoch=0)

    async with rt:
        await rt.utp.publish(_event(FaultSignal.NIC_PORT_FLAP, node="loner"))
        await _wait_for_processing(rt, 1)

    audit = rt.epe.escalation_audit()
    assert audit[0]["original_tier"] == "B0"
    # B0 can't handle (only 1 NIC) → should escalate
    assert audit[0]["escalated"]
    assert audit[0]["escalation_valid"]


# -----------------------------------------------------------------------
# Correlation window (§3.3)
# -----------------------------------------------------------------------

async def test_correlation_window_escalates_b1_burst_to_b4() -> None:
    """
    Three NODE_CRASH events in the same rack within the window → re-classify B4.
    """
    policy = OperatorPolicy(correlation_window_secs=1.0, correlation_node_threshold=3)
    rt = AegisRuntime(policy=policy)
    for i in range(6):
        rt.compute.register_neighbor(f"node{i}", f"node{(i+1) % 6}")
    rt.storage.write_tier3(epoch=0)

    async with rt:
        for i in range(4):  # 4 crashes in same rack → crosses threshold of 3
            await rt.utp.publish(
                TelemetryEvent(
                    rank=i, node=f"node{i}", fault_signal=FaultSignal.NODE_CRASH,
                    raw_payload={}, epoch=0, rack_id="rack0",
                )
            )
            await asyncio.sleep(0.05)

        await _wait_for_processing(rt, 4)

    audit = rt.epe.escalation_audit()
    b4_records = [r for r in audit if r["final_tier"] == "B4"]
    assert len(b4_records) >= 1, (
        "Correlation window did not trigger B1 → B4 re-classification"
    )
    for r in audit:
        assert r["escalation_valid"]
