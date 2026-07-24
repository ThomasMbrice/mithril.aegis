"""
E3 — KPIMeter wiring into the EPE (design.md §4 Layer E3, Phase 2).

Before this, KPIMeter.record() was defined but never called anywhere in
the codebase — aegis.status()["kpi"] always reported zero. These tests
verify every successful recovery now feeds the meter automatically, using
each layer's real measured recovery_secs (not a canned constant).
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


async def test_successful_recovery_records_kpi():
    policy = OperatorPolicy(correlation_window_secs=1.0, correlation_node_threshold=3, gpu_count=16)
    rt = AegisRuntime(policy=policy)
    rt.transport.register_node_nics("node0", ["nic0", "nic1"])

    async with rt:
        await rt.utp.publish(TelemetryEvent(
            rank=0, node="node0", fault_signal=FaultSignal.NIC_PORT_FLAP,
            raw_payload={}, epoch=0, nic_id="nic0",
        ))
        await _settle(rt, expected=1)

    summary = rt.kpi.summary()
    assert summary["faults_handled"] == 1
    assert summary["by_tier"]["B0"]["count"] == 1


async def test_kpi_uses_real_measured_recovery_time_not_zero_stub():
    """recovery_time_secs fed to KPIMeter comes from the layer's real clock."""
    policy = OperatorPolicy(correlation_window_secs=1.0, correlation_node_threshold=3)
    rt = AegisRuntime(policy=policy)
    rt.compute.register_neighbor("node0", "node1")

    async with rt:
        await rt.utp.publish(TelemetryEvent(
            rank=0, node="node0", fault_signal=FaultSignal.NODE_CRASH,
            raw_payload={}, epoch=0,
        ))
        await _settle(rt, expected=1)

    assert len(rt.kpi._records) == 1
    record = rt.kpi._records[0]
    # Real torch tensor ops take *some* measurable time, even on CPU/MPS.
    assert record.recovery_time_secs > 0.0
    assert record.recovery_time_secs < record.baseline_time_secs


async def test_failed_recovery_does_not_record_kpi():
    """Only successful recoveries feed the meter — no phantom savings on failure."""
    policy = OperatorPolicy(correlation_window_secs=1.0, correlation_node_threshold=3)
    rt = AegisRuntime(policy=policy)
    # Nothing registered anywhere — every tier will fail to handle this fault.

    async with rt:
        await rt.utp.publish(TelemetryEvent(
            rank=0, node="ghost", fault_signal=FaultSignal.NIC_PORT_FLAP,
            raw_payload={}, epoch=0, nic_id="nic0",
        ))
        await _settle(rt, expected=1)

    assert rt.kpi.summary()["faults_handled"] == 0


async def test_gpu_count_from_policy_scales_kpi():
    policy_small = OperatorPolicy(correlation_window_secs=1.0, correlation_node_threshold=3, gpu_count=1)
    policy_big = OperatorPolicy(correlation_window_secs=1.0, correlation_node_threshold=3, gpu_count=100)

    for policy, expected_scale in ((policy_small, 1), (policy_big, 100)):
        rt = AegisRuntime(policy=policy)
        rt.transport.register_node_nics("node0", ["nic0", "nic1"])
        async with rt:
            await rt.utp.publish(TelemetryEvent(
                rank=0, node="node0", fault_signal=FaultSignal.NIC_PORT_FLAP,
                raw_payload={}, epoch=0, nic_id="nic0",
            ))
            await _settle(rt, expected=1)
        record = rt.kpi._records[0]
        assert record.gpu_count == expected_scale
