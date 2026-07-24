"""
E4 — Operator dashboard (design.md §4 Layer E4, Phase 2).

Text/JSON report only, no daemon (ui.md §6 explicitly rejects a
daemon/sidecar) — these tests cover the in-process aegis.dashboard() API
and the underlying build_report()/render_text() functions it wraps.
"""

from __future__ import annotations

import asyncio

import pytest

import aegis
from aegis.dashboard import build_report, render_text
from aegis.policy.dsl import OperatorPolicy
from aegis.runtime import AegisRuntime
from chaos_inject.faults import FaultSpec
from chaos_inject.harness import ChaosHarness
from aegis.telemetry.events import FaultSignal


@pytest.fixture(autouse=True)
def reset_aegis():
    aegis._reset()
    yield
    aegis._reset()


async def _settle(rt: AegisRuntime, expected: int, timeout: float = 1.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while len(rt.epe.history) < expected:
        if asyncio.get_event_loop().time() >= deadline:
            break
        await asyncio.sleep(0.02)


def test_dashboard_before_init_raises():
    with pytest.raises(RuntimeError):
        aegis.dashboard()


def test_dashboard_invalid_fmt_raises():
    aegis.init()
    with pytest.raises(ValueError):
        aegis.dashboard(fmt="xml")


def test_dashboard_text_is_nonempty_string():
    aegis.init()
    report = aegis.dashboard()
    assert isinstance(report, str)
    assert "AEGIS Operator Dashboard" in report


def test_dashboard_json_has_expected_shape():
    aegis.init()
    report = aegis.dashboard(fmt="json")
    assert isinstance(report, dict)
    for key in ("current_epoch", "faults_processed", "by_tier", "kpi", "recent_faults"):
        assert key in report


async def test_build_report_reflects_real_fault_and_kpi():
    policy = OperatorPolicy(correlation_window_secs=1.0, correlation_node_threshold=3)
    rt = AegisRuntime(policy=policy)
    rt.transport.register_node_nics("node0", ["nic0", "nic1"])

    async with rt:
        chaos = ChaosHarness(rt.utp, rt.epoch_service)
        await chaos.inject(FaultSpec(
            fault_signal=FaultSignal.NIC_PORT_FLAP, rank=0, node="node0", nic_id="nic0",
        ))
        await _settle(rt, expected=1)

    report = build_report(rt)
    assert report["faults_processed"] == 1
    assert report["by_tier"]["B0"]["count"] == 1
    assert report["kpi"]["faults_handled"] == 1

    text = render_text(report)
    assert "B0" in text
    assert "1" in text


def test_dashboard_reports_zero_escalation_violations_for_clean_run():
    aegis.init()
    report = aegis.dashboard(fmt="json")
    assert report["escalation_invariant_violations"] == 0
