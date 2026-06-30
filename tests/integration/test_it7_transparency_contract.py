"""
IT-7 — Transparency contract test suite (ui.md §5).

Tests the five promises AEGIS makes to users:

  #1 No silent inactivity: after init(), all (non-disabled) hooks are active.
  #2 No silent semantic change: fidelity_flag on degraded checkpoints (covered
     by IT-4; referenced here for completeness).
  #3 Always escapable: disable() returns the job to native PyTorch path.
  #4 Always inspectable: status() / explain() expose every decision.
  #5 Fail-safe, not fail-active: a faulty subscriber does not kill the
     dispatch loop; the EPE still processes subsequent events.
  +  observe_only contract: EPE populates history but takes no recovery action.
"""

from __future__ import annotations

import asyncio

import pytest

import aegis
import aegis._state as _state
from aegis.policy.dsl import OperatorPolicy
from aegis.runtime import AegisRuntime
from chaos_inject.faults import FaultSpec
from chaos_inject.harness import ChaosHarness
from aegis.telemetry.events import FaultSignal


# ---------------------------------------------------------------------------
# Autouse fixture: guarantee clean state between tests


@pytest.fixture(autouse=True)
def reset_aegis():
    aegis._reset()
    yield
    aegis._reset()


# ---------------------------------------------------------------------------
# Helpers


def _full_runtime() -> AegisRuntime:
    """Return a started-but-not-yet-running runtime with 2-node topology."""
    rt = AegisRuntime(policy=OperatorPolicy(
        correlation_window_secs=1.0,
        correlation_node_threshold=3,
    ))
    rt.transport.register_node_nics("node0", ["nic0", "nic1"])
    rt.transport.register_node_nics("node1", ["nic0", "nic1"])
    rt.compute.register_neighbor("node0", "node1")
    rt.compute.register_neighbor("node1", "node0")
    rt.storage.write_tier1("node0", epoch=0)
    rt.storage.write_tier1("node1", epoch=0)
    rt.storage.write_tier2("node0", epoch=0)
    rt.storage.write_tier2("node1", epoch=0)
    rt.storage.write_tier3(epoch=0)
    rt.consensus.report_epoch(rank=0, epoch=0)
    rt.consensus.report_epoch(rank=1, epoch=0)
    return rt


async def _settle(rt: AegisRuntime, expected: int, timeout: float = 2.0) -> None:
    """Wait until the EPE has processed at least ``expected`` events."""
    deadline = asyncio.get_event_loop().time() + timeout
    while len(rt.epe.history) < expected:
        if asyncio.get_event_loop().time() >= deadline:
            break
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# Promise #1 — No silent inactivity


def test_promise1_all_hooks_active_after_init():
    """After init(), status() reports all five hooks as active."""
    aegis.init()
    s = aegis.status()
    assert s["initialized"] is True
    expected = {"transport", "compute", "checkpoint", "telemetry", "policy"}
    assert set(s["active_hooks"]) == expected


def test_promise1_disabled_hook_absent():
    """Hooks listed in disable= are absent; the rest are active."""
    aegis.init(disable=["checkpoint"])
    s = aegis.status()
    assert "checkpoint" not in s["active_hooks"]
    # Remaining four still active
    for hook in ("transport", "compute", "telemetry", "policy"):
        assert hook in s["active_hooks"]


def test_promise1_invalid_hook_raises_not_silently_skips():
    """A typo in disable= raises immediately — never silently accepted."""
    with pytest.raises(ValueError):
        aegis.init(disable=["chekpoint"])  # deliberate typo


# ---------------------------------------------------------------------------
# Promise #3 — Always escapable


def test_promise3_disable_stops_runtime():
    """disable() clears initialized state and nulls runtime."""
    aegis.init()
    assert _state.initialized is True
    aegis.disable()
    assert _state.initialized is False
    assert _state.runtime is None


def test_promise3_disable_is_idempotent():
    """Calling disable() twice does not raise."""
    aegis.init()
    aegis.disable()
    aegis.disable()  # should be a no-op


# ---------------------------------------------------------------------------
# Promise #4 — Always inspectable


def test_promise4_status_returns_expected_shape():
    """status() exposes epoch, hooks, mode, and kpi."""
    aegis.init()
    s = aegis.status()
    for key in ("initialized", "mode", "active_hooks", "current_epoch", "kpi"):
        assert key in s, f"Expected key {key!r} missing from status()"


def test_promise4_explain_answers_no_fault_gracefully():
    """explain() before any fault returns a human-readable message."""
    aegis.init()
    result = aegis.explain()
    assert "message" in result
    assert result["message"] == "No faults processed yet."


# ---------------------------------------------------------------------------
# Promise #5 — Fail-safe, not fail-active


async def test_promise5_faulty_subscriber_does_not_kill_dispatch():
    """
    A subscriber that always raises must not prevent the EPE from
    processing subsequent events via the UTP dispatch loop.
    """
    rt = _full_runtime()
    async with rt:
        # Insert a bad subscriber *before* the EPE subscriber
        def bad_subscriber(event):
            raise RuntimeError("subscriber intentionally broken")

        rt.utp._subscribers.insert(0, bad_subscriber)

        chaos = ChaosHarness(rt.utp, rt.epoch_service)
        await chaos.inject(FaultSpec(
            fault_signal=FaultSignal.NIC_PORT_FLAP,
            rank=0,
            node="node0",
            nic_id="nic0",
        ))
        await _settle(rt, expected=1)

        # The EPE must still have processed the event despite bad_subscriber
        assert len(rt.epe.history) == 1, (
            "EPE did not process the event — faulty subscriber killed dispatch"
        )


async def test_promise5_multiple_faults_after_bad_subscriber():
    """Verify multiple consecutive faults are all processed despite bad subscriber."""
    rt = _full_runtime()
    async with rt:
        def bad_subscriber(event):
            raise RuntimeError("always broken")

        rt.utp._subscribers.insert(0, bad_subscriber)

        chaos = ChaosHarness(rt.utp, rt.epoch_service)
        for i in range(3):
            await chaos.inject(FaultSpec(
                fault_signal=FaultSignal.NIC_PORT_FLAP,
                rank=i % 2,
                node=f"node{i % 2}",
                nic_id="nic0",
            ))

        await _settle(rt, expected=3)
        assert len(rt.epe.history) == 3


# ---------------------------------------------------------------------------
# observe_only contract


def test_observe_only_mode_is_recorded_in_state():
    """init(mode='observe_only') sets the EPE flag and status reflects mode."""
    aegis.init(mode="observe_only")
    assert _state.runtime.epe._observe_only is True  # type: ignore[union-attr]
    s = aegis.status()
    assert s["mode"] == "observe_only"


async def test_observe_only_populates_history_but_no_recovery():
    """
    In observe_only mode, the EPE appends to history (so explain() works)
    but the result message marks it as a no-action observation.
    """
    rt = _full_runtime()
    rt.epe._observe_only = True

    async with rt:
        chaos = ChaosHarness(rt.utp, rt.epoch_service)
        await chaos.inject(FaultSpec(
            fault_signal=FaultSignal.NIC_PORT_FLAP,
            rank=0,
            node="node0",
            nic_id="nic0",
        ))
        await _settle(rt, expected=1)

        assert len(rt.epe.history) == 1
        record = rt.epe.history[0]
        assert record.result is not None
        assert record.result.success is True
        assert "observe_only" in record.result.message


def test_observe_only_explain_flag():
    """aegis.explain() in observe_only mode has observe_only=True."""
    aegis.init(mode="observe_only")
    result = aegis.explain()
    assert result["observe_only"] is True


async def test_observe_only_vs_active_modes_via_public_api():
    """
    With mode='observe_only', init() starts without error and explain()
    correctly reports observe_only=True even after a no-fault state check.
    """
    aegis.init(mode="observe_only")
    s = aegis.status()
    assert s["initialized"] is True
    assert s["mode"] == "observe_only"

    ex = aegis.explain()
    assert ex["observe_only"] is True
    assert ex["message"] == "No faults processed yet."


# ---------------------------------------------------------------------------
# Full round-trip: init → inject → explain → disable


async def test_full_transparency_roundtrip():
    """
    End-to-end transparency contract:
    1. init()  → hooks active
    2. Inject a fault via the runtime UTP directly
    3. explain() → describes the fault
    4. disable() → state cleared
    """
    aegis.init()
    rt = _state.runtime
    assert rt is not None

    # Seed the runtime topology so the fault can be handled
    rt.transport.register_node_nics("node0", ["nic0", "nic1"])
    rt.storage.write_tier1("node0", epoch=0)
    rt.storage.write_tier2("node0", epoch=0)
    rt.storage.write_tier3(epoch=0)

    # Inject a fault through the live runtime's UTP
    loop = _state._loop
    assert loop is not None

    from aegis.telemetry.events import TelemetryEvent

    event = TelemetryEvent(
        rank=0,
        node="node0",
        fault_signal=FaultSignal.NIC_PORT_FLAP,
        raw_payload={"injected": True},
        nic_id="nic0",
        epoch=rt.epoch_service.current(),
        source="test",
    )
    future = asyncio.run_coroutine_threadsafe(rt.utp.publish(event), loop)
    future.result(timeout=2.0)

    # Wait for the EPE to process it
    deadline = asyncio.get_event_loop().time() + 2.0
    while len(rt.epe.history) == 0:
        if asyncio.get_event_loop().time() >= deadline:
            break
        await asyncio.sleep(0.05)

    assert len(rt.epe.history) == 1, "EPE did not process the injected fault"

    ex = aegis.explain()
    assert ex["signal"] == FaultSignal.NIC_PORT_FLAP.value
    assert ex["node"] == "node0"
    assert ex["observe_only"] is False

    aegis.disable()
    assert _state.initialized is False
